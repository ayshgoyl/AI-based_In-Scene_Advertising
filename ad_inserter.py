import cv2
import numpy as np
import os
import torch
import copy
import time
from ultralytics import YOLO, SAM, FastSAM

class AdPlacementSystem:
    def __init__(self, target_fps=None, logger=None):
        self.target_fps = target_fps
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log("info", "Configuration", "Compute device initialized", device=str(self.device))

        # 1. Initialize YOLOv8 Segmentation model for Object Tracking & Human Occlusion
        self._log("info", "EventFlow", "Loading YOLOv8-Seg model...")
        t0 = time.time()
        try:
            self.yolo_model = YOLO("yolov8n-seg.pt") 
            self._log("info", "Performance", "YOLOv8-Seg loaded successfully", duration_seconds=round(time.time() - t0, 3))
        except Exception as e:
            self._log("error", "Error", "YOLO init error", exception=str(e))

        # 2. Initialize SAM for exact Segmentation of the target surface
        self.sam_model = None
        self._log("info", "EventFlow", "Loading FastSAM model...")
        t0 = time.time()
        try:
            self.sam_model = FastSAM("FastSAM-s.pt")
            self._log("info", "Performance", "FastSAM loaded successfully", duration_seconds=round(time.time() - t0, 3))
        except Exception as e:
            self._log("error", "Error", "SAM init error", exception=str(e))

        # 3. Initialize MiDaS for Depth Estimation
        self._log("info", "EventFlow", "Loading MiDaS depth model...")
        t0 = time.time()
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.midas_transform = midas_transforms.small_transform
            self._log("info", "Performance", "MiDaS loaded successfully", duration_seconds=round(time.time() - t0, 3))
        except Exception as e:
            self._log("error", "Error", "MiDaS init error", exception=str(e))
            self.midas = None

    def _log(self, level, category, message, **kwargs):
        if self.logger:
            getattr(self.logger, level)(category, message, **kwargs)
        else:
            print(f"[{level.upper()}] [{category}] {message} | {kwargs}", flush=True)

    def process_video(self, video_path, logo_path, output_path, debug_dir=None):
        self._log("info", "EventFlow", "Opening video file for processing", video_path=video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log("error", "Error", "Failed to open video", video_path=video_path)
            return
            
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is None:
            self._log("error", "Error", "Failed to open logo", logo_path=logo_path)
            return
        if logo.shape[2] == 3:
            logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)
            
        logo_h, logo_w = logo.shape[:2]
        logo_pts = np.array([[0, 0], [logo_w - 1, 0], [logo_w - 1, logo_h - 1], [0, logo_h - 1]], dtype=np.float32)

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = self.target_fps if self.target_fps else original_fps
        frame_read_interval = int(original_fps / fps) if fps and original_fps >= fps else 1
        
        self._log("info", "Variable", "Video framerate calculated", original_fps=original_fps, target_fps=fps, frame_read_interval=frame_read_interval)

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # MASSIVE SPEED OPTIMIZATION: Process at 640p max width for CPU real-time speeds
        max_w = 640
        if original_width > max_w:
            scale = max_w / float(original_width)
            width = max_w
            height = int(original_height * scale)
        else:
            width = original_width
            height = original_height
            scale = 1.0
            
        self._log("info", "Variable", "Video dimensions resolved", original_dims=f"{original_width}x{original_height}", new_dims=f"{width}x{height}", scale=scale)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self._log("info", "EventFlow", "Starting main video processing loop", total_frames=total_frames)

        first_frame = True
        tracked_id = None
        target_obj_class = None
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_read_interval != 0:
                frame_idx += 1
                continue
                
            frame = cv2.resize(frame, (width, height))

            if debug_dir and first_frame:
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"extracted_frame_{frame_idx:05d}.jpg"), frame)

            # --- 1. Object Detection for Occlusion and Negative Space ---
            results = self.yolo_model.predict(frame, verbose=False)
            
            objects_mask = np.zeros((height, width), dtype=np.uint8)
            person_mask = np.zeros((height, width), dtype=np.uint8)
            
            if len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                if len(confs) > 0 and frame_idx % 5 == 0:
                    avg_conf = float(np.mean(confs))
                    self._log("info", "Performance", "YOLO object detection confidence", avg_confidence=round(avg_conf, 3), objects_detected=len(confs))
                
                for c_idx, cls_val in enumerate(classes):
                    m = masks[c_idx].astype(np.uint8)
                    m = cv2.resize(m, (width, height))
                    m_scaled = (m * 255).astype(np.uint8)
                    
                    objects_mask = cv2.bitwise_or(objects_mask, m_scaled)
                    if int(cls_val) == 0: # 0 is person
                        person_mask = cv2.bitwise_or(person_mask, m_scaled)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- 2. Surface/Banner Detection & Tracking ---
            if first_frame:
                p0 = None
                
                # A: Semantic Rectangularity via FastSAM
                # Analyzes all isolated objects in the scene and scores how physically rectangular they are 
                # (1.0 = perfect rectangle banner, 0.4 = human/tree). Prioritizes largest static rectangular objects.
                if self.sam_model is not None:
                    sam_res = self.sam_model(frame, verbose=False)
                    
                    best_score = 0
                    best_rect = None
                    screen_area = width * height
                    
                    if len(sam_res) > 0 and sam_res[0].masks is not None:
                        sam_masks = sam_res[0].masks.data.cpu().numpy()
                        
                        for m in sam_masks:
                            m_uint = (m * 255).astype(np.uint8)
                            m_uint = cv2.resize(m_uint, (width, height))
                            
                            contours, _ = cv2.findContours(m_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if not contours:
                                continue
                                
                            largest = max(contours, key=cv2.contourArea)
                            area = cv2.contourArea(largest)
                            
                            # Banner must be between 1% and 30% of the scene natively
                            if (screen_area * 0.01) < area < (screen_area * 0.30):
                                rect = cv2.minAreaRect(largest)
                                box_area = rect[1][0] * rect[1][1]
                                
                                if box_area > 0:
                                    rectangularity = area / box_area # 1.0 means perfectly rectangular mask
                                    
                                    if rectangularity > 0.65:
                                        # Ensure less than 15% overlap with dynamic people
                                        overlap = cv2.bitwise_and(m_uint, person_mask)
                                        if cv2.countNonZero(overlap) < (area * 0.15):
                                            if rectangularity > best_score:
                                                best_score = rectangularity
                                                best_rect = rect
                                                
                        if best_rect is not None:
                            p0 = cv2.boxPoints(best_rect).reshape(-1, 1, 2).astype(np.float32)
                            self._log("info", "Performance", "Surface detection confidence (rectangularity via FastSAM)", rectangularity_score=round(float(best_score), 3))

                # B: Fallback to empty wall space, restricted to a reasonable size, centered.
                if p0 is None:
                    wall_mask = cv2.bitwise_not(objects_mask)
                    kernel = np.ones((11, 11), np.uint8)
                    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel)

                    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        
                        # Artificially clamp the fallback background surface size so it never engulfs the whole screen
                        if cv2.contourArea(largest) > 1000:
                            rect = cv2.minAreaRect(largest)
                            
                            rect_center = rect[0]
                            rect_w, rect_h = rect[1]
                            
                            # Cap the max dimensions of a wall ad to 40% of the screen width
                            max_ad_w = width * 0.4
                            if rect_w > max_ad_w:
                                scale = max_ad_w / rect_w
                                rect_w *= scale
                                rect_h *= scale
                                
                            clamped_rect = (rect_center, (rect_w, rect_h), rect[2])
                            p0 = cv2.boxPoints(clamped_rect).reshape(-1, 1, 2).astype(np.float32)
                            
                # Order corner points for safe tracking and homography
                if p0 is not None:
                    surface_pts = p0.reshape(4, 2)
                    s = surface_pts.sum(axis=1)
                    diff = np.diff(surface_pts, axis=1)
                    ordered = np.zeros((4, 2), dtype=np.float32)
                    ordered[0] = surface_pts[np.argmin(s)]       # Top-left
                    ordered[2] = surface_pts[np.argmax(s)]       # Bottom-right
                    ordered[1] = surface_pts[np.argmin(diff)]    # Top-right
                    ordered[3] = surface_pts[np.argmax(diff)]    # Bottom-left
                    p0 = ordered.reshape(-1, 1, 2)
                    
                prev_gray = gray.copy()
                first_frame = False

            else:
                # Track the 4 plane corners across the video using Optical Flow
                if 'p0' in locals() and p0 is not None:
                    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
                    if st is not None and np.all(st == 1):
                        p0 = p1
                        if err is not None and frame_idx % 5 == 0:
                            avg_track_err = float(np.mean(err))
                            self._log("info", "Performance", "Optical flow tracking error (inversely tracks tracking confidence)", avg_tracking_error=round(avg_track_err, 3))
                prev_gray = gray.copy()

            # --- 3. Perspective, Depth, Blending ---
            if 'p0' in locals() and p0 is not None:
                H, _ = cv2.findHomography(logo_pts, p0.reshape(4, 2))
                
                if H is not None:
                    depth_map = None
                    if self.midas:
                        img_tensor = self.midas_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(self.device)
                        with torch.no_grad():
                            prediction = self.midas(img_tensor)
                            prediction = torch.nn.functional.interpolate(
                                prediction.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False
                            ).squeeze()
                        depth_map = prediction.cpu().numpy()
                        if depth_map.max() > depth_map.min():
                            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

                    # Blending & Rendering
                    frame = self.blend_ad(frame, logo, H, None, person_mask, depth_map)
                    
                    if debug_dir and frame_idx == 1:
                        warped_ad = cv2.warpPerspective(logo, H, (width, height))
                        cv2.imwrite(os.path.join(debug_dir, "warped_ad.png"), warped_ad)

            out.write(frame)
            if frame_idx % 5 == 0:
                self._log("info", "Performance", "Processed frames milestone", processed=frame_idx, total=total_frames, pct=round(frame_idx/max(1, total_frames)*100, 1))
            frame_idx += 1

        cap.release()
        out.release()
        self._log("info", "EventFlow", "Video processing loop completed successfully")

    def blend_ad(self, frame, logo, H, sam_mask, person_mask, depth_map):
        h, w = frame.shape[:2]
        
        warped_ad = cv2.warpPerspective(logo, H, (w, h))
        ad_alpha = warped_ad[:, :, 3]
        _, initial_mask = cv2.threshold(ad_alpha, 1, 255, cv2.THRESH_BINARY)
        
        if sam_mask is not None:
            initial_mask = cv2.bitwise_and(initial_mask, sam_mask)
            
        if person_mask is not None:
            initial_mask = cv2.bitwise_and(initial_mask, cv2.bitwise_not(person_mask))

        ad_mask_3c = cv2.cvtColor(initial_mask, cv2.COLOR_GRAY2BGR) / 255.0
        warped_ad_rgb = warped_ad[:, :, :3].astype(np.float32)
        
        if depth_map is not None:
            depth_factor = depth_map[:, :, np.newaxis]
            warped_ad_rgb = warped_ad_rgb * (0.4 + 0.6 * depth_factor)
            
        warped_ad_rgb = cv2.GaussianBlur(warped_ad_rgb, (3, 3), 0)

        frame_float = frame.astype(np.float32)
        result = frame_float * (1.0 - ad_mask_3c) + warped_ad_rgb * ad_mask_3c
        return result.astype(np.uint8)