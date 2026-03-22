import cv2
import numpy as np
import os

class AdPlacementSystem:
    def __init__(self, target_fps=None):
        self.target_fps = target_fps
        # Feature detector for tracking surfaces
        self.feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=10, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    def detect_surface(self, frame):
        """
        Detects a suitable surface (largest quadrilateral) for ad placement.
        Returns the 4 corner points of the detected surface.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # If our approximated contour has four points, we can assume it's a suitable surface
            if len(approx) == 4 and cv2.contourArea(approx) > 5000:
                # Reshape to a list of points
                pts = approx.reshape(4, 2).astype(np.float32)
                return self._order_points(pts)

        # Fallback: return a centered rectangle if no distinct quadrilateral is found
        h, w = frame.shape[:2]
        margin_x, margin_y = int(w * 0.2), int(h * 0.2)
        pts = np.array([
            [margin_x, margin_y],
            [w - margin_x, margin_y],
            [w - margin_x, h - margin_y],
            [margin_x, h - margin_y]
        ], dtype=np.float32)
        return pts

    def _order_points(self, pts):
        """Order points: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def get_features_in_poly(self, frame_gray, poly_pts):
        """Get trackable features strictly inside the detected polygon."""
        mask = np.zeros_like(frame_gray)
        cv2.fillConvexPoly(mask, poly_pts.astype(np.int32), 255)
        # Find corners inside the mask
        corners = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
        return corners

    def process_video(self, video_path, logo_path, output_path, debug_dir=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video at {video_path}")
            return

        # Read logo
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is None:
            print(f"Error opening logo at {logo_path}")
            return
            
        # Ensure logo has an alpha channel for blending
        if logo.shape[2] == 3:
            logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)

        logo_h, logo_w = logo.shape[:2]
        logo_pts = np.array([
            [0, 0],
            [logo_w - 1, 0],
            [logo_w - 1, logo_h - 1],
            [0, logo_h - 1]
        ], dtype=np.float32)

        # Video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = self.target_fps if self.target_fps else original_fps
        # Adjust frames if target_fps is set differently
        frame_read_interval = int(original_fps / fps) if fps and original_fps >= fps else 1

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use 'avc1' (H.264) so it's playable natively in browsers via the HTML5 <video> tag
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Starting video processing: {width}x{height} @ {fps} FPS")

        first_frame = True
        prev_gray = None
        p0 = None
        surface_pts = None

        # To handle occlusion, we store the first frame's appearance inside the polygon
        # And track its pixels over time.
        first_frame_gray = None
        first_surface_pts = None
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_read_interval != 0:
                frame_idx += 1
                continue
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                # Save every exact extracted frame
                cv2.imwrite(os.path.join(debug_dir, f"extracted_frame_{frame_idx:05d}.jpg"), frame)

            if first_frame:
                surface_pts = self.detect_surface(frame)
                first_surface_pts = surface_pts.copy()
                
                # Get features to track
                p0 = self.get_features_in_poly(frame_gray, surface_pts)
                
                if p0 is None or len(p0) < 4:
                    # Fallback to just tracking the 4 corners of the poly
                    p0 = surface_pts.reshape(-1, 1, 2)
                    
                first_frame_gray = frame_gray.copy()
                first_p0 = p0.copy()
                prev_gray = frame_gray.copy()
                
                # Calculate Homography and place ad
                H, _ = cv2.findHomography(logo_pts, surface_pts)
                
                if debug_dir:
                    os.makedirs(debug_dir, exist_ok=True)
                    # 2. Detected surface colored green
                    surface_vis = frame.copy()
                    cv2.fillConvexPoly(surface_vis, surface_pts.astype(np.int32), (0, 255, 0))
                    cv2.addWeighted(surface_vis, 0.4, frame, 0.6, 0, surface_vis)
                    cv2.imwrite(os.path.join(debug_dir, "detected_surface.jpg"), surface_vis)
                    
                    # 3. Warped brand image ad
                    warped_ad = cv2.warpPerspective(logo, H, (width, height))
                    cv2.imwrite(os.path.join(debug_dir, "warped_ad.png"), warped_ad)

                frame = self.place_ad(frame, logo, H)
                
                first_frame = False
            else:
                # Track points
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **self.lk_params)
                
                # Filter good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                H = None
                if len(good_new) >= 4:
                    # Find homography from FIRST frame to CURRENT frame using tracked features
                    # To avoid drift, we can track relative to previous frame or first frame
                    # Usually better to find homography of current features relative to original features
                    H_motion, inliers = cv2.findHomography(first_p0[st == 1], good_new, cv2.RANSAC, 5.0)
                    
                    if H_motion is not None:
                        # Update surface points based on new homography from first frame
                        current_surface_pts = cv2.perspectiveTransform(first_surface_pts.reshape(-1, 1, 2), H_motion)
                        surface_pts = current_surface_pts.reshape(4, 2)
                        
                        # Now find homography from logo to current surface
                        H, _ = cv2.findHomography(logo_pts, surface_pts)
                        
                if H is not None:
                    # Handle occlusion by frame differencing
                    # Warp current frame back to first frame's perspective to get "background" differences
                    H_inv = np.linalg.inv(H_motion)
                    warped_current = cv2.warpPerspective(frame_gray, H_inv, (width, height))
                    
                    # Difference inside the originally detected polygon area
                    mask = np.zeros_like(first_frame_gray)
                    cv2.fillConvexPoly(mask, first_surface_pts.astype(np.int32), 255)
                    
                    diff = cv2.absdiff(first_frame_gray, warped_current)
                    _, occlusion_mask_original = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
                    occlusion_mask_original = cv2.bitwise_and(occlusion_mask_original, occlusion_mask_original, mask=mask)
                    
                    # Morphological operations to clean up noises in occlusion mask
                    kernel = np.ones((5,5), np.uint8)
                    occlusion_mask_original = cv2.morphologyEx(occlusion_mask_original, cv2.MORPH_OPEN, kernel)
                    occlusion_mask_original = cv2.dilate(occlusion_mask_original, kernel, iterations=2)
                    
                    # Warp occlusion mask to current frame
                    occlusion_mask = cv2.warpPerspective(occlusion_mask_original, H_motion, (width, height))
                    
                    # Place ad considering occlusion
                    frame = self.place_ad(frame, logo, H, occlusion_mask)
                else:
                    print(f"Tracking lost at frame {frame_idx}. Attempting redetect...")
                    # Re-detect
                    surface_pts = self.detect_surface(frame)
                    first_surface_pts = surface_pts.copy()
                    p0 = self.get_features_in_poly(frame_gray, surface_pts)
                    if p0 is None or len(p0) < 4:
                        p0 = surface_pts.reshape(-1, 1, 2)
                    first_p0 = p0.copy()
                    prev_gray = frame_gray.copy()
                    first_frame_gray = frame_gray.copy()
                    
                    H, _ = cv2.findHomography(logo_pts, surface_pts)
                    if H is not None:
                         frame = self.place_ad(frame, logo, H)
                         
                    frame_idx += 1
                    out.write(frame)
                    continue

                # Now update the previous frame and previous points
                prev_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
                first_p0 = first_p0[st == 1].reshape(-1, 1, 2)
                
                # If too few points remain, re-detect features inside the updated polygon 
                # (but maintaining relation to first frame is tricky. For simplicity, we just reset if too few)
                if len(p0) < 10:
                     print(f"Features depleted at frame {frame_idx}. Redetecting inside current tracking box.")
                     new_features = self.get_features_in_poly(frame_gray, surface_pts)
                     if new_features is not None and len(new_features) > 0:
                         p0 = new_features
                         first_p0 = cv2.perspectiveTransform(new_features, H_inv).reshape(-1, 1, 2)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print("Processing finished.")

    def place_ad(self, frame, logo, H, occlusion_mask=None):
        h, w = frame.shape[:2]
        
        # Warp the ad
        warped_ad = cv2.warpPerspective(logo, H, (w, h))
        
        # Extract alpha channel to create a mask for the ad footprint
        ad_alpha = warped_ad[:, :, 3]
        _, ad_mask = cv2.threshold(ad_alpha, 1, 255, cv2.THRESH_BINARY)
        
        # If there's an occlusion mask, don't display the ad where occlusion occurs
        # occlusion_mask = 255 for occluding objects, 0 for background
        if occlusion_mask is not None:
            ad_mask = cv2.bitwise_and(ad_mask, cv2.bitwise_not(occlusion_mask))

        # Create a 3-channel version of the updated mask
        ad_mask_3c = cv2.cvtColor(ad_mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Strip alpha from warped ad for blending
        warped_ad_rgb = warped_ad[:, :, :3]
        
        # Blend the ad with the frame purely inside the ad_mask region
        result = frame * (1.0 - ad_mask_3c) + warped_ad_rgb * ad_mask_3c
        return result.astype(np.uint8)
