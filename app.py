import os
import uuid
import subprocess
import cv2
import imageio_ffmpeg
import tempfile
import mimetypes
import time
import json
import numpy as np
try:
    import pycocotools.mask as maskUtils
except ImportError:
    maskUtils = None
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

# Local imports
from ad_inserter import AdPlacementSystem
from logger import EnterpriseLogger

load_dotenv()
try:
    from supabase import create_client, Client
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    supabase_bucket = os.environ.get("SUPABASE_BUCKET", "outputs")
    supabase = create_client(url, key) if url and key else None
except Exception:
    supabase = None

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), "anti1_uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def get_video_metadata(filepath):
    try:
        size = os.path.getsize(filepath)
        
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return size, "unknown", 0
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        # Ensure we decode codec correctly
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        if not codec.strip():
            codec = "unknown"
            
        cap.release()
        return size, codec, round(fps, 2)
    except Exception:
        return os.path.getsize(filepath) if os.path.exists(filepath) else 0, "unknown", 0

def get_or_upload_file(local_path, remote_path, logger=None):
    if not os.path.exists(local_path):
        return ""
    if not supabase:
        return f"/static/outputs/{remote_path}"
    
    mime_type, _ = mimetypes.guess_type(local_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
        
    try:
        start_time = time.time()
        supabase.storage.from_(supabase_bucket).upload(
            path=remote_path,
            file=local_path,
            file_options={"content-type": mime_type, "upsert": "true"}
        )
        duration = round(time.time() - start_time, 3)
        if logger:
            logger.info("Performance", f"Cloud Upload {remote_path}", duration_seconds=duration, size_bytes=os.path.getsize(local_path))
        return supabase.storage.from_(supabase_bucket).get_public_url(remote_path)
    except Exception as e:
        if logger:
            logger.error("Error", f"Supabase upload failed for {local_path}", exception=str(e))
        print(f"Supabase upload failed for {local_path}: {e}")
        return f"/static/outputs/{remote_path}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prepare', methods=['POST'])
def prepare():
    if 'video' not in request.files or 'logo' not in request.files:
        return jsonify({"error": "Missing video or logo file"}), 400

    video_file = request.files['video']
    logo_file = request.files['logo']

    job_id = str(uuid.uuid4())
    job_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    os.makedirs(job_upload_dir, exist_ok=True)

    log_file_path = os.path.join(app.config['OUTPUT_FOLDER'], job_id, f"{job_id}.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger = EnterpriseLogger(log_file_path, job_id)

    logger.info("EventFlow", "Asset preparation started", endpoint="/prepare", method=request.method)
    logger.info("Security", "Incoming request details", ip=request.remote_addr, user_agent=str(request.user_agent))
    logger.info("Configuration", "Upload directories created", upload_dir=job_upload_dir)

    video_path = os.path.join(job_upload_dir, "input_video.mp4")
    logo_path = os.path.join(job_upload_dir, "input_logo.png")
    
    logger.info("EventFlow", "Saving uploaded files to disk")
    video_file.save(video_path)
    logo_file.save(logo_path)

    # 1-3. Initial video metadata
    initial_v_size, initial_v_codec, initial_v_fps = get_video_metadata(video_path)
    logger.info("Variable", "Initial video metadata", size=initial_v_size, codec=initial_v_codec, fps=initial_v_fps)
    
    # 4-5. Logo metadata
    logo_size = os.path.getsize(logo_path)
    logo_img = cv2.imread(logo_path)
    logo_dims = f"{logo_img.shape[1]}x{logo_img.shape[0]}" if logo_img is not None else "Unknown"
    logger.info("Variable", "Initial logo metadata", size=logo_size, dimensions=logo_dims)

    # 6-7. Convert using FFmpeg (H.264, 24fps)
    prepared_video_path = os.path.join(job_upload_dir, "prepared_video.mp4")
    logger.info("EventFlow", "Starting FFmpeg conversion to 24fps H.264")
    start_time = time.time()
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg_exe, '-y', '-i', video_path, '-c:v', 'libx264', '-r', '24', '-preset', 'fast', prepared_video_path], check=True, stderr=subprocess.PIPE)
        logger.info("Performance", "FFmpeg conversion completed", duration_seconds=round(time.time() - start_time, 2))
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode()
        logger.error("Error", "FFmpeg failed", stderr=err_msg)
        return jsonify({"error": f"FFmpeg failed: {err_msg}"}), 500
    except Exception as e:
        logger.error("Error", "Failed to execute FFmpeg", exception=str(e))
        return jsonify({"error": f"Failed to execute FFmpeg: {str(e)}"}), 500
        
    # 8-10. New video metadata
    new_v_size, new_v_codec, new_v_fps = get_video_metadata(prepared_video_path)
    logger.info("Variable", "Transformed video metadata", size=new_v_size, codec=new_v_codec, fps=new_v_fps)

    logger.info("EventFlow", "Asset preparation completed successfully")

    return jsonify({
        "job_id": job_id,
        "table": {
            "initial_video_size": f"{initial_v_size / (1024*1024):.2f} MB",
            "initial_video_codec": initial_v_codec,
            "initial_video_fps": initial_v_fps,
            "logo_size": f"{logo_size / 1024:.2f} KB",
            "logo_dims": logo_dims,
            "new_video_size": f"{new_v_size / (1024*1024):.2f} MB",
            "new_video_codec": new_v_codec,
            "new_video_fps": new_v_fps
        }
    })

@app.route('/process', methods=['POST'])
def process():
    job_id = request.form.get('job_id')
    if not job_id:
        return jsonify({"error": "No job ID provided"}), 400

    job_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    job_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    
    os.makedirs(job_output_dir, exist_ok=True)
    
    log_file_path = os.path.join(job_output_dir, f"{job_id}.log")
    logger = EnterpriseLogger(log_file_path, job_id)
    
    logger.info("EventFlow", "Processing started", endpoint="/process", method=request.method)
    logger.info("Configuration", "Output directory configured", output_dir=job_output_dir)

    # Use the 24fps H.264 prepared video!
    video_path = os.path.join(job_upload_dir, "prepared_video.mp4")
    logo_path = os.path.join(job_upload_dir, "input_logo.png")
    output_video_path = os.path.join(job_output_dir, "output.mp4")

    logger.info("EventFlow", "Initializing AdPlacementSystem")
    try:
        start_time = time.time()
        system = AdPlacementSystem(logger=logger)
        logger.info("EventFlow", "Starting video processing pipeline")
        system.process_video(video_path, logo_path, output_video_path, debug_dir=job_output_dir)
        logger.info("Performance", "Video processing pipeline completed", duration_seconds=round(time.time() - start_time, 2))
        
        logger.info("EventFlow", "Uploading extracted frames and results to Supabase")
        extracted_frames_urls = []
        if os.path.exists(job_output_dir):
            for file in sorted(os.listdir(job_output_dir)):
                if file.startswith("extracted_frame_") and file.endswith(".jpg"):
                    local_f = os.path.join(job_output_dir, file)
                    remote_f = f"{job_id}/{file}"
                    extracted_frames_urls.append(get_or_upload_file(local_f, remote_f, logger))
        
        det_surf_local = os.path.join(job_output_dir, "detected_surface.jpg")
        det_surf_url = get_or_upload_file(det_surf_local, f"{job_id}/detected_surface.jpg", logger)
        
        warped_ad_local = os.path.join(job_output_dir, "warped_ad.png")
        warped_ad_url = get_or_upload_file(warped_ad_local, f"{job_id}/warped_ad.png", logger)
        
        output_video_path = os.path.join(job_output_dir, "output.mp4")
        output_video_url = get_or_upload_file(output_video_path, f"{job_id}/output.mp4", logger)
        
        logger.info("EventFlow", "Uploading job execution log to Supabase")
        log_file_url = get_or_upload_file(log_file_path, f"{job_id}/{job_id}.log", logger)

        logger.info("EventFlow", "Processing complete, returning API payload")
        return jsonify({
            "message": "Processing complete",
            "extracted_frames": extracted_frames_urls,
            "detected_surface": det_surf_url,
            "warped_ad": warped_ad_url,
            "output_video": output_video_url,
            "execution_log": log_file_url
        })
    except Exception as e:
        import traceback
        err_traceback = traceback.format_exc()
        logger.error("Error", "Processing pipeline failed with exception", exception=str(e), traceback=err_traceback)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    job_id = request.form.get('job_id')
    if not job_id:
        return jsonify({"error": "No job ID provided"}), 400

    if 'ground_truth' not in request.files:
        return jsonify({"error": "Missing ground truth file"}), 400

    job_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    pred_path = os.path.join(job_output_dir, "predictions.json")
    
    if not os.path.exists(pred_path):
        return jsonify({"error": "Predictions not found for this job. Ensure you completed processing safely."}), 400
        
    gt_file = request.files['ground_truth']
    try:
        gt_data = json.load(gt_file)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400
        
    try:
        with open(pred_path, 'r') as f:
            pred_data = json.load(f)

        gt_masks = {}
        for ann in gt_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in gt_masks:
                gt_masks[img_id] = []
            seg = ann['segmentation']
            
            if isinstance(seg, dict) and 'counts' in seg:
                if isinstance(seg['counts'], str):
                    seg['counts'] = seg['counts'].encode('utf-8')
                gt_masks[img_id].append(seg)
            elif isinstance(seg, list):
                height = gt_data['images'][img_id]['height'] if img_id < len(gt_data['images']) else pred_data['height']
                width = gt_data['images'][img_id]['width'] if img_id < len(gt_data['images']) else pred_data['width']
                rle = maskUtils.frPyObjects(seg, height, width)
                gt_masks[img_id].append(rle[0])

        height = pred_data['height']
        width = pred_data['width']
        
        # Build image dimensions lookup
        image_dims = {img['id']: (img['width'], img['height']) for img in gt_data.get('images', [])}
        
        ious = []
        accuracies = []
        
        for i, pred in enumerate(pred_data.get('predictions', [])):
            img_id = list(gt_masks.keys())[i] if i < len(gt_masks) else None
            if img_id is None:
                break
                
            gt_rles = gt_masks[img_id]
            gt_w, gt_h = image_dims.get(img_id, (width, height))
            
            # Pred mask
            poly = np.array(pred['polygon'], dtype=np.float32)
            
            # Scale the predicted coordinates up to the original ground truth size
            scale_x = gt_w / float(width)
            scale_y = gt_h / float(height)
            poly[:, 0] *= scale_x
            poly[:, 1] *= scale_y
            
            flat_poly = poly.flatten().tolist()
            pred_rle = maskUtils.frPyObjects([flat_poly], gt_h, gt_w)[0]
            
            iou_matrix = maskUtils.iou([pred_rle], gt_rles, [0]*len(gt_rles))
            max_iou = float(np.max(iou_matrix)) if iou_matrix.size > 0 else 0.0
            ious.append(max_iou)
            
            pred_bin = maskUtils.decode(pred_rle)
            gt_bin = np.zeros((gt_h, gt_w), dtype=np.uint8)
            for rle in gt_rles:
                gt_bin = np.maximum(gt_bin, maskUtils.decode(rle))
                
            correct_pixels = np.sum(pred_bin == gt_bin)
            total_pixels = gt_h * gt_w
            acc = float(correct_pixels) / total_pixels
            accuracies.append(acc)

        mIoU = float(np.mean(ious)) if ious else 0.0
        pixel_acc = float(np.mean(accuracies)) if accuracies else 0.0
        mAP = float(np.mean([1 if iou > 0.5 else 0 for iou in ious])) if ious else 0.0
        
        log_file_path = os.path.join(job_output_dir, f"{job_id}.log")
        
        cloud_upload_time = 0.0
        cloud_upload_count = 0
        cloud_upload_bytes = 0
        
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r', encoding='utf-8') as lf:
                for line in lf:
                    if "[PERFORMANCE]" in line.upper() and "Cloud Upload" in line:
                        if "duration_seconds=" in line:
                            try:
                                dur_str = line.split("duration_seconds=")[1].split(",")[0].strip()
                                cloud_upload_time += float(dur_str)
                                cloud_upload_count += 1
                            except:
                                pass
                        if "size_bytes=" in line:
                            try:
                                size_str = line.split("size_bytes=")[1].split(",")[0].strip()
                                cloud_upload_bytes += int(size_str)
                            except:
                                pass

        metrics = {
            "mIOU": round(mIoU, 4),
            "PixelAccuracy": round(pixel_acc, 4),
            "mAP_50": round(mAP, 4),
            "Frames_Evaluated": len(ious),
            "Cloud_Upload_Count": cloud_upload_count,
            "Cloud_Upload_Time_sec": round(cloud_upload_time, 3),
            "Cloud_Upload_MB": round(cloud_upload_bytes / (1024 * 1024), 3)
        }
        
        log_file_path = os.path.join(job_output_dir, f"{job_id}.log")
        logger = EnterpriseLogger(log_file_path, job_id)
        
        logger.info("EventFlow", "Ground truth evaluation triggered")
        logger.info("Performance", "Metric evaluation complete", iou=metrics["mIOU"], accuracy=metrics["PixelAccuracy"], mAP=metrics["mAP_50"], frames_evaluated=metrics["Frames_Evaluated"], cloud_ops=metrics["Cloud_Upload_Count"])
        
        return jsonify(metrics)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Evaluation error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
