import os
import uuid
import subprocess
import cv2
import imageio_ffmpeg
import tempfile
import mimetypes
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

from ad_inserter import AdPlacementSystem

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

def get_or_upload_file(local_path, remote_path):
    if not os.path.exists(local_path):
        return ""
    if not supabase:
        return f"/static/outputs/{remote_path}"
    
    mime_type, _ = mimetypes.guess_type(local_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
        
    try:
        supabase.storage.from_(supabase_bucket).upload(
            path=remote_path,
            file=local_path,
            file_options={"content-type": mime_type, "upsert": "true"}
        )
        return supabase.storage.from_(supabase_bucket).get_public_url(remote_path)
    except Exception as e:
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

    video_path = os.path.join(job_upload_dir, "input_video.mp4")
    logo_path = os.path.join(job_upload_dir, "input_logo.png")
    
    video_file.save(video_path)
    logo_file.save(logo_path)

    # 1-3. Initial video metadata
    initial_v_size, initial_v_codec, initial_v_fps = get_video_metadata(video_path)
    
    # 4-5. Logo metadata
    logo_size = os.path.getsize(logo_path)
    logo_img = cv2.imread(logo_path)
    logo_dims = f"{logo_img.shape[1]}x{logo_img.shape[0]}" if logo_img is not None else "Unknown"

    # 6-7. Convert using FFmpeg (H.264, 24fps)
    prepared_video_path = os.path.join(job_upload_dir, "prepared_video.mp4")
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg_exe, '-y', '-i', video_path, '-c:v', 'libx264', '-r', '24', '-preset', 'fast', prepared_video_path], check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg failed: {e.stderr.decode()}"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to execute FFmpeg: {str(e)}"}), 500
        
    # 8-10. New video metadata
    new_v_size, new_v_codec, new_v_fps = get_video_metadata(prepared_video_path)

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

    # Use the 24fps H.264 prepared video!
    video_path = os.path.join(job_upload_dir, "prepared_video.mp4")
    logo_path = os.path.join(job_upload_dir, "input_logo.png")
    output_video_path = os.path.join(job_output_dir, "output.mp4")

    try:
        system = AdPlacementSystem()
        system.process_video(video_path, logo_path, output_video_path, debug_dir=job_output_dir)
        
        extracted_frames_urls = []
        if os.path.exists(job_output_dir):
            for file in sorted(os.listdir(job_output_dir)):
                if file.startswith("extracted_frame_") and file.endswith(".jpg"):
                    local_f = os.path.join(job_output_dir, file)
                    remote_f = f"{job_id}/{file}"
                    extracted_frames_urls.append(get_or_upload_file(local_f, remote_f))
        
        det_surf_local = os.path.join(job_output_dir, "detected_surface.jpg")
        det_surf_url = get_or_upload_file(det_surf_local, f"{job_id}/detected_surface.jpg")
        
        warped_ad_local = os.path.join(job_output_dir, "warped_ad.png")
        warped_ad_url = get_or_upload_file(warped_ad_local, f"{job_id}/warped_ad.png")
        
        output_video_url = get_or_upload_file(output_video_path, f"{job_id}/output.mp4")

        return jsonify({
            "message": "Processing complete",
            "extracted_frames": extracted_frames_urls,
            "detected_surface": det_surf_url,
            "warped_ad": warped_ad_url,
            "output_video": output_video_url
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
