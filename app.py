import os
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from ad_inserter import AdPlacementSystem

app = Flask(__name__)

# Configure upload and output directories using absolute paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'video' not in request.files or 'logo' not in request.files:
        return jsonify({"error": "Missing video or logo file"}), 400

    video_file = request.files['video']
    logo_file = request.files['logo']

    if video_file.filename == '' or logo_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    job_id = str(uuid.uuid4())
    job_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    job_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    
    os.makedirs(job_upload_dir, exist_ok=True)
    os.makedirs(job_output_dir, exist_ok=True)

    video_path = os.path.join(job_upload_dir, "input_video.mp4")
    logo_path = os.path.join(job_upload_dir, "input_logo.png")
    
    video_file.save(video_path)
    logo_file.save(logo_path)

    output_video_path = os.path.join(job_output_dir, "output.mp4")

    # Initialize the system and process the video
    try:
        system = AdPlacementSystem()
        # Pass job_output_dir as debug_dir to save intermediate frames
        system.process_video(video_path, logo_path, output_video_path, debug_dir=job_output_dir)
        
        extracted_frames = []
        if os.path.exists(job_output_dir):
            for file in sorted(os.listdir(job_output_dir)):
                if file.startswith("extracted_frame_") and file.endswith(".jpg"):
                    extracted_frames.append(f"/static/outputs/{job_id}/{file}")
        
        return jsonify({
            "message": "Processing complete",
            "extracted_frames": extracted_frames,
            "detected_surface": f"/static/outputs/{job_id}/detected_surface.jpg",
            "warped_ad": f"/static/outputs/{job_id}/warped_ad.png",
            "output_video": f"/static/outputs/{job_id}/output.mp4"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
