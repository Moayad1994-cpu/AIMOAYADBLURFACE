# app.py (v6.2 - Robust MAX_CONTENT_LENGTH Handling)
import os
import time
import traceback
import uuid
import json # For parsing manual regions
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for, current_app
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv() # Load environment variables first

# --- Configuration ---
UPLOAD_FOLDER_NAME = 'uploads'
ALLOWED_EXTENSIONS_IMG = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_EXTENSIONS_VID = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER_PATH = os.path.join(BASE_DIR, UPLOAD_FOLDER_NAME)

# --- Robust MAX_CONTENT_LENGTH Setting ---
MAX_UPLOAD_MB_DEFAULT = 100 # Default max upload size in MB
max_content_length_str = os.environ.get('MAX_CONTENT_LENGTH')
MAX_CONTENT_LENGTH = None

if max_content_length_str:
    try:
        # Try converting the environment variable value to an integer
        MAX_CONTENT_LENGTH = int(max_content_length_str)
        print(f"INFO: MAX_CONTENT_LENGTH found in environment: {MAX_CONTENT_LENGTH / (1024*1024):.0f} MB")
    except ValueError:
        print(f"WARNING: Invalid value '{max_content_length_str}' found for MAX_CONTENT_LENGTH in environment/'.env'. Using default.")
        MAX_CONTENT_LENGTH = None # Force using the default calculation

# If not set in environment or conversion failed, use the default calculation
if MAX_CONTENT_LENGTH is None:
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB_DEFAULT * 1024 * 1024
    print(f"INFO: MAX_CONTENT_LENGTH not in environment or invalid. Using default: {MAX_UPLOAD_MB_DEFAULT} MB ({MAX_CONTENT_LENGTH} bytes)")
# --- End MAX_CONTENT_LENGTH Setting ---

HAAR_CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (30, 30)

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_PATH
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH # Apply the determined limit
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'combined_blur_secret_standard_name_v4') # Change this!

# --- Global Variable for Haar Cascade Model ---
haar_cascade = None

# --- Helper Functions ---
# (allowed_file, check_folder_permissions, load_haar_cascade_on_startup, detect_faces_haar, blur_regions, process_uploaded_image, process_uploaded_video - all unchanged)
def allowed_file(filename, allowed_extensions): return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
def check_folder_permissions(folder_path):
    if not os.path.exists(folder_path):
        try: os.makedirs(folder_path, exist_ok=True); print(f"INFO: Created uploads directory: {folder_path}")
        except OSError as e: print(f"ERROR: Could not create uploads directory {folder_path}: {e}"); return False
    dummy_file_path = os.path.join(folder_path, f"perm_test_{uuid.uuid4().hex}.tmp")
    try:
        with open(dummy_file_path, 'w') as f: f.write('test')
        os.remove(dummy_file_path); print(f"INFO: Write permissions verified for folder: {folder_path}"); return True
    except Exception as e: print(f"!!!!!!!!!!!!!!!! ERROR: Cannot write to upload folder: {folder_path}\n       Reason: {e}\n       CHECK PERMISSIONS! !!!!!!!!!!!!!!!!"); return False
def load_haar_cascade_on_startup(cascade_filename):
    global haar_cascade; cascade_path = os.path.join(BASE_DIR, cascade_filename)
    if not os.path.exists(cascade_path): print(f"CRITICAL ERROR: Haar Cascade file '{cascade_filename}' not found at {cascade_path}"); return False
    try:
        haar_cascade = cv2.CascadeClassifier(cascade_path)
        if haar_cascade.empty(): print(f"ERROR: Failed to load Haar Cascade from {cascade_path}."); haar_cascade = None; return False
        print(f"INFO: Haar Cascade '{cascade_filename}' loaded successfully."); return True
    except Exception as e: print(f"ERROR loading Haar Cascade: {e}"); traceback.print_exc(); haar_cascade = None; return False
def detect_faces_haar(frame):
    global haar_cascade
    if haar_cascade is None: print("WARN: Haar Cascade not loaded, cannot auto-detect."); return np.empty((0, 4))
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); gray = cv2.equalizeHist(gray)
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=HAAR_SCALE_FACTOR, minNeighbors=HAAR_MIN_NEIGHBORS, minSize=HAAR_MIN_SIZE)
        return faces if isinstance(faces, np.ndarray) else np.empty((0, 4))
    except Exception as e: print(f"ERROR during Haar Cascade detection: {e}"); traceback.print_exc(); return np.empty((0, 4))
def blur_regions(frame, regions, blur_factor):
    processed_frame = frame.copy();
    try: regions_array = np.array(regions).astype(int)
    except (ValueError, TypeError): print("Warning: Could not convert regions to integer array."); return processed_frame
    if not isinstance(regions_array, np.ndarray) or regions_array.ndim != 2 or regions_array.shape[1] < 4 or len(regions_array) == 0: return processed_frame
    # print(f" INFO: Applying blur to {len(regions_array)} region(s).") # Can be verbose for video
    for region in regions_array:
        x, y, w, h = region[:4];
        if w <= 0 or h <= 0: continue
        y1=max(0, y); y2=min(frame.shape[0], y + h); x1=max(0, x); x2=min(frame.shape[1], x + w)
        if y2 <= y1 or x2 <= x1: continue
        roi = processed_frame[y1:y2, x1:x2]
        if roi.size == 0: continue
        base_kernel_dim = max(int(min(w, h) * blur_factor * 0.5), 7)
        kernel_size = base_kernel_dim // 2 * 2 + 1; kernel_size = max(5, kernel_size)
        try: blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), sigmaX=0, sigmaY=0); processed_frame[y1:y2, x1:x2] = blurred_roi
        except cv2.error as e: print(f"WARN: Blur failed. ROI:{roi.shape} K:({kernel_size},{kernel_size}) Err:{e}"); continue
    return processed_frame
def process_uploaded_image(input_path, output_path, blur_factor, manual_regions):
    print(f"--- Processing Image: {os.path.basename(input_path)} ---")
    method_used="Error"; regions_to_blur = [] ; num_regions_processed = 0
    try:
        img = cv2.imread(input_path);
        if img is None: raise IOError(f"Could not read input image: {input_path}")
        if manual_regions: print(f" INFO: Using {len(manual_regions)} manual region(s)."); regions_to_blur = manual_regions; method_used = "Manual"
        elif haar_cascade: print(f" INFO: No manual regions, using Haar auto-detection."); regions_to_blur = detect_faces_haar(img); method_used = "Automatic (Haar)"; print(f" INFO: Auto-detected {len(regions_to_blur)} face regions.")
        else: print(" INFO: No manual regions & Haar not loaded."); method_used = "None"
        blurred_img = blur_regions(img, regions_to_blur, blur_factor); num_regions_processed = len(regions_to_blur)
        success = cv2.imwrite(output_path, blurred_img);
        if not success: raise IOError(f"Could not write output image: {output_path}")
        print(f" INFO: Processed image saved: {output_path}")
        return True, num_regions_processed, method_used
    except Exception as e: print(f"ERROR processing image {input_path}: {e}"); traceback.print_exc(); return False, 0, "Error"
def process_uploaded_video(input_path, output_path, blur_factor, manual_regions):
    print(f"--- Processing Video: {os.path.basename(input_path)} ---")
    cap = None; out = None; regions_to_use = []; using_manual = False; total_detections = 0; method_used = "Error"; num_regions_processed = 0
    if manual_regions:
        print(f" INFO: Using {len(manual_regions)} manual region(s) statically."); using_manual = True; method_used = "Manual"
        try: regions_to_use = np.array(manual_regions).astype(int); num_regions_processed = len(regions_to_use)
        except: print("Warning: Could not convert manual regions to array."); regions_to_use = []
    elif haar_cascade: print(f" INFO: No manual regions, using Haar auto-detection per frame."); method_used = "Automatic (Haar)"
    else: print(" INFO: No manual regions & Haar not loaded."); method_used = "None"; regions_to_use = []
    try:
        cap = cv2.VideoCapture(input_path);
        if not cap.isOpened(): raise IOError(f"Cannot open video: {input_path}")
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps = cap.get(cv2.CAP_PROP_FPS);
        if fps <= 0 or fps > 240: print(f"Warn: Invalid FPS {fps}, using 30."); fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v'); print(f" INFO: Video props: {fw}x{fh} @ {fps:.2f} FPS")
        out = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh));
        if not out.isOpened(): raise IOError(f"Cannot open VideoWriter: {output_path}")
        fc = 0; start_time = time.time()
        while True:
            ret, frame = cap.read();
            if not ret: break
            fc += 1; processed_frame = frame; current_regions = []
            if using_manual: current_regions = regions_to_use
            elif haar_cascade: current_regions = detect_faces_haar(frame); total_detections += len(current_regions)
            else: current_regions = []
            if len(current_regions) > 0: processed_frame = blur_regions(frame, current_regions, blur_factor)
            out.write(processed_frame)
            if fc % 60 == 0: elapsed = time.time()-start_time; cur_fps = fc/elapsed if elapsed > 0 else 0; print(f"  INFO: Processed frame {fc}... ({cur_fps:.1f} FPS)")
        end_time = time.time()
        num_regions_reported = num_regions_processed if using_manual else total_detections
        print(f"INFO: Finished video. Frames: {fc}. Total auto detections (if used): {total_detections}. Manual regions used (if any): {len(regions_to_use) if using_manual else 0}")
        print(f"INFO: Output saved: {output_path}"); print(f"INFO: Time: {end_time - start_time:.2f}s")
        return True, num_regions_reported, method_used
    except Exception as e: print(f"ERROR processing video {input_path}: {e}"); traceback.print_exc(); return False, 0, "Error"
    finally:
        if cap and cap.isOpened(): cap.release(); print(" INFO: VideoCapture released.")
        if out and out.isOpened(): out.release(); print(" INFO: VideoWriter released.")

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # (Unchanged)
    print(f"INFO: Request for '/'. Attempting to render 'index.html'.")
    try: return render_template('index.html')
    except Exception as e: print(f"CRITICAL ERROR rendering template 'index.html': {e}"); traceback.print_exc(); return f"Internal Server Error: Could not render template.", 500

@app.route('/process', methods=['POST'])
def process_file():
    # (Unchanged)
    print("INFO: Received request for /process")
    global haar_cascade
    if 'file' not in request.files: print("ERROR: 'file' not in request.files"); return jsonify({"success": False, "error": "No file part."}), 400
    file = request.files['file']
    try: blur_factor = float(request.form.get('blurFactor', 0.4));
    except (ValueError, TypeError): blur_factor = 0.4; print("WARN: Using default blur factor 0.4")
    if not (0.05 <= blur_factor <= 1.0): blur_factor = max(0.05, min(1.0, blur_factor)); print(f"WARN: Clamped blur factor to {blur_factor:.2f}")
    manual_regions_json = request.form.get('manualRegions', '[]'); manual_regions = []; regions_were_provided = False
    try:
        parsed_regions = json.loads(manual_regions_json)
        if isinstance(parsed_regions, list) and len(parsed_regions) > 0:
            regions_were_provided = True
            for region in parsed_regions:
                if isinstance(region, (list, tuple)) and len(region) == 4: manual_regions.append(list(map(int, region[:4])))
            print(f"INFO: Received {len(manual_regions)} valid manual blur regions.")
        else: print(f"INFO: No valid manual regions received.")
    except Exception as e: print(f"WARN: Error processing manualRegions: {e}")
    if file.filename == '': print("ERROR: No file selected."); return jsonify({"success": False, "error": "No file selected."}), 400
    filename = secure_filename(file.filename); file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    is_image = file_ext in ALLOWED_EXTENSIONS_IMG; is_video = file_ext in ALLOWED_EXTENSIONS_VID
    if not is_image and not is_video: print(f"ERROR: Invalid file type '{file_ext}'."); return jsonify({"success": False, "error": f"Invalid file type '{file_ext}'."}), 415
    if not regions_were_provided and haar_cascade is None: print("ERROR: No manual regions & Haar cascade not loaded."); return jsonify({"success": False, "error": "Cannot process: No regions drawn & detector unavailable."}), 400
    try: os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
    except OSError as e: print(f"Error creating upload dir: {e}"); return jsonify({"success": False, "error": "Server directory error."}), 500
    unique_id = uuid.uuid4().hex; input_filename = f"{unique_id}_input.{file_ext}"
    output_ext = 'mp4' if is_video else file_ext; output_filename = f"{unique_id}_blurred.{output_ext}"
    input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], input_filename)
    output_path = os.path.join(current_app.config['UPLOAD_FOLDER'], output_filename)
    success = False; num_regions_processed = 0; method_used="Error"
    try:
        file.save(input_path); print(f"INFO: Uploaded media saved to: {input_path}")
        start_time = time.time()
        if is_image: success, num_regions_processed, method_used = process_uploaded_image(input_path, output_path, blur_factor, manual_regions)
        elif is_video: success, num_regions_processed, method_used = process_uploaded_video(input_path, output_path, blur_factor, manual_regions)
        end_time = time.time()
        if success:
             print("SUCCESS: Processing complete.")
             return jsonify({ "success": True, "filename": output_filename, "url": url_for('uploaded_file', filename=output_filename), "is_video": is_video, "processing_time": f"{end_time - start_time:.2f}", "regions_processed": num_regions_processed, "method_used": method_used }), 200
        else: print("ERROR: Processing function returned failure."); return jsonify({"success": False, "error": "Processing failed internally. See server logs."}), 500
    except Exception as e:
        print(f"ERROR: Exception during file processing route: {e}"); traceback.print_exc()
        return jsonify({"success": False, "error": f"Internal server error: {e}"}), 500
    finally:
        if os.path.exists(input_path):
            try: os.remove(input_path); print(f"INFO: Removed input file: {input_path}")
            except Exception as e_rem: print(f"WARNING: Could not remove input file {input_path}: {e_rem}")

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    # (Unchanged)
    safe_dir = os.path.abspath(current_app.config['UPLOAD_FOLDER'])
    safe_path = os.path.abspath(os.path.join(safe_dir, filename))
    if not safe_path.startswith(safe_dir): print(f"WARN: Forbidden access attempt: {filename}"); return "Forbidden", 403
    try: print(f"INFO: Serving file: {filename} from {safe_dir}"); return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename, as_attachment=False)
    except FileNotFoundError: print(f"ERROR: File not found requested: {filename}"); return "File not found", 404
    except Exception as e: print(f"ERROR serving file {filename}: {e}"); traceback.print_exc(); return "Error serving file", 500

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask Face Blur Application (Manual Draw + Haar Fallback)...")
    os.makedirs(UPLOAD_FOLDER_PATH, exist_ok=True); print(f"INFO: Upload folder ensured at: {UPLOAD_FOLDER_PATH}")
    if not check_folder_permissions(UPLOAD_FOLDER_PATH): print("\n!!! CRITICAL ERROR: Cannot write to upload folder. Uploads/Processing will fail. Check permissions. !!!\n")
    model_loaded = load_haar_cascade_on_startup(HAAR_CASCADE_FILENAME)
    if not model_loaded: print("\n!!! WARNING: Haar Cascade model failed to load. Automatic detection fallback disabled. !!!\n")
    # Use Waitress for better production serving if needed (pip install waitress)
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    app.run(debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true', host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))