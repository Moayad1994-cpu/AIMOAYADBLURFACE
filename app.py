# app.py (v6.3 - Enhanced Deployment Path Handling & Logging)
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
ALLOWED_EXTENSIONS_VID = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'ogg'} # Added webm/ogg

# Determine BASE_DIR using the location of this script file
# This is crucial for finding adjacent files like the Haar cascade in deployed environments
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Consider using a temporary directory for uploads in production environments
UPLOAD_FOLDER_PATH = os.path.join(BASE_DIR, UPLOAD_FOLDER_NAME)

# --- Robust MAX_CONTENT_LENGTH Setting ---
MAX_UPLOAD_MB_DEFAULT = 100 # Default max upload size in MB
max_content_length_str = os.environ.get('MAX_CONTENT_LENGTH')
MAX_CONTENT_LENGTH = None

if max_content_length_str:
    try:
        MAX_CONTENT_LENGTH = int(max_content_length_str)
        print(f"INFO: MAX_CONTENT_LENGTH found in environment: {MAX_CONTENT_LENGTH / (1024*1024):.0f} MB")
    except ValueError:
        print(f"WARNING: Invalid value '{max_content_length_str}' found for MAX_CONTENT_LENGTH in environment/'.env'. Using default.")
        MAX_CONTENT_LENGTH = None # Force using the default calculation

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
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secret_key_please_change_me') # Change this! Use environment variable in production.

# --- Global Variable for Haar Cascade Model ---
haar_cascade = None

# --- Helper Functions ---

def allowed_file(filename, allowed_extensions):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def check_folder_permissions(folder_path):
    """Checks if the application can write to the specified folder."""
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"INFO: Created directory: {folder_path}")
        except OSError as e:
            print(f"ERROR: Could not create directory {folder_path}: {e}")
            return False

    # Check basic write/execute permissions first (useful on Linux)
    if not os.access(folder_path, os.W_OK | os.X_OK):
         print(f"!!!!!!!!!!!!!!!! ERROR: Insufficient permissions (Write/Execute) for folder: {folder_path}. CHECK PERMISSIONS! !!!!!!!!!!!!!!!!");
         return False

    # Attempt to write a temporary file as a definitive check
    dummy_file_path = os.path.join(folder_path, f"perm_test_{uuid.uuid4().hex}.tmp")
    try:
        with open(dummy_file_path, 'w') as f:
            f.write('test')
        os.remove(dummy_file_path)
        print(f"INFO: Write permissions verified for folder: {folder_path}")
        return True
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!! ERROR: Cannot write to folder: {folder_path}\n       Reason: {e}\n       CHECK PERMISSIONS! !!!!!!!!!!!!!!!!");
        return False

def load_haar_cascade_on_startup(cascade_filename):
    """
    Loads the Haar Cascade classifier.
    Relies on the cascade_filename being in the same directory as this script (app.py).
    Adds enhanced logging for deployment debugging.
    """
    global haar_cascade
    cascade_path = os.path.join(BASE_DIR, cascade_filename)

    print(f"--- Attempting to load Haar Cascade ---")
    print(f"INFO: Script base directory (BASE_DIR): {BASE_DIR}")
    print(f"INFO: Expecting Haar Cascade filename: {cascade_filename}")
    print(f"INFO: Calculated absolute path for cascade: {cascade_path}")

    if not os.path.exists(cascade_path):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"CRITICAL ERROR: Haar Cascade file NOT FOUND at the expected path: {cascade_path}")
        print(f"Ensure '{cascade_filename}' is in the SAME directory as 'app.py' in your repository.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        haar_cascade = None
        return False

    if not os.access(cascade_path, os.R_OK):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"CRITICAL ERROR: Haar Cascade file FOUND but NO READ PERMISSION at: {cascade_path}")
        print(f"Check file permissions in your deployment environment.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        haar_cascade = None
        return False

    try:
        print(f"INFO: File exists and is readable. Attempting to load with cv2.CascadeClassifier...")
        haar_cascade = cv2.CascadeClassifier(cascade_path)
        if haar_cascade.empty():
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR: Failed to load Haar Cascade from {cascade_path}. File might be corrupted or invalid.")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            haar_cascade = None
            return False
        print(f"SUCCESS: Haar Cascade '{cascade_filename}' loaded successfully.")
        return True
    except cv2.error as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: OpenCV error while loading Haar Cascade from {cascade_path}: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        traceback.print_exc()
        haar_cascade = None
        return False
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: Unexpected error loading Haar Cascade from {cascade_path}: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        traceback.print_exc()
        haar_cascade = None
        return False

def detect_faces_haar(frame):
    """Detects faces in a frame using the loaded Haar Cascade."""
    global haar_cascade
    if haar_cascade is None:
        # This case should ideally be handled before calling, but good fallback
        print("WARN: detect_faces_haar called but Haar Cascade is not loaded.")
        return np.empty((0, 4)) # Return empty numpy array, shape (0, 4)
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = haar_cascade.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE
        )
        return faces if isinstance(faces, np.ndarray) else np.empty((0, 4))
    except cv2.error as e:
        print(f"ERROR during Haar Cascade detection (OpenCV error): {e}")
        # traceback.print_exc() # Can be too verbose
        return np.empty((0, 4))
    except Exception as e:
        print(f"ERROR during Haar Cascade detection (General error): {e}")
        traceback.print_exc()
        return np.empty((0, 4))

def blur_regions(frame, regions, blur_factor):
    """Applies Gaussian blur to specified regions in a frame."""
    processed_frame = frame.copy()
    try:
        regions_array = np.array(regions).astype(int)
    except (ValueError, TypeError):
        print("Warning: Could not convert regions to integer array. Skipping blur.")
        return processed_frame

    if not isinstance(regions_array, np.ndarray) or regions_array.ndim != 2 or regions_array.shape[1] < 4 or len(regions_array) == 0:
        return processed_frame

    for region in regions_array:
        x, y, w, h = region[:4]
        if w <= 0 or h <= 0: continue

        y1 = max(0, y)
        y2 = min(frame.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(frame.shape[1], x + w)

        if y2 <= y1 or x2 <= x1: continue

        roi = processed_frame[y1:y2, x1:x2]
        if roi.size == 0: continue

        base_kernel_dim = max(int(min(w, h) * blur_factor * 0.5), 7)
        kernel_size = (base_kernel_dim // 2 * 2) + 1
        kernel_size = max(5, kernel_size)

        try:
            blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), sigmaX=0, sigmaY=0)
            processed_frame[y1:y2, x1:x2] = blurred_roi
        except cv2.error as e:
            print(f"WARN: GaussianBlur failed. ROI:{roi.shape} K:({kernel_size},{kernel_size}) Err:{e}")
            continue
        except Exception as e:
            print(f"WARN: Unexpected error during blur for region: {e}")
            continue

    return processed_frame

def process_uploaded_image(input_path, output_path, blur_factor, manual_regions):
    """Processes a single uploaded image file."""
    print(f"--- Processing Image: {os.path.basename(input_path)} ---")
    method_used = "Error"
    regions_to_blur = []
    num_regions_processed = 0

    try:
        img = cv2.imread(input_path)
        if img is None:
            raise IOError(f"Could not read input image: {input_path}")

        # Determine which regions to use: manual first, then auto if available
        if manual_regions:
            print(f" INFO: Using {len(manual_regions)} manual region(s).")
            regions_to_blur = manual_regions
            method_used = "Manual"
        elif haar_cascade is not None:
            print(f" INFO: No manual regions, using Haar auto-detection.")
            regions_to_blur = detect_faces_haar(img)
            method_used = "Automatic (Haar)"
            print(f" INFO: Auto-detected {len(regions_to_blur)} face region(s).")
        else:
            # No manual regions AND cascade not loaded (should be caught earlier, but safety)
            print(" INFO: No manual regions provided & Haar cascade unavailable. No regions to blur.")
            regions_to_blur = []
            method_used = "None"

        # Apply blur if there are regions
        if len(regions_to_blur) > 0:
            blurred_img = blur_regions(img, regions_to_blur, blur_factor)
            num_regions_processed = len(regions_to_blur)
        else:
            blurred_img = img # No blur applied if no regions

        # Save the processed image
        success = cv2.imwrite(output_path, blurred_img)
        if not success:
            raise IOError(f"Could not write output image: {output_path}")

        print(f" INFO: Processed image saved: {output_path}")
        return True, num_regions_processed, method_used

    except Exception as e:
        print(f"ERROR processing image {os.path.basename(input_path)}: {e}")
        traceback.print_exc()
        return False, 0, "Error"


def process_uploaded_video(input_path, output_path, blur_factor, manual_regions):
    """Processes an uploaded video file frame by frame."""
    print(f"--- Processing Video: {os.path.basename(input_path)} ---")
    cap = None
    out = None
    regions_to_use = []
    using_manual = False
    total_detections = 0 # For counting auto-detections if used
    method_used = "Error"
    num_regions_processed_in_frame = 0 # For reporting at the end

    # Determine regions strategy: static manual or per-frame auto
    if manual_regions:
        print(f" INFO: Using {len(manual_regions)} manual region(s) statically for all frames.")
        using_manual = True
        method_used = "Manual"
        try:
            regions_to_use = np.array(manual_regions).astype(int)
            num_regions_processed_in_frame = len(regions_to_use)
        except (ValueError, TypeError):
            print("Warning: Could not convert manual regions to array. No regions will be used.")
            regions_to_use = []
            using_manual = False # Fallback as if no manual regions given
            method_used = "None" # Update method if manual failed
    elif haar_cascade is not None:
        print(f" INFO: No manual regions, using Haar auto-detection per frame.")
        method_used = "Automatic (Haar)"
        # regions_to_use will be determined per frame
    else:
        print(" INFO: No manual regions provided & Haar cascade unavailable. No regions will be blurred.")
        method_used = "None"
        # regions_to_use remains empty

    try:
        # Open video capture
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_path}")

        # Get video properties
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Handle potentially invalid FPS
        if fps <= 0 or fps > 240: # Set reasonable limits
            print(f"WARN: Invalid source FPS {fps:.2f} detected. Using 30 FPS for output.")
            fps = 30.0

        print(f" INFO: Video properties: {fw}x{fh} @ {fps:.2f} FPS, ~{total_frames} frames")

        # Define codec and create VideoWriter object (MP4 is generally compatible)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4 output
        out = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))
        if not out.isOpened():
             raise IOError(f"Cannot open VideoWriter for output: {output_path}")

        fc = 0 # Frame counter
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            fc += 1
            processed_frame = frame # Start with the original frame
            current_regions_for_blur = []

            if using_manual:
                current_regions_for_blur = regions_to_use
            elif method_used == "Automatic (Haar)": # Check if auto-detection is enabled
                current_regions_for_blur = detect_faces_haar(frame)
                total_detections += len(current_regions_for_blur)
            # else: current_regions_for_blur remains empty []

            # Apply blur if there are regions for this frame
            if len(current_regions_for_blur) > 0:
                processed_frame = blur_regions(frame, current_regions_for_blur, blur_factor)
                if not using_manual: # If auto-detecting, report count for this frame
                    num_regions_processed_in_frame = len(current_regions_for_blur)

            out.write(processed_frame)

            # Log progress periodically
            if fc % 100 == 0 or fc == 1: # Log every 100 frames and the first frame
                 elapsed = time.time() - start_time
                 cur_fps = fc / elapsed if elapsed > 0 else 0
                 progress = f"{fc}/{total_frames}" if total_frames > 0 else f"{fc}"
                 print(f"  Processed frame {progress}... ({cur_fps:.1f} FPS)")

        end_time = time.time()

        # Determine final count to report
        # If manual, report the static count. If auto, report total detections across frames.
        num_regions_reported = num_regions_processed_in_frame if using_manual else total_detections

        print(f"--- Video Processing Finished ---")
        print(f"INFO: Total frames processed: {fc}")
        if method_used == "Automatic (Haar)":
            print(f"INFO: Total automatic face detections across all frames: {total_detections}")
        elif using_manual:
             print(f"INFO: Static manual regions applied: {num_regions_processed_in_frame}")
        print(f"INFO: Output saved to: {output_path}")
        print(f"INFO: Total processing time: {end_time - start_time:.2f}s")

        return True, num_regions_reported, method_used

    except Exception as e:
        print(f"ERROR processing video {os.path.basename(input_path)}: {e}")
        traceback.print_exc()
        return False, 0, "Error"
    finally:
        # Ensure resources are released
        if cap and cap.isOpened():
            cap.release()
            print(" INFO: VideoCapture released.")
        if out and out.isOpened():
            out.release()
            print(" INFO: VideoWriter released.")


# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Serves the main HTML page."""
    print(f"INFO: Request received for '/'. Rendering 'index.html'.")
    try:
        # Pass max upload size to template if needed (optional)
        max_size_mb = app.config.get('MAX_CONTENT_LENGTH', 0) / (1024 * 1024)
        return render_template('index.html', max_upload_mb=max_size_mb)
    except Exception as e:
        print(f"CRITICAL ERROR rendering template 'index.html': {e}")
        traceback.print_exc()
        # Provide a minimal error page if template rendering fails
        return f"<h1>Internal Server Error</h1><p>Could not render the main application page. Please check server logs.</p>", 500

@app.route('/process', methods=['POST'])
def process_file():
    """Handles file upload, processing, and returns result info."""
    print("INFO: Received request for /process")
    global haar_cascade # Ensure access to the loaded model state

    # 1. Check if file part exists
    if 'file' not in request.files:
        print("ERROR: 'file' part missing in request.files")
        return jsonify({"success": False, "error": "No file part in the request."}), 400

    file = request.files['file']

    # 2. Check if filename is empty (no file selected)
    if file.filename == '':
        print("ERROR: No file selected (empty filename).")
        return jsonify({"success": False, "error": "No file selected."}), 400

    # 3. Validate file type
    filename = secure_filename(file.filename) # Sanitize filename
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    is_image = allowed_file(filename, ALLOWED_EXTENSIONS_IMG)
    is_video = allowed_file(filename, ALLOWED_EXTENSIONS_VID)

    if not is_image and not is_video:
        print(f"ERROR: Invalid file type '{file_ext}' for filename '{filename}'.")
        return jsonify({"success": False, "error": f"Invalid file type: '{file_ext}'. Allowed images: {ALLOWED_EXTENSIONS_IMG}, videos: {ALLOWED_EXTENSIONS_VID}"}), 415 # Unsupported Media Type

    # 4. Get parameters (blur factor, manual regions)
    try:
        blur_factor = float(request.form.get('blurFactor', 0.4))
        # Clamp blur factor to a reasonable range
        blur_factor = max(0.05, min(1.0, blur_factor))
        print(f"INFO: Using blur factor: {blur_factor:.2f}")
    except (ValueError, TypeError):
        blur_factor = 0.4 # Default value if conversion fails
        print("WARN: Invalid blur factor received. Using default 0.4")

    manual_regions_json = request.form.get('manualRegions', '[]')
    manual_regions = []
    regions_were_provided = False
    try:
        parsed_regions = json.loads(manual_regions_json)
        if isinstance(parsed_regions, list) and len(parsed_regions) > 0:
            valid_regions_count = 0
            for region in parsed_regions:
                # Validate each region structure and convert to int
                if isinstance(region, (list, tuple)) and len(region) >= 4:
                     try:
                         manual_regions.append(list(map(int, region[:4])))
                         valid_regions_count += 1
                     except (ValueError, TypeError):
                         print(f"WARN: Skipping invalid region data in manualRegions: {region}")
            if valid_regions_count > 0:
                 regions_were_provided = True
                 print(f"INFO: Received {len(manual_regions)} valid manual blur regions.")
        else:
            print(f"INFO: No manual regions provided or data was empty/invalid.")
    except json.JSONDecodeError:
        print(f"WARN: Could not decode manualRegions JSON: {manual_regions_json}")
    except Exception as e:
        print(f"WARN: Error processing manualRegions: {e}")

    # 5. Crucial Check: Can we proceed? Need manual regions OR a loaded cascade.
    if not regions_were_provided and haar_cascade is None:
        print("ERROR: Cannot process - No manual regions provided AND Haar cascade model is not loaded.")
        # Provide specific error message to the frontend
        return jsonify({"success": False, "error": "Processing unavailable: Please draw regions manually or wait for the administrator to fix the automatic detector."}), 400 # Bad Request or 503 Service Unavailable could also fit

    # 6. Prepare file paths
    try:
        # Ensure upload folder exists (might be redundant if checked at startup, but safe)
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create upload directory '{current_app.config['UPLOAD_FOLDER']}': {e}")
        return jsonify({"success": False, "error": "Server directory configuration error."}), 500

    unique_id = uuid.uuid4().hex
    input_filename = f"{unique_id}_input.{file_ext}"
    # Always output video as mp4 for better web compatibility
    output_ext = 'mp4' if is_video else file_ext
    output_filename = f"{unique_id}_blurred.{output_ext}"
    input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], input_filename)
    output_path = os.path.join(current_app.config['UPLOAD_FOLDER'], output_filename)

    # 7. Save uploaded file and process
    success = False
    num_regions_processed = 0
    method_used = "Error"
    try:
        file.save(input_path)
        print(f"INFO: Uploaded media saved to: {input_path}")

        start_time = time.time()
        if is_image:
            success, num_regions_processed, method_used = process_uploaded_image(input_path, output_path, blur_factor, manual_regions)
        elif is_video:
            success, num_regions_processed, method_used = process_uploaded_video(input_path, output_path, blur_factor, manual_regions)
        end_time = time.time()

        if success:
             print("SUCCESS: Processing complete.")
             # Generate URL for the processed file
             result_url = url_for('uploaded_file', filename=output_filename, _external=True) # Use _external=True if needed
             return jsonify({
                 "success": True,
                 "filename": output_filename, # Filename for download suggestion
                 "url": result_url,           # URL to access the file
                 "is_video": is_video,
                 "processing_time": f"{end_time - start_time:.2f}",
                 "regions_processed": num_regions_processed, # Count based on method
                 "method_used": method_used
             }), 200
        else:
            # Processing function returned failure, reason should be logged by the function
            print("ERROR: Processing function returned failure.")
            return jsonify({"success": False, "error": "Processing failed internally. See server logs for details."}), 500

    except Exception as e:
        # Catch any unexpected errors during file save or processing call
        print(f"ERROR: Exception during file processing route for {filename}: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": f"An internal server error occurred during processing."}), 500
    finally:
        # 8. Clean up the original uploaded file regardless of success/failure
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
                print(f"INFO: Removed temporary input file: {input_path}")
            except Exception as e_rem:
                # Log warning if cleanup fails, but don't fail the request
                print(f"WARNING: Could not remove temporary input file {input_path}: {e_rem}")


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serves files from the UPLOAD_FOLDER."""
    # Security: Ensure filename is safe and doesn't traverse directories
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
         print(f"WARN: Potentially unsafe filename requested: {filename}. Serving rejected.")
         return "Forbidden", 403

    # Security: Prevent path traversal by ensuring the path is within the UPLOAD_FOLDER
    safe_dir = os.path.abspath(current_app.config['UPLOAD_FOLDER'])
    safe_path = os.path.abspath(os.path.join(safe_dir, safe_filename))

    if not safe_path.startswith(safe_dir):
        print(f"WARN: Directory traversal attempt blocked for filename: {filename}")
        return "Forbidden", 403

    try:
        print(f"INFO: Serving file: {safe_filename} from {safe_dir}")
        return send_from_directory(
            current_app.config['UPLOAD_FOLDER'],
            safe_filename,
            as_attachment=False # Serve inline if possible (browser decides based on type)
        )
    except FileNotFoundError:
        print(f"ERROR: Requested file not found: {safe_filename}")
        return "File not found", 404
    except Exception as e:
        print(f"ERROR serving file {safe_filename}: {e}")
        traceback.print_exc()
        return "Error serving file", 500


# --- Main Execution ---
if __name__ == '__main__':
    print("======================================================")
    print(" Starting Flask Face Blur Application (v6.3) ")
    print("======================================================")

    # Check/Create Upload folder permissions *before* loading model
    # Note: On ephemeral filesystems (like Render free tier), this folder might reset.
    print(f"INFO: Ensuring upload folder exists at: {UPLOAD_FOLDER_PATH}")
    if not check_folder_permissions(UPLOAD_FOLDER_PATH):
        print("\n!!! CRITICAL ERROR: Cannot write to upload folder. Uploads/Processing will fail. Check permissions. !!!\n")
        # Consider exiting if uploads are essential and permission fails
        # exit(1)

    # Load the Haar Cascade model
    model_loaded = load_haar_cascade_on_startup(HAAR_CASCADE_FILENAME)
    if not model_loaded:
        print("\n!!! WARNING: Haar Cascade model failed to load. Automatic detection fallback disabled. !!!")
        print("!!! WARNING: Processing will ONLY work if manual regions are drawn by the user. !!!\n")
    else:
         print("\nINFO: Haar Cascade loaded. Automatic detection is available.\n")

    # Determine Flask run settings from environment variables
    flask_debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    # Render and other PaaS platforms set the PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    # Host '0.0.0.0' is necessary to accept connections from outside the container
    host = '0.0.0.0'

    print(f"--- Flask Server Settings ---")
    print(f" * Debug Mode: {flask_debug}")
    print(f" * Host: {host}")
    print(f" * Port: {port}")
    print(f" * Max Upload Size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f} MB")
    print(f"----------------------------")

    # Recommended for production: Use Waitress or Gunicorn (add to requirements.txt)
    # Example using Waitress:
    # if not flask_debug:
    #    try:
    #        from waitress import serve
    #        print("\nINFO: Starting server with Waitress production server...")
    #        serve(app, host=host, port=port)
    #    except ImportError:
    #        print("\nWARN: Waitress not installed. Falling back to Flask development server.")
    #        print("INFO: Consider installing Waitress for production: pip install waitress")
    #        app.run(debug=flask_debug, host=host, port=port)
    # else:
    #    print("\nINFO: Starting server with Flask development server (Debug Mode)...")
    #    app.run(debug=flask_debug, host=host, port=port)

    # Simpler approach: Always use Flask's built-in server (less performant for production)
    print(f"\nINFO: Starting Flask server ({'Debug' if flask_debug else 'Production'} mode)...")
    app.run(debug=flask_debug, host=host, port=port)
