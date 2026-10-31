
import os
import uuid
import json
import threading
from datetime import datetime, timezone

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from detect import process_video


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)
CORS(app)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
OUTPUT_DIR = os.path.join(BACKEND_DIR, "outputs")
HISTORY_FILE = os.path.join(BACKEND_DIR, "history.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Avoid conflicting history updates.
history_lock = threading.Lock()

# Prevent excessively large uploads. Adjust if required.
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# ============================================================
# HISTORY FUNCTIONS
# ============================================================

def load_history():
    if not os.path.isfile(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            history = json.load(file)

        return history if isinstance(history, list) else []

    except (OSError, json.JSONDecodeError):
        app.logger.exception("Could not load processing history.")
        return []


def save_history(history):
    temporary_file = HISTORY_FILE + ".tmp"

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)

    os.replace(temporary_file, HISTORY_FILE)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "RoadGuard backend is running"
    })


# ============================================================
# UPLOAD AND PROCESS VIDEO
# ============================================================

@app.route("/api/detect/video", methods=["POST"])
def detect_video():

    if "video" not in request.files:
        return jsonify({
            "error": "No video was received. Use the field name 'video'."
        }), 400

    uploaded_file = request.files["video"]

    if not uploaded_file.filename:
        return jsonify({
            "error": "Please select a video."
        }), 400

    original_filename = secure_filename(uploaded_file.filename)
    extension = os.path.splitext(original_filename)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Supported formats: MP4, AVI, MOV, and MKV."
        }), 400

    job_id = uuid.uuid4().hex

    input_path = os.path.join(
        UPLOAD_DIR,
        f"{job_id}{extension}"
    )

    output_filename = f"{job_id}_processed.mp4"

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    csv_filename = f"{job_id}_processed_detections.csv"

    csv_path = os.path.join(
        OUTPUT_DIR,
        csv_filename
    )

    try:
        # Save the uploaded video.
        uploaded_file.save(input_path)

        if (
            not os.path.isfile(input_path)
            or os.path.getsize(input_path) == 0
        ):
            return jsonify({
                "error": "The uploaded video is empty or could not be saved."
            }), 400

        app.logger.info(
            "Processing job %s: %s",
            job_id,
            original_filename
        )

        # Process video and convert output to H.264.
        results = process_video(input_path, output_path)

        if not isinstance(results, dict):
            raise RuntimeError(
                "The detection pipeline did not return valid results."
            )

        if not os.path.isfile(output_path):
            raise RuntimeError(
                "The processed video was not created."
            )

        # Respect the actual CSV filename returned by detect.py.
        returned_csv = results.get("output_csv")

        if returned_csv:
            candidate_csv = os.path.join(
                OUTPUT_DIR,
                os.path.basename(returned_csv)
            )

            if os.path.isfile(candidate_csv):
                csv_path = candidate_csv
                csv_filename = os.path.basename(candidate_csv)

        # Record the completed job.
        record = {
            "job_id": job_id,
            "original_filename": original_filename,
            "processed_video": output_filename,
            "csv_filename": csv_filename,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "results": results
        }

        with history_lock:
            history = load_history()
            history.insert(0, record)
            save_history(history)

        app.logger.info("Job %s completed successfully.", job_id)

        return jsonify({
            "job_id": job_id,
            "message": "Video processing completed successfully.",
            "results": results,
            "video_url": f"/api/videos/{output_filename}",
            "csv_url": f"/api/csv/{csv_filename}"
        }), 200

    except Exception as error:
        app.logger.exception(
            "Video processing failed for job %s.",
            job_id
        )

        return jsonify({
            "error": str(error),
            "job_id": job_id
        }), 500

    finally:
        # Remove only the temporary upload.
        # Keep the processed video and CSV for history.
        if os.path.isfile(input_path):
            try:
                os.remove(input_path)
            except OSError:
                app.logger.warning(
                    "Could not delete temporary upload: %s",
                    input_path
                )


# ============================================================
# HANDLE LARGE UPLOAD ERRORS
# ============================================================

@app.errorhandler(413)
def upload_too_large(error):
    return jsonify({
        "error": "The video exceeds the 500 MB upload limit."
    }), 413


# ============================================================
# SERVE PROCESSED VIDEO
# ============================================================

@app.route("/api/videos/<path:filename>", methods=["GET"])
def get_video(filename):

    safe_filename = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_filename)

    if not os.path.isfile(file_path):
        return jsonify({
            "error": "Processed video not found."
        }), 404

    response = send_from_directory(
        OUTPUT_DIR,
        safe_filename,
        mimetype="video/mp4",
        conditional=True
    )

    response.headers["Cache-Control"] = "no-cache"

    return response


# ============================================================
# DOWNLOAD DETECTION CSV
# ============================================================

@app.route("/api/csv/<path:filename>", methods=["GET"])
def get_csv(filename):

    safe_filename = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_filename)

    if not os.path.isfile(file_path):
        return jsonify({
            "error": "Detection CSV not found."
        }), 404

    return send_from_directory(
        OUTPUT_DIR,
        safe_filename,
        as_attachment=True,
        mimetype="text/csv"
    )


# ============================================================
# GET PROCESSING HISTORY
# ============================================================

@app.route("/api/history", methods=["GET"])
def get_history():

    with history_lock:
        history = load_history()

    return jsonify({
        "count": len(history),
        "history": history
    })


# ============================================================
# GET ONE PROCESSING RECORD
# ============================================================

@app.route("/api/history/<job_id>", methods=["GET"])
def get_history_item(job_id):

    with history_lock:
        history = load_history()

    for record in history:
        if record.get("job_id") == job_id:
            return jsonify(record)

    return jsonify({
        "error": "Processing record not found."
    }), 404


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("ROADGUARD BACKEND")
    print("Server:  http://127.0.0.1:5000")
    print("Health:  http://127.0.0.1:5000/api/health")
    print("History: http://127.0.0.1:5000/api/history")
    print("=" * 55)

    app.run(
        host="127.0.0.1",
    )
