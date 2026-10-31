
import os
import sys
import shutil
import subprocess

# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ============================================================
# IMPORT MAIN DETECTION PIPELINE
# ============================================================

try:
    from combined import process_video as combined_process_video
except ImportError as error:
    raise ImportError(
        "Could not import process_video from combined.py. "
        "Ensure combined.py is in the UE_PROJECT root and defines "
        "process_video(input_path, output_video_path, output_csv_path)."
    ) from error


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(input_path, output_path):
    """
    Process an uploaded video using combined.py, then convert
    the annotated video to browser-compatible H.264 MP4.
    """

    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    csv_path = (
        os.path.splitext(output_path)[0] + "_detections.csv"
    )

    print("\n" + "=" * 55)
    print("ROADGUARD: STARTING VIDEO PROCESSING")
    print("Input:", input_path)
    print("Output:", output_path)
    print("=" * 55)

    # --------------------------------------------------------
    # 1. Run the actual pothole/vehicle/depth pipeline
    # --------------------------------------------------------

    results = combined_process_video(
        input_path,
        output_path,
        csv_path
    )

    if not os.path.isfile(output_path):
        raise RuntimeError(
            "combined.py did not create the processed video."
        )

    if os.path.getsize(output_path) == 0:
        raise RuntimeError(
            "The processed video is empty."
        )

    if not isinstance(results, dict):
        raise RuntimeError(
            "combined.py must return a dictionary of results."
        )

    # --------------------------------------------------------
    # 2. Convert the annotated video to H.264
    # --------------------------------------------------------

    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg, restart VS Code, "
            "and verify that 'ffmpeg -version' works."
        )

    temporary_output = (
        os.path.splitext(output_path)[0] + "_h264.mp4"
    )

    command = [
        ffmpeg,
        "-y",
        "-i", output_path,
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        temporary_output
    ]

    print("Converting annotated video for browser playback...")

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=None
        )

        if completed.returncode != 0:
            details = (
                completed.stderr[-3000:]
                if completed.stderr
                else "Unknown FFmpeg error."
            )

            raise RuntimeError(
                f"FFmpeg conversion failed:\n{details}"
            )

        if (
            not os.path.isfile(temporary_output)
            or os.path.getsize(temporary_output) == 0
        ):
            raise RuntimeError(
                "FFmpeg did not create a valid H.264 video."
            )

        # Replace the original output only after successful conversion.
        os.replace(temporary_output, output_path)

    finally:
        if os.path.isfile(temporary_output):
            os.remove(temporary_output)

    # --------------------------------------------------------
    # 3. Return results to Flask
    # --------------------------------------------------------

    results["output_video"] = os.path.basename(output_path)
    results["output_csv"] = os.path.basename(csv_path)

    print("Video processing and conversion completed.")
    print("Processed video:", output_path)
    print("Detection CSV:", csv_path)

    return results

