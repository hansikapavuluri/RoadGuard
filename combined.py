import os
import csv
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# RoadGuard: upload-friendly processing pipeline.
# Import from backend/detect.py with:
# process_video(input_path, output_video_path, output_csv_path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POTHOLE_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
VEHICLE_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "yolo11n.pt"),
    os.path.join(BASE_DIR, "yolo11n.pt"),
]
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

POTHOLE_CONFIDENCE = 0.10  # Lower threshold to catch less-confident potholes
VEHICLE_CONFIDENCE = 0.30
MANUAL_VEHICLE_SPEED_KMH = 36.0  # Assumed speed, not measured from video
VEHICLE_CLASSES = [2, 3, 5, 7]   # COCO: car, motorcycle, bus, truck
PROCESS_EVERY_N_FRAMES = 5
DEPTH_EVERY_N_PROCESSED_FRAMES = 20
YOLO_IMAGE_SIZE = 650  # Higher resolution helps detect small/distant road defects
POTHOLE_ROI_LOWER_FRACTION = 0.45
FORWARD_CORRIDOR_X_MIN = 0.18
FORWARD_CORRIDOR_X_MAX = 0.82
FORWARD_CORRIDOR_Y_MIN = 0.28
FORWARD_CORRIDOR_Y_MAX = 0.94
DRAW_VEHICLE_BOXES = False  # Keep vehicle tracking, but avoid cluttering the video
MIN_VALID_DEPTH_M = 0.1
MAX_VALID_DEPTH_M = 80.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_round(value, digits=2, default=""):
    """Round numeric values safely, including numeric strings from video metadata."""
    if value is None or value == "":
        return default
    try:
        number = float(value)
        if not np.isfinite(number):
            return default
        return round(number, digits)
    except (TypeError, ValueError, OverflowError):
        return default

_pothole_model = None
_vehicle_model = None
_depth_processor = None
_depth_model = None


def _load_models():
    global _pothole_model, _vehicle_model, _depth_processor, _depth_model
    if _pothole_model is None:
        if not os.path.isfile(POTHOLE_MODEL_PATH):
            raise FileNotFoundError(
                f"Pothole model not found: {POTHOLE_MODEL_PATH}. "
                "Place best.pt in UE_PROJECT/models/."
            )
        _pothole_model = YOLO(POTHOLE_MODEL_PATH)

    if _vehicle_model is None:
        vehicle_path = next(
            (p for p in VEHICLE_MODEL_CANDIDATES if os.path.isfile(p)), None
        )
        _vehicle_model = YOLO(vehicle_path or "yolo11n.pt")

    if _depth_processor is None or _depth_model is None:
        _depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
        _depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME)
        _depth_model.to(DEVICE)
        _depth_model.eval()
    return _pothole_model, _vehicle_model, _depth_processor, _depth_model


def _clamp_box(box, width, height):
    x1, y1, x2, y2 = [int(v) for v in box]
    return (
        max(0, min(x1, width - 1)), max(0, min(y1, height - 1)),
        max(0, min(x2, width)), max(0, min(y2, height))
    )


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bottom_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


def _depth_stats(depth_map, box, lower_fraction=0.0):
    x1, y1, x2, y2 = box
    if lower_fraction > 0:
        y1 = int(y1 + (y2 - y1) * (1.0 - lower_fraction))
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    values = roi[np.isfinite(roi)]
    values = values[(values >= MIN_VALID_DEPTH_M) & (values <= MAX_VALID_DEPTH_M)]
    return float(np.median(values)) if values.size else None


def _label(frame, text, origin, color, scale=0.5, thickness=2):
    x, y = int(origin[0]), int(origin[1])
    y = max(20, min(y, frame.shape[0] - 8))
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _estimate_depth(frame, processor, model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
        depth = torch.nn.functional.interpolate(
            output.predicted_depth.unsqueeze(1),
            size=(frame.shape[0], frame.shape[1]),
            mode="bicubic", align_corners=False
        ).squeeze(1)
    return depth[0].float().cpu().numpy()


def _select_upcoming(potholes, width, height):
    """
    Select ONE candidate in the forward road corridor.
    This is a camera-view heuristic, not a calibrated lane/road geometry model.
    """
    choices = []
    for p in potholes:
        cx, cy = p["center"]
        in_forward_corridor = (
            width * FORWARD_CORRIDOR_X_MIN <= cx <= width * FORWARD_CORRIDOR_X_MAX
            and height * FORWARD_CORRIDOR_Y_MIN <= cy <= height * FORWARD_CORRIDOR_Y_MAX
        )
        if not in_forward_corridor:
            continue

        # Prefer nearer estimated camera depth. If depth is unavailable,
        # prioritize a candidate lower in the forward image and then confidence.
        if p["depth_m"] is not None:
            key = (0, p["depth_m"], -cy, -p["conf"])
        else:
            key = (1, 0, -cy, -p["conf"])
        choices.append((key, p))

    return min(choices, key=lambda item: item[0])[1] if choices else None


def _ttc(distance_m, speed_kmh):
    if distance_m is None or not np.isfinite(distance_m) or distance_m <= 0:
        return None
    if speed_kmh is None or not np.isfinite(speed_kmh) or speed_kmh <= 0:
        return None
    return float(distance_m / (speed_kmh / 3.6))


def _impact(pothole, width, height, speed_kmh):
    """
    Prototype risk score, NOT a validated prediction of vehicle damage.
    Bounding-box area is only an apparent-size proxy and changes with
    perspective, camera angle, distance and detector-box quality.
    Speed is currently assumed (36 km/h), not measured from the video.
    """
    x1, y1, x2, y2 = pothole["box"]
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_ratio = box_area / max(1, width * height)

    try:
        speed = float(speed_kmh)
    except (TypeError, ValueError):
        speed = 0.0
    if not np.isfinite(speed):
        speed = 0.0
    speed = min(max(speed, 0.0), 120.0)

    # 70 points for apparent pothole size and 30 for assumed vehicle speed.
    size_component = min(area_ratio / 0.025, 1.0) * 70.0
    speed_component = min(speed / 80.0, 1.0) * 30.0
    score = _safe_round(size_component + speed_component, 1, default=0.0)

    if score < 30:
        level = "LOW"
    elif score < 60:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return score, level


def _draw_warning(frame, pothole):
    """Draw a compact warning panel that leaves more of the road visible."""
    h, w = frame.shape[:2]
    panel_x1, panel_y1 = 12, 12
    panel_x2 = min(w - 12, max(420, int(w * 0.76)))
    panel_h = min(h - 12, 124 if pothole is not None else 100)

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y1 + panel_h),
                  (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    if pothole is None:
        _label(frame, "ROADGUARD | NO POTHOLE IN FORWARD CORRIDOR",
               (26, 38), (220, 220, 220), 0.52, 2)
        _label(frame, "No warning does not guarantee the road is clear.",
               (26, 66), (185, 185, 185), 0.42, 1)
        _label(frame, f"Assumed speed: {MANUAL_VEHICLE_SPEED_KMH:.1f} km/h",
               (26, 90), (200, 200, 200), 0.42, 1)
        return None

    distance = pothole.get("depth_m")
    ttc = _ttc(distance, MANUAL_VEHICLE_SPEED_KMH)
    score, level = _impact(pothole, w, h, MANUAL_VEHICLE_SPEED_KMH)

    _label(frame, "WARNING: UPCOMING POTHOLE DETECTED",
           (26, 39), (0, 105, 255), 0.55, 2)
    distance_text = (f"Estimated camera depth: {float(distance):.1f} m*"
                     if distance is not None else "Estimated camera depth: unavailable")
    _label(frame, distance_text, (26, 68), (0, 220, 255), 0.47, 2)
    _label(frame, f"Assumed speed: {MANUAL_VEHICLE_SPEED_KMH:.1f} km/h",
           (26, 94), (235, 235, 235), 0.40, 1)
    ttc_text = f"Time to reach: {ttc:.1f} s**" if ttc is not None else "Time to reach: unavailable"
    _label(frame, ttc_text, (26, 116), (235, 235, 235), 0.42, 1)

    # Put impact information on the right side when enough room is available.
    impact_x = max(300, int(w * 0.53))
    if impact_x < panel_x2 - 20:
        _label(frame, f"Prototype impact: {level} ({score:.1f}/100)",
               (impact_x, 116), (0, 190, 255), 0.40, 1)

    return {"distance_m": distance, "ttc_seconds": ttc,
            "impact_score": score, "impact_level": level}


def process_video(input_path, output_video_path, output_csv_path=None):
    """Process an uploaded video; return a JSON-serializable summary."""
    input_path = os.path.abspath(input_path)
    output_video_path = os.path.abspath(output_video_path)
    if output_csv_path is None:
        output_csv_path = os.path.splitext(output_video_path)[0] + "_detections.csv"
    output_csv_path = os.path.abspath(output_csv_path)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    pothole_model, vehicle_model, processor, depth_model = _load_models()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Input video has invalid dimensions.")

    # We write every 3rd source frame, so write at source_fps/3 to preserve
    # approximately the original video's playback duration. CSV timestamps use
    # source-video time and therefore align with the processed preview.
    output_fps = max(1.0, fps / PROCESS_EVERY_N_FRAMES)
    writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    csv_fields = [
        "frame", "timestamp_seconds", "object_type", "object_number", "track_id",
        "class_name", "confidence", "x1", "y1", "x2", "y2",
        "estimated_camera_to_pothole_depth_m", "object_median_depth_m",
        "nearest_pothole_depth_m", "estimated_depth_difference_m",
        "nearest_pothole_pixel_gap", "distance_status", "vehicle_speed_kmh",
        "estimated_time_to_reach_seconds", "impact_score_0_100", "impact_level",
        "potholes_in_frame", "vehicles_in_frame", "upcoming_pothole_detected",
        "frame_distance_m", "frame_ttc_seconds", "frame_impact_score",
        "frame_impact_level", "vehicle_speed_is_assumed"
    ]
    frame_no = processed_frames = 0
    pothole_frame_detections = vehicle_frame_detections = warning_frames = 0
    vehicle_track_ids = set()
    cached_depth = None
    depth_countdown = 0
    last_warning = None

    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            csv_writer.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_no += 1
                if (frame_no - 1) % PROCESS_EVERY_N_FRAMES:
                    continue
                timestamp = (frame_no - 1) / fps

                p_result = pothole_model.predict(
                    frame, conf=POTHOLE_CONFIDENCE, imgsz=YOLO_IMAGE_SIZE,
                    verbose=False
                )[0]
                v_result = vehicle_model.track(
                    source=frame, conf=VEHICLE_CONFIDENCE,
                    classes=VEHICLE_CLASSES, imgsz=YOLO_IMAGE_SIZE,
                    persist=True, tracker="bytetrack.yaml", verbose=False
                )[0]

                if cached_depth is None or depth_countdown <= 0:
                    cached_depth = _estimate_depth(frame, processor, depth_model)
                    depth_countdown = DEPTH_EVERY_N_PROCESSED_FRAMES - 1
                else:
                    depth_countdown -= 1
                depth_map = cached_depth

                potholes = []
                if p_result.boxes is not None:
                    for b in p_result.boxes:
                        box = _clamp_box(b.xyxy[0].detach().cpu().numpy(), width, height)
                        x1, y1, x2, y2 = box
                        if x2 <= x1 or y2 <= y1:
                            continue
                        p = {
                            "box": box, "center": _center(box),
                            "bottom": _bottom_center(box),
                            "conf": float(b.conf[0].item()),
                            "depth_m": _depth_stats(depth_map, box, POTHOLE_ROI_LOWER_FRACTION)
                        }
                        potholes.append(p)

                vehicles = []
                if v_result.boxes is not None:
                    for b in v_result.boxes:
                        box = _clamp_box(b.xyxy[0].detach().cpu().numpy(), width, height)
                        x1, y1, x2, y2 = box
                        if x2 <= x1 or y2 <= y1:
                            continue
                        cls_id = int(b.cls[0].item())
                        track_id = int(b.id[0].item()) if b.id is not None else None
                        if track_id is not None:
                            vehicle_track_ids.add(track_id)
                        vehicles.append({
                            "box": box, "center": _center(box), "bottom": _bottom_center(box),
                            "conf": float(b.conf[0].item()),
                            "class_name": vehicle_model.names.get(cls_id, str(cls_id)),
                            "track_id": track_id, "depth_m": _depth_stats(depth_map, box)
                        })

                upcoming = _select_upcoming(potholes, width, height)
                if upcoming:
                    # Highlight only the selected upcoming pothole in RED.
                    # Other detections remain in the CSV but are not drawn,
                    # preventing a frame full of overlapping boxes.
                    x1, y1, x2, y2 = upcoming["box"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    d = upcoming["depth_m"]
                    red_text = (
                        f"UPCOMING POTHOLE | {upcoming['conf']:.2f} | depth {d:.1f} m"
                        if d is not None
                        else f"UPCOMING POTHOLE | confidence {upcoming['conf']:.2f}"
                    )
                    label_y = y1 - 10 if y1 > 35 else y2 + 22
                    _label(frame, red_text, (x1, label_y), (0, 0, 255), 0.52, 2)
                    warning_frames += 1

                for i, p in enumerate(potholes, 1):
                    x1, y1, x2, y2 = p["box"]
                    d = p["depth_m"]
                    # Keep every pothole detection in CSV, but draw only the
                    # single selected upcoming pothole in the processed video.
                    # Give each detected pothole a prototype score even when no
                    # vehicle can be associated with it. This keeps pothole-level
                    # impact information available in the CSV/dashboard.
                    pothole_score, pothole_level = _impact(
                        p, width, height, MANUAL_VEHICLE_SPEED_KMH
                    )
                    pothole_ttc = _ttc(d, MANUAL_VEHICLE_SPEED_KMH)

                    csv_writer.writerow({
                        "frame": frame_no, "timestamp_seconds": _safe_round(timestamp, 3),
                        "object_type": "pothole", "object_number": i, "track_id": "",
                        "class_name": "pothole", "confidence": _safe_round(p["conf"], 4),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "estimated_camera_to_pothole_depth_m": _safe_round(d, 3) if d is not None else "",
                        "object_median_depth_m": _safe_round(d, 3) if d is not None else "",
                        "nearest_pothole_depth_m": "", "estimated_depth_difference_m": "",
                        "nearest_pothole_pixel_gap": "",
                        "distance_status": "model_estimate_needs_validation" if d is not None else "depth_unavailable",
                        "vehicle_speed_kmh": MANUAL_VEHICLE_SPEED_KMH,
                        "estimated_time_to_reach_seconds": _safe_round(pothole_ttc, 2) if pothole_ttc is not None else "",
                        "impact_score_0_100": pothole_score,
                        "impact_level": pothole_level
                    })

                for i, v in enumerate(vehicles, 1):
                    x1, y1, x2, y2 = v["box"]
                    vx, vy = v["bottom"]
                    possible = []
                    for p in potholes:
                        px, py = p["bottom"]
                        horizontal_gap = abs(px - vx)
                        same_corridor = horizontal_gap <= max((x2 - x1) * 1.5, width * 0.12)
                        ahead = py >= vy - (y2 - y1) * 0.25
                        if same_corridor and ahead:
                            possible.append((horizontal_gap + 0.35 * abs(py - vy), p,
                                             float(np.hypot(px - vx, py - vy))))
                    nearest = min(possible, key=lambda z: z[0]) if possible else None
                    if DRAW_VEHICLE_BOXES:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 0), 2)
                        track_text = f"ID {v['track_id']}" if v["track_id"] is not None else "Vehicle"
                        _label(frame, f"{track_text} {v['class_name']} {v['conf']:.2f}",
                               (x1, y1 - 8), (255, 150, 0), 0.46, 2)
                    nearest_depth = gap_px = depth_diff = None
                    ttc = impact_score = impact_level = ""
                    if nearest:
                        _, p, gap_px = nearest
                        px, py = p["bottom"]
                        cv2.line(frame, (int(vx), int(vy)), (int(px), int(py)), (0, 165, 255), 3)
                        cv2.circle(frame, (int(px), int(py)), 5, (0, 165, 255), -1)
                        nearest_depth = p["depth_m"]
                        if v["depth_m"] is not None and nearest_depth is not None:
                            depth_diff = abs(nearest_depth - v["depth_m"])
                        label = f"Pothole ahead: approx {nearest_depth:.1f} m from camera" if nearest_depth is not None else "Pothole ahead: depth unavailable"
                        _label(frame, label, (x1, min(height - 12, y2 + 20)), (0, 165, 255), 0.44, 2)
                        impact_score, impact_level = _impact(p, width, height, MANUAL_VEHICLE_SPEED_KMH)
                        ttc = _ttc(nearest_depth, MANUAL_VEHICLE_SPEED_KMH)
                    

                    csv_writer.writerow({
                        "frame": frame_no, "timestamp_seconds": _safe_round(timestamp, 3),
                        "object_type": "vehicle", "object_number": i,
                        "track_id": v["track_id"] if v["track_id"] is not None else "",
                        "class_name": v["class_name"], "confidence": _safe_round(v["conf"], 4),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "estimated_camera_to_pothole_depth_m": _safe_round(nearest_depth, 3) if nearest_depth is not None else "",
                        "object_median_depth_m": _safe_round(v["depth_m"], 3) if v["depth_m"] is not None else "",
                        "nearest_pothole_depth_m": _safe_round(nearest_depth, 3) if nearest_depth is not None else "",
                        "estimated_depth_difference_m": _safe_round(depth_diff, 3) if depth_diff is not None else "",
                        "nearest_pothole_pixel_gap": _safe_round(gap_px, 2) if gap_px is not None else "",
                        "distance_status": "rough_depth_difference_not_calibrated" if depth_diff is not None else ("model_estimate_needs_validation" if nearest_depth is not None else "no_associated_pothole"),
                        "vehicle_speed_kmh": MANUAL_VEHICLE_SPEED_KMH,
                        "estimated_time_to_reach_seconds": _safe_round(ttc, 2) if ttc is not None else "",
                        "impact_score_0_100": impact_score, "impact_level": impact_level
                    })

                warning = _draw_warning(frame, upcoming)
                # Keep the last non-empty warning for the overall job summary,
                # but write the CURRENT frame state separately for dashboard sync.
                if warning:
                    last_warning = warning

                summary_y = max(24, height - 66)
                cv2.rectangle(frame, (10, summary_y - 24),
                              (min(width - 10, 620), height - 10), (20, 20, 20), -1)
                _label(frame, f"Frame {frame_no}/{total_frames}", (22, summary_y),
                       (255, 255, 255), 0.48, 1)
                _label(frame, f"Potholes detected: {len(potholes)} | Vehicles tracked: {len(vehicles)}",
                       (22, summary_y + 23), (0, 255, 0), 0.46, 1)
                _label(frame, "*Depth-model estimate; validate against measured distances.",
                       (max(22, width - 560), height - 24), (255, 255, 255), 0.36, 1)

                # One summary row per processed frame makes it easy for the
                # frontend to update the intelligence panel in sync with video.
                frame_distance = warning.get("distance_m") if warning else None
                frame_ttc = warning.get("ttc_seconds") if warning else None
                frame_impact_score = warning.get("impact_score") if warning else None
                frame_impact_level = warning.get("impact_level") if warning else ""
                csv_writer.writerow({
                    "frame": frame_no,
                    "timestamp_seconds": _safe_round(timestamp, 3),
                    "object_type": "frame_summary",
                    "object_number": "",
                    "track_id": "",
                    "class_name": "frame_summary",
                    "confidence": "",
                    "x1": "", "y1": "", "x2": "", "y2": "",
                    "estimated_camera_to_pothole_depth_m": "",
                    "object_median_depth_m": "",
                    "nearest_pothole_depth_m": "",
                    "estimated_depth_difference_m": "",
                    "nearest_pothole_pixel_gap": "",
                    "distance_status": "model_estimate_needs_validation" if frame_distance is not None else "no_upcoming_pothole",
                    "vehicle_speed_kmh": MANUAL_VEHICLE_SPEED_KMH,
                    "estimated_time_to_reach_seconds": _safe_round(frame_ttc, 2) if frame_ttc is not None else "",
                    "impact_score_0_100": frame_impact_score if frame_impact_score is not None else "",
                    "impact_level": frame_impact_level,
                    "potholes_in_frame": len(potholes),
                    "vehicles_in_frame": len(vehicles),
                    "upcoming_pothole_detected": bool(upcoming),
                    "frame_distance_m": _safe_round(frame_distance, 3) if frame_distance is not None else "",
                    "frame_ttc_seconds": _safe_round(frame_ttc, 2) if frame_ttc is not None else "",
                    "frame_impact_score": frame_impact_score if frame_impact_score is not None else "",
                    "frame_impact_level": frame_impact_level,
                    "vehicle_speed_is_assumed": True,
                })

                writer.write(frame)
                processed_frames += 1
                pothole_frame_detections += len(potholes)
                vehicle_frame_detections += len(vehicles)

    finally:
        cap.release()
        writer.release()

    if processed_frames == 0 or not os.path.isfile(output_video_path) or os.path.getsize(output_video_path) == 0:
        raise RuntimeError("Video processing produced no output frames.")

    return {
        "status": "completed",
        "frames_processed": processed_frames,
        "source_frames": frame_no,
        "pothole_frame_detections": pothole_frame_detections,
        "vehicle_frame_detections": vehicle_frame_detections,
        "unique_vehicle_track_ids_observed": len(vehicle_track_ids),
        "warning_frames": warning_frames,
        "pothole_count_note": "Pothole frame detections are repeated observations across frames, not unique physical potholes.",
        "visualization_note": "Only the selected upcoming pothole is drawn in red; other potholes remain in CSV. Vehicle boxes are disabled by default.",
        "dashboard_csv_rows": "CSV includes one frame_summary row per processed frame for playback-synchronized dashboard updates.",
        "estimated_distance_m": last_warning.get("distance_m") if last_warning else None,
        "estimated_ttc_seconds": last_warning.get("ttc_seconds") if last_warning else None,
        "impact_score": last_warning.get("impact_score") if last_warning else None,
        "impact_level": last_warning.get("impact_level") if last_warning else None,
        "assumed_vehicle_speed_kmh": MANUAL_VEHICLE_SPEED_KMH,
        "distance_note": "Monocular depth estimate; validate against measured distances.",
        "ttc_note": "Uses assumed speed, not measured vehicle speed.",
        "impact_note": "Prototype heuristic, not a validated vehicle-damage prediction.",
        "output_video": os.path.basename(output_video_path),
        "output_csv": os.path.basename(output_csv_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoadGuard video processor")
    parser.add_argument("input_video", help="Path to a road video")
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    stem = os.path.splitext(os.path.basename(args.input_video))[0]
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    output_video = args.output_video or os.path.join(out_dir, f"{stem}_processed.mp4")
    output_csv = args.output_csv or os.path.splitext(output_video)[0] + "_detections.csv"
    print("Device:", DEVICE)
    print(process_video(args.input_video, output_video, output_csv))