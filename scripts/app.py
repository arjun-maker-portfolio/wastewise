import argparse
import json
import logging
import os
import signal
import sys
import time
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import uuid

try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    firebase_admin = None
    credentials = None
    firestore = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.food_classifier import classify_food, load_classifier
from src.vision_steel import (
    estimate_volume_from_contour,
    process_frame_steel_utensil,
    draw_detection_overlay,
)

UTC_TZ = timezone.utc
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
LOCAL_DATA_FILE = LOCAL_DATA_DIR / "utensil_sessions.json"
EMPTY_FOOD_LABELS = {"nothing", "none", "no_food", "empty", "empty_plate"}

shutdown_event = Event()

firebase_db = None
firebase_enabled = False


def setup_logging(log_file: str = None, verbose: bool = False):

    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def init_firebase(logger):

    global firebase_db, firebase_enabled

    if not FIREBASE_AVAILABLE:
        logger.warning("Firebase SDK not available. Install with: pip install firebase-admin")
        return False

    try:
        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH") or os.getenv("FIREBASE_CREDENTIALS_FILE")
        if not cred_path:
            cred_path = PROJECT_ROOT / "firebase-credentials.json"

        cred_path = Path(cred_path).expanduser()
        if not cred_path.is_absolute():
            cred_path = PROJECT_ROOT / cred_path

        if not cred_path.exists():
            logger.warning(f"Firebase credentials not found at: {cred_path}")
            logger.info(
                "Set FIREBASE_CREDENTIALS_PATH environment variable or place firebase-credentials.json in project root"
            )
            return False

        if not firebase_admin._apps:
            cred = credentials.Certificate(str(cred_path))
            firebase_admin.initialize_app(cred)

        firebase_db = firestore.client()
        firebase_enabled = True
        logger.info("Firestore initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        return False


def init_local_storage():

    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LOCAL_DATA_FILE.exists():
        save_local_data([])
    return LOCAL_DATA_FILE


def load_local_data():

    if LOCAL_DATA_FILE.exists():
        try:
            with open(LOCAL_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            raise RuntimeError(f"Cannot read local records: {LOCAL_DATA_FILE}") from exc
    return []


def save_local_data(data: list):

    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=LOCAL_DATA_DIR,
            prefix=".sessions-",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, LOCAL_DATA_FILE)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def save_to_firebase(collection: str, payload: dict, logger) -> str:

    if not firebase_enabled or firebase_db is None:
        return None

    try:
        doc_id = payload.pop("id", None)

        for key, value in payload.items():
            if isinstance(value, datetime):
                payload[key] = value.isoformat()

        if doc_id:
            firebase_db.collection(collection).document(doc_id).set(payload)
        else:
            doc_ref = firebase_db.collection(collection).add(payload)
            doc_id = doc_ref[1].id

        logger.debug(f"Saved to Firebase collection '{collection}': {doc_id}")
        return doc_id

    except Exception as e:
        logger.error(f"Failed to save to Firebase: {e}")
        return None


def save_session_to_local(session_data: dict, logger):

    try:
        now = datetime.now(UTC_TZ)
        doc_id = str(uuid.uuid4())

        payload = {
            "id": doc_id,
            "utensil_type": session_data.get("shape_type", "unknown"),
            "container_type": session_data.get("container_type"),
            "detection_type": "session",
            "foods": session_data.get("foods", []),
            "average_volume_ml": session_data.get("volume_ml"),
            "average_percent_fill": session_data.get("percent_fill"),
            "sample_count": session_data.get("sample_count", 1),
            "detected_food": session_data.get("food_label"),
            "food_confidence": session_data.get("food_confidence"),
            "started_at": str(session_data.get("started_at", now)),
            "ended_at": str(session_data.get("ended_at", now)),
            "created_at": str(now),
        }

        synced = False
        if firebase_enabled:
            cloud_id = save_to_firebase("utensil_sessions", payload.copy(), logger)
            if cloud_id:
                synced = True
                logger.info(f"Session saved to Firebase: {cloud_id}")

        try:
            data = load_local_data()
            payload["firebase_synced"] = synced
            data.append(payload)
            save_local_data(data)
            logger.info(f"Session saved to local JSON (failsafe): {doc_id}")
        except Exception as e:
            logger.error(f"Failed to save to local JSON failsafe: {e}")

        return doc_id

    except Exception as e:
        logger.error(f"Failed to save session: {e}")

    return None


def save_detection_to_local(result: dict, logger):

    try:
        now = datetime.now(UTC_TZ)
        doc_id = str(uuid.uuid4())

        payload = {
            "id": doc_id,
            "utensil_type": result.get("shape_type", "unknown"),
            "container_type": result.get("container_type"),
            "average_volume_ml": result.get("volume_ml"),
            "average_percent_fill": result.get("percent_fill"),
            "sample_count": 1,
            "detected_food": result.get("food_label"),
            "food_confidence": result.get("food_confidence"),
            "started_at": str(now),
            "ended_at": str(now),
            "created_at": str(now),
            "detection_type": "single_frame",
            "foods": result.get("foods", []),
        }

        synced = False
        if firebase_enabled:
            cloud_id = save_to_firebase("utensil_sessions", payload.copy(), logger)
            if cloud_id:
                synced = True
                logger.debug(f"Detection saved to Firebase: {cloud_id}")

        try:
            data = load_local_data()
            payload["firebase_synced"] = synced
            data.append(payload)
            save_local_data(data)
            logger.debug(f"Detection saved to local JSON (failsafe): {doc_id}")
        except Exception as e:
            logger.error(f"Failed to save to local JSON failsafe: {e}")

        return doc_id

    except Exception as e:
        logger.error(f"Failed to save detection: {e}")

    return None


class SessionTracker:
    def __init__(self, empty_frames_to_finalize: int = 3, tolerance_frames: int = 2):
        self.empty_frames_to_finalize = empty_frames_to_finalize
        self.tolerance_frames = tolerance_frames
        self.reset()

    def reset(self):
        self.active = False
        self.started_at = None
        self.last_seen_at = None
        self.volumes = []
        self.percents = []
        self.food_labels = []
        self.confidences = []
        self.food_samples = []
        self.shape_type = None
        self.container_type = None
        self.empty_streak = 0
        self.missed_frames = 0
        self.frame_count = 0

    def update(self, result: dict, food_label: str = None, food_confidence: float = None) -> dict:

        now = datetime.now(UTC_TZ)

        utensil_detected = result.get("utensil_detected", False)
        percent_fill = result.get("percent_fill")
        volume_ml = result.get("volume_ml")
        shape_type = result.get("shape_type")

        label_lower = food_label.lower() if food_label else ""
        empty_detected = label_lower in EMPTY_FOOD_LABELS or (
            percent_fill is not None and percent_fill < 2.0
        )

        if utensil_detected and not empty_detected:
            if not self.active:
                self.active = True
                self.started_at = now
                self.shape_type = shape_type
                self.container_type = result.get("container_type")

        if not self.active:
            self.empty_streak = 0
            return None

        if utensil_detected:
            self.missed_frames = 0
            self.last_seen_at = now
            self.frame_count += 1

            if empty_detected:
                self.empty_streak += 1
                if self.empty_streak >= self.empty_frames_to_finalize:
                    return self._finalize(now)
                return None

            self.empty_streak = 0

            self.food_samples.append(result.get("foods", []))

            if percent_fill is not None:
                self.percents.append(percent_fill)
            if volume_ml is not None:
                self.volumes.append(volume_ml)
            if food_label:
                self.food_labels.append(food_label)
                if food_confidence is not None:
                    self.confidences.append(food_confidence)

            return None

        else:
            self.missed_frames += 1

            if self.missed_frames > self.tolerance_frames:
                return self._finalize(now)

            return None

    def _finalize(self, ended_at: datetime) -> dict:
        if not self.active:
            self.reset()
            return None

        vols = [v for v in self.volumes if v is not None]
        fills = [p for p in self.percents if p is not None]

        if not vols and not fills:
            self.reset()
            return None

        avg_volume = sum(vols) / len(vols) if vols else None
        avg_percent = sum(fills) / len(fills) if fills else None

        sample_count = max(self.frame_count, len(vols), len(fills))

        food_label = None
        if self.food_labels:
            counts = Counter(self.food_labels)
            food_label = counts.most_common(1)[0][0]

        avg_conf = None
        if self.confidences:
            avg_conf = sum(self.confidences) / len(self.confidences)

        # combine food types across the session
        contributions = {}
        for regions in self.food_samples:
            for region in regions:
                label = region.get("food_label") or "unknown"
                item = contributions.setdefault(
                    label,
                    {
                        "food_label": label,
                        "area_mm2": 0.0,
                        "volume_ml": 0.0,
                        "percent_fill": 0.0,
                    },
                )
                for key in ("area_mm2", "volume_ml", "percent_fill"):
                    item[key] += region[key]
        for item in contributions.values():
            for key in ("area_mm2", "volume_ml", "percent_fill"):
                item[key] /= len(self.food_samples)

        session = {
            "foods": list(contributions.values()),
            "shape_type": self.shape_type,
            "container_type": self.container_type,
            "volume_ml": avg_volume,
            "percent_fill": avg_percent,
            "food_label": food_label,
            "food_confidence": avg_conf,
            "sample_count": sample_count,
            "started_at": self.started_at,
            "ended_at": self.last_seen_at or ended_at,
        }

        self.reset()
        return session

    def force_finalize(self) -> dict:
        if self.active:
            return self._finalize(datetime.now(UTC_TZ))
        return None


def classify_food_regions(frame, detection):
    food_mask = detection.get("food_mask")
    plate_mask = detection.get("utensil_mask")
    if food_mask is None or plate_mask is None:
        return []
    mask = ((food_mask > 0) & (plate_mask > 0)).astype(np.uint8)
    # classify disconnected food patches
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    plate_pixels = np.count_nonzero(plate_mask)
    scale = detection["volume_info"]["mm_per_px"]
    depth = detection["volume_info"]["effective_depth_mm"]
    foods = []
    for region_id in range(1, count):
        x, y, width, height, pixels = map(int, stats[region_id])
        region_mask = components[y : y + height, x : x + width] == region_id
        crop = frame[y : y + height, x : x + width].copy()
        crop[~region_mask] = 0
        pred = classify_food(crop)
        label, confidence = pred if pred else (None, None)
        area = pixels * scale**2
        foods.append(
            {
                "region_id": region_id,
                "bounding_rect": [x, y, width, height],
                "food_label": label,
                "food_confidence": confidence,
                "area_pixels": pixels,
                "area_mm2": area,
                "percent_fill": pixels / plate_pixels * 100.0,
                "volume_ml": area * depth / 1000.0,
            }
        )
    return foods


def process_frame(frame, dimension_mm: float, depth_mm: float, logger) -> dict:
    result = {
        "utensil_detected": False,
        "shape_type": None,
        "container_type": None,
        "percent_fill": None,
        "volume_ml": None,
        "food_label": None,
        "food_confidence": None,
        "foods": [],
    }
    try:
        detection = process_frame_steel_utensil(
            frame,
            known_dimension_mm=dimension_mm or 280.0,
            assumed_depth_mm=depth_mm,
            utensil_hint="round",
        )
        for key in (
            "utensil_detected",
            "shape_type",
            "container_type",
            "percent_fill",
            "volume_ml",
        ):
            result[key] = detection.get(key)
        result["detection"] = detection
        if result["utensil_detected"]:
            result["foods"] = classify_food_regions(frame, detection)
            if result["foods"]:
                largest = max(result["foods"], key=lambda food: food["area_pixels"])
                result["food_label"] = largest["food_label"]
                result["food_confidence"] = largest["food_confidence"]
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
    return result


def draw_frame_overlay(frame, result: dict) -> np.ndarray:

    detection = result.get("detection", {})
    if detection:
        display = draw_detection_overlay(frame, detection)
    else:
        display = frame.copy()

    y = 30
    if result.get("food_label"):
        conf = result.get("food_confidence", 0)
        text = f"Food: {result['food_label']} ({conf * 100:.0f}%)"
        cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25

    for food in result.get("foods", []):
        x, top, width, height = food["bounding_rect"]
        cv2.rectangle(display, (x, top), (x + width, top + height), (0, 255, 0), 1)
        text = f"{food['food_label'] or 'unknown'}: {food['volume_ml']:.0f} ml"
        cv2.putText(
            display, text, (x, max(15, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        display,
        timestamp,
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    return display


def signal_handler(signum, frame):

    print("\nShutdown signal received...")
    shutdown_event.set()


def run_detector(args, logger):

    storage_enabled = True
    if not args.dry_run:
        try:
            init_local_storage()
            logger.info(f"Local JSON storage initialized: {LOCAL_DATA_FILE}")
        except Exception as e:
            logger.error(f"Local storage initialization failed: {e}")
            if not args.dry_run:
                logger.warning("Continuing in dry-run mode (no saves)")
                args.dry_run = True
                storage_enabled = False

        if args.use_firebase:
            firebase_init = init_firebase(logger)
            if not firebase_init:
                logger.warning("Firebase initialization failed, using local JSON only")

    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        source_name = args.video
        is_video_file = True
    else:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"Camera {args.camera}"
        is_video_file = False

    if not cap.isOpened():
        logger.error(f"Failed to open {source_name}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video_file else 0

    logger.info(f"Source: {source_name}")
    if is_video_file:
        logger.info(f"Total frames: {total_frames}, FPS: {fps:.1f}")
    logger.info(f"Capture interval: {args.interval}s")
    logger.info(f"Dry run: {args.dry_run}")

    output_dir = None
    if args.save_frames:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving frames to: {output_dir}")

    tracker = SessionTracker(empty_frames_to_finalize=3)

    stats = {
        "frames_processed": 0,
        "sessions_saved": 0,
        "start_time": time.time(),
    }

    frame_step = max(1, int(fps * args.interval)) if is_video_file else 1
    frame_idx = 0
    last_capture = 0

    logger.info("Starting detection loop... (Press Ctrl+C to stop)")

    try:
        while not shutdown_event.is_set():
            if args.duration > 0:
                elapsed = time.time() - stats["start_time"]
                if elapsed >= args.duration:
                    logger.info(f"Duration limit reached ({args.duration}s)")
                    break

            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    logger.info("End of video file")
                    break
                else:
                    logger.warning("Failed to capture frame, retrying...")
                    time.sleep(0.5)
                    continue

            frame_idx += 1

            if is_video_file:
                if frame_idx % frame_step != 0:
                    continue
            else:
                now_time = time.time()
                if now_time - last_capture < args.interval:
                    continue
                last_capture = now_time

            result = process_frame(frame, args.dimension_mm, args.depth_mm, logger)
            stats["frames_processed"] += 1

            if result["utensil_detected"]:
                fill = result.get("percent_fill") or 0
                food = result.get("food_label") or "unknown"
                shape = result.get("shape_type") or "unknown"
                container = result.get("container_type") or "unknown"
                volume = result.get("volume_ml")
                vol_str = f", Vol: {volume:.0f}ml" if volume else ""
                logger.info(
                    f"Frame {frame_idx}: {shape}, {container}, Fill: {fill:.1f}%{vol_str}, Food: {food}"
                )
            else:
                logger.debug(f"Frame {frame_idx}: No utensil detected")

            session = tracker.update(
                result,
                food_label=result.get("food_label"),
                food_confidence=result.get("food_confidence"),
            )

            if session and not args.save_each_detection and not args.dry_run and storage_enabled:
                doc_id = save_session_to_local(session, logger)
                if doc_id:
                    stats["sessions_saved"] += 1
                    logger.info(
                        f"Session finalized - Food: {session.get('food_label')}, "
                        f"Avg Fill: {session.get('percent_fill', 0):.1f}%, "
                        f"Samples: {session.get('sample_count')}"
                    )

            if (
                args.save_each_detection
                and not args.dry_run
                and storage_enabled
                and result["utensil_detected"]
            ):
                doc_id = save_detection_to_local(result, logger)
                if doc_id:
                    stats["sessions_saved"] += 1

            if output_dir and result["utensil_detected"]:
                annotated = draw_frame_overlay(frame, result)
                filename = f"frame_{frame_idx:06d}_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(str(output_dir / filename), annotated)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")

    finally:
        final_session = tracker.force_finalize()
        if final_session and not args.save_each_detection and not args.dry_run and storage_enabled:
            save_session_to_local(final_session, logger)
            stats["sessions_saved"] += 1

        cap.release()

        elapsed = time.time() - stats["start_time"]
        logger.info("=" * 50)
        logger.info("Detection Summary")
        logger.info("=" * 50)
        logger.info(f"Run time: {elapsed:.1f}s")
        logger.info(f"Frames processed: {stats['frames_processed']}")
        logger.info(f"Sessions saved: {stats['sessions_saved']}")
        if stats["frames_processed"] > 0:
            logger.info(f"Avg rate: {stats['frames_processed'] / elapsed:.2f} frames/sec")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Headless Food Waste Detector for Raspberry Pi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # run with USB camera
  python scripts/app.py --camera 0

  # process video file (dry run)
  python scripts/app.py --video data/rice.mp4 --dry-run

  # run for 5 minutes, saving frames
  python scripts/app.py --duration 300 --save-frames

  # raspberry pi with pi camera
  python scripts/app.py --camera 0 --interval 2 --save-frames
""",
    )

    parser.add_argument(
        "--check-model",
        action="store_true",
        help="Load the classifier, run a synthetic input, and exit",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument(
        "--video", type=str, default=None, help="Process video file instead of camera"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0, help="Seconds between frame captures (default: 1.0)"
    )
    parser.add_argument(
        "--duration", type=float, default=0, help="Run for N seconds (0 = indefinitely, default: 0)"
    )
    parser.add_argument(
        "--dimension-mm",
        "--plate-diameter-mm",
        type=float,
        default=280,
        help="Assumed plate diameter in mm (default: 280)",
    )
    parser.add_argument(
        "--depth-mm", type=float, default=10, help="Assumed average food depth in mm (default: 10)"
    )
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frames to disk")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for saved frames (default: output)",
    )
    parser.add_argument(
        "--log-file", type=str, default=None, help="Log file path (default: console only)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't save to local storage (testing mode)"
    )
    parser.add_argument(
        "--save-each-detection",
        action="store_true",
        help="Save each frame's detection to local storage (instead of waiting for session finalization)",
    )
    parser.add_argument(
        "--use-firebase", action="store_true", help="Write to Firestore as well as local JSON"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    if not np.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be a positive finite number")
    if args.dimension_mm is not None and (
        not np.isfinite(args.dimension_mm) or args.dimension_mm <= 0
    ):
        parser.error("--dimension-mm must be a positive finite number")

    if not np.isfinite(args.depth_mm) or args.depth_mm <= 0:
        parser.error("--depth-mm must be a positive finite number")

    logger = setup_logging(args.log_file, args.verbose)

    try:
        classes = load_classifier()
        if args.check_model:
            probe = classify_food(np.zeros((224, 224, 3), dtype=np.uint8))
            if probe is None:
                raise RuntimeError("Classifier inference check failed")
        logger.info("Classifier ready: %s food classes", classes)
    except Exception:
        logger.exception("Cannot start food classifier")
        return 1
    if args.check_model:
        logger.info("Model load and inference check passed (not an accuracy test)")
        return 0

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 50)
    logger.info("Food Waste Detector - Headless Mode")
    logger.info("=" * 50)

    return run_detector(args, logger)


if __name__ == "__main__":
    sys.exit(main())
