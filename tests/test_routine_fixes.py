import argparse
import importlib.util
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("detector", ROOT / "scripts/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class RoutineFixes(unittest.TestCase):
    def setUp(self):
        app.shutdown_event.clear()
        self.logger = logging.getLogger("test")

    def test_interrupted_write_preserves_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            with (
                patch.object(app, "LOCAL_DATA_DIR", Path(directory)),
                patch.object(app, "LOCAL_DATA_FILE", path),
            ):
                app.save_local_data([{"id": "original"}])
                with patch.object(app.os, "replace", side_effect=OSError("interrupted")):
                    with self.assertRaises(OSError):
                        app.save_local_data([{"id": "new"}])
                self.assertEqual(app.load_local_data(), [{"id": "original"}])
                self.assertEqual(list(Path(directory).iterdir()), [path])
                path.write_text("{broken")
                with self.assertRaises(RuntimeError):
                    app.load_local_data()
                self.assertEqual(path.read_text(), "{broken")

    def test_container_metadata_reaches_storage(self):
        tracker = app.SessionTracker()
        tracker.update(
            {
                "utensil_detected": True,
                "shape_type": "rectangle",
                "container_type": "small_rectangle",
                "percent_fill": 50,
                "volume_ml": 100,
            }
        )
        session = tracker.force_finalize()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            with (
                patch.object(app, "LOCAL_DATA_DIR", Path(directory)),
                patch.object(app, "LOCAL_DATA_FILE", path),
                patch.object(app, "firebase_enabled", False),
            ):
                app.save_session_to_local(session, self.logger)
                app.save_detection_to_local(session, self.logger)
                records = json.loads(path.read_text())
                self.assertEqual([r["container_type"] for r in records], ["small_rectangle"] * 2)
                self.assertEqual(
                    [r["detection_type"] for r in records], ["session", "single_frame"]
                )

    def test_default_calibration_and_override(self):
        mask = np.ones((310, 340), dtype=np.uint8) * 255
        contour = np.array([[[0, 0]], [[339, 0]], [[339, 309]], [[0, 309]]], dtype=np.int32)
        food = mask.copy()
        food[155:] = 0
        detection = {
            "utensil_detected": True,
            "shape_type": "rectangle",
            "container_type": "big_square",
        }

        def estimate(frame, known_dimension_mm, **kwargs):
            percent, volume, _ = app.estimate_volume_from_contour(
                contour,
                mask,
                food,
                {"bounding_rect": (0, 0, 340, 310)},
                known_dimension_mm,
                kwargs["assumed_depth_mm"],
                kwargs["utensil_hint"],
            )
            return dict(detection, percent_fill=percent, volume_ml=volume)

        with (
            patch.object(app, "classify_food", return_value=None),
            patch.object(app, "process_frame_steel_utensil", side_effect=estimate),
        ):
            automatic = app.process_frame(None, None, 30, self.logger)["volume_ml"]
            explicit = app.process_frame(None, 280, 30, self.logger)["volume_ml"]
            override = app.process_frame(None, 200, 30, self.logger)["volume_ml"]
        self.assertAlmostEqual(automatic, explicit)
        self.assertAlmostEqual(override / automatic, (200 / 280) ** 2)

    def test_per_frame_mode_does_not_save_sessions(self):
        class Camera:
            def __init__(self):
                self.remaining = 2

            def isOpened(self):
                return True

            def get(self, key):
                return 30

            def read(self):
                self.remaining -= 1
                return (self.remaining >= 0, np.zeros((10, 10, 3), dtype=np.uint8))

            def release(self):
                pass

        result = {
            "utensil_detected": True,
            "shape_type": "rectangle",
            "container_type": "big_square",
            "volume_ml": 100,
            "percent_fill": 50,
            "food_label": "rice",
            "food_confidence": 0.9,
        }
        args = argparse.Namespace(
            dry_run=False,
            use_firebase=False,
            video="dummy.mp4",
            camera=0,
            interval=0.001,
            duration=0,
            save_frames=False,
            dimension_mm=None,
            depth_mm=30,
            save_each_detection=True,
        )
        with (
            patch.object(app, "init_local_storage"),
            patch.object(app.cv2, "VideoCapture", return_value=Camera()),
            patch.object(app, "process_frame", return_value=result),
            patch.object(app, "save_session_to_local") as sessions,
            patch.object(app, "save_detection_to_local", return_value="id") as frames,
        ):
            self.assertEqual(app.run_detector(args, self.logger), 0)
            self.assertEqual(frames.call_count, 2)
            sessions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
