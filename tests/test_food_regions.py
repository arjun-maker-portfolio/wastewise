import unittest
from unittest.mock import patch

import cv2
import numpy as np
from test_routine_fixes import app


class FoodRegions(unittest.TestCase):
    def test_separate_patches_keep_area_and_labels(self):
        frame = np.full((100, 100, 3), 80, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        mask[30:40, 30:50] = 255
        mask[60:70, 60:90] = 255
        detection = {
            "food_mask": mask,
            "utensil_mask": np.full_like(mask, 255),
            "volume_info": {"mm_per_px": 2, "effective_depth_mm": 10},
        }
        with patch.object(
            app, "classify_food", side_effect=[("rice", 0.9), ("rice", 0.8), ("dal_tadka", 0.85)]
        ) as classify:
            foods = app.classify_food_regions(frame, detection)
        self.assertEqual(classify.call_count, 3)
        self.assertEqual([f["area_pixels"] for f in foods], [100, 200, 300])
        self.assertEqual([f["volume_ml"] for f in foods], [4, 8, 12])
        tracker = app.SessionTracker()
        tracker.update(
            {"utensil_detected": True, "percent_fill": 6, "volume_ml": 24, "foods": foods}
        )
        contributions = {f["food_label"]: f for f in tracker.force_finalize()["foods"]}
        self.assertEqual(contributions["rice"]["area_mm2"], 1200)
        self.assertEqual(contributions["rice"]["volume_ml"], 12)
        self.assertEqual(contributions["dal_tadka"]["volume_ml"], 12)

    def test_mask_removes_background_and_excludes_pixels_outside_plate(self):
        frame = np.full((20, 20, 3), 123, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:8, 2:8] = 255
        mask[4:6, 4:6] = 0
        mask[15:18, 15:18] = 255
        plate = np.zeros_like(mask)
        plate[:10, :10] = 255
        detection = {
            "food_mask": mask,
            "utensil_mask": plate,
            "volume_info": {"mm_per_px": 1, "effective_depth_mm": 10},
        }
        with patch.object(app, "classify_food", return_value=("rice", 0.9)) as classify:
            foods = app.classify_food_regions(frame, detection)
            crop = classify.call_args.args[0]
        self.assertEqual(len(foods), 1)
        self.assertEqual(foods[0]["area_pixels"], 32)
        self.assertTrue(np.all(crop[2:4, 2:4] == 0))
        self.assertTrue(np.all(crop[0, 0] == 123))

    def test_depth_changes_volume_linearly(self):
        mask = np.zeros((301, 301), dtype=np.uint8)
        cv2.circle(mask, (150, 150), 140, 255, -1)
        contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        food = mask.copy()
        food[150:] = 0

        def volume(depth):
            return app.estimate_volume_from_contour(contour, mask, food, {}, 280, depth, "round")[1]

        self.assertAlmostEqual(volume(20), 2 * volume(10))
        with self.assertRaises(ValueError):
            volume(0)

    def test_empty_plate_skips_classifier(self):
        detection = {
            "food_mask": np.zeros((20, 20), dtype=np.uint8),
            "utensil_mask": np.ones((20, 20), dtype=np.uint8),
            "volume_info": {"mm_per_px": 1, "effective_depth_mm": 10},
        }
        with patch.object(app, "classify_food") as classify:
            self.assertEqual(
                app.classify_food_regions(np.zeros((20, 20, 3), dtype=np.uint8), detection), []
            )
            classify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
