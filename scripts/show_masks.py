import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vision_steel import detect_steel_utensil, segment_food_in_steel_utensil


def main():
    parser = argparse.ArgumentParser(description="Show segmentation stages without the classifier")
    parser.add_argument("image")
    parser.add_argument("--output", default="output/masks")
    args = parser.parse_args()
    frame = cv2.imread(args.image)
    if frame is None:
        parser.error("Cannot read image")
    if frame.shape[1] > 960:
        height = round(frame.shape[0] * 960 / frame.shape[1])
        frame = cv2.resize(frame, (960, height), interpolation=cv2.INTER_AREA)
    contour, plate, _ = detect_steel_utensil(frame, utensil_hint="round")
    if contour is None:
        parser.error("No round steel plate detected; try a clearer overhead view")
    final, info = segment_food_in_steel_utensil(frame, contour, plate, debug=True)
    stages = info["stages"]
    overlay = frame.copy()
    tint = np.full_like(frame, (60, 200, 60))
    mixed = cv2.addWeighted(frame, 0.55, tint, 0.45, 0)
    overlay[final > 0] = mixed[final > 0]
    cv2.drawContours(overlay, [contour], -1, (0, 210, 255), 2)
    panels = [
        ("original", frame),
        ("plate mask", plate),
        ("initial food mask", stages["raw"]),
        ("opening + closing", stages["cleanup"]),
        ("small regions removed", stages["filtered"]),
        ("final overlay", overlay),
    ]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for i, (label, img) in enumerate(panels):
        if not cv2.imwrite(str(out / f"{i + 1}_{label.replace(' ', '_')}.png"), img):
            raise OSError("Could not save stage image")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        tile = np.full((360, 480, 3), 245, dtype=np.uint8)
        scale = min(460 / img.shape[1], 290 / img.shape[0])
        width, height = round(img.shape[1] * scale), round(img.shape[0] * scale)
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        left, top = (480 - width) // 2, 55 + (290 - height) // 2
        tile[top : top + height, left : left + width] = resized
        cv2.putText(
            tile,
            f"{i + 1}. {label}",
            (18, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    board = np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))
    footer = np.full((46, board.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        footer,
        "masks computed by WasteWise",
        (18, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    board = np.vstack((board, footer))
    cv2.imwrite(str(out / "stages.png"), board)
    cv2.imwrite(str(out / "final_mask.png"), final)
    print(f"Saved segmentation stages to {out}")
    print(f"Food coverage: {info['coverage_percent']:.1f}% of detected plate")
    print("No classifier or manual food mask used. Input resized to at most 960 px wide.")


if __name__ == "__main__":
    main()
