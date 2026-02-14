"""Compare with and without built-in preprocessing."""

import cv2
import numpy as np
from pathlib import Path
from siamese_detector import SiameseDetector

TEST_DIRS = {
    "test1": ["IMG_0939.jpg", "IMG_0940.jpg", "IMG_0941.jpg"],
    "test2": ["IMG_0939.jpg", "IMG_0940.jpg", "IMG_0941.jpg"],
    "test3": ["IMG_0946.jpg", "IMG_0947.jpg", "IMG_0948.jpg"],
}

EXPECTED = [True, True, False]
BASE_DIR = Path("/Users/adam/Desktop/TreeHacks26/cv_detector/new_detector")
REF_DIR = BASE_DIR / "steps_jpg"
CROPPED_DIR = BASE_DIR / "cropped"


def manual_preprocess(image, crop_pct=0.15, enhance=True):
    """Manual preprocessing (like the tuning tests used)."""
    result = image.copy()

    if crop_pct > 0:
        h, w = result.shape[:2]
        margin_x = int(w * crop_pct)
        margin_y = int(h * crop_pct)
        result = result[margin_y:h-margin_y, margin_x:w-margin_x]

    if enhance:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        result = cv2.merge([l, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    return result


print("Testing with MANUAL preprocessing (like tuning script)")
print("Detector has NO built-in preprocessing")
print()

# Detector without preprocessing
detector = SiameseDetector(
    reference_dir=str(REF_DIR),
    cropped_dir=str(CROPPED_DIR),
    similarity_threshold=0.77,
    crop_percent=0.0,  # No built-in preprocessing
    use_enhancement=False
)

print("Test     | Img1 (YES) | Img2 (YES) | Img3 (NO) | Score")
print("-" * 60)

total = 0
for test_name, images in TEST_DIRS.items():
    results = []
    sims = []
    for i, img_name in enumerate(images):
        step = i + 1
        img_path = BASE_DIR / test_name / "jpg" / img_name
        image = cv2.imread(str(img_path))

        # Manual preprocessing
        processed = manual_preprocess(image, crop_pct=0.15, enhance=True)

        result = detector.verify_step(processed, step)
        results.append(result.is_match)
        sims.append(result.similarity)

    score = sum(1 for r, e in zip(results, EXPECTED) if r == e)
    total += score

    res_str = ['Y' if r else 'N' for r in results]
    sim_str = [f"{s:.0%}" for s in sims]
    print(f"{test_name}    | {res_str[0]}({sim_str[0]})  | {res_str[1]}({sim_str[1]})  | {res_str[2]}({sim_str[2]}) | {score}/3")

print(f"\nTotal: {total}/9")
