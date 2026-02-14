"""Find optimal threshold with CLAHE normalization across all brightness levels."""

import cv2
import numpy as np
from pathlib import Path
from siamese_detector import SiameseDetector

BASE_DIR = Path("/Users/adam/Desktop/TreeHacks26/cv_detector/new_detector")
REF_DIR = BASE_DIR / "steps_jpg"
CROPPED_DIR = BASE_DIR / "cropped"


def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def normalize_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.merge([l, a, b])
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def crop_center(image, crop_pct=0.15):
    h, w = image.shape[:2]
    margin_x = int(w * crop_pct)
    margin_y = int(h * crop_pct)
    return image[margin_y:h-margin_y, margin_x:w-margin_x]


ALL_TEST_IMAGES = [
    ("test1/jpg/IMG_0939.jpg", 1, True),
    ("test1/jpg/IMG_0940.jpg", 2, True),
    ("test1/jpg/IMG_0941.jpg", 3, False),
    ("test2/jpg/IMG_0939.jpg", 1, True),
    ("test2/jpg/IMG_0940.jpg", 2, True),
    ("test2/jpg/IMG_0941.jpg", 3, False),
    ("test3/jpg/IMG_0946.jpg", 1, True),
    ("test3/jpg/IMG_0947.jpg", 2, True),
    ("test3/jpg/IMG_0948.jpg", 3, False),
]

BRIGHTNESS_FACTORS = [0.5, 0.7, 1.0, 1.3, 1.5]

# Collect all similarity values
detector = SiameseDetector(
    reference_dir=str(REF_DIR),
    cropped_dir=str(CROPPED_DIR),
    similarity_threshold=0.0,  # Accept all to see raw values
    crop_percent=0.0,
    use_enhancement=False
)

yes_sims = []
no_sims = []

print("Collecting similarity values across brightness levels...")
for img_path, step, expected in ALL_TEST_IMAGES:
    image = cv2.imread(str(BASE_DIR / img_path))

    for factor in BRIGHTNESS_FACTORS:
        modified = adjust_brightness(image, factor)
        cropped = crop_center(modified)
        normalized = normalize_clahe(cropped)
        result = detector.verify_step(normalized, step)

        if expected:
            yes_sims.append(result.similarity)
        else:
            no_sims.append(result.similarity)

print()
print("=" * 60)
print("SIMILARITY VALUE DISTRIBUTION")
print("=" * 60)
print(f"YES cases (should match):     min={min(yes_sims):.1%}, max={max(yes_sims):.1%}, mean={np.mean(yes_sims):.1%}")
print(f"NO cases (should not match):  min={min(no_sims):.1%}, max={max(no_sims):.1%}, mean={np.mean(no_sims):.1%}")
print()

# Find optimal threshold
print("=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

# The gap between min YES and max NO
gap_lower = min(yes_sims)
gap_upper = max(no_sims)

print(f"Min YES similarity: {gap_lower:.1%}")
print(f"Max NO similarity:  {gap_upper:.1%}")

if gap_lower > gap_upper:
    print(f"✓ Good separation! Gap between {gap_upper:.1%} and {gap_lower:.1%}")
    optimal = (gap_lower + gap_upper) / 2
    print(f"Optimal threshold: {optimal:.2%}")
else:
    print(f"✗ Overlap! Some NO cases ({gap_upper:.1%}) are higher than some YES cases ({gap_lower:.1%})")
    print("Perfect separation not possible.")

print()
print("Testing thresholds:")
thresholds = np.arange(0.70, 0.85, 0.01)

best_thresh = 0
best_total = 0
best_yes = 0
best_no = 0

for thresh in thresholds:
    yes_correct = sum(1 for s in yes_sims if s >= thresh)
    no_correct = sum(1 for s in no_sims if s < thresh)
    total = yes_correct + no_correct

    if total >= best_total:
        best_total = total
        best_thresh = thresh
        best_yes = yes_correct
        best_no = no_correct

    # Only print interesting thresholds
    if total >= 40:
        print(f"  {thresh:.2f}: YES={yes_correct}/{len(yes_sims)}, NO={no_correct}/{len(no_sims)}, Total={total}/45 ({100*total/45:.0f}%)")

print()
print(f"OPTIMAL THRESHOLD: {best_thresh:.2f}")
print(f"  YES accuracy: {best_yes}/{len(yes_sims)} ({100*best_yes/len(yes_sims):.0f}%)")
print(f"  NO accuracy:  {best_no}/{len(no_sims)} ({100*best_no/len(no_sims):.0f}%)")
print(f"  Total:        {best_total}/45 ({100*best_total/45:.0f}%)")
