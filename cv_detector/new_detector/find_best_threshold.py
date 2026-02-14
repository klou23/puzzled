"""Find optimal threshold considering robustness."""

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


def simulate_distance(image, scale_factor):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    if scale_factor > 1:
        resized = cv2.resize(image, (new_w, new_h))
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        return resized[start_y:start_y+h, start_x:start_x+w]
    else:
        resized = cv2.resize(image, (new_w, new_h))
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        result = np.full_like(image, 128)
        result[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        return result


def preprocess(image, crop_pct=0.15):
    h, w = image.shape[:2]
    margin_x = int(w * crop_pct)
    margin_y = int(h * crop_pct)
    result = image[margin_y:h-margin_y, margin_x:w-margin_x]
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.merge([l, a, b])
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


# All test images across all test sets
ALL_TESTS = [
    # test1
    ("test1/jpg/IMG_0939.jpg", 1, True),
    ("test1/jpg/IMG_0940.jpg", 2, True),
    ("test1/jpg/IMG_0941.jpg", 3, False),
    # test2
    ("test2/jpg/IMG_0939.jpg", 1, True),
    ("test2/jpg/IMG_0940.jpg", 2, True),
    ("test2/jpg/IMG_0941.jpg", 3, False),
    # test3
    ("test3/jpg/IMG_0946.jpg", 1, True),
    ("test3/jpg/IMG_0947.jpg", 2, True),
    ("test3/jpg/IMG_0948.jpg", 3, False),
]

BRIGHTNESS_FACTORS = [0.6, 0.8, 1.0, 1.2, 1.4]  # Moderate range
SCALE_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]       # Moderate range

print("Finding optimal threshold with robustness in mind...")
print()

# First collect all similarity values
detector = SiameseDetector(
    reference_dir=str(REF_DIR),
    cropped_dir=str(CROPPED_DIR),
    similarity_threshold=0.0,
    crop_percent=0.0,
    use_enhancement=False
)

# Collect similarities for YES cases and NO cases
yes_sims = []
no_sims = []

for img_path, step, expected in ALL_TESTS:
    image = cv2.imread(str(BASE_DIR / img_path))

    # Normal
    processed = preprocess(image)
    result = detector.verify_step(processed, step)
    if expected:
        yes_sims.append(result.similarity)
    else:
        no_sims.append(result.similarity)

    # With brightness variations
    for factor in BRIGHTNESS_FACTORS:
        modified = adjust_brightness(image, factor)
        processed = preprocess(modified)
        result = detector.verify_step(processed, step)
        if expected:
            yes_sims.append(result.similarity)
        else:
            no_sims.append(result.similarity)

    # With scale variations
    for factor in SCALE_FACTORS:
        modified = simulate_distance(image, factor)
        processed = preprocess(modified)
        result = detector.verify_step(processed, step)
        if expected:
            yes_sims.append(result.similarity)
        else:
            no_sims.append(result.similarity)

print(f"YES cases (should be >=threshold): min={min(yes_sims):.1%}, max={max(yes_sims):.1%}, mean={np.mean(yes_sims):.1%}")
print(f"NO cases (should be <threshold):  min={min(no_sims):.1%}, max={max(no_sims):.1%}, mean={np.mean(no_sims):.1%}")
print()

# Find best threshold
print("Testing thresholds:")
thresholds = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80]

best_thresh = 0
best_score = 0
best_yes_acc = 0
best_no_acc = 0

for thresh in thresholds:
    yes_correct = sum(1 for s in yes_sims if s >= thresh)
    no_correct = sum(1 for s in no_sims if s < thresh)
    total = yes_correct + no_correct
    yes_acc = yes_correct / len(yes_sims) * 100
    no_acc = no_correct / len(no_sims) * 100

    print(f"  {thresh}: YES={yes_acc:.0f}% ({yes_correct}/{len(yes_sims)}), NO={no_acc:.0f}% ({no_correct}/{len(no_sims)}), Total={total}")

    if total > best_score:
        best_score = total
        best_thresh = thresh
        best_yes_acc = yes_acc
        best_no_acc = no_acc

print()
print(f"BEST THRESHOLD: {best_thresh}")
print(f"  YES accuracy: {best_yes_acc:.0f}%")
print(f"  NO accuracy:  {best_no_acc:.0f}%")
