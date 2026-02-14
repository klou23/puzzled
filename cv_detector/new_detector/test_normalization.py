"""
Test combined normalization techniques for brightness robustness.
"""

import cv2
import numpy as np
from pathlib import Path
from siamese_detector import SiameseDetector

BASE_DIR = Path("/Users/adam/Desktop/TreeHacks26/cv_detector/new_detector")
REF_DIR = BASE_DIR / "steps_jpg"
CROPPED_DIR = BASE_DIR / "cropped"


def adjust_brightness(image, factor):
    """Simulate different lighting conditions."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def normalize_clahe(image):
    """CLAHE on L channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.merge([l, a, b])
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def normalize_white_balance(image):
    """White balance - scale each channel to mean of 128."""
    result = image.astype(np.float32)
    for i in range(3):
        channel = result[:, :, i]
        mean = np.mean(channel)
        if mean > 0:
            result[:, :, i] = channel * (128 / mean)
    return np.clip(result, 0, 255).astype(np.uint8)


def normalize_combined_v1(image):
    """White balance THEN CLAHE."""
    img = normalize_white_balance(image)
    return normalize_clahe(img)


def normalize_combined_v2(image):
    """CLAHE THEN white balance."""
    img = normalize_clahe(image)
    return normalize_white_balance(img)


def normalize_percentile(image):
    """Stretch to 2nd-98th percentile per channel."""
    result = image.astype(np.float32)
    for i in range(3):
        channel = result[:, :, i]
        p2 = np.percentile(channel, 2)
        p98 = np.percentile(channel, 98)
        if p98 > p2:
            result[:, :, i] = (channel - p2) / (p98 - p2) * 255
    return np.clip(result, 0, 255).astype(np.uint8)


def normalize_percentile_clahe(image):
    """Percentile stretch then CLAHE."""
    img = normalize_percentile(image)
    return normalize_clahe(img)


def normalize_clahe_percentile(image):
    """CLAHE then percentile stretch."""
    img = normalize_clahe(image)
    return normalize_percentile(img)


def crop_center(image, crop_pct=0.15):
    h, w = image.shape[:2]
    margin_x = int(w * crop_pct)
    margin_y = int(h * crop_pct)
    return image[margin_y:h-margin_y, margin_x:w-margin_x]


# All test images from all test sets
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

NORMALIZERS = {
    "CLAHE only": normalize_clahe,
    "White Balance only": normalize_white_balance,
    "White Balance→CLAHE": normalize_combined_v1,
    "CLAHE→White Balance": normalize_combined_v2,
    "Percentile only": normalize_percentile,
    "Percentile→CLAHE": normalize_percentile_clahe,
    "CLAHE→Percentile": normalize_clahe_percentile,
}

detector = SiameseDetector(
    reference_dir=str(REF_DIR),
    cropped_dir=str(CROPPED_DIR),
    similarity_threshold=0.77,
    crop_percent=0.0,
    use_enhancement=False
)

print("TESTING NORMALIZATION COMBINATIONS FOR BRIGHTNESS ROBUSTNESS")
print("=" * 70)
print(f"Using ALL test images (9 images)")
print(f"Brightness factors: {BRIGHTNESS_FACTORS}")
print(f"Total tests per method: {len(ALL_TEST_IMAGES) * len(BRIGHTNESS_FACTORS)}")
print()

results = {}

for norm_name, norm_func in NORMALIZERS.items():
    correct = 0
    total = 0
    yes_correct = 0
    yes_total = 0
    no_correct = 0
    no_total = 0

    for img_path, step, expected in ALL_TEST_IMAGES:
        image = cv2.imread(str(BASE_DIR / img_path))

        for factor in BRIGHTNESS_FACTORS:
            modified = adjust_brightness(image, factor)
            cropped = crop_center(modified)
            normalized = norm_func(cropped)
            result = detector.verify_step(normalized, step)

            if result.is_match == expected:
                correct += 1
            total += 1

            if expected:
                yes_total += 1
                if result.is_match:
                    yes_correct += 1
            else:
                no_total += 1
                if not result.is_match:
                    no_correct += 1

    results[norm_name] = (correct, total, yes_correct, yes_total, no_correct, no_total)

# Sort by total correct
sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)

print(f"{'Method':<25} {'Total':>8} {'YES cases':>12} {'NO cases':>12}")
print("-" * 60)
for name, (c, t, yc, yt, nc, nt) in sorted_results:
    print(f"{name:<25} {c:>3}/{t} ({100*c/t:>2.0f}%) {yc:>3}/{yt} ({100*yc/yt:>2.0f}%) {nc:>3}/{nt} ({100*nc/nt:>2.0f}%)")

best_name = sorted_results[0][0]
best_correct = sorted_results[0][1][0]
best_total = sorted_results[0][1][1]
print()
print(f"BEST: {best_name} with {best_correct}/{best_total} ({100*best_correct/best_total:.0f}%)")
