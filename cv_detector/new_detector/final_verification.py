"""Final verification of the optimized Siamese detector."""

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


def preprocess(image, crop_pct=0.15):
    """Standard preprocessing: crop + CLAHE."""
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

print("=" * 70)
print("FINAL VERIFICATION - SIAMESE DETECTOR")
print("=" * 70)
print("Settings: threshold=0.81, crop=15%, CLAHE enhancement")
print()

detector = SiameseDetector(
    reference_dir=str(REF_DIR),
    cropped_dir=str(CROPPED_DIR),
    similarity_threshold=0.81,
    crop_percent=0.0,  # We'll apply preprocessing manually
    use_enhancement=False
)

print(f"Default threshold: {detector.similarity_threshold}")
print()

# Test 1: Normal conditions (no brightness change)
print("=" * 70)
print("TEST 1: NORMAL CONDITIONS")
print("=" * 70)
normal_correct = 0
for img_path, step, expected in ALL_TEST_IMAGES:
    image = cv2.imread(str(BASE_DIR / img_path))
    processed = preprocess(image)
    result = detector.verify_step(processed, step)

    correct = result.is_match == expected
    if correct:
        normal_correct += 1

    status = "✓" if correct else "✗"
    exp = "YES" if expected else "NO"
    got = "YES" if result.is_match else "NO"
    print(f"  {img_path.split('/')[-1]}: Step {step}, expect={exp}, got={got} ({result.similarity:.0%}) {status}")

print(f"\nNormal conditions: {normal_correct}/9 ({100*normal_correct/9:.0f}%)")

# Test 2: With brightness variations
print()
print("=" * 70)
print("TEST 2: BRIGHTNESS ROBUSTNESS")
print("=" * 70)
print(f"Testing with brightness factors: {BRIGHTNESS_FACTORS}")

total_correct = 0
yes_correct = 0
yes_total = 0
no_correct = 0
no_total = 0

for img_path, step, expected in ALL_TEST_IMAGES:
    image = cv2.imread(str(BASE_DIR / img_path))

    for factor in BRIGHTNESS_FACTORS:
        modified = adjust_brightness(image, factor)
        processed = preprocess(modified)
        result = detector.verify_step(processed, step)

        correct = result.is_match == expected
        if correct:
            total_correct += 1

        if expected:
            yes_total += 1
            if result.is_match:
                yes_correct += 1
        else:
            no_total += 1
            if not result.is_match:
                no_correct += 1

print(f"\nBrightness robustness: {total_correct}/45 ({100*total_correct/45:.0f}%)")
print(f"  YES cases (should match):    {yes_correct}/{yes_total} ({100*yes_correct/yes_total:.0f}%)")
print(f"  NO cases (should not match): {no_correct}/{no_total} ({100*no_correct/no_total:.0f}%)")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Normal conditions:     {normal_correct}/9 ({100*normal_correct/9:.0f}%)")
print(f"With brightness var:   {total_correct}/45 ({100*total_correct/45:.0f}%)")
print(f"False positives (NO→YES): {no_total - no_correct}/15")
print(f"False negatives (YES→NO): {yes_total - yes_correct}/30")
