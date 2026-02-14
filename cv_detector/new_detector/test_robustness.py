"""
Test robustness to brightness and scale variations.
Also verify our baseline is correct.
"""

import cv2
import numpy as np
from pathlib import Path
from siamese_detector import SiameseDetector

BASE_DIR = Path("/Users/adam/Desktop/TreeHacks26/cv_detector/new_detector")
REF_DIR = BASE_DIR / "steps_jpg"
CROPPED_DIR = BASE_DIR / "cropped"


def adjust_brightness(image, factor):
    """Adjust brightness. factor < 1 = darker, > 1 = brighter."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def simulate_distance(image, scale_factor):
    """Simulate closer (>1) or further (<1) camera distance."""
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    if scale_factor > 1:
        # Zoom in (closer) - resize up then center crop
        resized = cv2.resize(image, (new_w, new_h))
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        return resized[start_y:start_y+h, start_x:start_x+w]
    else:
        # Zoom out (further) - resize down then pad with gray
        resized = cv2.resize(image, (new_w, new_h))
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        result = np.full_like(image, 128)  # Gray padding
        result[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        return result


def preprocess(image, crop_pct=0.15):
    """Standard preprocessing - crop + CLAHE."""
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


# Test images with known expected results
TEST_IMAGES = [
    ("test1/jpg/IMG_0939.jpg", 1, True),   # Step 1, should match
    ("test1/jpg/IMG_0940.jpg", 2, True),   # Step 2, should match
    ("test1/jpg/IMG_0941.jpg", 3, False),  # Step 3, should NOT match
]

print("=" * 70)
print("BASELINE VERIFICATION")
print("=" * 70)

# Detector WITHOUT built-in preprocessing
detector = SiameseDetector(
    reference_dir=str(REF_DIR),
    cropped_dir=str(CROPPED_DIR),
    similarity_threshold=0.77,
    crop_percent=0.0,      # NO built-in crop
    use_enhancement=False  # NO built-in enhancement
)

print("\nBaseline (with manual preprocessing):")
for img_path, step, expected in TEST_IMAGES:
    image = cv2.imread(str(BASE_DIR / img_path))
    processed = preprocess(image)  # Manual preprocessing
    result = detector.verify_step(processed, step)
    match = "✓" if result.is_match == expected else "✗"
    exp_str = "YES" if expected else "NO"
    got_str = "YES" if result.is_match else "NO"
    print(f"  Step {step}: expect={exp_str}, got={got_str} ({result.similarity:.0%}) {match}")


print("\n" + "=" * 70)
print("BRIGHTNESS VARIATION TEST")
print("=" * 70)
print("Factors: 0.5=very dark, 0.7=dark, 1.0=normal, 1.3=bright, 1.5=very bright")

BRIGHTNESS_FACTORS = [0.5, 0.7, 1.0, 1.3, 1.5]

brightness_correct = 0
brightness_total = 0

for img_path, step, expected in TEST_IMAGES:
    image = cv2.imread(str(BASE_DIR / img_path))
    exp_str = "YES" if expected else "NO"

    print(f"\nStep {step} (expect {exp_str}):")
    for factor in BRIGHTNESS_FACTORS:
        # Apply brightness change BEFORE preprocessing
        modified = adjust_brightness(image, factor)
        processed = preprocess(modified)
        result = detector.verify_step(processed, step)

        correct = result.is_match == expected
        if correct:
            brightness_correct += 1
        brightness_total += 1

        status = "✓" if correct else "✗"
        print(f"  {factor}: {'YES' if result.is_match else 'NO'} ({result.similarity:.0%}) {status}")


print("\n" + "=" * 70)
print("DISTANCE/SCALE VARIATION TEST")
print("=" * 70)
print("Factors: 0.7=far, 0.85=medium-far, 1.0=normal, 1.15=medium-close, 1.3=close")

SCALE_FACTORS = [0.7, 0.85, 1.0, 1.15, 1.3]

scale_correct = 0
scale_total = 0

for img_path, step, expected in TEST_IMAGES:
    image = cv2.imread(str(BASE_DIR / img_path))
    exp_str = "YES" if expected else "NO"

    print(f"\nStep {step} (expect {exp_str}):")
    for factor in SCALE_FACTORS:
        # Apply scale change BEFORE preprocessing
        modified = simulate_distance(image, factor)
        processed = preprocess(modified)
        result = detector.verify_step(processed, step)

        correct = result.is_match == expected
        if correct:
            scale_correct += 1
        scale_total += 1

        status = "✓" if correct else "✗"
        print(f"  {factor}: {'YES' if result.is_match else 'NO'} ({result.similarity:.0%}) {status}")


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Brightness robustness: {brightness_correct}/{brightness_total} ({100*brightness_correct/brightness_total:.0f}%)")
print(f"Scale robustness:      {scale_correct}/{scale_total} ({100*scale_correct/scale_total:.0f}%)")
print(f"Overall robustness:    {brightness_correct + scale_correct}/{brightness_total + scale_total} ({100*(brightness_correct + scale_correct)/(brightness_total + scale_total):.0f}%)")
