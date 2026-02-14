"""
Test with preprocessing and threshold tuning.
Expected results: YES, YES, NO for all test sets.
"""

import cv2
import numpy as np
from pathlib import Path

# Test directories and their image order
TEST_DIRS = {
    "test1": ["IMG_0939.jpg", "IMG_0940.jpg", "IMG_0941.jpg"],
    "test2": ["IMG_0939.jpg", "IMG_0940.jpg", "IMG_0941.jpg"],
    "test3": ["IMG_0946.jpg", "IMG_0947.jpg", "IMG_0948.jpg"],
}

EXPECTED = [True, True, False]  # YES, YES, NO

BASE_DIR = Path("/Users/adam/Desktop/TreeHacks26/cv_detector/new_detector")
REF_DIR = BASE_DIR / "steps_jpg"
CROPPED_DIR = BASE_DIR / "cropped"
MODEL_PATH = BASE_DIR / "lego_model.pth"


def preprocess_image(image: np.ndarray, method: str = "none") -> np.ndarray:
    """Apply preprocessing to input image."""
    if method == "none":
        return image

    result = image.copy()

    if method == "crop_center":
        # Crop to center 70% to focus on the Lego
        h, w = result.shape[:2]
        margin_x, margin_y = int(w * 0.15), int(h * 0.15)
        result = result[margin_y:h-margin_y, margin_x:w-margin_x]

    elif method == "enhance":
        # Enhance contrast and saturation
        # Convert to LAB and enhance L channel
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        result = cv2.merge([l, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    elif method == "crop_and_enhance":
        # Both
        h, w = result.shape[:2]
        margin_x, margin_y = int(w * 0.15), int(h * 0.15)
        result = result[margin_y:h-margin_y, margin_x:w-margin_x]

        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        result = cv2.merge([l, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    elif method == "resize_small":
        # Resize to smaller size (reduces noise, focuses on overall shape)
        result = cv2.resize(result, (224, 224))

    elif method == "blur_and_crop":
        # Slight blur to reduce noise, then crop
        result = cv2.GaussianBlur(result, (3, 3), 0)
        h, w = result.shape[:2]
        margin_x, margin_y = int(w * 0.15), int(h * 0.15)
        result = result[margin_y:h-margin_y, margin_x:w-margin_x]

    return result


def calculate_score(results: list, expected: list) -> int:
    """Calculate how many predictions match expected."""
    return sum(1 for r, e in zip(results, expected) if r == e)


def test_siamese_thresholds():
    """Test Siamese model with different thresholds and preprocessing."""
    from siamese_detector import SiameseDetector

    print("\n" + "=" * 70)
    print("SIAMESE MODEL - THRESHOLD & PREPROCESSING TUNING")
    print("=" * 70)
    print("Expected: [YES, YES, NO]")

    thresholds = [0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]
    preprocess_methods = ["none", "crop_center", "enhance", "crop_and_enhance"]

    best_score = 0
    best_config = None

    for preprocess in preprocess_methods:
        print(f"\n--- Preprocessing: {preprocess} ---")

        for thresh in thresholds:
            detector = SiameseDetector(
                reference_dir=str(REF_DIR),
                cropped_dir=str(CROPPED_DIR),
                similarity_threshold=thresh
            )

            total_correct = 0
            all_results = []

            for test_name, images in TEST_DIRS.items():
                results = []
                for i, img_name in enumerate(images):
                    step = i + 1
                    img_path = BASE_DIR / test_name / "jpg" / img_name
                    image = cv2.imread(str(img_path))

                    # Apply preprocessing
                    image = preprocess_image(image, preprocess)

                    result = detector.verify_step(image, step)
                    results.append(result.is_match)

                score = calculate_score(results, EXPECTED)
                total_correct += score
                all_results.append(results)

            if total_correct > best_score:
                best_score = total_correct
                best_config = (thresh, preprocess, all_results)

            result_str = " | ".join([
                f"test{i+1}: {['YES' if r else 'NO' for r in res]}"
                for i, res in enumerate(all_results)
            ])
            print(f"  thresh={thresh:.2f}: {total_correct}/9 correct | {result_str}")

    print(f"\n*** BEST: threshold={best_config[0]}, preprocess={best_config[1]}, score={best_score}/9 ***")
    return best_config


def test_similarity_thresholds():
    """Test Similarity model with different thresholds and preprocessing."""
    from similarity_detector import SimilarityDetector

    print("\n" + "=" * 70)
    print("SIMILARITY MODEL - THRESHOLD & PREPROCESSING TUNING")
    print("=" * 70)
    print("Expected: [YES, YES, NO]")

    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    preprocess_methods = ["none", "crop_center", "enhance"]

    best_score = 0
    best_config = None

    for preprocess in preprocess_methods:
        print(f"\n--- Preprocessing: {preprocess} ---")

        for thresh in thresholds:
            detector = SimilarityDetector(
                reference_dir=str(REF_DIR),
                cropped_dir=str(CROPPED_DIR),
                similarity_threshold=thresh
            )

            total_correct = 0
            all_results = []

            for test_name, images in TEST_DIRS.items():
                results = []
                for i, img_name in enumerate(images):
                    step = i + 1
                    img_path = BASE_DIR / test_name / "jpg" / img_name
                    image = cv2.imread(str(img_path))

                    image = preprocess_image(image, preprocess)

                    result = detector.verify_step(image, step)
                    results.append(result.is_match)

                score = calculate_score(results, EXPECTED)
                total_correct += score
                all_results.append(results)

            if total_correct > best_score:
                best_score = total_correct
                best_config = (thresh, preprocess, all_results)

            result_str = " | ".join([
                f"test{i+1}: {['YES' if r else 'NO' for r in res]}"
                for i, res in enumerate(all_results)
            ])
            print(f"  thresh={thresh:.2f}: {total_correct}/9 correct | {result_str}")

    print(f"\n*** BEST: threshold={best_config[0]}, preprocess={best_config[1]}, score={best_score}/9 ***")
    return best_config


def test_cnn_confidence():
    """Test CNN with different confidence thresholds."""
    from cnn_detector import CNNLegoDetector

    print("\n" + "=" * 70)
    print("CNN MODEL - CONFIDENCE THRESHOLD TUNING")
    print("=" * 70)
    print("Expected: [YES, YES, NO]")
    print("\nNote: CNN predicts which step, so we check if predicted == target")

    detector = CNNLegoDetector(
        model_path=str(MODEL_PATH),
        reference_dir=str(REF_DIR)
    )

    preprocess_methods = ["none", "crop_center", "enhance"]

    best_score = 0
    best_config = None

    for preprocess in preprocess_methods:
        print(f"\n--- Preprocessing: {preprocess} ---")

        total_correct = 0
        all_results = []
        all_details = []

        for test_name, images in TEST_DIRS.items():
            results = []
            details = []
            for i, img_name in enumerate(images):
                step = i + 1
                img_path = BASE_DIR / test_name / "jpg" / img_name
                image = cv2.imread(str(img_path))

                image = preprocess_image(image, preprocess)

                result = detector.verify_step(image, step)
                results.append(result.is_match)
                details.append(f"pred=s{result.predicted_step}({result.confidence:.0%})")

            score = calculate_score(results, EXPECTED)
            total_correct += score
            all_results.append(results)
            all_details.append(details)

        if total_correct > best_score:
            best_score = total_correct
            best_config = (preprocess, all_results)

        for i, (res, det) in enumerate(zip(all_results, all_details)):
            res_str = ['YES' if r else 'NO' for r in res]
            print(f"  test{i+1}: {res_str} | {det}")
        print(f"  Score: {total_correct}/9")

    print(f"\n*** BEST: preprocess={best_config[0]}, score={best_score}/9 ***")
    return best_config


if __name__ == "__main__":
    print("LEGO VERIFICATION - THRESHOLD TUNING")
    print("Expected for all tests: [YES, YES, NO]")

    # Test each model
    cnn_best = test_cnn_confidence()
    similarity_best = test_similarity_thresholds()
    siamese_best = test_siamese_thresholds()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"CNN:        preprocess={cnn_best[0]}")
    print(f"Similarity: threshold={similarity_best[0]}, preprocess={similarity_best[1]}")
    print(f"Siamese:    threshold={siamese_best[0]}, preprocess={siamese_best[1]}")
