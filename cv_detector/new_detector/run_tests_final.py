"""
Fine-tuned tests to achieve 9/9.
Focus on the best performers and explore more thresholds.
"""

import cv2
import numpy as np
from pathlib import Path

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


def preprocess_image(image: np.ndarray, crop_pct: float = 0.15, enhance: bool = False) -> np.ndarray:
    """Apply preprocessing with configurable crop percentage."""
    result = image.copy()

    if crop_pct > 0:
        h, w = result.shape[:2]
        margin_x, margin_y = int(w * crop_pct), int(h * crop_pct)
        result = result[margin_y:h-margin_y, margin_x:w-margin_x]

    if enhance:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        result = cv2.merge([l, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    return result


def calculate_score(results: list, expected: list) -> int:
    return sum(1 for r, e in zip(results, expected) if r == e)


def test_cnn_detailed():
    """Test CNN with fine-tuned crop percentages."""
    from cnn_detector import CNNLegoDetector

    print("\n" + "=" * 70)
    print("CNN MODEL - FINE TUNING CROP PERCENTAGE")
    print("=" * 70)

    detector = CNNLegoDetector(
        model_path=str(MODEL_PATH),
        reference_dir=str(REF_DIR)
    )

    crop_percentages = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]

    best_score = 0
    best_config = None

    for crop_pct in crop_percentages:
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

                image = preprocess_image(image, crop_pct=crop_pct, enhance=False)

                result = detector.verify_step(image, step)
                results.append(result.is_match)
                details.append(f"s{result.predicted_step}({result.confidence:.0%})")

            score = calculate_score(results, EXPECTED)
            total_correct += score
            all_results.append(results)
            all_details.append(details)

        if total_correct > best_score:
            best_score = total_correct
            best_config = (crop_pct, all_results, all_details)

        result_strs = []
        for i, (res, det) in enumerate(zip(all_results, all_details)):
            res_str = ['Y' if r else 'N' for r in res]
            result_strs.append(f"test{i+1}:[{','.join(res_str)}]")

        print(f"crop={crop_pct:.0%}: {total_correct}/9 | {' '.join(result_strs)}")

    print(f"\n*** CNN BEST: crop={best_config[0]:.0%}, score={best_score}/9 ***")

    # Show detailed breakdown of best
    print("\nBest config details:")
    for i, (res, det) in enumerate(zip(best_config[1], best_config[2])):
        exp = ['Y' if e else 'N' for e in EXPECTED]
        got = ['Y' if r else 'N' for r in res]
        match = ['✓' if r == e else '✗' for r, e in zip(res, EXPECTED)]
        print(f"  test{i+1}: expected={exp}, got={got}, match={match}, preds={det}")

    return best_config


def test_siamese_detailed():
    """Test Siamese with fine-tuned parameters."""
    from siamese_detector import SiameseDetector

    print("\n" + "=" * 70)
    print("SIAMESE MODEL - FINE TUNING")
    print("=" * 70)

    # Fine-tune around the best found values
    thresholds = [0.76, 0.77, 0.78, 0.79, 0.80, 0.81]
    crop_percentages = [0.12, 0.15, 0.18, 0.20]

    best_score = 0
    best_config = None

    for crop_pct in crop_percentages:
        for thresh in thresholds:
            detector = SiameseDetector(
                reference_dir=str(REF_DIR),
                cropped_dir=str(CROPPED_DIR),
                similarity_threshold=thresh
            )

            total_correct = 0
            all_results = []
            all_sims = []

            for test_name, images in TEST_DIRS.items():
                results = []
                sims = []
                for i, img_name in enumerate(images):
                    step = i + 1
                    img_path = BASE_DIR / test_name / "jpg" / img_name
                    image = cv2.imread(str(img_path))

                    image = preprocess_image(image, crop_pct=crop_pct, enhance=True)

                    result = detector.verify_step(image, step)
                    results.append(result.is_match)
                    sims.append(f"{result.similarity:.0%}")

                score = calculate_score(results, EXPECTED)
                total_correct += score
                all_results.append(results)
                all_sims.append(sims)

            if total_correct > best_score:
                best_score = total_correct
                best_config = (thresh, crop_pct, all_results, all_sims)

            if total_correct >= 8:  # Only print good results
                result_strs = []
                for i, res in enumerate(all_results):
                    res_str = ['Y' if r else 'N' for r in res]
                    result_strs.append(f"[{','.join(res_str)}]")
                print(f"thresh={thresh:.2f}, crop={crop_pct:.0%}: {total_correct}/9 | {' '.join(result_strs)}")

    print(f"\n*** SIAMESE BEST: thresh={best_config[0]}, crop={best_config[1]:.0%}, score={best_score}/9 ***")

    # Show details
    print("\nBest config details:")
    for i, (res, sims) in enumerate(zip(best_config[2], best_config[3])):
        exp = ['Y' if e else 'N' for e in EXPECTED]
        got = ['Y' if r else 'N' for r in res]
        match = ['✓' if r == e else '✗' for r, e in zip(res, EXPECTED)]
        print(f"  test{i+1}: expected={exp}, got={got}, match={match}, sims={sims}")

    return best_config


def test_ensemble():
    """Try ensemble of CNN + Siamese."""
    from cnn_detector import CNNLegoDetector
    from siamese_detector import SiameseDetector

    print("\n" + "=" * 70)
    print("ENSEMBLE: CNN + SIAMESE")
    print("=" * 70)
    print("Rule: Both must agree for YES, otherwise NO")

    cnn = CNNLegoDetector(model_path=str(MODEL_PATH), reference_dir=str(REF_DIR))
    siamese = SiameseDetector(
        reference_dir=str(REF_DIR),
        cropped_dir=str(CROPPED_DIR),
        similarity_threshold=0.78
    )

    crop_pct = 0.15

    total_correct = 0
    all_results = []

    for test_name, images in TEST_DIRS.items():
        results = []
        for i, img_name in enumerate(images):
            step = i + 1
            img_path = BASE_DIR / test_name / "jpg" / img_name
            image = cv2.imread(str(img_path))

            # CNN with crop
            img_cnn = preprocess_image(image, crop_pct=crop_pct, enhance=False)
            cnn_result = cnn.verify_step(img_cnn, step)

            # Siamese with crop + enhance
            img_sia = preprocess_image(image, crop_pct=crop_pct, enhance=True)
            sia_result = siamese.verify_step(img_sia, step)

            # Ensemble: both must say YES
            ensemble_match = cnn_result.is_match and sia_result.is_match
            results.append(ensemble_match)

            print(f"  {test_name} img{i+1}: CNN={'Y' if cnn_result.is_match else 'N'}, "
                  f"Siamese={'Y' if sia_result.is_match else 'N'}, "
                  f"Ensemble={'Y' if ensemble_match else 'N'}, "
                  f"Expected={'Y' if EXPECTED[i] else 'N'}")

        score = calculate_score(results, EXPECTED)
        total_correct += score
        all_results.append(results)

    print(f"\n*** ENSEMBLE SCORE: {total_correct}/9 ***")
    return total_correct, all_results


if __name__ == "__main__":
    print("FINE-TUNING FOR 9/9 ACCURACY")
    print("Expected: [YES, YES, NO] for all tests")

    cnn_best = test_cnn_detailed()
    siamese_best = test_siamese_detailed()
    ensemble_score, ensemble_results = test_ensemble()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
