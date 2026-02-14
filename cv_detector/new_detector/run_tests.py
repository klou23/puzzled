"""
Run all verification models on test images.

Test scenario:
- User starts at step 0 (hasn't completed anything)
- Submits image 1 -> should match step 1? (YES = step 1 done correctly)
- Submits image 2 -> should match step 2? (YES = step 2 done correctly)
- Submits image 3 -> should match step 3? (YES = step 3 done correctly)
"""

import cv2
from pathlib import Path

# Test directories and their image order
TEST_DIRS = {
    "test1": ["IMG_0939.jpg", "IMG_0940.jpg", "IMG_0941.jpg"],
    "test2": ["IMG_0939.jpg", "IMG_0940.jpg", "IMG_0941.jpg"],
    "test3": ["IMG_0946.jpg", "IMG_0947.jpg", "IMG_0948.jpg"],
}

BASE_DIR = Path("/Users/adam/Desktop/TreeHacks26/cv_detector/new_detector")
REF_DIR = BASE_DIR / "steps_jpg"
CROPPED_DIR = BASE_DIR / "cropped"
MODEL_PATH = BASE_DIR / "lego_model.pth"


def test_cnn():
    """Test CNN model."""
    from cnn_detector import CNNLegoDetector

    print("\n" + "=" * 60)
    print("CNN MODEL RESULTS")
    print("=" * 60)

    detector = CNNLegoDetector(
        model_path=str(MODEL_PATH),
        reference_dir=str(REF_DIR)
    )

    results = {}

    for test_name, images in TEST_DIRS.items():
        print(f"\n{test_name}:")
        results[test_name] = []

        for i, img_name in enumerate(images):
            step = i + 1  # Steps are 1-indexed
            img_path = BASE_DIR / test_name / "jpg" / img_name
            image = cv2.imread(str(img_path))

            if image is None:
                print(f"  Image {i+1}: Could not load {img_path}")
                results[test_name].append(None)
                continue

            result = detector.verify_step(image, step)
            match = "YES" if result.is_match else "NO"
            results[test_name].append(result.is_match)
            print(f"  Image {i+1} vs Step {step}: {match} (predicted: s{result.predicted_step}, conf: {result.confidence:.1%})")

    return results


def test_similarity():
    """Test Similarity model."""
    from similarity_detector import SimilarityDetector

    print("\n" + "=" * 60)
    print("SIMILARITY MODEL RESULTS")
    print("=" * 60)

    detector = SimilarityDetector(
        reference_dir=str(REF_DIR),
        cropped_dir=str(CROPPED_DIR),
        similarity_threshold=0.75
    )

    results = {}

    for test_name, images in TEST_DIRS.items():
        print(f"\n{test_name}:")
        results[test_name] = []

        for i, img_name in enumerate(images):
            step = i + 1
            img_path = BASE_DIR / test_name / "jpg" / img_name
            image = cv2.imread(str(img_path))

            if image is None:
                print(f"  Image {i+1}: Could not load {img_path}")
                results[test_name].append(None)
                continue

            result = detector.verify_step(image, step)
            match = "YES" if result.is_match else "NO"
            results[test_name].append(result.is_match)
            print(f"  Image {i+1} vs Step {step}: {match} (similarity: {result.similarity:.1%})")

    return results


def test_siamese():
    """Test Siamese model (without training, using pre-trained features)."""
    from siamese_detector import SiameseDetector

    print("\n" + "=" * 60)
    print("SIAMESE MODEL RESULTS (pre-trained, no fine-tuning)")
    print("=" * 60)

    # Use without trained weights - just pre-trained ResNet features
    detector = SiameseDetector(
        reference_dir=str(REF_DIR),
        cropped_dir=str(CROPPED_DIR),
        similarity_threshold=0.7
    )

    results = {}

    for test_name, images in TEST_DIRS.items():
        print(f"\n{test_name}:")
        results[test_name] = []

        for i, img_name in enumerate(images):
            step = i + 1
            img_path = BASE_DIR / test_name / "jpg" / img_name
            image = cv2.imread(str(img_path))

            if image is None:
                print(f"  Image {i+1}: Could not load {img_path}")
                results[test_name].append(None)
                continue

            result = detector.verify_step(image, step)
            match = "YES" if result.is_match else "NO"
            results[test_name].append(result.is_match)
            print(f"  Image {i+1} vs Step {step}: {match} (similarity: {result.similarity:.1%})")

    return results


if __name__ == "__main__":
    print("LEGO STEP VERIFICATION TEST")
    print("Testing: User submits 3 images sequentially")
    print("Expected: Image 1 matches Step 1, Image 2 matches Step 2, Image 3 matches Step 3")

    cnn_results = test_cnn()
    similarity_results = test_similarity()
    siamese_results = test_siamese()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nFormat: [Image1→Step1, Image2→Step2, Image3→Step3]")
    print("YES = Step completed correctly, NO = Step not matched\n")

    for test_name in TEST_DIRS:
        print(f"{test_name}:")
        cnn = ["YES" if r else "NO" for r in cnn_results.get(test_name, [])]
        sim = ["YES" if r else "NO" for r in similarity_results.get(test_name, [])]
        sia = ["YES" if r else "NO" for r in siamese_results.get(test_name, [])]
        print(f"  CNN:        [{', '.join(cnn)}]")
        print(f"  Similarity: [{', '.join(sim)}]")
        print(f"  Siamese:    [{', '.join(sia)}]")
