"""
Bounding Box Extractor for Lego pieces.
Detects and crops the Lego assembly from the background.
Supports both automatic detection and manual annotations.
"""

import cv2
import json
import numpy as np
from pathlib import Path


# Global cache for annotations
_annotations_cache = {}


def load_annotations(annotations_file) -> dict:
    """Load manual annotations from file."""
    global _annotations_cache
    annotations_file = str(annotations_file)

    if annotations_file not in _annotations_cache:
        path = Path(annotations_file)
        if path.exists():
            with open(path) as f:
                _annotations_cache[annotations_file] = json.load(f)
        else:
            _annotations_cache[annotations_file] = {}

    return _annotations_cache[annotations_file]


def crop_with_annotation(image: np.ndarray, bbox: list, padding: int = 10) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop image using a manual annotation."""
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]

    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_w - x, w + 2 * padding)
    h = min(img_h - y, h + 2 * padding)

    cropped = image[y:y+h, x:x+w]
    return cropped, (x, y, w, h)


def extract_bounding_box(image: np.ndarray, padding: int = 20) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Extract bounding box around the Lego piece(s) in an image.

    Args:
        image: BGR image (from cv2.imread)
        padding: Extra pixels around the detected object

    Returns:
        cropped_image: The cropped region containing the Lego
        bbox: (x, y, w, h) of the bounding box
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # The background appears to be tan/brown cardboard
    # We'll detect non-background pixels

    # Method 1: Edge detection + contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding works well for objects on uniform backgrounds
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Also try Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Combine both methods
    combined = cv2.bitwise_or(thresh, edges)

    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.dilate(combined, kernel, iterations=2)
    combined = cv2.erode(combined, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Return full image if no contours found
        h, w = image.shape[:2]
        return image, (0, 0, w, h)

    # Find the largest contour (should be the Lego piece)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add padding
    img_h, img_w = image.shape[:2]
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_w - x, w + 2 * padding)
    h = min(img_h - y, h + 2 * padding)

    # Crop the image
    cropped = image[y:y+h, x:x+w]

    return cropped, (x, y, w, h)


def extract_bounding_box_color(image: np.ndarray, padding: int = 30) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Alternative method using color-based segmentation.
    Better for detecting colored Lego pieces on tan background.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for tan/brown background (to exclude)
    # Tan background in HSV: low saturation, medium-high value
    lower_tan = np.array([10, 20, 100])
    upper_tan = np.array([30, 120, 220])

    # Create mask for background
    bg_mask = cv2.inRange(hsv, lower_tan, upper_tan)

    # Invert to get foreground (Lego pieces)
    fg_mask = cv2.bitwise_not(bg_mask)

    # Clean up the mask
    kernel = np.ones((7, 7), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        h, w = image.shape[:2]
        return image, (0, 0, w, h)

    # Get bounding box that encompasses all significant contours
    min_area = 500  # Minimum contour area to consider
    significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not significant_contours:
        largest = max(contours, key=cv2.contourArea)
        significant_contours = [largest]

    # Combine all significant contours
    all_points = np.vstack(significant_contours)
    x, y, w, h = cv2.boundingRect(all_points)

    # Add padding
    img_h, img_w = image.shape[:2]
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_w - x, w + 2 * padding)
    h = min(img_h - y, h + 2 * padding)

    cropped = image[y:y+h, x:x+w]
    return cropped, (x, y, w, h)


def process_image(image_path: str, output_dir: str = None, method: str = "combined") -> np.ndarray:
    """
    Process a single image and optionally save the cropped result.

    Args:
        image_path: Path to the input image
        output_dir: Directory to save cropped images (optional)
        method: "edge", "color", or "combined"
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if method == "edge":
        cropped, bbox = extract_bounding_box(image)
    elif method == "color":
        cropped, bbox = extract_bounding_box_color(image)
    else:
        # Combined: try both and pick the one with smaller bbox (tighter fit)
        cropped1, bbox1 = extract_bounding_box(image)
        cropped2, bbox2 = extract_bounding_box_color(image)

        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]

        # Pick the one that's not too small but tighter
        min_area = 0.01 * image.shape[0] * image.shape[1]

        if area1 > min_area and area2 > min_area:
            cropped, bbox = (cropped1, bbox1) if area1 < area2 else (cropped2, bbox2)
        elif area1 > min_area:
            cropped, bbox = cropped1, bbox1
        else:
            cropped, bbox = cropped2, bbox2

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = Path(image_path).name
        output_path = Path(output_dir) / f"cropped_{filename}"
        cv2.imwrite(str(output_path), cropped)

    return cropped, bbox


def batch_process(input_dir: str, output_dir: str, method: str = "combined"):
    """Process all JPG images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_file in sorted(input_path.glob("*.jpg")):
        try:
            cropped, bbox = process_image(str(img_file), str(output_path), method)
            print(f"Processed {img_file.name}: bbox={bbox}")
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract bounding boxes from Lego images")
    parser.add_argument("--input", "-i", default="./steps_jpg", help="Input directory")
    parser.add_argument("--output", "-o", default="./cropped", help="Output directory")
    parser.add_argument("--method", "-m", choices=["edge", "color", "combined"],
                        default="combined", help="Detection method")

    args = parser.parse_args()
    batch_process(args.input, args.output, args.method)
