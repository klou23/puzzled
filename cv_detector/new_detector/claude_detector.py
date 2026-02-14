"""
Lego Step Verification using Claude API.
Compares a captured image against reference images for each step.
"""

import anthropic
import base64
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from bbox_extractor import extract_bounding_box_color, load_annotations, crop_with_annotation


@dataclass
class VerificationResult:
    step: int
    is_match: bool
    confidence: str  # "high", "medium", "low"
    explanation: str


class ClaudeLegoDetector:
    """Lego step verification using Claude's vision capabilities."""

    def __init__(self, api_key: str, reference_dir: str, cropped_dir: str = None):
        """
        Initialize the detector.

        Args:
            api_key: Anthropic API key
            reference_dir: Directory containing step reference images (s1-0.jpg, s2-0.jpg, etc.)
            cropped_dir: Directory containing manually cropped images (optional)
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.reference_dir = Path(reference_dir)
        self.cropped_dir = Path(cropped_dir) if cropped_dir else None
        self.annotations = {}

        # Try to load manual annotations
        if self.cropped_dir:
            annotations_file = self.cropped_dir / "annotations.json"
            self.annotations = load_annotations(annotations_file)

        self.current_step = 1
        self.total_steps = self._count_steps()
        self.reference_images = self._load_reference_images()

    def _count_steps(self) -> int:
        """Count the number of unique steps in the reference directory."""
        steps = set()
        for f in self.reference_dir.glob("s*-*.jpg"):
            step_num = f.stem.split("-")[0][1:]  # Extract number from "s1-0"
            steps.add(int(step_num))
        return max(steps) if steps else 0

    def _load_reference_images(self) -> dict[int, list[Path]]:
        """Load all reference images grouped by step number."""
        references = {}
        for f in sorted(self.reference_dir.glob("s*-*.jpg")):
            step_num = int(f.stem.split("-")[0][1:])
            if step_num not in references:
                references[step_num] = []
            references[step_num].append(f)
        return references

    def _image_to_base64(self, image_path) -> str:
        """Convert an image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _numpy_to_base64(self, image: np.ndarray) -> str:
        """Convert a numpy image array to base64 string."""
        _, buffer = cv2.imencode(".jpg", image)
        return base64.standard_b64encode(buffer).decode("utf-8")

    def verify_step(self, captured_image: np.ndarray, target_step: int = None) -> VerificationResult:
        """
        Verify if the captured image matches the target step.

        Args:
            captured_image: BGR image from camera/file
            target_step: Step number to verify against (defaults to current_step)

        Returns:
            VerificationResult with match status and explanation
        """
        if target_step is None:
            target_step = self.current_step

        if target_step not in self.reference_images:
            return VerificationResult(
                step=target_step,
                is_match=False,
                confidence="low",
                explanation=f"No reference images found for step {target_step}"
            )

        # Crop the captured image to focus on the Lego
        cropped, _ = extract_bounding_box_color(captured_image)
        captured_b64 = self._numpy_to_base64(cropped)

        # Get reference image (prefer cropped version if available)
        ref_path = self.reference_images[target_step][0]
        cropped_ref_path = None
        if self.cropped_dir:
            cropped_ref_path = self.cropped_dir / f"cropped_{ref_path.name}"

        if cropped_ref_path and cropped_ref_path.exists():
            ref_b64 = self._image_to_base64(cropped_ref_path)
        else:
            ref_b64 = self._image_to_base64(ref_path)

        # Create the prompt for Claude
        prompt = f"""You are a Lego assembly verification assistant. Compare these two images:

IMAGE 1 (Reference): This shows what step {target_step} of the Lego assembly should look like when completed correctly.

IMAGE 2 (Captured): This is what the user has built.

Analyze if the captured image matches the reference for step {target_step}.

Consider:
1. Are the same Lego pieces present?
2. Are they assembled in the same configuration?
3. Is the overall shape/structure matching?
4. Minor differences in angle/lighting are acceptable.

Respond in this exact format:
MATCH: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
EXPLANATION: [Brief explanation of your assessment]"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": ref_b64
                                }
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": captured_b64
                                }
                            }
                        ]
                    }
                ]
            )

            # Parse the response
            response_text = response.content[0].text
            lines = response_text.strip().split("\n")

            is_match = False
            confidence = "low"
            explanation = response_text

            for line in lines:
                if line.startswith("MATCH:"):
                    is_match = "YES" in line.upper()
                elif line.startswith("CONFIDENCE:"):
                    conf = line.split(":")[1].strip().upper()
                    if conf in ["HIGH", "MEDIUM", "LOW"]:
                        confidence = conf.lower()
                elif line.startswith("EXPLANATION:"):
                    explanation = line.split(":", 1)[1].strip()

            return VerificationResult(
                step=target_step,
                is_match=is_match,
                confidence=confidence,
                explanation=explanation
            )

        except Exception as e:
            return VerificationResult(
                step=target_step,
                is_match=False,
                confidence="low",
                explanation=f"API error: {str(e)}"
            )

    def advance_step(self) -> bool:
        """Move to the next step if available."""
        if self.current_step < self.total_steps:
            self.current_step += 1
            return True
        return False

    def reset(self):
        """Reset to step 1."""
        self.current_step = 1

    def get_status(self) -> dict:
        """Get current progress status."""
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": f"{self.current_step}/{self.total_steps}"
        }


class LegoAssemblyApp:
    """Main application for Lego assembly verification with state progression."""

    def __init__(self, api_key: str, reference_dir: str, camera_index: int = 0, cropped_dir: str = None):
        self.detector = ClaudeLegoDetector(api_key, reference_dir, cropped_dir)
        self.camera_index = camera_index
        self.cap = None

    def start_camera(self):
        """Initialize the camera."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")

    def stop_camera(self):
        """Release the camera."""
        if self.cap:
            self.cap.release()

    def capture_frame(self) -> np.ndarray:
        """Capture a single frame from the camera."""
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera not initialized")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame

    def verify_current_step(self, frame: np.ndarray = None) -> VerificationResult:
        """
        Verify the current step.

        Args:
            frame: Optional pre-captured frame. If None, captures from camera.
        """
        if frame is None:
            frame = self.capture_frame()
        return self.detector.verify_step(frame)

    def try_advance(self, frame: np.ndarray = None) -> tuple[bool, VerificationResult]:
        """
        Try to verify and advance to the next step.

        Returns:
            (advanced, result): Whether we advanced and the verification result
        """
        result = self.verify_current_step(frame)
        if result.is_match and result.confidence in ["high", "medium"]:
            advanced = self.detector.advance_step()
            return advanced, result
        return False, result

    def run_interactive(self):
        """Run an interactive session with the camera."""
        self.start_camera()
        print(f"\nLego Assembly Verification")
        print(f"Total steps: {self.detector.total_steps}")
        print(f"\nControls:")
        print("  SPACE - Verify current step")
        print("  R     - Reset to step 1")
        print("  Q     - Quit\n")

        try:
            while True:
                frame = self.capture_frame()

                # Draw status overlay
                status = self.detector.get_status()
                cv2.putText(
                    frame,
                    f"Step {status['current_step']}/{status['total_steps']}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    "SPACE=Verify, R=Reset, Q=Quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )

                cv2.imshow("Lego Assembly", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    print(f"\nVerifying step {self.detector.current_step}...")
                    advanced, result = self.try_advance(frame)

                    print(f"  Match: {result.is_match}")
                    print(f"  Confidence: {result.confidence}")
                    print(f"  {result.explanation}")

                    if advanced:
                        if self.detector.current_step > self.detector.total_steps:
                            print("\n*** ASSEMBLY COMPLETE! ***")
                        else:
                            print(f"\n  -> Advanced to step {self.detector.current_step}")
                elif key == ord('r'):
                    self.detector.reset()
                    print("\nReset to step 1")

        finally:
            self.stop_camera()
            cv2.destroyAllWindows()


def verify_from_file(api_key: str, reference_dir: str, image_path: str, step: int):
    """Verify a single image file against a specific step."""
    detector = ClaudeLegoDetector(api_key, reference_dir)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read {image_path}")
        return

    result = detector.verify_step(image, step)
    print(f"\nVerification Result for Step {step}:")
    print(f"  Match: {result.is_match}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Explanation: {result.explanation}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lego Step Verification with Claude API")
    parser.add_argument("--api-key", "-k", required=True, help="Anthropic API key")
    parser.add_argument("--reference", "-r", default="./steps_jpg", help="Reference images directory")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
    parser.add_argument("--image", "-i", help="Single image to verify (instead of camera)")
    parser.add_argument("--step", "-s", type=int, default=1, help="Step number to verify against")

    args = parser.parse_args()

    if args.image:
        verify_from_file(args.api_key, args.reference, args.image, args.step)
    else:
        app = LegoAssemblyApp(args.api_key, args.reference, args.camera)
        app.run_interactive()
