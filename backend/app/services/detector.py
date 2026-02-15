"""
Step Verification Service using Claude API.
Compares captured images against reference images.
"""

import anthropic
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


@dataclass
class VerificationResult:
    step: int
    is_match: bool
    confidence: str  # "high", "medium", "low"
    explanation: str


class StepDetector:
    """Step verification using Claude's vision capabilities."""

    def __init__(self, reference_dir: str, api_key: str = None):
        """
        Initialize the detector.

        Args:
            reference_dir: Directory containing step reference images
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.reference_dir = Path(reference_dir)
        self.reference_images = self._load_reference_images()

    def _load_reference_images(self) -> dict[int, list[Path]]:
        """Load all reference images grouped by step number."""
        references = {}
        # Support multiple naming patterns
        for pattern in ["s*-*.jpg", "step*.jpg", "*.jpg"]:
            for f in sorted(self.reference_dir.glob(pattern)):
                # Try to extract step number from filename
                name = f.stem.lower()
                if name.startswith("s") and "-" in name:
                    step_num = int(name.split("-")[0][1:])
                elif name.startswith("step"):
                    step_num = int("".join(filter(str.isdigit, name)) or 0)
                else:
                    continue

                if step_num not in references:
                    references[step_num] = []
                references[step_num].append(f)

        return references

    def _image_to_base64(self, image_path: Path) -> str:
        """Convert an image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _bytes_to_base64(self, image_bytes: bytes) -> str:
        """Convert image bytes to base64 string."""
        return base64.standard_b64encode(image_bytes).decode("utf-8")

    def verify_step(
        self,
        captured_image: bytes,
        target_step: int,
        task_context: str = "assembly"
    ) -> VerificationResult:
        """
        Verify if the captured image matches the target step.

        Args:
            captured_image: Image bytes (JPEG)
            target_step: Step number to verify against
            task_context: Description of the task type

        Returns:
            VerificationResult with match status and explanation
        """
        if target_step not in self.reference_images:
            return VerificationResult(
                step=target_step,
                is_match=False,
                confidence="low",
                explanation=f"No reference images found for step {target_step}"
            )

        captured_b64 = self._bytes_to_base64(captured_image)
        ref_path = self.reference_images[target_step][0]
        ref_b64 = self._image_to_base64(ref_path)

        prompt = f"""You are a step verification assistant. Compare these two images:

IMAGE 1 (Reference): This shows what step {target_step} of the {task_context} should look like when completed correctly.

IMAGE 2 (Captured): This is what the user has done.

Analyze if the captured image matches the reference for step {target_step}.

Consider:
1. Are the same components/pieces present?
2. Are they arranged in the same configuration?
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
                            {"type": "text", "text": prompt},
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

    def get_total_steps(self) -> int:
        """Get the total number of steps."""
        return max(self.reference_images.keys()) if self.reference_images else 0


# Singleton instance (lazy loaded)
_detector: Optional[StepDetector] = None


def get_detector() -> StepDetector:
    """Get or create the detector instance."""
    global _detector
    if _detector is None:
        reference_dir = os.getenv("REFERENCE_DIR", "./reference_images")
        _detector = StepDetector(reference_dir)
    return _detector
