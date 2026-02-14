"""
Lego Step Verification using Feature Similarity.
No training required - uses pre-trained embeddings to compare images.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image


@dataclass
class VerificationResult:
    step: int
    is_match: bool
    confidence: float
    similarity: float
    explanation: str


class SimilarityDetector:
    """
    Lego step verification using cosine similarity of image embeddings.

    No training required! Uses pre-trained ResNet features to compare
    the captured image against reference images for the current step.
    """

    def __init__(self, reference_dir: str = None, cropped_dir: str = None,
                 similarity_threshold: float = 0.75, device: str = None):
        """
        Initialize the detector.

        Args:
            reference_dir: Directory with original step images (fallback)
            cropped_dir: Directory with cropped reference images (preferred)
            similarity_threshold: Minimum similarity to consider a match (0-1)
            device: 'cuda', 'mps', or 'cpu'
        """
        self.device = device or self._get_device()
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.cropped_dir = Path(cropped_dir) if cropped_dir else None
        self.similarity_threshold = similarity_threshold

        self.current_step = 1
        self.total_steps = 0
        self.reference_images = {}  # step -> list of image paths
        self.reference_embeddings = {}  # step -> list of embeddings

        # Load pre-trained model for feature extraction
        print(f"Loading feature extractor on {self.device}...")
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the classification head - we just want features
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load reference images and compute embeddings
        self._load_references()
        print(f"Loaded {self.total_steps} steps with {sum(len(v) for v in self.reference_images.values())} reference images")

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_references(self):
        """Load reference images and compute their embeddings."""
        # Prefer cropped images
        search_dir = self.cropped_dir if self.cropped_dir and self.cropped_dir.exists() else self.reference_dir

        if not search_dir or not search_dir.exists():
            print("Warning: No reference directory found")
            return

        # Find all step images
        pattern = "cropped_s*-*.jpg" if "cropped" in str(search_dir) else "s*-*.jpg"

        for img_path in sorted(search_dir.glob(pattern)):
            # Extract step number
            name = img_path.stem.replace("cropped_", "")
            step_num = int(name.split("-")[0][1:])

            if step_num not in self.reference_images:
                self.reference_images[step_num] = []
                self.reference_embeddings[step_num] = []

            self.reference_images[step_num].append(img_path)

            # Compute embedding
            embedding = self._compute_embedding_from_path(img_path)
            self.reference_embeddings[step_num].append(embedding)

        self.total_steps = max(self.reference_images.keys()) if self.reference_images else 0

    def _compute_embedding_from_path(self, image_path: Path) -> torch.Tensor:
        """Compute embedding for an image file."""
        image = Image.open(image_path).convert('RGB')
        return self._compute_embedding(image)

    def _compute_embedding(self, pil_image: Image.Image) -> torch.Tensor:
        """Compute embedding for a PIL image."""
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(input_tensor)
            embedding = embedding.squeeze()  # Remove batch and spatial dims
            embedding = F.normalize(embedding, dim=0)  # L2 normalize

        return embedding

    def _compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings."""
        return torch.dot(embedding1, embedding2).item()

    def verify_step(self, captured_image: np.ndarray, target_step: int = None) -> VerificationResult:
        """
        Verify if captured image matches the target step.

        Args:
            captured_image: BGR image from OpenCV
            target_step: Step to verify (defaults to current_step)

        Returns:
            VerificationResult with match status and similarity score
        """
        if target_step is None:
            target_step = self.current_step

        if target_step not in self.reference_embeddings:
            return VerificationResult(
                step=target_step,
                is_match=False,
                confidence=0.0,
                similarity=0.0,
                explanation=f"No reference images for step {target_step}"
            )

        # Convert captured image to PIL
        captured_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(captured_rgb)

        # Compute embedding for captured image
        captured_embedding = self._compute_embedding(pil_image)

        # Compare against all reference images for this step
        similarities = []
        for ref_embedding in self.reference_embeddings[target_step]:
            sim = self._compute_similarity(captured_embedding, ref_embedding)
            similarities.append(sim)

        # Use max similarity (best match among references)
        max_similarity = max(similarities)
        avg_similarity = sum(similarities) / len(similarities)

        # Determine if it's a match
        is_match = max_similarity >= self.similarity_threshold

        # Confidence based on how far above/below threshold
        if is_match:
            # Scale 0.75-1.0 to 0.5-1.0 confidence
            confidence = 0.5 + 0.5 * (max_similarity - self.similarity_threshold) / (1 - self.similarity_threshold)
        else:
            # Scale 0-0.75 to 0-0.5 confidence
            confidence = 0.5 * max_similarity / self.similarity_threshold

        confidence = min(1.0, max(0.0, confidence))

        if is_match:
            explanation = f"Matches step {target_step} (similarity: {max_similarity:.2%})"
        else:
            explanation = f"Does not match step {target_step} (similarity: {max_similarity:.2%}, need {self.similarity_threshold:.0%})"

        return VerificationResult(
            step=target_step,
            is_match=is_match,
            confidence=confidence,
            similarity=max_similarity,
            explanation=explanation
        )

    def advance_step(self) -> bool:
        """Move to the next step."""
        if self.current_step < self.total_steps:
            self.current_step += 1
            return True
        return False

    def reset(self):
        """Reset to step 1."""
        self.current_step = 1

    def get_status(self) -> dict:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": f"{self.current_step}/{self.total_steps}"
        }


class SimilarityAssemblyApp:
    """Interactive app using similarity-based verification."""

    def __init__(self, reference_dir: str = None, cropped_dir: str = None,
                 camera_index: int = 0, threshold: float = 0.75):
        self.detector = SimilarityDetector(
            reference_dir=reference_dir,
            cropped_dir=cropped_dir,
            similarity_threshold=threshold
        )
        self.camera_index = camera_index
        self.cap = None

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")

    def stop_camera(self):
        if self.cap:
            self.cap.release()

    def capture_frame(self) -> np.ndarray:
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera not initialized")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame

    def run_interactive(self):
        """Run interactive verification session."""
        self.start_camera()

        print(f"\nSimilarity-Based Lego Verification")
        print(f"Total steps: {self.detector.total_steps}")
        print(f"Threshold: {self.detector.similarity_threshold:.0%}")
        print(f"\nControls:")
        print("  SPACE  - Verify current step")
        print("  UP/DOWN- Adjust threshold")
        print("  R      - Reset to step 1")
        print("  Q      - Quit\n")

        try:
            while True:
                frame = self.capture_frame()
                display = frame.copy()

                # Draw status
                status = self.detector.get_status()
                cv2.putText(display, f"Step {status['current_step']}/{status['total_steps']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display, f"Threshold: {self.detector.similarity_threshold:.0%}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(display, "SPACE=Verify, R=Reset, Q=Quit",
                            (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Lego Verification", display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord(' '):
                    print(f"\nVerifying step {self.detector.current_step}...")
                    result = self.detector.verify_step(frame)

                    print(f"  Similarity: {result.similarity:.2%}")
                    print(f"  Match: {result.is_match}")
                    print(f"  {result.explanation}")

                    if result.is_match:
                        if self.detector.advance_step():
                            print(f"  -> Advanced to step {self.detector.current_step}")
                        else:
                            print("  *** ASSEMBLY COMPLETE! ***")

                elif key == ord('r'):
                    self.detector.reset()
                    print("\nReset to step 1")
                elif key == 82:  # Up arrow
                    self.detector.similarity_threshold = min(0.95, self.detector.similarity_threshold + 0.05)
                    print(f"Threshold: {self.detector.similarity_threshold:.0%}")
                elif key == 84:  # Down arrow
                    self.detector.similarity_threshold = max(0.5, self.detector.similarity_threshold - 0.05)
                    print(f"Threshold: {self.detector.similarity_threshold:.0%}")

        finally:
            self.stop_camera()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Similarity-based Lego verification")
    parser.add_argument("--reference", "-r", default="./steps_jpg", help="Reference images dir")
    parser.add_argument("--cropped", "-x", default="./cropped", help="Cropped images dir")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
    parser.add_argument("--threshold", "-t", type=float, default=0.75, help="Similarity threshold")
    parser.add_argument("--image", "-i", help="Test single image")
    parser.add_argument("--step", "-s", type=int, default=1, help="Step to verify against")

    args = parser.parse_args()

    if args.image:
        detector = SimilarityDetector(
            reference_dir=args.reference,
            cropped_dir=args.cropped,
            similarity_threshold=args.threshold
        )
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not read {args.image}")
        else:
            result = detector.verify_step(image, args.step)
            print(f"\nVerification Result:")
            print(f"  Step: {result.step}")
            print(f"  Similarity: {result.similarity:.2%}")
            print(f"  Match: {result.is_match}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  {result.explanation}")
    else:
        app = SimilarityAssemblyApp(
            reference_dir=args.reference,
            cropped_dir=args.cropped,
            camera_index=args.camera,
            threshold=args.threshold
        )
        app.run_interactive()
