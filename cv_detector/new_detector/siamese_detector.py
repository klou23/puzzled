"""
Lego Step Verification using Siamese Network.
Learns to compare image pairs and determine if they match.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


@dataclass
class VerificationResult:
    step: int
    is_match: bool
    confidence: float
    similarity: float
    explanation: str


class SiamesePairDataset(Dataset):
    """
    Dataset that generates pairs of images for Siamese training.

    For each sample, we return:
    - img1: Anchor image
    - img2: Either same-step (positive) or different-step (negative) image
    - label: 1 if same step, 0 if different
    """

    def __init__(self, image_dir: str, cropped_dir: str = None,
                 pairs_per_image: int = 10, transform=None):
        self.image_dir = Path(image_dir)
        self.cropped_dir = Path(cropped_dir) if cropped_dir else None
        self.pairs_per_image = pairs_per_image
        self.transform = transform

        # Group images by step
        self.step_images = {}  # step_num -> list of image paths
        self._load_images()

        # Generate pairs
        self.pairs = []
        self._generate_pairs()

    def _load_images(self):
        """Load and group images by step number."""
        # Prefer cropped images
        search_dir = self.cropped_dir if self.cropped_dir and self.cropped_dir.exists() else self.image_dir
        pattern = "cropped_s*-*.jpg" if "cropped" in str(search_dir) else "s*-*.jpg"

        for img_path in sorted(search_dir.glob(pattern)):
            name = img_path.stem.replace("cropped_", "")
            step_num = int(name.split("-")[0][1:])

            if step_num not in self.step_images:
                self.step_images[step_num] = []
            self.step_images[step_num].append(img_path)

    def _generate_pairs(self):
        """Generate positive and negative pairs."""
        all_steps = list(self.step_images.keys())

        for step in all_steps:
            images = self.step_images[step]
            other_steps = [s for s in all_steps if s != step]

            for img_path in images:
                # Generate positive pairs (same step)
                for _ in range(self.pairs_per_image // 2):
                    # Pair with same image (will be augmented differently)
                    self.pairs.append((img_path, img_path, 1))

                    # Pair with another image from same step if available
                    if len(images) > 1:
                        other = random.choice([p for p in images if p != img_path])
                        self.pairs.append((img_path, other, 1))

                # Generate negative pairs (different step)
                for _ in range(self.pairs_per_image // 2):
                    if other_steps:
                        neg_step = random.choice(other_steps)
                        neg_img = random.choice(self.step_images[neg_step])
                        self.pairs.append((img_path, neg_img, 0))

        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


class SiameseNetwork(nn.Module):
    """
    Siamese Network using shared ResNet backbone.

    Takes two images, extracts features from each using the same network,
    then compares the features to produce a similarity score.
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()

        # Use pre-trained ResNet18 as backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove classification layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Freeze early layers for transfer learning
        for param in list(self.features.parameters())[:50]:
            param.requires_grad = False

    def forward_one(self, x):
        """Extract embedding for one image."""
        x = self.features(x)
        x = self.embedding(x)
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        """
        Forward pass for a pair of images.
        Returns embeddings for both images.
        """
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.

    For similar pairs: minimize distance
    For dissimilar pairs: push apart if distance < margin
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # Euclidean distance
        distance = F.pairwise_distance(emb1, emb2)

        # Contrastive loss
        # label=1: similar, minimize distance
        # label=0: dissimilar, push apart
        loss = label * distance.pow(2) + \
               (1 - label) * F.relu(self.margin - distance).pow(2)

        return loss.mean()


class SiameseDetector:
    """
    Lego step verification using trained Siamese network.
    """

    def __init__(self, model_path: str = None, reference_dir: str = None,
                 cropped_dir: str = None, similarity_threshold: float = 0.81,
                 device: str = None, crop_percent: float = 0.15,
                 use_enhancement: bool = True):
        self.device = device or self._get_device()
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.cropped_dir = Path(cropped_dir) if cropped_dir else None
        self.similarity_threshold = similarity_threshold
        self.crop_percent = crop_percent
        self.use_enhancement = use_enhancement

        self.current_step = 1
        self.total_steps = 0
        self.reference_images = {}
        self.reference_embeddings = {}

        # Initialize model
        self.model = SiameseNetwork()
        self.model = self.model.to(self.device)

        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()

        # Image preprocessing (no augmentation for inference)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load reference images
        self._load_references()

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_references(self):
        """Load reference images and compute embeddings."""
        search_dir = self.cropped_dir if self.cropped_dir and self.cropped_dir.exists() else self.reference_dir

        if not search_dir or not search_dir.exists():
            print("Warning: No reference directory found")
            return

        pattern = "cropped_s*-*.jpg" if "cropped" in str(search_dir) else "s*-*.jpg"

        for img_path in sorted(search_dir.glob(pattern)):
            name = img_path.stem.replace("cropped_", "")
            step_num = int(name.split("-")[0][1:])

            if step_num not in self.reference_images:
                self.reference_images[step_num] = []
                self.reference_embeddings[step_num] = []

            self.reference_images[step_num].append(img_path)

            # Compute embedding
            embedding = self._compute_embedding(img_path)
            self.reference_embeddings[step_num].append(embedding)

        self.total_steps = max(self.reference_images.keys()) if self.reference_images else 0
        print(f"Loaded {self.total_steps} steps with {sum(len(v) for v in self.reference_images.values())} reference images")

    def _compute_embedding(self, image_path: Path) -> torch.Tensor:
        """Compute embedding for an image file."""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.forward_one(input_tensor)

        return embedding.squeeze(0)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply recommended preprocessing: center crop and CLAHE enhancement.

        Call this on captured images before verify_step() for best results.
        Optimal settings: crop_percent=0.15, use_enhancement=True

        Args:
            image: BGR image from OpenCV

        Returns:
            Preprocessed BGR image
        """
        result = image.copy()

        # Center crop
        if self.crop_percent > 0:
            h, w = result.shape[:2]
            margin_x = int(w * self.crop_percent)
            margin_y = int(h * self.crop_percent)
            result = result[margin_y:h-margin_y, margin_x:w-margin_x]

        # CLAHE enhancement
        if self.use_enhancement:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            result = cv2.merge([l, a, b])
            result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

        return result

    def _compute_embedding_from_numpy(self, image: np.ndarray) -> torch.Tensor:
        """Compute embedding for a numpy image."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.forward_one(input_tensor)

        return embedding.squeeze(0)

    def verify_step(self, captured_image: np.ndarray, target_step: int = None) -> VerificationResult:
        """
        Verify if captured image matches the target step.
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

        # Compute embedding for captured image
        captured_emb = self._compute_embedding_from_numpy(captured_image)

        # Compare against all reference embeddings for this step
        similarities = []
        for ref_emb in self.reference_embeddings[target_step]:
            # Cosine similarity (embeddings are normalized)
            sim = torch.dot(captured_emb, ref_emb).item()
            similarities.append(sim)

        max_similarity = max(similarities)

        # Determine match
        is_match = max_similarity >= self.similarity_threshold

        # Confidence scaling
        if is_match:
            confidence = 0.5 + 0.5 * (max_similarity - self.similarity_threshold) / (1 - self.similarity_threshold)
        else:
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
        if self.current_step < self.total_steps:
            self.current_step += 1
            return True
        return False

    def reset(self):
        self.current_step = 1

    def get_status(self) -> dict:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": f"{self.current_step}/{self.total_steps}"
        }

    def train(self, epochs: int = 30, batch_size: int = 16,
              learning_rate: float = 0.001, save_path: str = "siamese_model.pth"):
        """
        Train the Siamese network.
        """
        print(f"\nTraining Siamese Network")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print()

        # Training augmentation with heavy brightness variation
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(25),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            # Heavy brightness/contrast variation to learn invariance
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # Random grayscale to learn shape over color
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.2)
        ])

        # Create dataset
        search_dir = self.cropped_dir if self.cropped_dir and self.cropped_dir.exists() else self.reference_dir
        dataset = SiamesePairDataset(
            image_dir=str(self.reference_dir),
            cropped_dir=str(self.cropped_dir) if self.cropped_dir else None,
            pairs_per_image=20,
            transform=train_transform
        )

        print(f"Training pairs: {len(dataset)}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        # Loss and optimizer
        criterion = ContrastiveLoss(margin=1.0)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for img1, img2, labels in pbar:
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                emb1, emb2 = self.model(img1, img2)
                loss = criterion(emb1, emb2, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Compute accuracy
                with torch.no_grad():
                    distances = F.pairwise_distance(emb1, emb2)
                    predictions = (distances < 0.5).float()
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * correct / total:.1f}%"
                })

            scheduler.step()

            avg_loss = total_loss / len(dataloader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%")

        # Save model
        torch.save(self.model.state_dict(), save_path)
        print(f"\nModel saved to {save_path}")

        self.model.eval()

        # Recompute reference embeddings with trained model
        self.reference_embeddings = {}
        for step, paths in self.reference_images.items():
            self.reference_embeddings[step] = []
            for path in paths:
                emb = self._compute_embedding(path)
                self.reference_embeddings[step].append(emb)


class SiameseAssemblyApp:
    """Interactive app using Siamese network verification."""

    def __init__(self, model_path: str = None, reference_dir: str = None,
                 cropped_dir: str = None, camera_index: int = 0,
                 threshold: float = 0.7):
        self.detector = SiameseDetector(
            model_path=model_path,
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

        print(f"\nSiamese Network Lego Verification")
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

                cv2.imshow("Lego Verification (Siamese)", display)

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
                    self.detector.similarity_threshold = max(0.3, self.detector.similarity_threshold - 0.05)
                    print(f"Threshold: {self.detector.similarity_threshold:.0%}")

        finally:
            self.stop_camera()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Siamese Network Lego Verification")
    parser.add_argument("--reference", "-r", default="./steps_jpg", help="Reference images dir")
    parser.add_argument("--cropped", "-x", default="./cropped", help="Cropped images dir")
    parser.add_argument("--model", "-m", default="siamese_model.pth", help="Model file path")
    parser.add_argument("--train", "-t", action="store_true", help="Train the model")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Training epochs")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    parser.add_argument("--image", "-i", help="Test single image")
    parser.add_argument("--step", "-s", type=int, default=1, help="Step to verify against")

    args = parser.parse_args()

    if args.train:
        detector = SiameseDetector(
            reference_dir=args.reference,
            cropped_dir=args.cropped,
            similarity_threshold=args.threshold
        )
        detector.train(epochs=args.epochs, save_path=args.model)

    if args.image:
        model_path = args.model if Path(args.model).exists() else None
        if not model_path and not args.train:
            print(f"Error: Model file not found: {args.model}")
            print("Run with --train first to create a model.")
        else:
            detector = SiameseDetector(
                model_path=model_path,
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
    elif not args.train:
        model_path = args.model if Path(args.model).exists() else None
        app = SiameseAssemblyApp(
            model_path=model_path,
            reference_dir=args.reference,
            cropped_dir=args.cropped,
            camera_index=args.camera,
            threshold=args.threshold
        )
        app.run_interactive()
