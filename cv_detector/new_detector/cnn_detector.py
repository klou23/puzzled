"""
Lego Step Verification using a Custom CNN Model.
Uses transfer learning with a pre-trained ResNet for step classification.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from bbox_extractor import extract_bounding_box_color, load_annotations, crop_with_annotation


@dataclass
class VerificationResult:
    step: int
    is_match: bool
    confidence: float
    predicted_step: int
    explanation: str


class LegoStepDataset(Dataset):
    """Dataset for Lego step images with heavy augmentation for small datasets."""

    def __init__(self, image_dir: str = None, transform=None, augment: bool = True,
                 cropped_dir: str = None, augment_factor: int = 20):
        self.image_dir = Path(image_dir) if image_dir else None
        self.cropped_dir = Path(cropped_dir) if cropped_dir else None
        self.transform = transform
        self.augment = augment
        self.augment_factor = augment_factor  # How many augmented versions per image
        self.images = []
        self.labels = []
        self.step_to_idx = {}
        self.idx_to_step = {}

        # Load manual annotations if available
        self.annotations = {}
        if self.cropped_dir:
            annotations_file = self.cropped_dir / "annotations.json"
            self.annotations = load_annotations(annotations_file)

        self._load_images()

    def _load_images(self):
        """Load all images and create step mappings."""
        steps = set()

        # First, find all cropped images directly
        if self.cropped_dir and self.cropped_dir.exists():
            for f in self.cropped_dir.glob("cropped_s*-*.jpg"):
                # Extract step number from "cropped_s1-0.jpg" -> "s1" -> 1
                name_part = f.stem.replace("cropped_", "")  # "s1-0"
                step_num = int(name_part.split("-")[0][1:])  # 1
                steps.add(step_num)

        # Fallback to image_dir if no cropped images
        if not steps and self.image_dir:
            for f in self.image_dir.glob("s*-*.jpg"):
                step_num = int(f.stem.split("-")[0][1:])
                steps.add(step_num)

        # Create mappings
        for idx, step in enumerate(sorted(steps)):
            self.step_to_idx[step] = idx
            self.idx_to_step[idx] = step

        # Load cropped images first (preferred)
        if self.cropped_dir and self.cropped_dir.exists():
            for f in sorted(self.cropped_dir.glob("cropped_s*-*.jpg")):
                name_part = f.stem.replace("cropped_", "")
                step_num = int(name_part.split("-")[0][1:])
                self.images.append(str(f))
                self.labels.append(self.step_to_idx[step_num])

        # If no cropped images, fall back to original images
        if not self.images and self.image_dir:
            for f in sorted(self.image_dir.glob("s*-*.jpg")):
                step_num = int(f.stem.split("-")[0][1:])
                self.images.append(str(f))
                self.labels.append(self.step_to_idx[step_num])

    def __len__(self):
        # Return more samples if augmenting
        return len(self.images) * (self.augment_factor if self.augment else 1)

    def __getitem__(self, idx):
        actual_idx = idx % len(self.images)
        img_path = self.images[actual_idx]
        label = self.labels[actual_idx]

        # Load image
        image = cv2.imread(img_path)

        # If it's already a cropped image, use it directly
        if "cropped_" in img_path:
            cropped = image
        else:
            # Check if we have manual annotation for this image
            img_name = Path(img_path).name
            if img_name in self.annotations:
                cropped, _ = crop_with_annotation(image, self.annotations[img_name])
            else:
                cropped, _ = extract_bounding_box_color(image)

        # Convert BGR to RGB for PIL
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)

        if self.transform:
            pil_image = self.transform(pil_image)

        return pil_image, label

    @property
    def num_classes(self):
        return len(self.step_to_idx)


class LegoStepClassifier(nn.Module):
    """CNN classifier for Lego assembly steps using transfer learning."""

    def __init__(self, num_classes: int):
        super().__init__()

        # Use a pre-trained ResNet18 as the backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze early layers for transfer learning
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class CNNLegoDetector:
    """Lego step verification using custom CNN model."""

    def __init__(self, model_path: str = None, reference_dir: str = None, device: str = None):
        """
        Initialize the detector.

        Args:
            model_path: Path to saved model weights (optional)
            reference_dir: Directory with reference images for training
            device: 'cuda', 'mps', or 'cpu'
        """
        self.device = device or self._get_device()
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.model = None
        self.step_to_idx = {}
        self.idx_to_step = {}
        self.current_step = 1
        self.total_steps = 0

        # Image transforms for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Heavy augmentation for training with small dataset
        # Using CenterCrop after rotation to ensure object stays in frame
        self.train_transform = transforms.Compose([
            # Resize larger to allow for rotation without cutting off
            transforms.Resize((280, 280)),
            # Random rotation - use fill to avoid black corners
            transforms.RandomRotation(
                degrees=25,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=128  # Gray fill for rotated corners
            ),
            # Center crop to get clean 224x224 after rotation
            transforms.CenterCrop(224),
            # Random horizontal flip
            transforms.RandomHorizontalFlip(p=0.5),
            # Random vertical flip (lego can be viewed from different angles)
            transforms.RandomVerticalFlip(p=0.3),
            # Brightness, contrast, saturation, hue adjustments
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.1
            ),
            # Random affine for slight scale/translate variations
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                fill=128
            ),
            # Gaussian blur occasionally
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.2),
            # Convert to tensor
            transforms.ToTensor(),
            # Normalize with ImageNet stats
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            # Random erasing for robustness (simulates occlusion)
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
        ])

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        elif reference_dir:
            self._setup_from_reference()

    def _get_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _setup_from_reference(self):
        """Setup step mappings from reference directory."""
        steps = set()
        for f in self.reference_dir.glob("s*-*.jpg"):
            step_num = int(f.stem.split("-")[0][1:])
            steps.add(step_num)

        for idx, step in enumerate(sorted(steps)):
            self.step_to_idx[step] = idx
            self.idx_to_step[idx] = step

        self.total_steps = len(steps)

    def train(self, epochs: int = 100, batch_size: int = 16, learning_rate: float = 0.0005,
              save_path: str = "lego_model.pth", cropped_dir: str = None):
        """
        Train the model on reference images.

        With only 37 images across 11 classes, we use heavy augmentation
        and a small learning rate with patience-based early stopping.
        """
        # Use cropped_dir if provided, otherwise look for ./cropped
        if cropped_dir is None:
            cropped_dir = self.reference_dir.parent / "cropped" if self.reference_dir else Path("./cropped")

        print(f"Training on device: {self.device}")
        print(f"Looking for cropped images in: {cropped_dir}")

        # Create dataset with heavy augmentation
        dataset = LegoStepDataset(
            image_dir=str(self.reference_dir) if self.reference_dir else None,
            transform=self.train_transform,
            augment=True,
            cropped_dir=str(cropped_dir),
            augment_factor=25  # 25x augmentation for small dataset
        )

        self.step_to_idx = dataset.step_to_idx
        self.idx_to_step = dataset.idx_to_step
        self.total_steps = dataset.num_classes

        print(f"Found {len(dataset.images)} base images across {dataset.num_classes} classes")
        print(f"With augmentation: {len(dataset)} training samples")

        # Data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        # Initialize model
        self.model = LegoStepClassifier(dataset.num_classes).to(self.device)

        # Loss and optimizer with weight decay for regularization
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing helps with small datasets
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Cosine annealing scheduler for smooth learning rate decay
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )

        # Training loop with early stopping
        best_acc = 0
        patience = 20
        patience_counter = 0

        print(f"\nStarting training for {epochs} epochs...")
        print("-" * 60)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            num_batches = len(dataloader)

            for batch_idx, (images, labels) in enumerate(dataloader):
                # Progress bar
                progress = (batch_idx + 1) / num_batches
                bar_len = 30
                filled = int(bar_len * progress)
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f'\rEpoch {epoch+1:3d}/{epochs} [{bar}] {batch_idx+1}/{num_batches}', end='', flush=True)
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            scheduler.step()

            acc = 100. * correct / total
            avg_loss = running_loss / len(dataloader)
            current_lr = optimizer.param_groups[0]['lr']

            # Save best model
            if acc > best_acc:
                best_acc = acc
                self.save_model(save_path)
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress (clear the progress bar line first)
            print(f'\rEpoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:6.2f}% | Best: {best_acc:6.2f}% | LR: {current_lr:.6f}    ')

            # Early stopping
            if patience_counter >= patience and acc > 90:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        print("-" * 50)
        print(f"Training complete! Best accuracy: {best_acc:.2f}%")
        print(f"Model saved to: {save_path}")

        # Reload best model
        self.load_model(save_path)

    def save_model(self, path: str):
        """Save model weights and mappings."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'step_to_idx': self.step_to_idx,
            'idx_to_step': self.idx_to_step,
            'total_steps': self.total_steps
        }, path)

    def load_model(self, path: str):
        """Load model weights and mappings."""
        checkpoint = torch.load(path, map_location=self.device)
        self.step_to_idx = checkpoint['step_to_idx']
        self.idx_to_step = checkpoint['idx_to_step']
        self.total_steps = checkpoint['total_steps']

        self.model = LegoStepClassifier(len(self.step_to_idx)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {path}")

    def predict(self, image: np.ndarray) -> tuple[int, float]:
        """
        Predict the step number for an image.

        Args:
            image: BGR image from OpenCV

        Returns:
            (predicted_step, confidence)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")

        # Crop and preprocess
        cropped, _ = extract_bounding_box_color(image)
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)

        # Transform
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = probabilities.max(1)

        predicted_step = self.idx_to_step[predicted_idx.item()]
        return predicted_step, confidence.item()

    def verify_step(self, captured_image: np.ndarray, target_step: int = None) -> VerificationResult:
        """
        Verify if the captured image matches the target step.

        Args:
            captured_image: BGR image from camera/file
            target_step: Step number to verify against (defaults to current_step)

        Returns:
            VerificationResult with match status
        """
        if target_step is None:
            target_step = self.current_step

        predicted_step, confidence = self.predict(captured_image)
        is_match = predicted_step == target_step

        if is_match:
            if confidence > 0.8:
                explanation = f"High confidence match for step {target_step}"
            elif confidence > 0.5:
                explanation = f"Medium confidence match for step {target_step}"
            else:
                explanation = f"Low confidence match for step {target_step}"
        else:
            explanation = f"Image appears to be step {predicted_step}, not step {target_step}"

        return VerificationResult(
            step=target_step,
            is_match=is_match,
            confidence=confidence,
            predicted_step=predicted_step,
            explanation=explanation
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


class CNNLegoAssemblyApp:
    """Main application for Lego assembly verification using CNN."""

    def __init__(self, model_path: str = None, reference_dir: str = None, camera_index: int = 0):
        self.detector = CNNLegoDetector(model_path=model_path, reference_dir=reference_dir)
        self.camera_index = camera_index
        self.cap = None

    def train_model(self, epochs: int = 50, save_path: str = "lego_model.pth"):
        """Train the model on reference images."""
        self.detector.train(epochs=epochs, save_path=save_path)

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
        """Verify the current step."""
        if frame is None:
            frame = self.capture_frame()
        return self.detector.verify_step(frame)

    def try_advance(self, frame: np.ndarray = None) -> tuple[bool, VerificationResult]:
        """Try to verify and advance to the next step."""
        result = self.verify_current_step(frame)
        if result.is_match and result.confidence > 0.5:
            advanced = self.detector.advance_step()
            return advanced, result
        return False, result

    def run_interactive(self):
        """Run an interactive session with the camera."""
        if self.detector.model is None:
            print("No model loaded. Training first...")
            self.train_model()

        self.start_camera()
        print(f"\nLego Assembly Verification (CNN)")
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

                cv2.imshow("Lego Assembly (CNN)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    print(f"\nVerifying step {self.detector.current_step}...")
                    advanced, result = self.try_advance(frame)

                    print(f"  Predicted: Step {result.predicted_step}")
                    print(f"  Match: {result.is_match}")
                    print(f"  Confidence: {result.confidence:.2%}")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lego Step Verification with Custom CNN")
    parser.add_argument("--reference", "-r", default="./steps_jpg", help="Reference images directory")
    parser.add_argument("--model", "-m", default="lego_model.pth", help="Model file path")
    parser.add_argument("--train", "-t", action="store_true", help="Train the model")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Training epochs")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
    parser.add_argument("--image", "-i", help="Single image to verify")
    parser.add_argument("--step", "-s", type=int, default=1, help="Step to verify against")

    args = parser.parse_args()

    if args.train:
        detector = CNNLegoDetector(reference_dir=args.reference)
        detector.train(epochs=args.epochs, save_path=args.model)
    elif args.image:
        detector = CNNLegoDetector(model_path=args.model, reference_dir=args.reference)
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not read {args.image}")
        else:
            result = detector.verify_step(image, args.step)
            print(f"\nVerification Result:")
            print(f"  Target Step: {result.step}")
            print(f"  Predicted Step: {result.predicted_step}")
            print(f"  Match: {result.is_match}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  {result.explanation}")
    else:
        app = CNNLegoAssemblyApp(model_path=args.model, reference_dir=args.reference, camera_index=args.camera)
        app.run_interactive()
