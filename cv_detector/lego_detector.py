"""
Lego Assembly Step Detector — Local CV with Augmented Templates

For each step, we:
  1. Extract the Lego piece crop from the reference image (red-bg isolation)
  2. Generate augmented variants: rotations, brightness shifts, scales
  3. Compute shape descriptors for each variant
  4. Match video frames by finding the closest descriptor to any variant of next_step

State machine: only ever checks the NEXT expected step. Advances after
CONFIRM_FRAMES consecutive matches. Never skips.
"""

import cv2
import json
import os
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AUGMENT_ROTATIONS  = list(range(-30, 31, 10))   # degrees
AUGMENT_BRIGHTNESS = [-40, -20, 0, 20, 40]      # pixel offset
AUGMENT_SCALES     = [0.75, 0.9, 1.0, 1.1, 1.25]
DESCRIPTOR_SIZE    = 128                          # resize crop to this before descriptors


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    frame_number: int
    completed_step: Optional[int]
    waiting_for_step: int
    confidence: float        # 0–1 (1 = perfect match)
    is_error: bool
    error_message: Optional[str]
    lego_bbox: Optional[tuple]
    reason: str


@dataclass
class AssemblyState:
    next_step: int = 1
    completed_steps: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    match_buffer: list = field(default_factory=list)
    CONFIRM_FRAMES: int = 3
    MATCH_THRESHOLD: float = 0.72    # score above this = match


# ---------------------------------------------------------------------------
# Lego isolation
# ---------------------------------------------------------------------------

def extract_lego_crop(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Isolate the Lego piece from the red-fabric background.
    Returns a tight crop of just the piece, or None if not found.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ih, iw = img.shape[:2]

    # Suppress the red background — anything strongly red is background
    # Red wraps around in HSV (hue ~0 and ~180)
    red_lo1 = cv2.inRange(hsv, np.array([0,  80,  60]), np.array([12, 255, 255]))
    red_lo2 = cv2.inRange(hsv, np.array([165, 80, 60]), np.array([180,255, 255]))
    red_mask = cv2.bitwise_or(red_lo1, red_lo2)

    # Black bottom strip (table edge) — also background
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))

    bg_mask = cv2.bitwise_or(red_mask, black_mask)
    fg_mask = cv2.bitwise_not(bg_mask)

    # Clean up
    kernel = np.ones((7, 7), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = ih * iw
    valid = [c for c in contours
             if img_area * 0.001 < cv2.contourArea(c) < img_area * 0.40]
    if not valid:
        return None

    # Union bounding box of all valid contours
    xs  = [cv2.boundingRect(c)[0] for c in valid]
    ys  = [cv2.boundingRect(c)[1] for c in valid]
    x2s = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid]
    y2s = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid]
    x1 = max(0, min(xs)  - 20)
    y1 = max(0, min(ys)  - 20)
    x2 = min(iw, max(x2s) + 20)
    y2 = min(ih, max(y2s) + 20)

    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def get_lego_bbox(img: np.ndarray) -> Optional[tuple]:
    """Return (x, y, w, h) bounding box in original image coords."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ih, iw = img.shape[:2]

    red_lo1 = cv2.inRange(hsv, np.array([0,  80,  60]), np.array([12, 255, 255]))
    red_lo2 = cv2.inRange(hsv, np.array([165, 80, 60]), np.array([180,255, 255]))
    red_mask = cv2.bitwise_or(red_lo1, red_lo2)
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
    bg_mask = cv2.bitwise_or(red_mask, black_mask)
    fg_mask = cv2.bitwise_not(bg_mask)

    kernel = np.ones((7, 7), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = ih * iw
    valid = [c for c in contours
             if img_area * 0.001 < cv2.contourArea(c) < img_area * 0.40]
    if not valid:
        return None

    xs  = [cv2.boundingRect(c)[0] for c in valid]
    ys  = [cv2.boundingRect(c)[1] for c in valid]
    x2s = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid]
    y2s = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid]
    x1 = max(0, min(xs)  - 20)
    y1 = max(0, min(ys)  - 20)
    x2 = min(iw, max(x2s) + 20)
    y2 = min(ih, max(y2s) + 20)
    return (x1, y1, x2 - x1, y2 - y1)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_crop(crop: np.ndarray) -> list[np.ndarray]:
    """
    Generate augmented variants of a Lego crop:
    rotations × brightness shifts × scales.
    Returns list of uint8 BGR images, all resized to DESCRIPTOR_SIZE.
    """
    variants = []
    h, w = crop.shape[:2]

    for scale in AUGMENT_SCALES:
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        scaled = cv2.resize(crop, (sw, sh))
        cx, cy = sw // 2, sh // 2

        for angle in AUGMENT_ROTATIONS:
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            # Compute new bounding box after rotation so nothing is clipped
            cos_a = abs(M[0, 0]); sin_a = abs(M[0, 1])
            nw = int(sh * sin_a + sw * cos_a)
            nh = int(sh * cos_a + sw * sin_a)
            M[0, 2] += (nw - sw) / 2
            M[1, 2] += (nh - sh) / 2
            rotated = cv2.warpAffine(scaled, M, (nw, nh),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(128, 80, 80))  # neutral red-ish bg

            for brightness in AUGMENT_BRIGHTNESS:
                adjusted = np.clip(rotated.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
                resized  = cv2.resize(adjusted, (DESCRIPTOR_SIZE, DESCRIPTOR_SIZE))
                variants.append(resized)

    return variants


# ---------------------------------------------------------------------------
# Shape descriptor
# ---------------------------------------------------------------------------

def compute_descriptor(crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a shape descriptor vector from a Lego crop image.
    Components:
      - 7 Hu moments (log-scaled, rotation-invariant)
      - aspect ratio
      - extent (filled area / bbox area)
      - solidity (filled area / convex hull area)
      - normalized contour perimeter
      - number of significant child contours (approximation of stud count)
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Threshold to get Lego silhouette
    # The piece is lighter than the reddish background fill
    _, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour = piece outline
    main = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main)
    if area < 100:
        return None

    # Hu moments
    moments = cv2.moments(main)
    hu = cv2.HuMoments(moments).flatten()
    # Log-scale for stability
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    # Shape stats
    bx, by, bw, bh = cv2.boundingRect(main)
    aspect = bw / (bh + 1e-6)
    extent = area / (bw * bh + 1e-6)

    hull = cv2.convexHull(main)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    perimeter = cv2.arcLength(main, True)
    norm_perim = perimeter / (2 * (bw + bh) + 1e-6)

    # Count stud-like sub-contours (small circles inside the main contour)
    n_studs = 0
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            if cnt is main:
                continue
            ca = cv2.contourArea(cnt)
            if 50 < ca < area * 0.05:
                n_studs += 1
    stud_feat = min(n_studs / 20.0, 1.0)  # normalize

    desc = np.concatenate([
        hu_log,
        [aspect, extent, solidity, norm_perim, stud_feat]
    ])
    return desc.astype(np.float32)


def descriptor_distance(d1: np.ndarray, d2: np.ndarray) -> float:
    """Normalized L2 distance → similarity score in [0, 1]."""
    diff = d1 - d2
    dist = float(np.sqrt(np.dot(diff, diff)))
    # Convert distance to similarity; tune divisor empirically
    return 1.0 / (1.0 + dist * 0.3)


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class LegoStepDetector:

    def __init__(self, reference_dir: str):
        self.reference_dir = Path(reference_dir)
        self.ref_images: dict[int, np.ndarray] = {}
        # For each step: list of descriptors (one per augmented variant)
        self.ref_descriptors: dict[int, list[np.ndarray]] = {}
        self.state = AssemblyState()
        self._load_references()

    # ------------------------------------------------------------------
    # Loading + augmentation
    # ------------------------------------------------------------------

    def _load_references(self):
        files = sorted(
            [f for f in self.reference_dir.glob("IMG_*.jpg")
             if not f.name.startswith('_')]
        )
        if not files:
            raise ValueError(f"No reference images in {self.reference_dir}")

        for i, path in enumerate(files, start=1):
            img = cv2.imread(str(path))
            if img is None:
                continue
            self.ref_images[i] = img

            crop = extract_lego_crop(img)
            if crop is None:
                print(f"  Step {i}: WARNING — could not isolate Lego from {path.name}")
                self.ref_descriptors[i] = []
                continue

            variants = augment_crop(crop)
            descs = [d for d in (compute_descriptor(v) for v in variants) if d is not None]
            self.ref_descriptors[i] = descs
            print(f"  Step {i}: {path.name}  → {len(descs)} augmented descriptors")

        print(f"\nLoaded {len(self.ref_images)} steps\n")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_step(self, frame: np.ndarray, step: int) -> tuple[float, str]:
        """
        Score how well `frame` matches `step`.
        Returns (best_similarity 0–1, reason_string).
        """
        descs = self.ref_descriptors.get(step, [])
        if not descs:
            return 0.0, "No descriptors for this step"

        # Extract Lego crop from the video frame
        crop = extract_lego_crop(frame)
        if crop is None:
            return 0.0, "No Lego piece detected in frame"

        frame_desc = compute_descriptor(crop)
        if frame_desc is None:
            return 0.0, "Could not compute frame descriptor"

        # Best match across all augmented reference variants
        best = max(descriptor_distance(frame_desc, rd) for rd in descs)
        reason = f"Best similarity to step {step}: {best:.3f}"
        return float(best), reason

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def reset(self):
        self.state = AssemblyState()

    def process_frame(self, frame: np.ndarray, frame_number: int = 0) -> DetectionResult:
        state = self.state
        total = len(self.ref_images)
        next_step = state.next_step

        if next_step > total:
            return DetectionResult(frame_number, None, next_step, 1.0,
                                   False, None, None, "Build complete!")

        bbox = get_lego_bbox(frame)
        score, reason = self._score_step(frame, next_step)

        passes = score >= state.MATCH_THRESHOLD
        state.match_buffer.append(passes)
        if len(state.match_buffer) > state.CONFIRM_FRAMES:
            state.match_buffer.pop(0)

        confirmed = (len(state.match_buffer) == state.CONFIRM_FRAMES
                     and all(state.match_buffer))

        completed_step = None
        is_error, error_message = False, None

        if confirmed:
            completed_step = next_step
            state.completed_steps.append(next_step)
            state.next_step += 1
            state.match_buffer.clear()
            print(f"  [frame {frame_number:>5}] ✓ Step {next_step} confirmed "
                  f"({score:.2f}): {reason}")

        return DetectionResult(frame_number, completed_step, state.next_step,
                               score, is_error, error_message, bbox, reason)

    # ------------------------------------------------------------------
    # Video analysis
    # ------------------------------------------------------------------

    def analyze_video(self, video_path: str, sample_rate: int = 5,
                      output_path: Optional[str] = None) -> list:
        """
        sample_rate=5 → check every 5th frame (~6x/sec at 30fps). No API cost.
        """
        self.reset()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        results = []
        frame_number = 0
        last_result = None

        print(f"Analyzing {video_path}  ({total} frames @ {fps:.0f}fps)")
        print(f"Sampling every {sample_rate} frames\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % sample_rate == 0:
                last_result = self.process_frame(frame, frame_number)
                results.append(last_result)
                t = frame_number / fps
                print(f"  t={t:5.1f}s  step?={last_result.waiting_for_step}  "
                      f"score={last_result.confidence:.3f}  {last_result.reason}")

            if writer and last_result:
                writer.write(self._draw_overlay(frame.copy(), last_result))

            frame_number += 1

        cap.release()
        if writer:
            writer.release()

        self._print_summary()
        return results

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------

    def _draw_overlay(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        h, w = frame.shape[:2]
        thresh = self.state.MATCH_THRESHOLD

        if result.lego_bbox:
            x, y, bw, bh = result.lego_bbox
            color = (0, 255, 0) if result.confidence >= thresh else (0, 180, 255)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 3)
            cv2.putText(frame, f"Step {result.waiting_for_step}?",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        total = len(self.ref_images)
        done  = len(self.state.completed_steps)
        cv2.putText(frame,
                    f"Waiting for step {result.waiting_for_step}/{total}   done: {done}",
                    (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        bar_w = int(350 * result.confidence)
        bar_color = (0, 200, 0) if result.confidence >= thresh else (0, 150, 255)
        cv2.rectangle(frame, (15, 48), (15 + bar_w, 68), bar_color, -1)
        cv2.rectangle(frame, (15, 48), (365, 68), (100, 100, 100), 1)
        cv2.putText(frame, f"{result.confidence:.0%}", (370, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        target = result.waiting_for_step
        if target in self.ref_images:
            thumb = cv2.resize(self.ref_images[target], (100, 100))
            tx, ty = w - 115, 85
            frame[ty:ty+100, tx:tx+100] = thumb
            border = (0, 200, 0) if result.confidence >= thresh else (0, 150, 255)
            cv2.rectangle(frame, (tx-2, ty-2), (tx+102, ty+102), border, 2)
            cv2.putText(frame, f"target: step {target}",
                        (tx-5, ty-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if result.is_error:
            cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 140), -1)
            cv2.putText(frame, f"ERROR: {result.error_message}",
                        (15, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self):
        total  = len(self.ref_images)
        done   = len(self.state.completed_steps)
        errors = self.state.errors

        print("\n" + "=" * 55)
        print("  ASSEMBLY DETECTION SUMMARY")
        print("=" * 55)
        print(f"  Steps completed : {done} / {total}")
        if self.state.completed_steps:
            print(f"  Steps confirmed : {self.state.completed_steps}")
        print(f"  Errors detected : {len(errors)}")
        if errors:
            for fn, msg in errors:
                print(f"    Frame {fn:>6}: {msg}")
        if done == total and not errors:
            print("\n  *** BUILD COMPLETED CORRECTLY! ***")
        elif done < total:
            print(f"\n  Build incomplete — stopped at step {done + 1}.")
        print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python lego_detector.py <video.mp4> [output.mp4] [sample_rate]")
        sys.exit(1)

    video_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    sample_rate = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    detector = LegoStepDetector("./steps_jpg")
    detector.analyze_video(video_path, sample_rate=sample_rate, output_path=output_path)
