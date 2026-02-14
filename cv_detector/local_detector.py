"""
Lego Assembly Step Detector — Local CV with Augmented Templates

For each step, we:
  1. Extract the Lego piece crop from the reference image (red-bg isolation)
  2. Generate augmented variants: rotations, brightness shifts, scales
  3. Compute shape descriptors for each variant
  4. Match video frames by finding the closest descriptor to any variant of next_step

State machine: strictly sequential — only ever checks the NEXT expected step.
Advances only after CONFIRM_FRAMES consecutive matches. Never skips steps.
"""

import cv2
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
DESCRIPTOR_SIZE    = 128                          # resize crop before descriptors


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    frame_number: int
    completed_step: Optional[int]   # step just confirmed, or None
    waiting_for_step: int
    confidence: float               # 0–1 (1 = perfect match)
    lego_bbox: Optional[tuple]      # (x, y, w, h) in original frame
    reason: str


@dataclass
class AssemblyState:
    next_step: int = 1
    completed_steps: list = field(default_factory=list)
    match_buffer: list = field(default_factory=list)
    CONFIRM_FRAMES: int = 3
    MATCH_THRESHOLD: float = 0.72


# ---------------------------------------------------------------------------
# Lego isolation (red-background subtraction)
# ---------------------------------------------------------------------------

def _fg_mask(img: np.ndarray) -> np.ndarray:
    """Return binary foreground mask — everything that is NOT the red bg or black border."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red background (hue wraps around 0/180)
    red1 = cv2.inRange(hsv, np.array([0,   80, 60]), np.array([12,  255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 80, 60]), np.array([180, 255, 255]))
    black = cv2.inRange(hsv, np.array([0, 0, 0]),    np.array([180, 255,  60]))
    bg = cv2.bitwise_or(red1, cv2.bitwise_or(red2, black))
    fg = cv2.bitwise_not(bg)
    k = np.ones((7, 7), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    return fg


def extract_lego_crop(img: np.ndarray) -> Optional[np.ndarray]:
    """Return a tight crop of the Lego piece, or None if not found."""
    fg = _fg_mask(img)
    ih, iw = img.shape[:2]
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours
             if ih * iw * 0.001 < cv2.contourArea(c) < ih * iw * 0.40]
    if not valid:
        return None
    rects = [cv2.boundingRect(c) for c in valid]
    x1 = max(0,  min(r[0]        for r in rects) - 20)
    y1 = max(0,  min(r[1]        for r in rects) - 20)
    x2 = min(iw, max(r[0] + r[2] for r in rects) + 20)
    y2 = min(ih, max(r[1] + r[3] for r in rects) + 20)
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def get_lego_bbox(img: np.ndarray) -> Optional[tuple]:
    """Return (x, y, w, h) bounding box in original image coordinates."""
    fg = _fg_mask(img)
    ih, iw = img.shape[:2]
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours
             if ih * iw * 0.001 < cv2.contourArea(c) < ih * iw * 0.40]
    if not valid:
        return None
    rects = [cv2.boundingRect(c) for c in valid]
    x1 = max(0,  min(r[0]        for r in rects) - 20)
    y1 = max(0,  min(r[1]        for r in rects) - 20)
    x2 = min(iw, max(r[0] + r[2] for r in rects) + 20)
    y2 = min(ih, max(r[1] + r[3] for r in rects) + 20)
    return (x1, y1, x2 - x1, y2 - y1)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_crop(crop: np.ndarray) -> list:
    """
    Generate augmented variants of a Lego crop.
    Covers: 5 scales × 7 rotations × 5 brightness = 175 variants per step.
    """
    variants = []
    h, w = crop.shape[:2]

    for scale in AUGMENT_SCALES:
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        scaled = cv2.resize(crop, (sw, sh))
        cx, cy = sw // 2, sh // 2

        for angle in AUGMENT_ROTATIONS:
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
            nw = int(sh * sin_a + sw * cos_a)
            nh = int(sh * cos_a + sw * sin_a)
            M[0, 2] += (nw - sw) / 2
            M[1, 2] += (nh - sh) / 2
            rotated = cv2.warpAffine(scaled, M, (nw, nh),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(120, 75, 75))

            for brightness in AUGMENT_BRIGHTNESS:
                adjusted = np.clip(rotated.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
                variants.append(cv2.resize(adjusted, (DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)))

    return variants


# ---------------------------------------------------------------------------
# Shape descriptor
# ---------------------------------------------------------------------------

def compute_descriptor(crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute a rotation-invariant shape descriptor vector:
      - 7 log-scaled Hu moments
      - aspect ratio, extent, solidity, normalised perimeter, stud count
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    main = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main)
    if area < 100:
        return None

    # Hu moments (log-scaled for numerical stability)
    hu = cv2.HuMoments(cv2.moments(main)).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    # Bounding box shape stats
    _, _, bw, bh = cv2.boundingRect(main)
    aspect    = bw / (bh + 1e-6)
    extent    = area / (bw * bh + 1e-6)
    hull_area = cv2.contourArea(cv2.convexHull(main))
    solidity  = area / (hull_area + 1e-6)
    perim     = cv2.arcLength(main, True)
    norm_perim = perim / (2 * (bw + bh) + 1e-6)

    # Stud count: small filled sub-contours inside the piece
    n_studs = sum(
        1 for c in contours
        if c is not main and 50 < cv2.contourArea(c) < area * 0.05
    )
    stud_feat = min(n_studs / 20.0, 1.0)

    return np.array([*hu_log, aspect, extent, solidity, norm_perim, stud_feat],
                    dtype=np.float32)


def similarity(d1: np.ndarray, d2: np.ndarray) -> float:
    """L2-distance converted to a [0, 1] similarity score."""
    dist = float(np.linalg.norm(d1 - d2))
    return 1.0 / (1.0 + dist * 0.3)


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class LegoStepDetector:

    def __init__(self, reference_dir: str):
        self.reference_dir = Path(reference_dir)
        self.ref_images: dict[int, np.ndarray] = {}
        self.ref_descriptors: dict[int, list] = {}  # step → [descriptor, ...]
        self.state = AssemblyState()
        self._load_references()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_references(self):
        files = sorted(f for f in self.reference_dir.glob("IMG_*.jpg")
                       if not f.name.startswith('_'))
        if not files:
            raise ValueError(f"No reference images in {self.reference_dir}")

        for step, path in enumerate(files, start=1):
            img = cv2.imread(str(path))
            if img is None:
                continue
            self.ref_images[step] = img

            crop = extract_lego_crop(img)
            if crop is None:
                print(f"  Step {step}: WARNING — could not isolate piece from {path.name}")
                self.ref_descriptors[step] = []
                continue

            variants = augment_crop(crop)
            descs = [d for d in (compute_descriptor(v) for v in variants) if d is not None]
            self.ref_descriptors[step] = descs
            print(f"  Step {step}: {path.name}  → {len(descs)} augmented descriptors")

        print(f"\nLoaded {len(self.ref_images)} reference steps\n")

    # ------------------------------------------------------------------
    # Scoring (only ever called for next_step — enforces sequential order)
    # ------------------------------------------------------------------

    def _score_frame(self, frame: np.ndarray, step: int) -> tuple:
        descs = self.ref_descriptors.get(step, [])
        if not descs:
            return 0.0, "No descriptors for this step"

        crop = extract_lego_crop(frame)
        if crop is None:
            return 0.0, "No Lego piece detected (hands covering or out of frame)"

        frame_desc = compute_descriptor(crop)
        if frame_desc is None:
            return 0.0, "Could not compute descriptor"

        best = max(similarity(frame_desc, rd) for rd in descs)
        return float(best), f"similarity={best:.3f}"

    # ------------------------------------------------------------------
    # State machine — STRICTLY sequential, never skips
    # ------------------------------------------------------------------

    def reset(self):
        self.state = AssemblyState()

    def process_frame(self, frame: np.ndarray, frame_number: int = 0) -> DetectionResult:
        state     = self.state
        total     = len(self.ref_images)
        next_step = state.next_step

        # All steps confirmed — nothing left to do
        if next_step > total:
            return DetectionResult(frame_number, None, next_step, 1.0, None,
                                   "Build complete!")

        bbox          = get_lego_bbox(frame)
        score, reason = self._score_frame(frame, next_step)

        passes = score >= state.MATCH_THRESHOLD
        state.match_buffer.append(passes)
        if len(state.match_buffer) > state.CONFIRM_FRAMES:
            state.match_buffer.pop(0)

        # Confirm only after CONFIRM_FRAMES consecutive passes
        confirmed = (len(state.match_buffer) == state.CONFIRM_FRAMES
                     and all(state.match_buffer))

        completed_step = None
        if confirmed:
            completed_step = next_step
            state.completed_steps.append(next_step)
            state.next_step += 1          # advance to next step only now
            state.match_buffer.clear()
            print(f"  [frame {frame_number:>5}] ✓ Step {next_step} CONFIRMED  {reason}")

        return DetectionResult(frame_number, completed_step, state.next_step,
                               score, bbox, reason)

    # ------------------------------------------------------------------
    # Video pipeline
    # ------------------------------------------------------------------

    def analyze_video(self, video_path: str, sample_rate: int = 5,
                      output_path: Optional[str] = None) -> list:
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

        results, last_result = [], None
        frame_number = 0

        print(f"Video: {video_path}  ({total} frames @ {fps:.0f} fps)")
        print(f"Sampling every {sample_rate} frames  (~{total // sample_rate} checks)\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % sample_rate == 0:
                last_result = self.process_frame(frame, frame_number)
                results.append(last_result)
                t = frame_number / fps
                print(f"  t={t:5.1f}s  waiting_for=step{last_result.waiting_for_step}  "
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
    # HUD overlay
    # ------------------------------------------------------------------

    def _draw_overlay(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        h, w  = frame.shape[:2]
        thresh = self.state.MATCH_THRESHOLD

        # Bounding box around detected piece
        if result.lego_bbox:
            x, y, bw, bh = result.lego_bbox
            color = (0, 255, 0) if result.confidence >= thresh else (0, 180, 255)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 3)
            cv2.putText(frame, f"Step {result.waiting_for_step}?",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Top HUD bar
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        done = len(self.state.completed_steps)
        cv2.putText(frame,
                    f"Waiting: step {result.waiting_for_step}/{len(self.ref_images)}  "
                    f"confirmed: {done}",
                    (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        bar_w = int(350 * min(result.confidence, 1.0))
        bar_color = (0, 200, 0) if result.confidence >= thresh else (0, 150, 255)
        cv2.rectangle(frame, (15, 48), (15 + bar_w, 68), bar_color, -1)
        cv2.rectangle(frame, (15, 48), (365, 68), (100, 100, 100), 1)
        cv2.putText(frame, f"{result.confidence:.0%}", (370, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Reference thumbnail (top-right)
        target = result.waiting_for_step
        if target in self.ref_images:
            thumb = cv2.resize(self.ref_images[target], (100, 100))
            tx, ty = w - 115, 85
            frame[ty:ty+100, tx:tx+100] = thumb
            border = (0, 200, 0) if result.confidence >= thresh else (0, 150, 255)
            cv2.rectangle(frame, (tx-2, ty-2), (tx+102, ty+102), border, 2)
            cv2.putText(frame, f"target: step {target}",
                        (tx-5, ty-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self):
        total = len(self.ref_images)
        done  = len(self.state.completed_steps)
        print("\n" + "=" * 55)
        print("  ASSEMBLY DETECTION SUMMARY")
        print("=" * 55)
        print(f"  Steps confirmed : {self.state.completed_steps}")
        print(f"  Progress        : {done} / {total}")
        if done == total:
            print("\n  *** BUILD COMPLETED CORRECTLY! ***")
        else:
            print(f"\n  Stopped at step {done + 1} (expected {total} total).")
        print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python local_detector.py <video> [output.mp4] [sample_rate]")
        sys.exit(1)

    video_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    sample_rate = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    detector = LegoStepDetector("./steps_jpg")
    detector.analyze_video(video_path, sample_rate=sample_rate, output_path=output_path)
