"""
Manual Bounding Box Annotator for Lego images.
Draw a persistent, resizable bounding box and save when ready.
"""

import cv2
import json
import numpy as np
from pathlib import Path


class BBoxAnnotator:
    """Interactive bounding box annotation tool with persistent, resizable boxes."""

    def __init__(self, image_dir: str, output_dir: str = None):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir) if output_dir else self.image_dir.parent / "cropped"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.annotations_file = self.output_dir / "annotations.json"
        self.annotations = self._load_annotations()

        self.images = sorted(self.image_dir.glob("s*-*.jpg"))
        self.current_idx = 0

        # Box state (x1, y1, x2, y2) in display coordinates
        self.box = None
        self.scale = 1.0
        self.original_image = None
        self.display_image = None

        # Interaction state
        self.mode = None  # None, 'drawing', 'moving', 'resizing'
        self.active_handle = None  # Which handle is being dragged
        self.drag_start = None  # Starting point of drag
        self.box_at_drag_start = None  # Box state when drag started

        # Handle size for corners and edges
        self.handle_size = 15

    def _load_annotations(self) -> dict:
        """Load existing annotations if available."""
        if self.annotations_file.exists():
            with open(self.annotations_file) as f:
                return json.load(f)
        return {}

    def _save_annotations(self):
        """Save annotations to file."""
        with open(self.annotations_file, "w") as f:
            json.dump(self.annotations, f, indent=2)

    def _get_handle_at(self, x: int, y: int) -> str | None:
        """Check if point is near a handle. Returns handle name or None."""
        if not self.box:
            return None

        x1, y1, x2, y2 = self.box
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2  # Midpoints
        hs = self.handle_size

        # Check corners first (priority)
        handles = {
            'nw': (x1, y1), 'ne': (x2, y1),
            'sw': (x1, y2), 'se': (x2, y2),
            'n': (mx, y1), 's': (mx, y2),
            'w': (x1, my), 'e': (x2, my)
        }

        for handle, (hx, hy) in handles.items():
            if abs(x - hx) < hs and abs(y - hy) < hs:
                return handle
        return None

    def _is_inside_box(self, x: int, y: int) -> bool:
        """Check if point is inside the box (but not on a handle)."""
        if not self.box:
            return False
        x1, y1, x2, y2 = self.box
        margin = self.handle_size
        return (x1 + margin) < x < (x2 - margin) and (y1 + margin) < y < (y2 - margin)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        # Offset for info bar
        y = y - 60
        if y < 0:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)

            if self.box:
                self.box_at_drag_start = self.box

                handle = self._get_handle_at(x, y)
                if handle:
                    self.mode = 'resizing'
                    self.active_handle = handle
                elif self._is_inside_box(x, y):
                    self.mode = 'moving'
                else:
                    # Start drawing new box (replaces old one)
                    self.mode = 'drawing'
                    self.box = None
            else:
                # No box exists, start drawing
                self.mode = 'drawing'

        elif event == cv2.EVENT_MOUSEMOVE and self.mode:
            if self.mode == 'drawing':
                # Preview the box while drawing
                sx, sy = self.drag_start
                self.box = (min(sx, x), min(sy, y), max(sx, x), max(sy, y))

            elif self.mode == 'moving' and self.box_at_drag_start:
                # Move the entire box
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                x1, y1, x2, y2 = self.box_at_drag_start
                self.box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

            elif self.mode == 'resizing' and self.box_at_drag_start:
                # Resize based on which handle is being dragged
                x1, y1, x2, y2 = self.box_at_drag_start
                h = self.active_handle

                if 'n' in h:
                    y1 = y
                if 's' in h:
                    y2 = y
                if 'w' in h:
                    x1 = x
                if 'e' in h:
                    x2 = x

                # Ensure valid box (min size)
                if x2 - x1 > 20 and y2 - y1 > 20:
                    self.box = (x1, y1, x2, y2)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.mode == 'drawing' and self.box:
                # Finalize box - ensure it's normalized
                x1, y1, x2, y2 = self.box
                self.box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

                # Ensure minimum size
                if (self.box[2] - self.box[0]) < 20 or (self.box[3] - self.box[1]) < 20:
                    self.box = self.box_at_drag_start  # Restore previous if too small

            self.mode = None
            self.active_handle = None
            self.drag_start = None
            self.box_at_drag_start = None

    def _get_cursor_type(self, x: int, y: int) -> str:
        """Determine what cursor should be shown (for visual feedback)."""
        if not self.box:
            return 'cross'

        handle = self._get_handle_at(x, y)
        if handle:
            if handle in ['nw', 'se']:
                return 'nwse'
            elif handle in ['ne', 'sw']:
                return 'nesw'
            elif handle in ['n', 's']:
                return 'ns'
            elif handle in ['e', 'w']:
                return 'ew'
        elif self._is_inside_box(x, y):
            return 'move'
        return 'cross'

    def _draw_box(self, image: np.ndarray) -> np.ndarray:
        """Draw the bounding box with handles on the image."""
        display = image.copy()

        if self.box:
            x1, y1, x2, y2 = [int(v) for v in self.box]

            # Clamp to image bounds for drawing
            h, w = display.shape[:2]
            x1_draw, y1_draw = max(0, x1), max(0, y1)
            x2_draw, y2_draw = min(w, x2), min(h, y2)

            # Draw semi-transparent overlay outside the box
            overlay = display.copy()
            mask = np.ones_like(display) * 40
            mask[y1_draw:y2_draw, x1_draw:x2_draw] = 0
            display = cv2.subtract(display, mask.astype(np.uint8))

            # Draw box outline (thick green)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate handle positions
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            handles = [
                (x1, y1), (mx, y1), (x2, y1),  # Top row
                (x1, my), (x2, my),             # Middle row
                (x1, y2), (mx, y2), (x2, y2)   # Bottom row
            ]

            # Draw handles
            hs = 8
            for hx, hy in handles:
                # White filled square with green border
                cv2.rectangle(display,
                              (hx - hs, hy - hs),
                              (hx + hs, hy + hs),
                              (255, 255, 255), -1)
                cv2.rectangle(display,
                              (hx - hs, hy - hs),
                              (hx + hs, hy + hs),
                              (0, 200, 0), 2)

            # Show dimensions
            box_w, box_h = x2 - x1, y2 - y1
            label = f"{int(box_w / self.scale)}x{int(box_h / self.scale)}"
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return display

    def _load_current_image(self):
        """Load and scale the current image."""
        img_path = self.images[self.current_idx]
        self.original_image = cv2.imread(str(img_path))

        if self.original_image is None:
            return False

        # Scale for display
        max_dim = 800
        h, w = self.original_image.shape[:2]
        if max(h, w) > max_dim:
            self.scale = max_dim / max(h, w)
            new_size = (int(w * self.scale), int(h * self.scale))
            self.display_image = cv2.resize(self.original_image, new_size)
        else:
            self.scale = 1.0
            self.display_image = self.original_image.copy()

        # Load existing annotation if available
        img_name = img_path.name
        if img_name in self.annotations:
            # Convert from original to display coordinates
            ox, oy, ow, oh = self.annotations[img_name]
            self.box = (
                int(ox * self.scale),
                int(oy * self.scale),
                int((ox + ow) * self.scale),
                int((oy + oh) * self.scale)
            )
        else:
            self.box = None

        return True

    def _save_current_box(self):
        """Save the current box annotation and cropped image."""
        if not self.box:
            print("No box to save!")
            return False

        img_path = self.images[self.current_idx]
        x1, y1, x2, y2 = self.box

        # Convert from display to original coordinates
        x = int(x1 / self.scale)
        y = int(y1 / self.scale)
        w = int((x2 - x1) / self.scale)
        h = int((y2 - y1) / self.scale)

        # Clamp to image bounds
        img_h, img_w = self.original_image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        # Store annotation
        self.annotations[img_path.name] = [x, y, w, h]
        self._save_annotations()

        # Crop and save
        cropped = self.original_image[y:y+h, x:x+w]
        output_path = self.output_dir / f"cropped_{img_path.name}"
        cv2.imwrite(str(output_path), cropped)

        print(f"Saved: {img_path.name} -> bbox=[{x}, {y}, {w}, {h}]")
        return True

    def run(self):
        """Run the annotation interface."""
        if not self.images:
            print("No images found in", self.image_dir)
            return

        print("\nBounding Box Annotator")
        print("=" * 40)
        print(f"Images: {len(self.images)}")
        print(f"Already annotated: {len(self.annotations)}")
        print()
        print("Controls:")
        print("  Draw box       - Click and drag on empty area")
        print("  Move box       - Drag from inside the box")
        print("  Resize box     - Drag the corner/edge handles")
        print("  S              - Save and go to next")
        print("  N / Right      - Next image (without saving)")
        print("  P / Left       - Previous image")
        print("  C              - Clear box")
        print("  Q              - Quit")
        print()

        cv2.namedWindow("Annotator", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Annotator", self._mouse_callback)

        self._load_current_image()

        while True:
            if self.display_image is None:
                self.current_idx = (self.current_idx + 1) % len(self.images)
                self._load_current_image()
                continue

            # Draw the display
            display = self._draw_box(self.display_image)

            # Create info bar
            img_path = self.images[self.current_idx]
            bar_height = 60
            bar_width = display.shape[1]
            info_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
            info_bar[:] = (40, 40, 40)  # Dark gray background

            # Image info
            info = f"[{self.current_idx + 1}/{len(self.images)}] {img_path.name}"
            cv2.putText(info_bar, info, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Step and status
            step_num = img_path.stem.split("-")[0]
            if img_path.name in self.annotations:
                status = "SAVED"
                status_color = (0, 255, 0)
            elif self.box:
                status = "Box ready - press S to save"
                status_color = (0, 200, 255)
            else:
                status = "Draw a box"
                status_color = (100, 100, 255)

            cv2.putText(info_bar, f"Step: {step_num}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(info_bar, status, (120, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

            # Controls hint
            cv2.putText(info_bar, "[S]ave [N]ext [P]rev [C]lear [Q]uit",
                        (bar_width - 280, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            # Combine info bar and display
            combined = np.vstack([info_bar, display])
            cv2.imshow("Annotator", combined)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('s'):
                if self._save_current_box():
                    self.current_idx = (self.current_idx + 1) % len(self.images)
                    self._load_current_image()

            elif key in [ord('n'), 83, 3]:  # N or Right arrow
                self.current_idx = (self.current_idx + 1) % len(self.images)
                self._load_current_image()

            elif key in [ord('p'), 81, 2]:  # P or Left arrow
                self.current_idx = (self.current_idx - 1) % len(self.images)
                self._load_current_image()

            elif key == ord('c'):
                self.box = None

        cv2.destroyAllWindows()
        print(f"\nDone! Annotations: {self.annotations_file}")
        print(f"Cropped images: {self.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual bounding box annotator")
    parser.add_argument("--input", "-i", default="./steps_jpg", help="Input image directory")
    parser.add_argument("--output", "-o", default="./cropped", help="Output directory for crops")

    args = parser.parse_args()
    annotator = BBoxAnnotator(args.input, args.output)
    annotator.run()
