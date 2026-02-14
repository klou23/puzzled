"""
Real-time Lego Assembly Detector (webcam mode)

Wraps LegoStepDetector for live webcam use.
"""

import cv2
from lego_detector import LegoStepDetector
import argparse


def run_webcam(reference_dir: str, camera_index: int = 0):
    detector = LegoStepDetector(reference_dir)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: could not open camera {camera_index}")
        return

    print("Real-time Lego assembly detector running.")
    print("Press 'q' to quit, 'r' to reset.\n")

    frame_number = 0
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every 10 frames to keep UI responsive
        if frame_number % 10 == 0:
            last_result = detector.process_frame(frame, frame_number)

        if last_result is not None:
            frame = detector._draw_overlay(frame, last_result)

        cv2.imshow("Lego Assembly Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            last_result = None
            print("State reset.")

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    if last_result:
        detector._print_summary([])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", "-r", default="./steps_jpg")
    parser.add_argument("--camera", "-c", type=int, default=0)
    args = parser.parse_args()
    run_webcam(args.reference, args.camera)
