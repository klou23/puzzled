"""
Lego Assembly Verification App
Main entry point supporting Claude API, CNN, and similarity-based verification.
"""

import argparse
import cv2
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Lego Assembly Step Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Claude API (webcam):
  python main.py --mode claude --api-key YOUR_KEY

  # Run with Claude API (manual image input):
  python main.py --mode claude --api-key YOUR_KEY --manual

  # Verify a single image with Claude:
  python main.py --mode claude --api-key YOUR_KEY --image test.jpg --step 3

  # Run similarity-based (no training needed):
  python main.py --mode similarity

  # Run with CNN model:
  python main.py --mode cnn --train
        """
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["claude", "cnn", "similarity"],
        default="claude",
        help="Verification mode: 'claude' (API), 'cnn' (trained model), or 'similarity' (no training)"
    )
    parser.add_argument(
        "--api-key", "-k",
        help="Anthropic API key (required for claude mode)"
    )
    parser.add_argument(
        "--reference", "-r",
        default="./steps_jpg",
        help="Directory containing reference step images"
    )
    parser.add_argument(
        "--model",
        default="lego_model.pth",
        help="Path to CNN model file"
    )
    parser.add_argument(
        "--train", "-t",
        action="store_true",
        help="Train the CNN model before running"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera index for live capture"
    )
    parser.add_argument(
        "--image", "-i",
        help="Single image file to verify (instead of camera)"
    )
    parser.add_argument(
        "--step", "-s",
        type=int,
        default=1,
        help="Step number to verify against (for single image mode)"
    )
    parser.add_argument(
        "--cropped", "-x",
        default="./cropped",
        help="Directory containing manually cropped reference images"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for similarity mode (0-1)"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Manual image input mode (type image paths instead of webcam)"
    )

    args = parser.parse_args()

    # Validate reference directory
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"Error: Reference directory not found: {args.reference}")
        print("Run convert_heic.py first to create JPG images.")
        return 1

    # Count available steps
    steps = set()
    for f in ref_path.glob("s*-*.jpg"):
        step_num = int(f.stem.split("-")[0][1:])
        steps.add(step_num)

    if not steps:
        print(f"Error: No step images found in {args.reference}")
        print("Expected format: s1-0.jpg, s1-1.jpg, s2-0.jpg, etc.")
        return 1

    print(f"\nLego Assembly Verification")
    print(f"=" * 40)
    print(f"Mode: {args.mode.upper()}")
    print(f"Reference directory: {args.reference}")
    print(f"Steps available: {len(steps)} (s1 to s{max(steps)})")
    print(f"Total images: {len(list(ref_path.glob('s*-*.jpg')))}")

    # Check for cropped images
    cropped_path = Path(args.cropped)
    if cropped_path.exists():
        cropped_count = len(list(cropped_path.glob("cropped_*.jpg")))
        print(f"Cropped images: {cropped_count}")
    else:
        cropped_path = None
    print()

    if args.mode == "claude":
        if not args.api_key:
            print("Error: --api-key required for Claude mode")
            return 1

        from claude_detector import ClaudeLegoDetector, LegoAssemblyApp, verify_from_file

        if args.image:
            # Single image verification
            verify_from_file(args.api_key, args.reference, args.image, args.step)
        elif args.manual:
            # Manual image input mode
            run_manual_claude(args.api_key, args.reference, str(cropped_path) if cropped_path else None)
        else:
            # Webcam mode
            app = LegoAssemblyApp(
                args.api_key, args.reference, args.camera,
                str(cropped_path) if cropped_path else None
            )
            app.run_interactive()

    elif args.mode == "similarity":
        from similarity_detector import SimilarityDetector, SimilarityAssemblyApp

        if args.image:
            detector = SimilarityDetector(
                reference_dir=args.reference,
                cropped_dir=str(cropped_path) if cropped_path else None,
                similarity_threshold=args.threshold
            )
            image = cv2.imread(args.image)
            if image is None:
                print(f"Error: Could not read {args.image}")
                return 1

            result = detector.verify_step(image, args.step)
            print(f"\nVerification Result:")
            print(f"  Step: {result.step}")
            print(f"  Similarity: {result.similarity:.2%}")
            print(f"  Match: {result.is_match}")
            print(f"  {result.explanation}")
        else:
            app = SimilarityAssemblyApp(
                reference_dir=args.reference,
                cropped_dir=str(cropped_path) if cropped_path else None,
                camera_index=args.camera,
                threshold=args.threshold
            )
            app.run_interactive()

    else:  # CNN mode
        from cnn_detector import CNNLegoDetector, CNNLegoAssemblyApp

        if args.train:
            print("Training CNN model...")
            detector = CNNLegoDetector(reference_dir=args.reference)
            detector.train(epochs=args.epochs, save_path=args.model)
            print()

        if args.image:
            model_path = args.model if Path(args.model).exists() else None
            if not model_path and not args.train:
                print(f"Error: Model file not found: {args.model}")
                print("Run with --train first to create a model.")
                return 1

            detector = CNNLegoDetector(model_path=model_path, reference_dir=args.reference)
            image = cv2.imread(args.image)
            if image is None:
                print(f"Error: Could not read {args.image}")
                return 1

            result = detector.verify_step(image, args.step)
            print(f"\nVerification Result:")
            print(f"  Target Step: {result.step}")
            print(f"  Predicted Step: {result.predicted_step}")
            print(f"  Match: {result.is_match}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  {result.explanation}")
        else:
            model_path = args.model if Path(args.model).exists() else None
            app = CNNLegoAssemblyApp(
                model_path=model_path,
                reference_dir=args.reference,
                camera_index=args.camera
            )
            app.run_interactive()

    return 0


def run_manual_claude(api_key: str, reference_dir: str, cropped_dir: str = None):
    """Run Claude verification with manual image path input."""
    from claude_detector import ClaudeLegoDetector

    detector = ClaudeLegoDetector(api_key, reference_dir, cropped_dir)

    print(f"\nManual Image Input Mode")
    print(f"Total steps: {detector.total_steps}")
    print(f"Current step: {detector.current_step}")
    print()
    print("Commands:")
    print("  <image_path>  - Verify image against current step")
    print("  step <n>      - Jump to step n")
    print("  status        - Show current status")
    print("  reset         - Reset to step 1")
    print("  quit          - Exit")
    print()

    while True:
        try:
            user_input = input(f"[Step {detector.current_step}/{detector.total_steps}] Enter image path: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            detector.reset()
            print(f"Reset to step 1")
            continue
        elif user_input.lower() == 'status':
            status = detector.get_status()
            print(f"Current step: {status['current_step']}/{status['total_steps']}")
            continue
        elif user_input.lower().startswith('step '):
            try:
                step = int(user_input.split()[1])
                if 1 <= step <= detector.total_steps:
                    detector.current_step = step
                    print(f"Jumped to step {step}")
                else:
                    print(f"Invalid step. Must be 1-{detector.total_steps}")
            except (ValueError, IndexError):
                print("Usage: step <number>")
            continue

        # Treat input as image path
        image_path = Path(user_input)
        if not image_path.exists():
            print(f"File not found: {user_input}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {user_input}")
            continue

        print(f"Verifying against step {detector.current_step}...")
        result = detector.verify_step(image)

        print(f"  Match: {result.is_match}")
        print(f"  Confidence: {result.confidence}")
        print(f"  {result.explanation}")

        if result.is_match and result.confidence in ["high", "medium"]:
            if detector.advance_step():
                print(f"  -> Advanced to step {detector.current_step}")
            else:
                print("  *** ASSEMBLY COMPLETE! ***")
                break


if __name__ == "__main__":
    exit(main())
