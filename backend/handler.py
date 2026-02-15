"""
RunPod Serverless Handler
Handles verification requests in RunPod serverless environment.
"""

import base64
import os
import runpod
from app.services.detector import StepDetector

# Initialize detector once (reused across requests)
detector = None


def get_detector_instance():
    global detector
    if detector is None:
        reference_dir = os.getenv("REFERENCE_DIR", "./reference_images")
        detector = StepDetector(reference_dir)
    return detector


def handler(event):
    """
    RunPod handler function.

    Expected input:
    {
        "input": {
            "image_base64": "...",  # Base64 encoded JPEG image
            "step_id": 1            # Step number to verify
        }
    }

    Returns:
    {
        "match": true/false,
        "confidence": "high/medium/low",
        "feedback": "explanation...",
        "step": 1
    }
    """
    try:
        input_data = event.get("input", {})

        # Validate input
        image_b64 = input_data.get("image_base64")
        step_id = input_data.get("step_id")

        if not image_b64:
            return {"error": "Missing image_base64"}
        if step_id is None:
            return {"error": "Missing step_id"}

        # Decode image
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            return {"error": f"Invalid base64 image: {str(e)}"}

        # Get detector and verify
        det = get_detector_instance()
        result = det.verify_step(image_bytes, int(step_id))

        return {
            "match": result.is_match,
            "confidence": result.confidence,
            "feedback": result.explanation,
            "step": result.step
        }

    except Exception as e:
        return {"error": str(e)}


# For local testing
if __name__ == "__main__":
    # Test with a sample event
    test_event = {
        "input": {
            "image_base64": "",  # Would be actual base64 image
            "step_id": 1
        }
    }
    print("RunPod handler ready")
    print("To test locally, run: python -c \"import handler; print(handler.handler({'input': {...}}))\"")


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
