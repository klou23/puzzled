"""
Step Verification Routes
Handles image upload and verification against reference steps.
"""

import logging
from datetime import datetime
from typing import Optional
from collections import deque

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

from app.services.detector import get_detector, VerificationResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/verify", tags=["verification"])

# Store recent verifications for dashboard (last 20)
recent_verifications: deque = deque(maxlen=20)


class VerifyResponse(BaseModel):
    match: bool
    confidence: str
    feedback: str
    step: int
    timestamp: str


class VerificationLog(BaseModel):
    step: int
    match: bool
    confidence: str
    feedback: str
    timestamp: str


@router.post("/verify-step", response_model=VerifyResponse)
async def verify_step(
    image: UploadFile = File(...),
    stepId: int = Form(...)
):
    """
    Verify a captured image against a reference step.

    - **image**: JPEG image file from camera
    - **stepId**: The step number to verify against
    """
    logger.info(f"Received verification request for step {stepId}")

    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image bytes
        image_bytes = await image.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")

        # Get detector and verify
        detector = get_detector()
        result: VerificationResult = detector.verify_step(image_bytes, stepId)

        timestamp = datetime.now().isoformat()

        # Log for dashboard
        log_entry = VerificationLog(
            step=result.step,
            match=result.is_match,
            confidence=result.confidence,
            feedback=result.explanation,
            timestamp=timestamp
        )
        recent_verifications.append(log_entry.model_dump())

        logger.info(f"Verification result: match={result.is_match}, confidence={result.confidence}")

        return VerifyResponse(
            match=result.is_match,
            confidence=result.confidence,
            feedback=result.explanation,
            step=result.step,
            timestamp=timestamp
        )

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.get("/status")
async def get_status():
    """Get detector status and info."""
    try:
        detector = get_detector()
        return {
            "status": "ready",
            "total_steps": detector.get_total_steps(),
            "reference_dir": str(detector.reference_dir)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/recent", response_model=list[VerificationLog])
async def get_recent_verifications():
    """Get recent verification attempts for dashboard."""
    return list(recent_verifications)


@router.get("/dashboard")
async def dashboard():
    """Simple HTML dashboard showing recent verifications."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Verification Dashboard</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
            h1 { color: #333; }
            .log { padding: 12px; margin: 8px 0; border-radius: 8px; }
            .match { background: #d4edda; border-left: 4px solid #28a745; }
            .no-match { background: #f8d7da; border-left: 4px solid #dc3545; }
            .step { font-weight: bold; }
            .confidence { color: #666; font-size: 0.9em; }
            .time { color: #999; font-size: 0.8em; }
            .empty { color: #999; text-align: center; padding: 40px; }
        </style>
    </head>
    <body>
        <h1>Verification Dashboard</h1>
        <p>Auto-refreshes every 5 seconds</p>
        <div id="logs">
    """

    if not recent_verifications:
        html += '<div class="empty">No verifications yet. Waiting for requests...</div>'
    else:
        for log in reversed(recent_verifications):
            css_class = "match" if log["match"] else "no-match"
            status = "✓ MATCH" if log["match"] else "✗ NO MATCH"
            html += f"""
            <div class="log {css_class}">
                <span class="step">Step {log["step"]}</span> - {status}
                <span class="confidence">({log["confidence"]} confidence)</span>
                <br><span>{log["feedback"]}</span>
                <br><span class="time">{log["timestamp"]}</span>
            </div>
            """

    html += """
        </div>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)
