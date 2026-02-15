"""
Help Request API Routes
Endpoints for creating Zoom meetings for hackathon help requests
"""

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.models.schemas import ErrorResponse, HelpRequestResponse
from app.services.email_service import EmailService
from app.services.zoom_client import ZoomClient

router = APIRouter()
logger = logging.getLogger(__name__)


def get_zoom_client(settings: Settings = Depends(get_settings)) -> ZoomClient:
    """
    Dependency injection for ZoomClient.

    Args:
        settings: Application settings

    Returns:
        ZoomClient: Configured Zoom API client
    """
    return ZoomClient(
        account_id=settings.zoom_account_id,
        client_id=settings.zoom_client_id,
        client_secret=settings.zoom_client_secret,
    )


def get_email_service(settings: Settings = Depends(get_settings)) -> Optional[EmailService]:
    """
    Dependency injection for EmailService.
    Returns None if email is not configured.

    Args:
        settings: Application settings

    Returns:
        EmailService or None: Configured email service or None
    """
    if not settings.email_configured:
        logger.debug("Email not configured, skipping email service initialization")
        return None

    return EmailService(
        smtp_host=settings.smtp_host,
        smtp_port=settings.smtp_port,
        smtp_user=settings.smtp_user,
        smtp_password=settings.smtp_password,
        from_email=settings.smtp_from_email,
        from_name=settings.smtp_from_name,
        use_tls=settings.smtp_use_tls,
    )


@router.post(
    "/api/help",
    response_model=HelpRequestResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Zoom API error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    tags=["help"],
)
async def create_help_meeting(
    zoom_client: ZoomClient = Depends(get_zoom_client),
    email_service: Optional[EmailService] = Depends(get_email_service),
    settings: Settings = Depends(get_settings),
):
    """
    Create a Zoom meeting for hackathon help request.
    Sends email notification to on-call engineer.

    Returns meeting join URL and start URL for the host.
    Meeting is created instantly with settings optimized for quick help sessions.

    Returns:
        HelpRequestResponse: Meeting URLs and ID
    """
    try:
        # Create Zoom meeting
        meeting_data = await zoom_client.create_meeting(topic="Hackathon Help Request")
        logger.info(f"Created meeting: {meeting_data['meeting_id']}")

        # Send email notification (graceful failure)
        if email_service:
            try:
                await email_service.send_meeting_invitation(
                    recipient=settings.help_request_recipient,
                    meeting_data=meeting_data,
                )
                logger.info(f"Email sent to {settings.help_request_recipient}")
            except Exception as email_error:
                logger.error(f"Email notification failed: {email_error}")
                # Continue anyway - email failure shouldn't block help request

        return HelpRequestResponse(
            joinUrl=meeting_data["join_url"],
            startUrl=meeting_data["start_url"],
            meetingId=str(meeting_data["meeting_id"]),
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"Zoom API error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create Zoom meeting: {str(e)}"
        )
    except httpx.RequestError as e:
        logger.error(f"Network error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Unable to reach Zoom API. Please try again later.",
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
