"""
Help Request API Routes
Endpoints for creating Zoom meetings for hackathon help requests
"""

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.models.schemas import ErrorResponse, HelpRequestResponse
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
):
    """
    Create a Zoom meeting for hackathon help request.

    Returns meeting join URL and start URL for the host.
    Meeting is created instantly with settings optimized for quick help sessions.

    Returns:
        HelpRequestResponse: Meeting URLs and ID
    """
    try:
        meeting_data = await zoom_client.create_meeting(topic="Hackathon Help Request")

        logger.info(f"Created meeting: {meeting_data['meeting_id']}")

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
