"""
Pydantic Models
Request and response schemas for API endpoints
"""

from pydantic import BaseModel, HttpUrl


class HelpRequestResponse(BaseModel):
    """Response schema for /api/help endpoint"""

    joinUrl: HttpUrl
    startUrl: HttpUrl | None = None  # Optional, for hosts
    meetingId: str


class ErrorResponse(BaseModel):
    """Standard error response schema"""

    error: str
    detail: str | None = None
