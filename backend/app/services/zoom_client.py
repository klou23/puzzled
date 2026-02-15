"""
Zoom API Client
Handles Server-to-Server OAuth and meeting creation
"""

import base64
from datetime import datetime, timedelta
from typing import Optional

import httpx


class ZoomClient:
    """
    Zoom Server-to-Server OAuth Client

    Handles OAuth token acquisition and meeting creation.
    Tokens are cached in memory with automatic refresh on expiry.
    """

    TOKEN_URL = "https://zoom.us/oauth/token"
    API_BASE_URL = "https://api.zoom.us/v2"

    def __init__(self, account_id: str, client_id: str, client_secret: str):
        """
        Initialize Zoom client with OAuth credentials.

        Args:
            account_id: Zoom Account ID
            client_id: OAuth Client ID
            client_secret: OAuth Client Secret
        """
        self.account_id = account_id
        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    async def _get_access_token(self) -> str:
        """
        Get access token using Server-to-Server OAuth flow.
        Caches token until expiry (with 5-minute safety buffer).

        Returns:
            str: Valid access token

        Raises:
            httpx.HTTPStatusError: If OAuth request fails
        """
        # Return cached token if still valid
        if self._access_token and self._token_expiry:
            if datetime.now() < self._token_expiry:
                return self._access_token

        # Create Basic Auth header
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        params = {
            "grant_type": "account_credentials",
            "account_id": self.account_id,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.TOKEN_URL, headers=headers, params=params, timeout=10.0
            )
            response.raise_for_status()

            data = response.json()
            self._access_token = data["access_token"]
            # Set expiry 5 minutes before actual expiry for safety
            expires_in = data.get("expires_in", 3600)
            self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)

            return self._access_token

    async def create_meeting(
        self, topic: str = "Hackathon Help Request", duration: int = 40
    ) -> dict:
        """
        Create a Zoom meeting and return join/start URLs.

        Args:
            topic: Meeting title
            duration: Meeting duration in minutes

        Returns:
            dict with keys: join_url, start_url, meeting_id, password

        Raises:
            httpx.HTTPStatusError: If API request fails
            httpx.RequestError: If network error occurs
        """
        token = await self._get_access_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Meeting configuration for instant meeting
        payload = {
            "topic": topic,
            "type": 1,  # Instant meeting
            "duration": duration,
            "settings": {
                "host_video": True,
                "participant_video": True,
                "join_before_host": True,
                "waiting_room": False,
                "auto_recording": "none",
            },
        }

        # Create meeting for "me" (the authenticated account)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_BASE_URL}/users/me/meetings",
                headers=headers,
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()

            meeting_data = response.json()

            return {
                "join_url": meeting_data["join_url"],
                "start_url": meeting_data["start_url"],
                "meeting_id": meeting_data["id"],
                "password": meeting_data.get("password", ""),
            }
