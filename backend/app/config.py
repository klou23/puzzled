"""
Application Configuration
Manages environment variables using Pydantic Settings
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Zoom OAuth Credentials (optional for verify-only mode)
    zoom_account_id: str = ""
    zoom_client_id: str = ""
    zoom_client_secret: str = ""

    # Anthropic API Key
    anthropic_api_key: str = ""

    # Reference images directory
    reference_dir: str = "./reference_images"

    # CORS Configuration
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # Application Settings
    environment: str = "development"
    log_level: str = "INFO"

    # Email Configuration (Optional)
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from_email: Optional[str] = None
    smtp_from_name: Optional[str] = "TreeHacks26 Help System"
    smtp_use_tls: bool = True

    # Notification Recipients
    help_request_recipient: str = "jasonyi@unc.edu"

    @property
    def email_configured(self) -> bool:
        """Check if email is properly configured"""
        return all([
            self.smtp_host,
            self.smtp_user,
            self.smtp_password,
            self.smtp_from_email
        ])

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance (singleton pattern).

    Returns:
        Settings: Application configuration
    """
    return Settings()
