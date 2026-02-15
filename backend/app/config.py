"""
Application Configuration
Manages environment variables using Pydantic Settings
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Zoom OAuth Credentials
    zoom_account_id: str
    zoom_client_id: str
    zoom_client_secret: str

    # CORS Configuration
    cors_origins: str = "http://localhost:5173"

    # Application Settings
    environment: str = "development"
    log_level: str = "INFO"

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
