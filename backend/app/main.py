"""
FastAPI Application Entry Point
Main application with CORS configuration and route registration
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import help

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

settings = get_settings()

app = FastAPI(
    title="TreeHacks26 API",
    description="Backend API for Hackathon Help System with Zoom Integration",
    version="1.0.0",
)

# CORS Configuration
origins = settings.cors_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(help.router)


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and deployment verification.

    Returns:
        dict: Health status and environment information
    """
    return {"status": "healthy", "environment": settings.environment}


@app.get("/")
async def root():
    """
    Root endpoint with API information.

    Returns:
        dict: Welcome message and available endpoints
    """
    return {
        "message": "TreeHacks26 API - Hackathon Help System",
        "docs": "/docs",
        "health": "/health",
    }
