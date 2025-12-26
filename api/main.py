# api/main.py
"""
FastAPI application entry point.

Responsibilities:
- Initialize FastAPI app
- Configure global middleware (CORS)
- Register API routers
- Expose basic health and service metadata endpoints

This module intentionally contains no business logic.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import router


def create_app() -> FastAPI:
    """
    Application factory.

    Using a factory pattern makes the app:
    - Easier to test
    - Easier to configure for different environments
    - Cleaner for production deployments (Gunicorn/Uvicorn)
    """
    app = FastAPI(
        title="AI Fraud Agents Public API",
        description="RAG + Analytics AI Engine for Fraud Intelligence",
        version="1.0.0",
    )

    configure_middleware(app)
    register_routes(app)

    return app


def configure_middleware(app: FastAPI) -> None:
    """
    Configure global middleware.

    Note:
    CORS is intentionally permissive because this API
    is exposed as a public portfolio/demo service.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def register_routes(app: FastAPI) -> None:
    """
    Register all API routers and system endpoints.
    """
    app.include_router(router)

    @app.get("/health", tags=["system"])
    def health_check():
        """
        Lightweight health check endpoint.
        Used for uptime monitoring and deployment validation.
        """
        return {"status": "ok"}

    @app.get("/", tags=["system"])
    def root():
        """
        Service metadata endpoint.
        Useful for quick manual verification.
        """
        return {
            "service": "AI Fraud Agents API",
            "status": "running",
            "docs": "/docs",
            "health": "/health",
        }


# ASGI application instance
app = create_app()
