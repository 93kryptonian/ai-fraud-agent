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

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import router
from api.signals import register_signals_routes
from src.observability.live import configure_from_env as configure_live_signals
from src.safety.rate_limit import RateLimitMiddleware


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
    # Opt-in (SIGNALS_ENABLED); a no-op by default. The /signals endpoint is
    # registered (with /metrics) only when the feed is enabled AND SIGNALS_TOKEN is set.
    if configure_live_signals() is not None:
        register_signals_routes(app)

    return app


def configure_middleware(app: FastAPI) -> None:
    """
    Configure global middleware.

    Note:
    CORS is intentionally permissive (any origin) because this API
    is exposed as a public portfolio/demo service. `allow_credentials`
    is False — no cookies/sessions are used, and the combination of
    allow_origins=["*"] with allow_credentials=True is invalid per the
    CORS spec (browsers reject it) as well as an unnecessary risk.

    A simple per-IP rate limiter guards against runaway LLM cost from
    a single client (see src/safety/rate_limit.py for limitations).

    Middleware order matters: Starlette wraps middleware so the LAST
    one added to the app is the OUTERMOST layer. CORSMiddleware is
    added last so it wraps the rate limiter — this way CORS headers
    are still attached to a 429 response, and the CORS preflight
    (OPTIONS) short-circuits before it can ever be counted/blocked by
    the rate limiter.
    """
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=int(os.getenv("RATE_LIMIT_PER_MINUTE", "20")),
        window_seconds=60,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
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
