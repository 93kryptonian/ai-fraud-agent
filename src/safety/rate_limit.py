# src/safety/rate_limit.py
"""
Lightweight per-client rate limiting.

Design principles:
- No external dependencies (stdlib only) — safe to add without touching
  requirements.txt
- In-memory fixed-window counter — good enough to blunt casual cost-abuse
  on a single-process demo deployment
- Fails open on internal errors (never blocks legitimate traffic due to
  a bug in the limiter itself)
- Observable (M8.4): a blocked request emits one bounded
  `rate_limit.blocked` event labelled by route only. Telemetry never changes
  the limiter's decision: a failure while recording a block still returns 429.

Known limitation:
State is per-process. Behind multiple Render instances/workers this
limit is effectively multiplied by the instance count. For real
production use, replace with a shared store (Redis) — see
docs/operations.md.
"""

import os
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.observability.events import emit_event
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Behind a reverse proxy (Render) request.client.host is the proxy, so every
# user would share one bucket. When TRUST_FORWARDED_FOR=true we read the
# client from X-Forwarded-For, counting from the RIGHT: each trusted proxy
# appends the address it saw, so the entry TRUSTED_PROXY_HOPS from the end is
# the real client. The leftmost entry is client-controlled and spoofable.
TRUST_FORWARDED_FOR = os.getenv("TRUST_FORWARDED_FOR", "false").lower() == "true"
TRUSTED_PROXY_HOPS = max(1, int(os.getenv("TRUSTED_PROXY_HOPS", "1")))


def get_client_ip(request: Request) -> str:
    if TRUST_FORWARDED_FOR:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            parts = [p.strip() for p in xff.split(",") if p.strip()]
            if parts:
                return parts[-min(TRUSTED_PROXY_HOPS, len(parts))]
    return request.client.host if request.client else "unknown"


# Bounded route domain for rate_limit.blocked events (same values as the
# request events' `route`, plus "other"). The raw path is mapped HERE, before
# any event is built, so no event ever carries a raw path.
_APPLICATION_ROUTES = frozenset({"/query", "/rag", "/analytics"})
# Telemetry/control-plane endpoints are not application traffic (M8.2/M8.3
# self-exclusion): a throttled scraper must not inflate the "other" bucket.
_CONTROL_PLANE_ROUTES = frozenset({"/signals", "/metrics"})
OTHER_ROUTE = "other"


def map_route(path: str) -> Optional[str]:
    """
    Raw request path -> bounded route for rate_limit.blocked, or None when no
    event should be emitted (control-plane paths). Exact match with a
    trailing slash tolerated; everything else is "other".
    """
    normalized = path.rstrip("/") or "/"
    if normalized in _CONTROL_PLANE_ROUTES:
        return None
    if normalized in _APPLICATION_ROUTES:
        return normalized
    return OTHER_ROUTE


def _record_block(path: str, count: int) -> None:
    """
    Log and emit for one blocked request. Never raises: a telemetry failure
    must not change the limiter's decision (the caller still returns 429).
    Deliberately records no client identifier (no IP, hash or key): the raw
    client address is held in memory only (contract §6).
    """
    try:
        route = map_route(path)
        # Control-plane paths still get a log line (operators want to know a
        # scraper is throttled) but no event: they are not application traffic.
        logger.warning(f"[rate_limit] Blocked route={route or 'control_plane'} count={count}")
        if route is None:
            return
        emit_event(
            "rate_limit.blocked", step="rate_limit", status="blocked",
            metadata={"route": route},
        )
    except Exception:
        try:
            logger.exception("[rate_limit] Failed to record blocked request")
        except Exception:
            pass


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Fixed-window rate limiter keyed by client IP.
    """

    def __init__(self, app, requests_per_window: int = 20, window_seconds: int = 60):
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        # ip -> (window_start_epoch, count)
        self._buckets: Dict[str, Tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    async def dispatch(self, request: Request, call_next):
        try:
            client_ip = get_client_ip(request)
            now = time.time()
            window_start, count = self._buckets[client_ip]

            if now - window_start >= self.window_seconds:
                window_start, count = now, 0

            count += 1
            self._buckets[client_ip] = (window_start, count)

            if count > self.requests_per_window:
                retry_after = max(0, int(self.window_seconds - (now - window_start)))
                _record_block(request.url.path, count)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded. Please slow down.",
                        "retry_after_seconds": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )
        except Exception:
            # Never let the limiter itself take down the API
            logger.exception("[rate_limit] Limiter failed — allowing request through")

        return await call_next(request)
