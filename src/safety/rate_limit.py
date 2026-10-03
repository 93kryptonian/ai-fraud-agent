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

Known limitation:
State is per-process. Behind multiple Render instances/workers this
limit is effectively multiplied by the instance count. For real
production use, replace with a shared store (Redis) — see
docs/operations.md.
"""

import os
import time
from collections import defaultdict
from typing import Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

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
                logger.warning(f"[rate_limit] Blocked ip={client_ip} count={count}")
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
