# api/signals.py
"""
Operator endpoint for live signals (M8.2).

    GET /signals   Authorization: Bearer <SIGNALS_TOKEN>   ->   M7.3 snapshot

Access semantics (docs/observability-contract.md §14):
- The route is registered ONLY when the live feed is enabled and a token is
  configured, so a disabled deployment returns a plain 404 and the endpoint
  does not appear in the OpenAPI schema. The handler re-checks both at
  request time (defence in depth).
- Enabled + missing/invalid credentials -> 401, generic body,
  `WWW-Authenticate: Bearer`. There is no redacted public variant.
- Token comparison is constant-time. The token is never logged, echoed in a
  response, or placed in an event.
- This handler deliberately emits NO request events and never sets a
  request_id: the signals endpoint must not observe itself, or a frequent
  scraper would dominate requests_total and request latency. That is an
  explicit rule, not a side effect.
- The existing rate limiter and CORS middleware apply as for every route.
"""

import hmac
import os
from typing import Optional

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse

from src.observability import live
from src.utils.logger import get_logger

logger = get_logger(__name__)

_NOT_FOUND = {"detail": "Not Found"}
_NO_STORE = {"Cache-Control": "no-store"}


def _configured_token() -> Optional[str]:
    return os.getenv("SIGNALS_TOKEN") or None


def _presented_token(authorization: Optional[str]) -> Optional[str]:
    """The bearer credential, or None for a missing/malformed header."""
    if not authorization:
        return None
    scheme, _, credentials = authorization.partition(" ")
    credentials = credentials.strip()
    if scheme.lower() != "bearer" or not credentials:
        return None
    return credentials


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error": "Unauthorized"},
        headers={"WWW-Authenticate": "Bearer", **_NO_STORE},
    )


def signals_endpoint(authorization: Optional[str] = Header(default=None)):
    token = _configured_token()
    collector = live.get()
    if token is None or collector is None:
        return JSONResponse(status_code=404, content=_NOT_FOUND)

    presented = _presented_token(authorization)
    if presented is None or not hmac.compare_digest(
        presented.encode("utf-8"), token.encode("utf-8")
    ):
        return _unauthorized()

    try:
        return JSONResponse(content=collector.snapshot(), headers=_NO_STORE)
    except Exception:
        logger.error("[signals] snapshot failed", exc_info=True)
        return JSONResponse(
            status_code=503, content={"error": "Signals unavailable"}, headers=_NO_STORE
        )


def register_signals_route(app: FastAPI) -> bool:
    """
    Register GET /signals if (and only if) it can serve: the live feed is
    enabled and SIGNALS_TOKEN is set. Returns whether it was registered.
    """
    if not live.is_enabled():
        return False

    if _configured_token() is None:
        logger.warning(
            "[signals] SIGNALS_ENABLED=true but SIGNALS_TOKEN is not set; "
            "the /signals endpoint stays closed"
        )
        return False

    app.add_api_route(
        "/signals",
        signals_endpoint,
        methods=["GET"],
        tags=["system"],
        summary="Live operational signals (operator, bearer token)",
    )
    return True
