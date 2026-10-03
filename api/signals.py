# api/signals.py
"""
Operator endpoints for live signals (M8.2 / M8.3).

    GET /signals   Authorization: Bearer <SIGNALS_TOKEN>   ->   M7.3 snapshot (JSON)
    GET /metrics   Authorization: Bearer <SIGNALS_TOKEN>   ->   Prometheus text 0.0.4

Both share one gate and one set of access rules. /signals is the analytical
JSON snapshot; /metrics is a separate representation of the same snapshot
(src/observability/exposition.py) — /signals is not a disguised Prometheus
endpoint.

Access semantics (docs/observability-contract.md §14):
- The routes are registered ONLY when the live feed is enabled and a token is
  configured, so a disabled deployment returns a plain 404 and the endpoints
  do not appear in the OpenAPI schema. The handler re-checks both at
  request time (defence in depth).
- Enabled + missing/invalid credentials -> 401, generic body,
  `WWW-Authenticate: Bearer`. There is no redacted public variant.
- Token comparison is constant-time. The token is never logged, echoed in a
  response, or placed in an event.
- These handlers deliberately emit NO request events and never set a
  request_id: the signals endpoints must not observe themselves, or a frequent
  scraper would dominate requests_total and request latency. That is an
  explicit rule, not a side effect.
- The existing rate limiter and CORS middleware apply as for every route.
"""

import hmac
import os
from typing import Optional

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse, Response

from src.observability import live
from src.observability.exposition import CONTENT_TYPE, render_prometheus
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


def _gate(authorization: Optional[str]):
    """
    Shared access gate. Returns (collector, None) when the request may be
    served, or (None, error_response) for 404 (closed) / 401 (bad credentials).
    """
    token = _configured_token()
    collector = live.get()
    if token is None or collector is None:
        return None, JSONResponse(status_code=404, content=_NOT_FOUND)

    presented = _presented_token(authorization)
    if presented is None or not hmac.compare_digest(
        presented.encode("utf-8"), token.encode("utf-8")
    ):
        return None, _unauthorized()
    return collector, None


def _unavailable() -> JSONResponse:
    return JSONResponse(
        status_code=503, content={"error": "Signals unavailable"}, headers=_NO_STORE
    )


def signals_endpoint(authorization: Optional[str] = Header(default=None)):
    collector, error = _gate(authorization)
    if error is not None:
        return error

    try:
        return JSONResponse(content=collector.snapshot(), headers=_NO_STORE)
    except Exception:
        logger.error("[signals] snapshot failed", exc_info=True)
        return _unavailable()


def metrics_endpoint(authorization: Optional[str] = Header(default=None)):
    collector, error = _gate(authorization)
    if error is not None:
        return error

    try:
        # snapshot() takes the aggregator lock and releases it on return;
        # rendering (sorting, cumulative buckets, string building) happens
        # OUTSIDE that critical section.
        snapshot = collector.snapshot()
        body = render_prometheus(snapshot)
        return Response(content=body, media_type=CONTENT_TYPE, headers=_NO_STORE)
    except Exception:
        logger.error("[signals] metrics rendering failed", exc_info=True)
        return _unavailable()


def register_signals_routes(app: FastAPI) -> bool:
    """
    Register GET /signals and GET /metrics if (and only if) they can serve:
    the live feed is enabled and SIGNALS_TOKEN is set. Returns whether they
    were registered.
    """
    if not live.is_enabled():
        return False

    if _configured_token() is None:
        logger.warning(
            "[signals] SIGNALS_ENABLED=true but SIGNALS_TOKEN is not set; "
            "the /signals and /metrics endpoints stay closed"
        )
        return False

    app.add_api_route(
        "/signals",
        signals_endpoint,
        methods=["GET"],
        tags=["system"],
        summary="Live operational signals (operator, bearer token)",
    )
    app.add_api_route(
        "/metrics",
        metrics_endpoint,
        methods=["GET"],
        tags=["system"],
        summary="Prometheus text exposition of live signals (operator, bearer token)",
        response_class=Response,
    )
    return True
