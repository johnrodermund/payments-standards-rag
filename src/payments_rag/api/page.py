"""The one web page (question box, cited answer, escalation, flows list) and its script and stylesheet.

No inline script or style, so the Content-Security-Policy can forbid both: model output is inserted with
textContent only (see static/app.js), and the CSP is the second line of defence if that ever slips.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "public, max-age=300",
}


@lru_cache(maxsize=8)
def _asset(name: str) -> bytes:
    return resources.files("payments_rag.api").joinpath("static", name).read_bytes()


@router.get("/", include_in_schema=False)
def page() -> Response:
    return Response(_asset("index.html"), media_type="text/html; charset=utf-8", headers=SECURITY_HEADERS)


@router.get("/app.js", include_in_schema=False)
def script() -> Response:
    return Response(_asset("app.js"), media_type="text/javascript; charset=utf-8", headers=SECURITY_HEADERS)


@router.get("/app.css", include_in_schema=False)
def stylesheet() -> Response:
    return Response(_asset("app.css"), media_type="text/css; charset=utf-8", headers=SECURITY_HEADERS)
