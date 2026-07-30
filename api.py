"""FastAPI entry point for the deployable newsroom service.

A thin adapter over :func:`factloop.graph.run_newsroom` (the same pipeline the
Streamlit ``app.py`` drives locally), exposing it as a rate-limited, CORS-scoped
HTTP endpoint. All secrets come from the server environment via ``get_settings``;
nothing is accepted from the caller but the keyword itself.
"""

from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from factloop.config import get_settings
from factloop.graph import run_newsroom
from factloop.schemas import GeoArticle
from factloop.state import NewsroomState


class GenerateRequest(BaseModel):
    keyword: str


class Source(BaseModel):
    title: str
    url: str


class GenerateResponse(BaseModel):
    status: Literal["ok", "no_results", "qc_blocked"]
    language: str | None = None
    coverage_days: int | None = None
    article: GeoArticle | None = None
    sources: list[Source] = []
    unverified_claims: list[str] = []


def _client_ip(request: Request) -> str:
    """Real caller IP, trusting ``X-Forwarded-For`` because we sit behind a proxy.

    On Render the socket peer is the platform's proxy, so ``request.client.host``
    would collapse every visitor onto one key and rate-limit them as a single
    client. The leftmost forwarded entry is the original caller.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


_settings = get_settings()
# headers_enabled writes the standard X-RateLimit-* headers and, on a 429, the
# Retry-After header, so clients can see and respect their remaining budget.
limiter = Limiter(key_func=_client_ip, default_limits=[], headers_enabled=True)

app = FastAPI(title="factloop-newsroom")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS is enforced by the browser only; it keeps the endpoint off other sites'
# pages but does nothing against scripted clients. The rate limiter is what
# actually guards the paid pipeline.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def _sources(state: NewsroomState) -> list[Source]:
    return [
        Source(title=item.source_title, url=item.source_url)
        for item in state["fact_file"].items
    ]


@app.post("/generate", response_model=GenerateResponse)
@limiter.limit(_settings.rate_limit)
def generate(request: Request, response: Response, payload: GenerateRequest) -> GenerateResponse:
    # ``response`` is declared so slowapi can attach the X-RateLimit-* headers to
    # it; FastAPI then merges those onto the serialized body.
    #
    # A plain ``def`` on purpose: run_newsroom is synchronous and calls
    # asyncio.run internally, which raises if invoked from a running event loop.
    # FastAPI runs sync handlers in a threadpool, so there is no loop to clash with.
    try:
        state = run_newsroom(payload.keyword)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if state.get("no_results"):
        return GenerateResponse(status="no_results", language=state.get("language"))
    if state.get("qc_blocked"):
        return GenerateResponse(
            status="qc_blocked",
            language=state.get("language"),
            unverified_claims=state["fact_check"].unverified_claims,
        )
    return GenerateResponse(
        status="ok",
        language=state.get("language"),
        coverage_days=state.get("coverage_days"),
        article=state["geo"],
        sources=_sources(state),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
