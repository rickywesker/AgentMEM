"""HTTP surface: the Add/Search contract the evaluation platform drives.

Contract rules that shape this file (API guide, "Error Handling"):
  - Add returns 200 only after the write is durable *and* searchable.
  - success must be boolean true; request_id/user_id/session_id must echo.
  - Search must always return a ``data`` array, even when empty.
  - 400/422 are not retried by the platform, so a malformed request must fail
    loudly rather than be guessed at — but an *internal* fault should surface
    as 5xx, which the platform does retry.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from . import extract, ingest, retrieve
from .config import API_KEY, LLM_MAX_CONCURRENCY, MODELS
from .llm import LLMError, embed
from .schemas import AddRequest, AddResponse, Memory, SearchRequest, SearchResponse
from .store import Store

log = logging.getLogger("agentmem")

store = Store()
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await store.connect()
    state["http"] = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=10.0),
        # Must exceed LLM_MAX_CONCURRENCY or the pool, not the semaphore,
        # becomes the real limit — and pool waits are invisible in logs.
        limits=httpx.Limits(
            max_connections=LLM_MAX_CONCURRENCY * 2,
            max_keepalive_connections=LLM_MAX_CONCURRENCY,
        ),
    )
    state["llm_slots"] = asyncio.Semaphore(LLM_MAX_CONCURRENCY)
    log.info("agentmem ready (llm concurrency %d)", LLM_MAX_CONCURRENCY)
    try:
        yield
    finally:
        await state["http"].aclose()
        await store.close()


app = FastAPI(title="AgentMEM", version="0.1.0", lifespan=lifespan)


async def authorize(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    """Accepts the platform's Bearer/Token header or X-Api-Key.

    Open when no key is configured, which is what the public smoke mode with
    auth ``none`` needs.
    """
    if not API_KEY:
        return
    presented = x_api_key or ""
    if authorization:
        parts = authorization.split(None, 1)
        presented = parts[1].strip() if len(parts) == 2 else authorization.strip()
    if presented != API_KEY:
        raise HTTPException(status_code=401, detail="invalid credentials")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


async def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Embed, or give up quietly.

    Embeddings are an enhancement over the lexical index, not a precondition
    for it. A provider outage must degrade recall, never fail an Add — the
    platform does not retry a write it already accepted.
    """
    if not MODELS.embed.base_url or not texts:
        return None
    try:
        async with state["llm_slots"]:
            return await embed(state["http"], MODELS.embed, texts)
    except LLMError as error:
        log.warning("embedding unavailable, indexing lexically only: %s", error)
        return None


@app.post("/add", response_model=AddResponse, dependencies=[Depends(authorize)])
async def add(request: AddRequest) -> AddResponse:
    anchor = ingest.chunk_anchor(request)
    rows = ingest.message_rows(request)

    # Facts are the precision layer; raw turns underneath are the recall floor.
    # Extraction failing degrades the former and leaves the latter intact.
    async with state["llm_slots"]:
        facts = await extract.extract(state["http"], request, anchor)
    if facts:
        rows += ingest.fact_rows(request, facts, anchor, offset=len(rows))

    if rows:
        vectors = await embed_texts([row[5] for row in rows])
        if vectors is not None:
            rows = ingest.with_embeddings(rows, vectors)
        # Awaited, not scheduled: the contract requires the data to be
        # searchable before this handler returns.
        await store.write(rows)
    return AddResponse(
        success=True,
        request_id=request.request_id,
        user_id=request.user_id,
        session_id=request.session_id,
    )


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(authorize)])
async def search(request: SearchRequest) -> SearchResponse:
    records = await store.load_user(request.user_id)
    if not records:
        return SearchResponse(data=[])

    query_vector = None
    vectors = await embed_texts([request.query])
    if vectors:
        query_vector = np.asarray(vectors[0], dtype=np.float32)

    pool = retrieve.candidates(
        records, request.query, query_vector, options=request.options
    )
    limit = min(retrieve.DEFAULT_RETURN, request.top_k)
    selected = retrieve.trim(pool, limit=limit)

    return SearchResponse(
        data=[
            Memory(
                id=item.record.id,
                content=item.record.content,
                score=item.score,
                created_at=item.record.created_at.isoformat() if item.record.created_at else None,
            )
            for item in selected
        ]
    )


@app.exception_handler(Exception)
async def unhandled(request, exc: Exception) -> JSONResponse:
    """Surface faults as 5xx so the platform's retry path can absorb them."""
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=503, content={"error": "internal error"})
