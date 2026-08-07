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
from .config import API_KEY, CHAR_BUDGET, CHOICE_CHAR_BUDGET, LLM_MAX_CONCURRENCY, MODELS
from .llm import LLMError, embed
from .schemas import AddRequest, AddResponse, Memory, SearchRequest, SearchResponse
from .store import Store

# uvicorn configures its own loggers and leaves the root one bare, so without
# this every log.info here goes nowhere — including the line that says which
# retrieval is running. Only log.warning was surviving, via logging.lastResort.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")

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
    # Say which retrieval is actually running. An unset gear disables its stage
    # silently — that is deliberate, but it makes a misconfiguration
    # indistinguishable from a choice, and the difference is seven points.
    if MODELS.embed.base_url:
        log.info("retrieval: hybrid, embedding with %s", MODELS.embed.model)
    else:
        log.warning("retrieval: LEXICAL ONLY — EMBED_API_BASE is unset")
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
    """Liveness, plus which retrieval stages are actually configured.

    Reported because nothing else over HTTP can see it. A service with no
    embeddings answers every request correctly and passes every contract check
    — the only other witness is a count of non-null embeddings in the database,
    which needs a shell on the host. That gap already hid a live deployment
    that was silently lexical-only.
    """
    return {
        "status": "ok",
        "retrieval": "hybrid" if MODELS.embed.base_url else "lexical-only",
        "embed_model": MODELS.embed.model or None,
        "extract_model": MODELS.extract.model or None,
    }


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

    # Facts index the turns; they are never returned in place of them. An
    # extraction failure costs matching quality and nothing else, which is why
    # it is allowed to fail — the platform does not retry a write it accepted.
    async with state["llm_slots"]:
        facts = await extract.extract(state["http"], request, anchor)
    if facts:
        sources = ingest.attribute(facts, rows)
        rows += ingest.fact_rows(request, facts, anchor, offset=len(rows), sources=sources)

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
    # Facts are index entries, not answers: a fact that ranks hands its slot
    # to the turn it came from.
    pool = retrieve.resolve_sources(pool, {record.id: record for record in records})
    limit = min(retrieve.DEFAULT_RETURN, request.top_k)
    # A choice question announces itself by carrying options, and it wants a
    # much tighter context than an open-ended one — see CHOICE_CHAR_BUDGET.
    budget = CHOICE_CHAR_BUDGET if request.options else CHAR_BUDGET
    selected = retrieve.trim(pool, limit=limit, char_budget=budget)

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
