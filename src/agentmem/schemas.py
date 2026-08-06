"""Add/Search wire contract.

Field names and shapes are fixed by the Agent Memory Leaderboard API guide.
A contract violation is a 400/422 that the platform does *not* retry, so these
models are deliberately strict about what is required and permissive about
extra keys the platform may add later.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str
    # Unix milliseconds. Optional per the contract, but when present it is the
    # only reliable anchor for resolving relative time expressions at ingest.
    timestamp: int | None = None


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str
    messages: list[Message]
    user_id: str
    session_id: str


class AddResponse(BaseModel):
    """All four fields must echo the request exactly, or the run aborts."""

    success: bool
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str
    user_id: str
    # Present only for choice questions. Legitimate retrieval conditioning:
    # it is part of the documented Search input.
    options: list[str] | None = None
    top_k: int = Field(default=100, ge=1, le=100)


class Memory(BaseModel):
    id: str
    content: str
    score: float | None = None
    # CL-Bench renders retrieved memories as "- [{created_at}] {text}", so this
    # field reaches the answer model verbatim on the coding track.
    created_at: str | None = None


class SearchResponse(BaseModel):
    """Ordered most- to least-relevant. The platform preserves our order."""

    data: list[Memory]
