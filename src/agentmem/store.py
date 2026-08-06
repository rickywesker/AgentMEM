"""Postgres-backed memory store.

Why no vector index: ``user_id`` is one conversation, so a user's whole memory
set is a few hundred rows. Search pulls the set with a single indexed lookup
and scores it in-process with numpy. An ANN index over 200 vectors costs more
than the brute-force dot product it replaces.

Why Postgres and not SQLite: the platform writes with 64 concurrent workers,
and SQLite serialises writers behind one lock.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import asyncpg
import numpy as np

from .config import DATABASE_URL, EMBED_DIM

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    request_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ,
    ordinal     INTEGER NOT NULL,
    embedding   BYTEA
);
-- Facts point at the turn they were derived from. A fact is an index entry,
-- not something we hand to the answer model, so a fact that ranks resolves to
-- its source turn on the way out.
ALTER TABLE memories ADD COLUMN IF NOT EXISTS source_id TEXT;
CREATE INDEX IF NOT EXISTS memories_user_idx ON memories (user_id);
-- request_id is the platform's idempotency key: a retried Add must not
-- duplicate rows.
CREATE UNIQUE INDEX IF NOT EXISTS memories_dedup_idx ON memories (request_id, kind, ordinal);
"""


@dataclass(slots=True)
class Record:
    id: str
    content: str
    kind: str
    created_at: datetime | None
    ordinal: int
    embedding: np.ndarray | None
    # Set on facts only: the id of the turn the fact was derived from.
    source_id: str | None = None


def memory_id(request_id: str, kind: str, ordinal: int) -> str:
    digest = hashlib.sha256(f"{request_id}\x00{kind}\x00{ordinal}".encode()).hexdigest()
    return f"mem_{digest[:24]}"


def pack(vector: list[float] | np.ndarray | None) -> bytes | None:
    if vector is None:
        return None
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


class Store:
    def __init__(self, dsn: str = DATABASE_URL):
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        # min_size covers the platform's 64 Add workers without reconnect churn.
        self._pool = await asyncpg.create_pool(self._dsn, min_size=8, max_size=48)
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("store is not connected")
        return self._pool

    async def write(self, rows: list[tuple]) -> int:
        """Insert memory rows. Idempotent on (request_id, kind, ordinal).

        Rows are (id, user_id, session_id, request_id, kind, content,
        created_at, ordinal, embedding, source_id).
        """
        if not rows:
            return 0
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                """
                INSERT INTO memories
                    (id, user_id, session_id, request_id, kind, content,
                     created_at, ordinal, embedding, source_id)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                ON CONFLICT (request_id, kind, ordinal) DO NOTHING
                """,
                rows,
            )
        return len(rows)

    async def load_user(self, user_id: str) -> list[Record]:
        """Every memory for one user, in source order.

        This is the only read path, and it never crosses ``user_id``. The
        competition treats cross-user retrieval as disqualifying, so the
        boundary lives here rather than in each caller.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, kind, created_at, ordinal, embedding, source_id
                FROM memories WHERE user_id = $1 ORDER BY ordinal
                """,
                user_id,
            )
        return [
            Record(
                id=row["id"],
                content=row["content"],
                kind=row["kind"],
                created_at=row["created_at"],
                ordinal=row["ordinal"],
                embedding=unpack(row["embedding"]),
                source_id=row["source_id"],
            )
            for row in rows
        ]

    async def count(self, user_id: str) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT count(*) FROM memories WHERE user_id = $1", user_id
            )


def to_datetime(epoch_ms: int | None) -> datetime | None:
    if epoch_ms is None:
        return None
    try:
        return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def embedding_matrix(records: list[Record]) -> tuple[np.ndarray, list[int]]:
    """Stack available embeddings and report which records they belong to."""
    indices: list[int] = []
    vectors: list[np.ndarray] = []
    for position, record in enumerate(records):
        if record.embedding is not None:
            indices.append(position)
            vectors.append(record.embedding)
    if not vectors:
        return np.zeros((0, EMBED_DIM), dtype=np.float32), []
    return np.vstack(vectors), indices
