"""LongMemEval-S → platform Add/Search call stream.

Why this dataset earns its place in the harness: it shares LoCoMo's answer and
judge contract byte-for-byte, so the same tuning applies, but it probes two
things LoCoMo does not. ``temporal-reasoning`` is a third of its questions —
our weakest category — and ``knowledge-update`` tests whether a later fact
correctly supersedes an earlier one, which is the memory-governance capability
the competition scores and LoCoMo never exercises.

Structurally it is the opposite shape from LoCoMo. Each *question* owns its own
haystack of ~54 sessions, most of them irrelevant distractors, so ``user_id``
is per question rather than per conversation.

Upstream: https://huggingface.co/datasets/xiaowu0162/longmemeval (longmemeval_s)
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from .locomo import CHUNK_MESSAGES, CHUNK_WORDS, Chunk, Question, _chunk_messages

# "2023/05/20 (Sat) 02:21"
_STAMP = re.compile(r"(\d{4})/(\d{2})/(\d{2})(?:\s*\([^)]*\))?\s*(\d{2}):(\d{2})")

__all__ = ["CHUNK_MESSAGES", "CHUNK_WORDS", "Chunk", "Question", "load"]


def _epoch_ms(stamp: str) -> int | None:
    match = _STAMP.search(stamp or "")
    if not match:
        return None
    year, month, day, hour, minute = (int(g) for g in match.groups())
    try:
        # Anchored to UTC, like the LoCoMo adapter: reading a naive value as
        # machine-local time silently shifts early-morning sessions a day back.
        moment = datetime(year, month, day, hour, minute, tzinfo=UTC)
    except ValueError:
        return None
    return int(moment.timestamp() * 1000)


def load(
    path: str | Path, run_id: str = "local", limit: int = 0
) -> tuple[list[Chunk], list[Question]]:
    """Build the call stream.

    ``limit`` caps the number of questions, and therefore the number of
    haystacks ingested. The full set is ~21k chunks at ~43 per haystack, so
    capping is what keeps an iteration cycle at minutes rather than an hour.

    The cap samples round-robin across question types. The file is grouped by
    type, so taking the first N would ingest one category and silently report
    it as the dataset score.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if 0 < limit < len(records):
        grouped: dict[str, list[dict]] = {}
        for record in records:
            grouped.setdefault(str(record.get("question_type", "unknown")), []).append(record)
        picked: list[dict] = []
        rounds = 0
        while len(picked) < limit:
            progressed = False
            for group in grouped.values():
                if rounds < len(group) and len(picked) < limit:
                    picked.append(group[rounds])
                    progressed = True
            if not progressed:
                break
            rounds += 1
        records = picked

    chunks: list[Chunk] = []
    questions: list[Question] = []

    for record in records:
        question_id = str(record["question_id"])
        user_id = f"eval:{run_id}:longmemeval:{question_id}"

        messages: list[dict] = []
        sessions = record.get("haystack_sessions", [])
        dates = record.get("haystack_dates", [])
        session_ids = record.get("haystack_session_ids", [])
        for index, session in enumerate(sessions):
            epoch = _epoch_ms(dates[index] if index < len(dates) else "")
            source = session_ids[index] if index < len(session_ids) else None
            for offset, turn in enumerate(session):
                text = str(turn.get("content", "")).strip()
                if not text:
                    continue
                role = str(turn.get("role", "user"))
                messages.append(
                    {
                        "role": role,
                        # Speaker is carried inside the content so attribution
                        # survives retrieval, matching the LoCoMo adapter.
                        "content": f"{role}: {text}",
                        "timestamp": None if epoch is None else epoch + offset * 60_000,
                        "_session": source,
                    }
                )

        for position, batch in enumerate(_chunk_messages(messages)):
            chunks.append(
                Chunk(
                    request_id=f"eval:{run_id}:longmemeval_s:{question_id}:chunk-{position}",
                    user_id=user_id,
                    session_id=f"eval:{run_id}:sample:{question_id}",
                    messages=batch,
                )
            )

        questions.append(
            Question(
                id=question_id,
                user_id=user_id,
                question=str(record["question"]),
                gold=str(record["answer"]),
                # Question types are strings here, not integers. Mapped to a
                # stable index so the shared report path can group by them.
                category=_CATEGORIES.setdefault(
                    str(record.get("question_type", "unknown")), len(_CATEGORIES) + 1
                ),
                speaker_1_name="user",
                speaker_2_name="assistant",
                evidence=list(record.get("answer_session_ids", [])),
            )
        )

    return chunks, questions


# Populated on load; exposed so the runner can label the report.
_CATEGORIES: dict[str, int] = {}


def category_names() -> dict[int, str]:
    return {index: name for name, index in _CATEGORIES.items()}
