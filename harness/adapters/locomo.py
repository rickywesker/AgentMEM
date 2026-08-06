"""LoCoMo → platform Add/Search call stream.

Reproduces the platform's ingestion shape as described in the API guide:
one ``user_id`` per conversation, 20-message chunks, Unix-millisecond
timestamps, ids in the ``eval:<run_id>:...`` namespace.

Upstream dataset: https://github.com/snap-research/locomo (data/locomo10.json)
The leaderboard runs a refined variant we do not have, so absolute scores here
are indicative, not predictive. Relative deltas between our own configurations
are what this harness is for.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

CHUNK_MESSAGES = 20  # platform default
CHUNK_WORDS = 2000  # platform splits at whichever limit trips first

# LoCoMo session stamps look like "1:56 pm on 8 May, 2023".
_SESSION_DT = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<ampm>am|pm)\s+on\s+(?P<d>\d{1,2})\s+(?P<mon>\w+),?\s+(?P<y>\d{4})",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    request_id: str
    user_id: str
    session_id: str
    messages: list[dict]


@dataclass
class Question:
    id: str
    user_id: str
    question: str
    gold: str
    category: int
    speaker_1_name: str
    speaker_2_name: str
    evidence: list[str] = field(default_factory=list)


def _session_epoch_ms(stamp: str) -> int | None:
    match = _SESSION_DT.search(stamp or "")
    if not match:
        return None
    hour = int(match["h"]) % 12
    if match["ampm"].lower() == "pm":
        hour += 12
    try:
        dt = datetime.strptime(
            f"{match['d']} {match['mon']} {match['y']} {hour}:{match['m']}", "%d %B %Y %H:%M"
        )
    except ValueError:
        return None
    # Anchor to UTC rather than letting `.timestamp()` read the naive value as
    # machine-local time. On a UTC+8 host that shifted every "12:xx am"
    # session back a day — 17 of LoCoMo's 288 sessions — and the judge grades
    # day granularity exactly, so a one-day slip is a lost temporal point.
    return int(dt.replace(tzinfo=UTC).timestamp() * 1000)


def _chunk_messages(messages: list[dict]) -> list[list[dict]]:
    """Split at 20 messages or 2,000 words, whichever trips first."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    words = 0
    for message in messages:
        count = len(str(message["content"]).split())
        if current and (len(current) >= CHUNK_MESSAGES or words + count > CHUNK_WORDS):
            chunks.append(current)
            current, words = [], 0
        current.append(message)
        words += count
    if current:
        chunks.append(current)
    return chunks


def load(path: str | Path, run_id: str = "local") -> tuple[list[Chunk], list[Question]]:
    samples = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    questions: list[Question] = []

    for index, sample in enumerate(samples):
        conversation = sample["conversation"]
        speaker_a = conversation.get("speaker_a", "speaker 1")
        speaker_b = conversation.get("speaker_b", "speaker 2")
        user_id = f"eval:{run_id}:locomo:conv-{index}"
        session_id = f"eval:{run_id}:sample:{index}"

        session_keys = sorted(
            (k for k in conversation if k.startswith("session_") and k[8:].isdigit()),
            key=lambda k: int(k[8:]),
        )
        messages: list[dict] = []
        for key in session_keys:
            epoch = _session_epoch_ms(conversation.get(f"{key}_date_time", ""))
            for offset, turn in enumerate(conversation[key]):
                messages.append(
                    {
                        # The platform sends conversational memories as user
                        # turns; the speaker is carried inside the content so
                        # attribution survives into retrieval.
                        "role": "user",
                        "content": f"{turn['speaker']}: {turn['text']}",
                        # One minute apart inside a session keeps ordering
                        # stable without inventing precision we do not have.
                        "timestamp": None if epoch is None else epoch + offset * 60_000,
                        "_dia_id": turn.get("dia_id"),
                        "_session": key,
                    }
                )

        for position, batch in enumerate(_chunk_messages(messages)):
            chunks.append(
                Chunk(
                    request_id=f"eval:{run_id}:locomo_refined:conv-{index}:chunk-{position}",
                    user_id=user_id,
                    session_id=session_id,
                    messages=batch,
                )
            )

        for qa_index, qa in enumerate(sample.get("qa", [])):
            # Category 5 is adversarial: the answerable field is
            # `adversarial_answer` and `answer` is absent.
            gold = qa.get("answer", qa.get("adversarial_answer"))
            if gold is None:
                continue
            questions.append(
                Question(
                    id=f"conv-{index}#q{qa_index:04d}",
                    user_id=user_id,
                    question=str(qa["question"]),
                    gold=str(gold),
                    category=int(qa.get("category", 0)),
                    speaker_1_name=speaker_a,
                    speaker_2_name=speaker_b,
                    evidence=list(qa.get("evidence", [])),
                )
            )

    return chunks, questions
