"""PersonaMem v1 → platform Add/Search call stream.

The third leaderboard dataset we can obtain, and the one that probes what the
other two barely touch: stated preferences, how they change over time, and why
they changed. Every question type is some form of "what does this user
actually want".

Two things make its contract different from LoCoMo and LongMemEval, and both
change what a good answer looks like:

* Retrieved memories are injected as **chat messages**, not into a
  ``<memories>`` block inside one user turn.
* Scoring is **deterministic string matching** on ``(a)``/``(b)``/``(c)``/``(d)``
  after a ``<final_answer>`` marker — no LLM judge, and a response naming more
  than one option letter is simply wrong.

Structurally it is prefix-scoped: 37 shared conversations, and each question
sees only the first ``end_index_in_shared_context`` messages of one of them.
``user_id`` is therefore per (context, prefix) pair, of which there are 222.

Upstream: https://huggingface.co/datasets/bowen-upenn/PersonaMem
  questions_32k.csv, shared_contexts_32k.jsonl
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .locomo import Chunk, Question, _chunk_messages

__all__ = ["load", "category_names"]

_CATEGORIES: dict[str, int] = {}

# PersonaMem carries no timestamps. Messages are spaced a day apart from a
# fixed epoch so ordering survives, without implying a real calendar date the
# answer model might try to quote.
_EPOCH_MS = 1_672_531_200_000  # 2023-01-01T00:00:00Z
_DAY_MS = 86_400_000


def _load_contexts(path: Path) -> dict[str, list[dict]]:
    contexts: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            contexts.update(json.loads(line))
    return contexts


def load(
    questions_csv: str | Path,
    contexts_jsonl: str | Path,
    run_id: str = "local",
    limit: int = 0,
) -> tuple[list[Chunk], list[Question]]:
    rows = list(csv.DictReader(Path(questions_csv).open(encoding="utf-8")))
    contexts = _load_contexts(Path(contexts_jsonl))

    if 0 < limit < len(rows):
        # Round-robin across question types; the file is grouped by type, so
        # taking a prefix would ingest one category and call it the dataset.
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["question_type"], []).append(row)
        picked: list[dict] = []
        depth = 0
        while len(picked) < limit:
            progressed = False
            for group in grouped.values():
                if depth < len(group) and len(picked) < limit:
                    picked.append(group[depth])
                    progressed = True
            if not progressed:
                break
            depth += 1
        rows = picked

    chunks: list[Chunk] = []
    questions: list[Question] = []
    ingested: set[tuple[str, int]] = set()

    for row in rows:
        context_id = row["shared_context_id"]
        end_index = int(row["end_index_in_shared_context"])
        context = contexts.get(context_id)
        if not context:
            continue

        scope = (context_id, end_index)
        user_id = f"eval:{run_id}:personamem:{context_id[:12]}-{end_index}"

        # Many questions share a prefix; ingest each distinct one once.
        if scope not in ingested:
            ingested.add(scope)
            messages = []
            for position, message in enumerate(context[:end_index]):
                text = str(message.get("content", "")).strip()
                if not text:
                    continue
                role = str(message.get("role", "user"))
                messages.append(
                    {
                        "role": "user" if role == "system" else role,
                        "content": text,
                        "timestamp": _EPOCH_MS + position * _DAY_MS,
                    }
                )
            for part, batch in enumerate(_chunk_messages(messages)):
                chunks.append(
                    Chunk(
                        request_id=f"eval:{run_id}:personamem:{context_id[:12]}-{end_index}:chunk-{part}",
                        user_id=user_id,
                        session_id=f"eval:{run_id}:sample:{context_id[:12]}",
                        messages=batch,
                    )
                )

        question = Question(
            id=str(row["question_id"]),
            user_id=user_id,
            question=str(row["user_question_or_message"]),
            gold=str(row["correct_answer"]),
            category=_CATEGORIES.setdefault(row["question_type"], len(_CATEGORIES) + 1),
            speaker_1_name="user",
            speaker_2_name="assistant",
        )
        # The official answer prompt requires the original options string
        # verbatim, not a reconstructed list — so it is carried whole.
        question.evidence = [str(row["all_options"])]
        # Search, by contrast, takes options as an array. Splitting on the
        # "(a) " markers is what lets the service recognise a choice question
        # at all, which is how it knows to tighten the context budget.
        question.options = [
            part.strip().rstrip('",]')
            for part in re.split(r'(?=\("?[a-d]"?\))', str(row["all_options"]))
            if part.strip().startswith("(")
        ]
        questions.append(question)

    return chunks, questions


def category_names() -> dict[int, str]:
    return {index: name for name, index in _CATEGORIES.items()}
