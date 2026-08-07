"""Measure retrieval recall against LoCoMo's annotated evidence.

End-to-end accuracy conflates two very different failures: not retrieving the
evidence, and retrieving it but answering wrongly anyway. Only the first is
ours to fix — the answer model and its prompt are fixed by the platform.

LoCoMo annotates each question with the dialogue turns that support it, so the
split is directly measurable. Retrieval makes no LLM calls beyond embedding the
query, which makes this far cheaper than a scored run.

  python -m harness.recall --system http://localhost:8080 --limit 600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from .adapters import locomo, longmemeval  # noqa: E402

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def longmemeval_evidence(path: str | Path, limit: int) -> dict[str, list[str]]:
    """question_id -> the turn texts of its answer-bearing sessions.

    LongMemEval annotates which *sessions* hold the answer rather than which
    turns, so this is a looser target than LoCoMo's: a question counts as
    fully covered when something from every answer session came back.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if 0 < limit < len(records):
        grouped: dict[str, list[dict]] = {}
        for record in records:
            grouped.setdefault(str(record.get("question_type", "?")), []).append(record)
        picked, depth = [], 0
        while len(picked) < limit:
            progressed = False
            for group in grouped.values():
                if depth < len(group) and len(picked) < limit:
                    picked.append(group[depth])
                    progressed = True
            if not progressed:
                break
            depth += 1
        records = picked

    out: dict[str, list[str]] = {}
    for record in records:
        wanted = set(record.get("answer_session_ids", []))
        ids = record.get("haystack_session_ids", [])
        texts: list[str] = []
        for index, session in enumerate(record.get("haystack_sessions", [])):
            if index < len(ids) and ids[index] in wanted:
                for turn in session:
                    text = str(turn.get("content", "")).strip()
                    if len(text) > 40:
                        texts.append(text)
        out[str(record["question_id"])] = texts
    return out


def evidence_texts(path: str | Path) -> dict[str, dict[str, str]]:
    """dia_id -> turn text, per conversation."""
    samples = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for index, sample in enumerate(samples):
        conversation = sample["conversation"]
        table: dict[str, str] = {}
        for key in conversation:
            if key.startswith("session_") and key[8:].isdigit():
                for turn in conversation[key]:
                    if turn.get("dia_id"):
                        table[turn["dia_id"]] = turn["text"]
        out[f"conv-{index}"] = table
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="http://localhost:8080")
    parser.add_argument("--dataset", default="locomo", choices=["locomo", "longmemeval"])
    parser.add_argument("--data-path", default="harness/datasets/locomo/data/locomo10.json")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    if args.dataset == "longmemeval":
        path = "harness/datasets/longmemeval/longmemeval_s.json"
        _, questions = longmemeval.load(path, limit=args.limit)
        by_question = longmemeval_evidence(path, args.limit)
        labels = longmemeval.category_names()
        scoped = [q for q in questions if by_question.get(q.id)]
    else:
        _, questions = locomo.load(args.data_path)
        table = evidence_texts(args.data_path)
        labels = CATEGORIES
        scoped = [q for q in questions if q.evidence][: args.limit]
        by_question = {
            q.id: [
                table.get(q.id.split("#")[0], {}).get(d, "")
                for d in q.evidence
            ]
            for q in scoped
        }

    hits: dict[int, list[float]] = defaultdict(list)
    full: dict[int, list[bool]] = defaultdict(list)
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=180.0) as client:

        async def one(question: locomo.Question) -> None:
            async with semaphore:
                response = await client.post(
                    f"{args.system.rstrip('/')}/search",
                    json={
                        "query": question.question,
                        "user_id": question.user_id,
                        "top_k": args.top_k,
                    },
                )
                blob = " \n".join(m["content"] for m in response.json().get("data", []))

            wanted = [w for w in by_question.get(question.id, []) if w]
            if not wanted:
                return
            # A turn counts as retrieved when a distinctive slice of it appears
            # in the returned text; records carry a date prefix and speaker
            # name, so exact equality would undercount.
            found = sum(1 for w in wanted if w[:60].strip() and w[:60].strip() in blob)
            hits[question.category].append(found / len(wanted))
            full[question.category].append(found == len(wanted))

        await asyncio.gather(*(one(q) for q in scoped))

    print(f"{'category':28s}{'n':>5s}{'evidence recall':>17s}{'all evidence present':>22s}")
    every: list[float] = []
    for category in sorted(hits):
        values = hits[category]
        every.extend(values)
        complete = 100 * sum(full[category]) / len(full[category])
        print(
            f"{str(labels.get(category, category)):28s}{len(values):5d}"
            f"{100 * sum(values) / len(values):16.1f}%{complete:21.1f}%"
        )
    if every:
        print(f"{'ALL':28s}{len(every):5d}{100 * sum(every) / len(every):16.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
