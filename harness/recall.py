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

from .adapters import locomo  # noqa: E402

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


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
    parser.add_argument("--data-path", default="harness/datasets/locomo/data/locomo10.json")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    _, questions = locomo.load(args.data_path)
    table = evidence_texts(args.data_path)
    scoped = [q for q in questions if q.evidence][: args.limit]

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

            conversation = question.id.split("#")[0]
            wanted = [table.get(conversation, {}).get(d, "") for d in question.evidence]
            wanted = [w for w in wanted if w]
            if not wanted:
                return
            # A turn counts as retrieved when a distinctive slice of it appears
            # in the returned text; records carry a date prefix and speaker
            # name, so exact equality would undercount.
            found = sum(1 for w in wanted if w[:60].strip() and w[:60].strip() in blob)
            hits[question.category].append(found / len(wanted))
            full[question.category].append(found == len(wanted))

        await asyncio.gather(*(one(q) for q in scoped))

    print(f"{'category':14s}{'n':>5s}{'evidence recall':>17s}{'all evidence present':>22s}")
    every: list[float] = []
    for category in sorted(hits):
        values = hits[category]
        every.extend(values)
        complete = 100 * sum(full[category]) / len(full[category])
        print(
            f"{CATEGORIES.get(category, category):14s}{len(values):5d}"
            f"{100 * sum(values) / len(values):16.1f}%{complete:21.1f}%"
        )
    if every:
        print(f"{'ALL':14s}{len(every):5d}{100 * sum(every) / len(every):16.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
