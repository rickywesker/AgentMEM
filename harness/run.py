"""Offline replica of the platform pipeline: add → search → answer → judge.

This exists because a full evaluation can only be submitted once every three
months. Every configuration change must be measured here first.

The answer and judge stages use the platform's verbatim prompts (contracts.py)
and its fixed parameters (gpt-4o-mini, temperature 0, no max_tokens). Answers
and judgements are cached on disk keyed by prompt hash, so re-running after a
retrieval change only pays for the questions whose retrieved context moved.

  python -m harness.run --dataset locomo --limit 200 --system http://localhost:8080
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmem.config import Gear  # noqa: E402
from agentmem.llm import chat  # noqa: E402

from . import contracts  # noqa: E402
from .adapters import locomo  # noqa: E402

CACHE = Path(__file__).resolve().parent / ".cache"


# Platform-fixed external contract. These model names are pinned deliberately:
# the harness is only useful if it mirrors what the platform actually runs.
def _answer_gear() -> Gear:
    return Gear(
        base_url=os.environ.get("ANSWER_API_BASE", "").rstrip("/"),
        api_key=os.environ.get("ANSWER_API_KEY", ""),
        model=os.environ.get("ANSWER_MODEL", "gpt-4o-mini"),
    )


def _judge_gear() -> Gear:
    return Gear(
        base_url=os.environ.get("JUDGE_API_BASE", "").rstrip("/"),
        api_key=os.environ.get("JUDGE_API_KEY", ""),
        model=os.environ.get("JUDGE_MODEL", "gpt-4o-mini"),
    )


def _cache_get(kind: str, key: str) -> str | None:
    path = CACHE / kind / f"{key}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _cache_put(kind: str, key: str, value: str) -> None:
    path = CACHE / kind / f"{key}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _hash(text: str, *, salt: str = "") -> str:
    return hashlib.sha256((salt + "\x00" + text).encode("utf-8")).hexdigest()[:32]


@dataclass
class Result:
    id: str
    category: int
    question: str
    gold: str
    generated: str
    label: str
    n_memories: int
    context_chars: int


class HttpSystem:
    """Drives a running Add/Search service over the wire."""

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 1200.0):
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def add(self, chunk: locomo.Chunk) -> None:
        payload = {
            "request_id": chunk.request_id,
            "user_id": chunk.user_id,
            "session_id": chunk.session_id,
            "messages": [
                {k: v for k, v in m.items() if not k.startswith("_")} for m in chunk.messages
            ],
        }
        response = await self.client.post(f"{self.base_url}/add", json=payload)
        response.raise_for_status()
        body = response.json()
        # The platform treats each of these as a hard contract error that stops
        # the run, so the harness must be at least as strict.
        success = body.get("success")
        assert success is True, f"success must be boolean true, got {success!r}"
        for field in ("request_id", "user_id", "session_id"):
            assert body.get(field) == payload[field], f"{field} must echo exactly"

    async def search(self, question: locomo.Question, top_k: int = 100) -> list[dict]:
        payload = {"query": question.question, "user_id": question.user_id, "top_k": top_k}
        response = await self.client.post(f"{self.base_url}/search", json=payload)
        response.raise_for_status()
        body = response.json()
        assert isinstance(body.get("data"), list), "response must contain a data array"
        return body["data"][:top_k]

    async def aclose(self) -> None:
        await self.client.aclose()


def format_context(memories: list[dict], style: str) -> str:
    """Render retrieved memories the way the platform's answer stage would.

    The platform's formatter for the textual datasets lives in its private
    deployment code — only the CL-Bench one is public, and it renders
    ``- [created_at] text``. So ``stamped`` is the best guess and ``bare`` is
    the pessimistic case where ``created_at`` is dropped entirely. Scoring
    under both is how we check that a configuration does not secretly depend
    on a formatting detail we cannot see.
    """
    if style == "bare":
        return "\n".join(f"- {m['content']}" for m in memories)
    return "\n".join(
        f"- [{m['created_at']}] {m['content']}" if m.get("created_at") else f"- {m['content']}"
        for m in memories
    )


def build_answer_item(
    question: locomo.Question, memories: list[dict], style: str = "stamped"
) -> dict:
    """Assemble the record the platform hands to the answer model.

    Search returns one flat, user-scoped list, so it lands in the speaker-1
    slot via the platform's ``retrieved_context`` fallback and speaker 2 stays
    empty. Mirrored here exactly.
    """
    block = format_context(memories, style)
    return {
        "id": question.id,
        "question": question.question,
        "gold_answer": question.gold,
        "speaker_1_name": question.speaker_1_name,
        "speaker_2_name": question.speaker_2_name,
        "retrieved_context": block,
    }


async def score_one(
    client: httpx.AsyncClient,
    system: HttpSystem,
    question: locomo.Question,
    top_k: int,
    semaphore: asyncio.Semaphore,
    style: str = "stamped",
) -> Result:
    async with semaphore:
        memories = await system.search(question, top_k=top_k)
        item = build_answer_item(question, memories, style)

        answer_prompt = contracts.render_answer_prompt(item)
        key = _hash(answer_prompt, salt=_answer_gear().model)
        generated = _cache_get("answer", key)
        if generated is None:
            # Platform sends no max_tokens. Matching that matters: a truncated
            # answer would score differently from the real pipeline.
            generated = await chat(client, _answer_gear(), answer_prompt, temperature=0.0)
            _cache_put("answer", key, generated)

        judge_prompt = contracts.render_accuracy_prompt(item, generated)
        key = _hash(judge_prompt, salt=_judge_gear().model)
        raw = _cache_get("judge", key)
        if raw is None:
            raw = await chat(client, _judge_gear(), judge_prompt, temperature=0.0)
            _cache_put("judge", key, raw)
        try:
            label = contracts.parse_judge_label(raw)
        except ValueError:
            label = "WRONG"

        return Result(
            id=question.id,
            category=question.category,
            question=question.question,
            gold=question.gold,
            generated=generated,
            label=label,
            n_memories=len(memories),
            context_chars=len(item["retrieved_context"]),
        )


def stratified_sample(
    questions: list[locomo.Question], limit: int, seed: int
) -> list[locomo.Question]:
    """Keep the category mix of the full set so subsampled scores stay comparable."""
    if limit <= 0 or limit >= len(questions):
        return questions
    by_category: dict[int, list[locomo.Question]] = defaultdict(list)
    for question in questions:
        by_category[question.category].append(question)
    rng = random.Random(seed)
    picked: list[locomo.Question] = []
    for _, group in sorted(by_category.items()):
        rng.shuffle(group)
        share = max(1, round(limit * len(group) / len(questions)))
        picked.extend(group[:share])
    rng.shuffle(picked)
    return picked[:limit]


def report(results: list[Result]) -> dict:
    correct = sum(r.label == "CORRECT" for r in results)
    overall = 100.0 * correct / len(results) if results else 0.0
    per_category: dict[str, dict] = {}
    for category in sorted({r.category for r in results}):
        group = [r for r in results if r.category == category]
        hits = sum(r.label == "CORRECT" for r in group)
        per_category[str(category)] = {
            "n": len(group),
            "score": round(100.0 * hits / len(group), 2),
        }
    memories = [r.n_memories for r in results] or [0]
    chars = [r.context_chars for r in results] or [0]
    return {
        "n": len(results),
        "overall": round(overall, 2),
        "per_category": per_category,
        "avg_memories_returned": round(sum(memories) / len(memories), 1),
        "avg_context_chars": round(sum(chars) / len(chars)),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locomo")
    parser.add_argument("--data-path", default="harness/datasets/locomo/data/locomo10.json")
    parser.add_argument("--system", default="http://localhost:8080")
    parser.add_argument("--system-key", default=os.environ.get("AGENTMEM_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=200, help="0 = all questions")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument(
        "--context-style",
        choices=["stamped", "bare"],
        default="stamped",
        help="how retrieved memories are rendered into the answer prompt; "
        "'bare' drops created_at, the pessimistic case for the platform's "
        "unpublished textual formatter",
    )
    parser.add_argument("--tag", default="run", help="label for the saved report")
    args = parser.parse_args()

    if args.dataset != "locomo":
        raise SystemExit(f"adapter for {args.dataset!r} not wired yet")

    chunks, questions = locomo.load(args.data_path)
    sampled = stratified_sample(questions, args.limit, args.seed)
    print(
        f"loaded {len(chunks)} chunks, {len(questions)} questions "
        f"({Counter(q.category for q in questions)}); evaluating {len(sampled)}"
    )

    system = HttpSystem(args.system, args.system_key)
    try:
        if not args.skip_ingest:
            ingest = asyncio.Semaphore(args.concurrency)

            async def add_one(chunk: locomo.Chunk) -> None:
                async with ingest:
                    await system.add(chunk)

            await asyncio.gather(*(add_one(c) for c in chunks))
            print(f"ingested {len(chunks)} chunks")

        semaphore = asyncio.Semaphore(args.concurrency)
        async with httpx.AsyncClient(timeout=180.0) as client:
            results = await asyncio.gather(
                *(
                    score_one(client, system, q, args.top_k, semaphore, args.context_style)
                    for q in sampled
                )
            )
    finally:
        await system.aclose()

    summary = report(list(results))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out = Path("runs") / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "summary": summary,
                "config": vars(args),
                "results": [vars(r) for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
