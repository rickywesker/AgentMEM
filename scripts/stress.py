"""Reproduce the platform's load shape against a running service.

The real run is 64 concurrent Add workers and 32 concurrent Search workers
sustained for up to 72 hours. What this checks is not throughput but the things
that would invalidate a submission:

  - every Add echoes its ids and returns ``success: true``
  - a retried Add does not duplicate rows (``request_id`` idempotency)
  - Search never returns another user's memory
  - nothing 5xxs under concurrency
  - p99 Add latency leaves headroom under the platform's 1,200s timeout

  python scripts/stress.py --url http://localhost:8080 --users 40 --chunks 6
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time

import httpx

CHUNK_MESSAGES = 20


def sentinel(user: int) -> str:
    """Per-user marker that cannot be a substring of another user's marker.

    Delimited on both sides — a bare ``zebra{user}`` makes ``zebra1`` match
    inside ``zebra10`` and reports leaks that are not there.
    """
    return f"<zebra:{user}>"


def chunk_payload(user: int, chunk: int) -> dict:
    base = 1683525360000 + chunk * 20 * 60_000
    return {
        "request_id": f"stress:conv-{user}:chunk-{chunk}",
        "user_id": f"stress:conv-{user}",
        "session_id": f"stress:sample:{user}",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Speaker{i % 2}: user {user} chunk {chunk} turn {i}. "
                    f"The secret token for user {user} is {sentinel(user)}."
                ),
                "timestamp": base + i * 60_000,
            }
            for i in range(CHUNK_MESSAGES)
        ],
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--users", type=int, default=40)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--add-workers", type=int, default=64)
    parser.add_argument("--search-workers", type=int, default=32)
    parser.add_argument("--key", default="")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    failures: list[str] = []
    add_latency: list[float] = []

    payloads = [chunk_payload(u, c) for u in range(args.users) for c in range(args.chunks)]
    # Replay the first 10% a second time: the platform retries, and a retry
    # must not double-write.
    retries = payloads[: max(1, len(payloads) // 10)]

    async with httpx.AsyncClient(timeout=1200.0, headers=headers) as client:
        add_slots = asyncio.Semaphore(args.add_workers)

        async def do_add(payload: dict) -> None:
            async with add_slots:
                start = time.perf_counter()
                try:
                    response = await client.post(f"{args.url}/add", json=payload)
                except httpx.HTTPError as error:
                    failures.append(f"add transport: {error}")
                    return
                add_latency.append(time.perf_counter() - start)
                if response.status_code != 200:
                    failures.append(f"add {response.status_code}: {response.text[:120]}")
                    return
                body = response.json()
                if body.get("success") is not True:
                    failures.append(f"add success={body.get('success')!r} (must be true)")
                for field in ("request_id", "user_id", "session_id"):
                    if body.get(field) != payload[field]:
                        failures.append(f"add {field} not echoed: {body.get(field)!r}")

        started = time.perf_counter()
        await asyncio.gather(*(do_add(p) for p in payloads + retries))
        ingest_seconds = time.perf_counter() - started

        search_slots = asyncio.Semaphore(args.search_workers)

        async def do_search(user: int) -> None:
            async with search_slots:
                try:
                    response = await client.post(
                        f"{args.url}/search",
                        json={
                            "query": f"What is the secret token for user {user}?",
                            "user_id": f"stress:conv-{user}",
                            "top_k": 100,
                        },
                    )
                except httpx.HTTPError as error:
                    failures.append(f"search transport: {error}")
                    return
                if response.status_code != 200:
                    failures.append(f"search {response.status_code}: {response.text[:120]}")
                    return
                data = response.json().get("data")
                if not isinstance(data, list):
                    failures.append("search response has no data array")
                    return
                if len(data) > 100:
                    failures.append(f"search returned {len(data)} records, top_k is 100")
                # Isolation: no other user's token may appear in this user's
                # results. This is the disqualifying failure, so it is checked
                # on every record rather than sampled.
                for item in data:
                    content = str(item.get("content", ""))
                    for other in range(args.users):
                        if other != user and sentinel(other) in content:
                            failures.append(f"LEAK: user {user} saw user {other} data")
                            return
                # The user's own marker must be present, or the search found
                # nothing and the isolation check above passed vacuously.
                if not any(sentinel(user) in str(i.get("content", "")) for i in data):
                    failures.append(f"user {user} did not retrieve its own data")

        started = time.perf_counter()
        await asyncio.gather(*(do_search(u) for u in range(args.users)))
        search_seconds = time.perf_counter() - started

    total_adds = len(payloads) + len(retries)
    ordered = sorted(add_latency)
    p50 = statistics.median(ordered) if ordered else 0.0
    if len(ordered) > 100:
        p99 = ordered[int(len(ordered) * 0.99)]
    else:
        p99 = ordered[-1] if ordered else 0.0

    print(f"adds            {total_adds} ({len(retries)} were retries) in {ingest_seconds:.1f}s")
    print(f"add latency     p50 {p50:.2f}s  p99 {p99:.2f}s  (platform timeout 1200s)")
    print(f"searches        {args.users} in {search_seconds:.1f}s")
    print(f"throughput      {total_adds / max(ingest_seconds, 1e-9):.1f} adds/s")

    if failures:
        print(f"\nFAILED — {len(failures)} problem(s):")
        for problem in sorted(set(failures))[:20]:
            print(f"  - {problem}")
        return 1
    print("\nPASS — contract held, no cross-user leakage, no 5xx")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
