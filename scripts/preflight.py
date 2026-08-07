"""Check the live endpoint against every contract rule before spending a smoke run.

Smoke is limited to one attempt per hour, and the platform treats a contract
error as fatal rather than retryable — a missing `success`, a mismatched id, or
an absent `data` array stops the run outright. Everything checked here is
something that would burn an hour to discover.

  python scripts/preflight.py --url https://host --key KEY
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


def chunk(request_id: str, user_id: str, session_id: str) -> dict:
    return {
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
        "messages": [
            {
                "role": "user",
                "content": "Caroline: I adopted a dog named Biscuit on 7 May 2023.",
                "timestamp": 1683525360000,
            },
            {
                "role": "assistant",
                "content": "Melanie: Lovely. What breed is Biscuit?",
                "timestamp": 1683525420000,
            },
        ],
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--key", default="")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    auth = {"Authorization": f"Token {args.key}"} if args.key else {}
    results: list[tuple[str, str, str]] = []

    def check(status: str, name: str, detail: str = "") -> None:
        results.append((status, name, detail))

    # trust_env=False so a shell proxy cannot answer for the evaluator. Every
    # check below passed against Railway while the evaluator, on a mainland
    # network, could not open a TCP connection at all — the requests were going
    # out through HTTPS_PROXY. A green preflight has to mean the path the
    # platform uses works, not that some path works.
    proxies = [name for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY") if os.environ.get(name)]
    if proxies:
        check(WARN, "proxy env ignored", f"{', '.join(proxies)} set; testing the direct path")

    async with httpx.AsyncClient(timeout=180.0, trust_env=False) as client:
        # --- health -------------------------------------------------------
        try:
            response = await client.get(f"{base}/health")
            ok = 200 <= response.status_code < 300
            check(PASS if ok else FAIL, "GET /health returns 2xx", f"HTTP {response.status_code}")
        except httpx.HTTPError as error:
            check(FAIL, "GET /health reachable", str(error)[:90])
            return report(results)

        # --- auth ---------------------------------------------------------
        if args.key:
            for label, headers in [
                ("Authorization: Token", {"Authorization": f"Token {args.key}"}),
                ("Authorization: Bearer", {"Authorization": f"Bearer {args.key}"}),
                ("X-Api-Key", {"X-Api-Key": args.key}),
            ]:
                response = await client.post(
                    f"{base}/search",
                    headers=headers,
                    json={"query": "q", "user_id": "preflight:none", "top_k": 10},
                )
                check(
                    PASS if response.status_code == 200 else FAIL,
                    f"accepts {label}",
                    f"HTTP {response.status_code}",
                )
            response = await client.post(
                f"{base}/search",
                headers={"Authorization": "Token wrong-key"},
                json={"query": "q", "user_id": "preflight:none", "top_k": 10},
            )
            check(
                PASS if response.status_code == 401 else FAIL,
                "rejects a wrong key with 401",
                f"HTTP {response.status_code}",
            )
        else:
            check(WARN, "auth not exercised", "no --key supplied")

        # --- add ----------------------------------------------------------
        payload = chunk(
            "eval:preflight:locomo_refined:conv-0:chunk-0",
            "eval:preflight:locomo:conv-0",
            "eval:preflight:sample:0",
        )
        response = await client.post(f"{base}/add", headers=auth, json=payload)
        check(
            PASS if response.status_code == 200 else FAIL,
            "POST /add returns 200",
            f"HTTP {response.status_code}",
        )
        body = response.json() if response.status_code == 200 else {}
        check(
            PASS if body.get("success") is True else FAIL,
            "success is boolean true",
            f"got {body.get('success')!r} ({type(body.get('success')).__name__})",
        )
        for field in ("request_id", "user_id", "session_id"):
            check(
                PASS if body.get(field) == payload[field] else FAIL,
                f"{field} echoed exactly",
                f"got {body.get(field)!r}",
            )

        # A retried Add must not double-write.
        await client.post(f"{base}/add", headers=auth, json=payload)
        response = await client.post(
            f"{base}/search",
            headers=auth,
            json={"query": "Biscuit dog", "user_id": payload["user_id"], "top_k": 100},
        )
        data = response.json().get("data", [])
        ids = [item["id"] for item in data]
        check(
            PASS if len(ids) == len(set(ids)) else FAIL,
            "retried Add does not duplicate records",
            f"{len(ids)} records, {len(set(ids))} unique",
        )

        # --- search shape --------------------------------------------------
        check(PASS if isinstance(data, list) else FAIL, "response carries a data array")
        check(PASS if data else FAIL, "ingested content is retrievable", f"{len(data)} records")
        if data:
            first = data[0]
            check(
                PASS if isinstance(first.get("id"), str) and first["id"] else FAIL,
                "each record has a non-empty string id",
            )
            check(
                PASS if isinstance(first.get("content"), str) and first["content"] else FAIL,
                "each record has non-empty content",
            )
            score = first.get("score")
            check(
                PASS if score is None or isinstance(score, int | float) else FAIL,
                "score is numeric or absent",
                f"got {score!r}",
            )
            ordered = [d.get("score") for d in data if d.get("score") is not None]
            check(
                PASS if ordered == sorted(ordered, reverse=True) else FAIL,
                "records are ordered most- to least-relevant",
            )

        # --- top_k is a ceiling ---------------------------------------------
        response = await client.post(
            f"{base}/search",
            headers=auth,
            json={"query": "Biscuit", "user_id": payload["user_id"], "top_k": 1},
        )
        returned = len(response.json().get("data", []))
        check(
            PASS if returned <= 1 else FAIL,
            "top_k is respected as a ceiling",
            f"{returned} for top_k=1",
        )

        # --- unknown user is empty, not an error -----------------------------
        response = await client.post(
            f"{base}/search",
            headers=auth,
            json={"query": "anything", "user_id": "eval:preflight:never-written", "top_k": 100},
        )
        check(
            PASS if response.status_code == 200 and response.json().get("data") == [] else FAIL,
            "unknown user_id returns an empty data array, not an error",
            f"HTTP {response.status_code}",
        )

        # --- malformed input must be a 4xx, not a 500 -------------------------
        response = await client.post(f"{base}/add", headers=auth, json={"request_id": "only"})
        check(
            PASS if 400 <= response.status_code < 500 else FAIL,
            "malformed Add is 4xx (a 5xx would be retried pointlessly)",
            f"HTTP {response.status_code}",
        )

        # --- optional fields the platform may send ---------------------------
        response = await client.post(
            f"{base}/add",
            headers=auth,
            json={
                "request_id": "eval:preflight:x:chunk-1",
                "user_id": "eval:preflight:locomo:conv-1",
                "session_id": "eval:preflight:sample:1",
                "messages": [{"role": "user", "content": "no timestamp on this turn"}],
                "unexpected_field": "the platform may add fields later",
            },
        )
        check(
            PASS if response.status_code == 200 else FAIL,
            "tolerates a missing timestamp and unknown fields",
            f"HTTP {response.status_code}",
        )

        # --- choice questions carry options ----------------------------------
        response = await client.post(
            f"{base}/search",
            headers=auth,
            json={
                "query": "What is the dog called?",
                "options": ["A. Biscuit", "B. Rover"],
                "user_id": payload["user_id"],
                "top_k": 100,
            },
        )
        check(
            PASS if response.status_code == 200 else FAIL,
            "accepts the options field on choice questions",
            f"HTTP {response.status_code}",
        )

    return report(results)


def report(results: list[tuple[str, str, str]]) -> int:
    width = max(len(name) for _, name, _ in results)
    for status, name, detail in results:
        mark = {PASS: "  ok ", FAIL: "FAIL ", WARN: "warn "}[status]
        print(f"{mark}{name:<{width}}  {detail}")
    failures = sum(1 for status, _, _ in results if status == FAIL)
    print()
    if failures:
        print(f"NOT READY — {failures} contract violation(s). Fix before spending a smoke run.")
        return 1
    print(f"READY — {len(results)} checks passed. Safe to trigger smoke.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
