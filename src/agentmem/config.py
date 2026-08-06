"""Runtime configuration.

Each LLM tier is configured independently, so swapping a provider is an env
change rather than a code change. Call sites reference the tier
(``MODELS.extract``), never a model name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Gear:
    """One LLM tier: endpoint, credential, and model, resolved from env."""

    base_url: str
    api_key: str
    model: str


def _gear(prefix: str) -> Gear:
    return Gear(
        base_url=os.environ.get(f"{prefix}_API_BASE", "").rstrip("/"),
        api_key=os.environ.get(f"{prefix}_API_KEY", ""),
        model=os.environ.get(f"{prefix}_MODEL", ""),
    )


class MODELS:
    """LLM gears. Add a gear here rather than a model name in a call site."""

    extract = _gear("EXTRACT")
    embed = _gear("EMBED")


EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))

# How many memories Search returns, and how many candidates the recall stage
# fuses before trimming. Both swept against the offline harness; see
# runs/sweep-*.json. POOL_N must exceed RETURN_N or it silently becomes the
# real return limit.
RETURN_N = int(os.environ.get("AGENTMEM_RETURN_N", "20"))
POOL_N = int(os.environ.get("AGENTMEM_POOL_N", "60"))

# Upper bound on the characters Search hands back, applied on top of RETURN_N.
#
# A fixed record count is the wrong unit across datasets. LoCoMo turns average
# ~147 characters, so a hundred of them is a 15k-character context the answer
# model handles comfortably. LongMemEval turns are ShareGPT-length, and the
# same hundred records produce ~112k characters — roughly 28k tokens aimed at
# gpt-4o-mini. Budgeting characters makes the returned count adapt to how long
# the records actually are, instead of being tuned to whichever dataset was
# measured last. 0 disables the budget.
#
# 50,000 is a hedge, not a tuned optimum. It is a no-op on LoCoMo, whose
# hundred records come to ~17k characters, and it clamps LongMemEval to the
# middle of its curve. The LongMemEval sample is too small to prove a gain;
# what the budget buys is that no dataset with long records can quietly hand
# the answer model a 100k-character prompt. 0 disables it.
CHAR_BUDGET = int(os.environ.get("AGENTMEM_CHAR_BUDGET", "50000"))

# Share of the returned slots reserved for extracted facts; the rest go to raw
# dated turns. Swept, not guessed — see runs/share-*.json.
FACT_SHARE = float(os.environ.get("AGENTMEM_FACT_SHARE", "0.5"))

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://agentmem:agentmem@localhost:5432/agentmem"
)

API_KEY = os.environ.get("AGENTMEM_API_KEY", "")
HOST = os.environ.get("AGENTMEM_HOST", "0.0.0.0")
PORT = int(os.environ.get("AGENTMEM_PORT", "8080"))

# Ceiling on in-flight LLM calls. The platform drives Add with 64 workers and
# each Add makes an extraction call and an embedding call, so an unthrottled
# service presents ~128 concurrent requests to the provider and starts
# collecting 429s. Throttling here costs latency; being rate-limited costs the
# extraction layer, because a failed extraction degrades to raw turns.
LLM_MAX_CONCURRENCY = int(os.environ.get("AGENTMEM_LLM_CONCURRENCY", "48"))
