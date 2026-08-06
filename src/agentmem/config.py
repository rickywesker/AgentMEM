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
