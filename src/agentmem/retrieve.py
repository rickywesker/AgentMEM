"""Query-time retrieval: lexical + dense recall, fused, then trimmed.

Two things worth knowing before changing anything here.

1. **Return the full ``top_k``.** The tempting move is to return a small,
   precise set, on the theory that a weak answer model plus a strict judge
   punishes noise. Measured, that is backwards: score rises monotonically with
   the number of records returned, 41.3 at three records to 60.7 at a hundred.
   A user's corpus is only a few hundred short records, so a hundred of them is
   not noise, it is coverage. What the trim controls is the *mix*, not the size.

2. **Ranking selects and orders; it never writes.** No stage here composes
   text or consults an expected answer. The competition forbids answering
   inside Search or dressing an answer up as a memory record, and that is a
   disqualifying rule rather than a scoring one.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .config import FACT_SHARE, POOL_N, RETURN_N
from .store import Record, embedding_matrix

# Chosen against the harness, not by intuition: see runs/ for the sweep.
DEFAULT_RETURN = RETURN_N
DEFAULT_POOL = POOL_N
# Safety net for queries that match nothing lexically or densely — not a
# target size. See the backfill note in `candidates`.
MIN_CANDIDATES = 5
RRF_K = 60

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    """a an and are as at be been being but by did do does for from had has have he her hers him
    his how i if in into is it its me my of on or our ours she so than that the their theirs them
    then there these they this to too was we were what when where which who whom why will with you
    your yours""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass(slots=True)
class Scored:
    record: Record
    score: float


class BM25:
    """Okapi BM25 over one user's memories.

    Rebuilt per query rather than cached: a user's corpus is a few hundred
    short documents, so the build is microseconds and there is no cache
    invalidation to get wrong when Add and Search interleave.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.lengths = [len(doc) for doc in corpus]
        self.avg_length = (sum(self.lengths) / len(corpus)) if corpus else 0.0
        self.frequencies = [Counter(doc) for doc in corpus]
        document_count = len(corpus)
        appearances: Counter[str] = Counter()
        for doc in corpus:
            appearances.update(set(doc))
        self.idf = {
            term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in appearances.items()
        }

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.corpus), dtype=np.float32)
        if not self.corpus or self.avg_length == 0:
            return out
        for index, frequencies in enumerate(self.frequencies):
            length = self.lengths[index]
            total = 0.0
            for term in query:
                count = frequencies.get(term, 0)
                if not count:
                    continue
                denominator = count + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                total += self.idf.get(term, 0.0) * count * (self.k1 + 1) / denominator
            out[index] = total
        return out


def reciprocal_rank_fusion(rankings: list[list[int]], size: int, k: int = RRF_K) -> list[int]:
    """Fuse ranked index lists. Rank-based, so lexical and cosine scales never
    have to be reconciled."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank + 1)
    return [index for index, _ in sorted(fused.items(), key=lambda kv: -kv[1])][:size]


def _order(scores: np.ndarray, limit: int) -> list[int]:
    if scores.size == 0:
        return []
    top = np.argsort(-scores)[:limit]
    return [int(i) for i in top if scores[i] > 0]


def candidates(
    records: list[Record],
    query: str,
    query_vector: np.ndarray | None,
    *,
    options: list[str] | None = None,
    pool: int = DEFAULT_POOL,
    floor: int = MIN_CANDIDATES,
) -> list[Scored]:
    """Recall stage: fuse BM25 and cosine into one candidate pool.

    ``options`` is documented Search input for choice questions. Folding it
    into the lexical query is retrieval conditioning — it widens recall toward
    the vocabulary the question is about. It does not pick an option.
    """
    if not records:
        return []

    lexical_query = tokenize(query)
    if options:
        for option in options:
            lexical_query.extend(tokenize(option))

    corpus = [tokenize(record.content) for record in records]
    lexical = BM25(corpus).scores(lexical_query)
    rankings = [_order(lexical, pool)]

    if query_vector is not None:
        matrix, indices = embedding_matrix(records)
        if matrix.shape[0]:
            normalised = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
            unit_query = query_vector / (np.linalg.norm(query_vector) + 1e-9)
            cosine = normalised @ unit_query.astype(np.float32)
            dense = np.full(len(records), -np.inf, dtype=np.float32)
            for position, record_index in enumerate(indices):
                dense[record_index] = cosine[position]
            order = np.argsort(-dense)[:pool]
            rankings.append([int(i) for i in order if np.isfinite(dense[i])])

    fused = reciprocal_rank_fusion(rankings, pool)

    # Backfill guards the degenerate case only: a query whose vocabulary misses
    # every record, with the dense path unavailable or missing too, which would
    # otherwise hand the answer model an empty context. Normal thin results are
    # left thin — `trim` decides the final size.
    if len(fused) < floor:
        already = set(fused)
        # Recency-biased, matching answer-prompt rule 8 ("prefer the most
        # recent supported memory").
        for index in reversed(range(len(records))):
            if len(fused) >= floor:
                break
            if index not in already:
                fused.append(index)

    total = len(fused)
    return [
        Scored(record=records[index], score=round(1.0 - rank / max(total, 1), 6))
        for rank, index in enumerate(fused)
    ]


def _dedup(items: list[Scored], seen: set[str]) -> list[Scored]:
    kept: list[Scored] = []
    for item in items:
        fingerprint = " ".join(tokenize(item.record.content)[:12])
        if fingerprint and fingerprint in seen:
            continue
        if fingerprint:
            seen.add(fingerprint)
        kept.append(item)
    return kept


def trim(
    scored: list[Scored],
    limit: int = DEFAULT_RETURN,
    fact_share: float = FACT_SHARE,
) -> list[Scored]:
    """Cut the pool to a fixed slot budget, split between the two indexes.

    The two record kinds are good at different questions, and letting one
    starve the other costs more than it gains. Facts are distilled and
    self-contained, and they carry single-hop and multi-hop questions. Raw
    turns are verbatim and individually dated, and they carry temporal
    questions — and questions with no answer at all, where a polished fact
    invites the answer model to confabulate and a raw turn does not.

    Measured with facts unrestricted: single-hop +14 and multi-hop +10 against
    the lexical baseline, but temporal -17 and adversarial -26, because facts
    won every tie and displaced the dated turns (runs/v3-conddate.json). The
    quota keeps both signals present. ``fact_share`` is swept, not guessed.
    """
    ordered = [item for _, item in sorted(enumerate(scored), key=lambda p: (-p[1].score, p[0]))]
    fact_cap = round(limit * fact_share)

    seen: set[str] = set()
    facts = _dedup([i for i in ordered if i.record.kind == "fact"], seen)[:fact_cap]
    messages = _dedup([i for i in ordered if i.record.kind != "fact"], seen)[: limit - len(facts)]
    # Either side may under-fill; give the slack back to the other.
    if len(facts) + len(messages) < limit:
        spare = limit - len(facts) - len(messages)
        taken = {id(i) for i in facts} | {id(i) for i in messages}
        facts += _dedup([i for i in ordered if id(i) not in taken], seen)[:spare]

    merged = facts + messages
    # Restore global relevance order so the platform sees a ranked list.
    return sorted(merged, key=lambda i: -i.score)[:limit]
