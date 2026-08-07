"""Query-time retrieval: lexical + dense recall, fused, then trimmed.

Two things worth knowing before changing anything here.

1. **Return the full ``top_k``.** The tempting move is to return a small,
   precise set, on the theory that a weak answer model plus a strict judge
   punishes noise. Measured, that is backwards: score rises monotonically with
   the number of records returned, 41.3 at three records to 60.7 at a hundred.
   A user's corpus is only a few hundred short records, so a hundred of them is
   not noise, it is coverage. What the trim controls is the *mix*, not the size.

2. **Relevance order, not conversation order.** Presenting the chosen records
   chronologically is free and looks obviously right — the answer prompt asks
   the model to resolve relative times and prefer recent memories, and a
   hundred records sorted by score arrive temporally scrambled. It gained 1.4
   on LoCoMo, where multi-hop rose 7.1 and temporal fell 8.6, and lost 2.0 on
   LongMemEval, where the equivalent category did not move at all. Keeping the
   top few by relevance and sorting only the tail bounced non-monotonically
   across a range wider than the effect. The idea does not survive a second
   dataset.

3. **Ranking selects and orders; it never writes.** No stage here composes
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

from .config import CHAR_BUDGET, FACT_SHARE, POOL_N, RETURN_N
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


_MONTHS = {
    name.lower(): number
    for number, name in enumerate(
        "January February March April May June July August September "
        "October November December".split(),
        start=1,
    )
}
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_MONTH_WORD = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b", re.IGNORECASE)
# "most recent" and "first" are temporal constraints too, and unlike "last
# month" they need no anchor date to resolve.
_LATEST = re.compile(
    r"\b(most recent|latest|last time|mostly recently|nowadays|these days)\b", re.I
)
_EARLIEST = re.compile(r"\b(first time|for the first time|originally|initially|at first)\b", re.I)


@dataclass(slots=True)
class Temporal:
    year: int | None = None
    month: int | None = None
    prefer: str | None = None  # "latest" | "earliest"


def temporal_constraint(query: str) -> Temporal | None:
    """Pull a resolvable time constraint out of the question, or nothing.

    Search receives no question date — only the query, the options and the
    user — so relative expressions like "last month" have no anchor and are
    deliberately not guessed at. What is resolvable without one: an explicit
    year, a named month, and orderings like "the first time" or "most
    recently".

    Returning None when nothing is found matters. A query with no temporal
    intent must not pick up a recency bias it never asked for.
    """
    year_match = _YEAR.search(query)
    month_match = _MONTH_WORD.search(query)
    if _LATEST.search(query):
        prefer = "latest"
    elif _EARLIEST.search(query):
        prefer = "earliest"
    else:
        prefer = None
    if not (year_match or month_match or prefer):
        return None
    return Temporal(
        year=int(year_match.group()) if year_match else None,
        month=_MONTHS[month_match.group(1).lower()] if month_match else None,
        prefer=prefer,
    )


def temporal_ranking(records: list[Record], want: Temporal, limit: int) -> list[int]:
    """Rank by agreement with the question's time constraint.

    This is a third ranking fused alongside lexical and dense, never a hard
    filter. A filter would zero out recall whenever the parse is wrong, and
    being wrong is cheap here only because RRF treats this as one vote.
    """
    matches: list[tuple] = []
    for index, record in enumerate(records):
        when = record.created_at
        if when is None:
            continue
        if want.year is not None and when.year != want.year:
            continue
        if want.month is not None and when.month != want.month:
            continue
        matches.append((when, index))
    if not matches:
        return []
    # Within the window, order by whichever end the question asked for;
    # default to recent, matching answer-prompt rule 8.
    matches.sort(key=lambda pair: pair[0], reverse=want.prefer != "earliest")
    return [index for _, index in matches[:limit]]


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

    # Third signal, only when the question actually carries a time constraint.
    # created_at is a real timestamp; leaving it out of ranking meant dates
    # were only ever matched as ordinary words by BM25.
    want = temporal_constraint(query)
    if want is not None:
        dated = temporal_ranking(records, want, pool)
        if dated:
            rankings.append(dated)

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


def resolve_sources(scored: list[Scored], by_id: dict[str, Record]) -> list[Scored]:
    """Swap a ranked fact for the turn it was derived from.

    A fact earns its place in the index, not in the answer prompt. Extracted
    sentences are dense and easy to match against, which is exactly what
    retrieval wants; they are also polished enough to read like conclusions,
    which is what makes the answer model confabulate on questions the
    conversation never answers.

    The first attempt put facts and turns in one pool competing for the same
    hundred slots, and every fact that won displaced a dated verbatim turn —
    temporal fell 17 points and adversarial 26. Using facts as keys and
    returning their sources keeps the matching benefit without spending a
    single return slot on derived text.

    Two facts from one turn collapse to that turn once, at the better rank.
    Unattributed facts are returned as themselves rather than dropped.
    """
    resolved: list[Scored] = []
    seen: set[str] = set()
    for item in scored:
        record = item.record
        if record.kind == "fact" and record.source_id:
            record = by_id.get(record.source_id, record)
        if record.id in seen:
            continue
        seen.add(record.id)
        resolved.append(Scored(record=record, score=item.score))
    return resolved


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


def _apply_char_budget(items: list[Scored], budget: int) -> list[Scored]:
    """Drop the tail once the returned text exceeds ``budget`` characters.

    Applied after ranking, so the records that survive are the most relevant
    ones. Always keeps at least one record — an empty context scores 10.4
    against 63.4, so a single over-long memory still beats none.
    """
    if budget <= 0:
        return items
    kept: list[Scored] = []
    used = 0
    for item in items:
        length = len(item.record.content)
        if kept and used + length > budget:
            break
        kept.append(item)
        used += length
    return kept


def trim(
    scored: list[Scored],
    limit: int = DEFAULT_RETURN,
    fact_share: float = FACT_SHARE,
    char_budget: int = CHAR_BUDGET,
) -> list[Scored]:
    """Cut the pool to a slot budget, split between the two indexes.

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
    # Restore global relevance order so the platform sees a ranked list, then
    # cap the total text — record count alone is the wrong unit when record
    # length varies by an order of magnitude between datasets.
    ranked = sorted(merged, key=lambda i: -i.score)[:limit]
    return _apply_char_budget(ranked, char_budget)
