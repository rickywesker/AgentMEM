# AgentMEM

A memory system for the [Agent Memory Leaderboard 2026](https://agentmemories.ai/competition/),
academic methods board, textual track.

The platform drives two endpoints — `POST /add` to write, `POST /search` to
retrieve — and handles answering and scoring itself with fixed parameters
(`gpt-4o-mini`, temperature 0, `top_k` 100). So the only thing a submission
actually controls is **what text ends up in the returned records, and in what
order**. Everything below follows from that.

## Architecture

```
POST /add     chunk of ≤20 turns
                ├─ turn rows    one per turn, verbatim, date-stamped
                └─ fact rows    extracted sentences, each pointing back at
                                the turn it came from
              both embedded, written synchronously to Postgres

POST /search  load every record for this user_id
                ├─ BM25         over the user's corpus
                ├─ cosine       brute force, no ANN index
                └─ created_at   only when the question names a time
              fused with RRF
                → facts resolve to their source turn
                → deduplicated → top 100, capped at 50k characters
```

**Facts are keys, never values.** A fact that ranks hands its slot to the turn
it was extracted from, so the answer model only ever reads verbatim
conversation. This is the difference between extraction helping and hurting —
see below.

**No vector index.** `user_id` scopes retrieval to a single conversation, so a
user's whole corpus is a few hundred rows. One indexed lookup pulls the set and
numpy scores it in-process. An ANN index over 800 vectors costs more than the
brute-force dot product it would replace.

**Postgres, not SQLite.** The platform ingests with 64 concurrent workers and
SQLite serialises writers behind a single lock.

**Raw turns, indexed verbatim.** Each turn keeps its speaker and carries a
day-granularity date inlined into the text — not only in the `created_at`
field. The platform's formatter for the textual datasets is not public, so a
record that relies on `created_at` being rendered might arrive at the answer
model with no date at all, and the judge grades time granularity exactly.

**Every LLM call on the write path is optional.** Embedding and extraction both
degrade to "skip it" on failure rather than failing the write, because `Add`
returning non-200 is not something the platform retries indefinitely, and a
lexical-only index still answers most questions.

## What the measurements said

All numbers are LoCoMo, n=150, scored with the platform's verbatim answer and
judge prompts (`harness/contracts.py`). Reproduce with `harness/run.py`.

**Returning more is better, up to the contract ceiling.** Score rises
monotonically with the number of records returned — 41.3 at 3 records, 60.7 at
100. The instinct to return a small, precise set is wrong here: with a corpus
of a few hundred short records, returning 100 is not noisy, it is thorough.

| records returned | 3 | 6 | 10 | 20 | 40 | 60 | 100 |
|---|---|---|---|---|---|---|---|
| score | 41.3 | 48.7 | 49.3 | 52.7 | 54.0 | 57.3 | **60.7** |

**Returning extracted facts as memories is net negative.** This is the wiring
the system shipped with first, and it did not survive measurement. Extracting
self-contained sentences at ingest and returning them *alongside* the raw turns
loses two points overall against not extracting at all:

| category | no extraction | facts indexed, not returned | facts in 75% of slots | n |
|---|---|---|---|---|
| multi-hop | 53.5 | 49.3 | 53.5 | 71 |
| temporal | **58.0** | 55.6 | 44.4 | 81 |
| open-domain | 37.5 | 37.5 | 41.7 | 24 |
| single-hop | **81.6** | 81.1 | 80.2 | 212 |
| adversarial | 45.5 | 45.5 | 44.6 | 112 |
| **overall** | **63.6** | 62.4 | 60.8 | 500 |
| avg context | 17,298 chars | 14,613 | 15,458 | |

The mechanism is in the last row. Facts do not merely fail to help — they
*displace*. They compete with raw turns for the fixed candidate pool, so the
hundred records that come back are drawn from a worse selection, and the
answer model gets less of the conversation. Temporal questions lose the most,
because they are carried by the per-turn dates that get pushed out.

Note the middle column: extraction is a loss even when the extracted facts are
never returned, purely from consuming pool slots.

An earlier measurement at n=150 appeared to show facts gaining +12 on
single-hop. That was wrong on two counts — the baseline it was compared against
had no embeddings, and at n=150 a single category holds ~64 questions, where
one standard error is about 6 points. The finding did not survive a larger
sample against a matched baseline.

The displacement diagnosis is what eventually fixed extraction rather than
killing it. If facts lose by taking slots, stop giving them slots — index them
and return their sources instead. That is the shipped design, and the
before/after is further down.

**Dense retrieval and fact extraction are substitutes, and the system needs one
of them.** Measured next to extraction, embeddings looked worthless — 60.7
against 61.3, inside the noise. Measured with extraction off, they are worth
seven points:

| configuration | LoCoMo n=500 |
|---|---|
| embeddings, no extraction | **62.2–63.4** |
| BM25 + gpt-4o-mini extraction, no embeddings | 57.8 |
| BM25 alone | 55.6 |

Both do the same job — bridging the gap between the words in a question and the
words the conversation actually used — so whichever is added second finds
little left to contribute. The trap is reading the first comparison alone,
concluding embeddings are optional, and shipping plain BM25 seven points down.

This also decides the fallback if the academic board's "Add/Search must use
gpt-4o-mini" rule turns out to cover embedding models: not plain BM25, but
BM25 plus gpt-4o-mini extraction, which recovers 2.2 of the lost points using
only the mandated model. Switching between the two is environment variables,
not code.

What is left is deliberately plain: index every turn verbatim with its date,
retrieve with BM25 and cosine, return the top hundred. Each thing that was
added on top of that had to justify itself against the harness, and most did
not.

**The result does not depend on how the platform renders `created_at`.** The
formatter for the textual datasets is not published, so the same index was
scored twice: once rendered `- [created_at] content`, once with `created_at`
dropped entirely. The gap is 1.6 points overall (63.6 → 62.0), and temporal —
the category that should care most — loses 3.7. That is the payoff for inlining
the date into `content` rather than relying on the field: the pessimistic case
degrades instead of collapsing.

For scale, an empty index scores **10.4** on the same 500 questions. That is
what the answer model produces with no memory at all, and it is the floor
everything above is measured against.

**Extraction was wired wrong, not wrong in principle.** Facts and turns shared
one candidate pool and competed for the same hundred return slots, so every
fact that won displaced a dated verbatim turn. Using facts as *keys* — indexed
and matched against, but resolving to their source turn on the way out — keeps
the matching benefit and spends no return slot on derived text:

| wiring (same extractor, same facts) | LoCoMo n=500 |
|---|---|
| no extraction | 64.0 |
| facts returned as memories | 60.2 |
| **facts as keys, turns returned** | **64.4** |

Against no extraction the net is small: +0.4 on LoCoMo and +2.7 on LongMemEval
(n=150), pooling to **+0.9 over 650 questions** — inside the noise floor. It
ships on anyway, for two reasons that are about risk rather than the mean. It
is never negative overall on either dataset, and its largest single-category
effect is single-session-preference at +16, which is the entire shape of
PersonaMem — one of the five leaderboard datasets and one we cannot measure.
Set `EXTRACT_API_BASE` empty to turn it off; that is the configuration the
64.0 above was measured on.

The honest counterweight: knowledge-update fell 12 points at n=150, extraction
triples ingest time, and neither the gain nor the loss clears one standard
error.

**Every extractor was tried, not just a cheap one.** An earlier version of this
finding used `gpt-4o-mini` and concluded extraction does not work at all. That
was too broad twice over — the wiring was wrong, *and* extraction quality
tracks model strength:

| extractor | LoCoMo n=500 | ingest wall time |
|---|---|---|
| none | **63.4** | 124s |
| gpt-5-mini | 63.2 | 736s |
| claude-opus-4-8 | 62.4 | 453s |
| gemini-2.5-pro | 62.2 | 763s |
| deepseek-chat | 61.8 | 263s |
| gpt-4o-mini | 60.2 | ~200s |

Those numbers are all from the *wrong wiring*, with facts occupying return
slots. They are kept here because they show the gradient is real: a weak
extractor is clearly worse than a strong one either way. `gpt-4o-mini` is what
ships, since the academic board requires it and the structural fix matters more
than the model.

### How much of this is real

Running the same configuration three times gave 63.6, 63.4, and 62.2. The
answer model is at temperature 0 and answers are cached by prompt hash, so that
spread comes from retrieval nondeterminism producing slightly different
contexts. **Run-to-run variance is about ±1.4 points**, on top of ±2.2 points of
sampling error at n=500.

So the honest reading of everything above is:

- **Solid**: return count matters enormously on LoCoMo (41.3 → 60.7); a weak
  extractor hurts; the empty-index floor is 10.4.
- **Directional**: strong extractors land level with no extraction; dense
  retrieval is neutral; long contexts hurt on LongMemEval.
- **Not established**: any single comparison under about 3 points, including
  the exact choice of `fact_share` and the precise character budget.

Configurations were chosen to be safe across that uncertainty rather than to
win a specific number.

## Competition compliance

| Rule | How it is met |
|---|---|
| Search must not generate answers | Ranking selects and orders existing records. No stage composes text at query time. |
| No answers disguised as memories | Every returned record traces to a stored row derived from the ingested corpus. |
| No hardcoded answers | No benchmark answer, question, or id appears in `src/`. |
| No cross-sample state | `user_id` is the only read scope; `store.load_user` never crosses it. |
| Data used only for this run, deleted within 30 days | Corpora and run artifacts are gitignored; teardown drops the database. |
| Reuse must be disclosed | See Attribution. |

Corpus text is treated as untrusted input: the extraction prompt delivers it in
a data block with an explicit instruction that anything resembling a command is
to be recorded as something a speaker said, not acted on.

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
docker compose up -d
```

```bash
cp .env.example .env
```

Then start the service and score it against the offline harness:

```bash
.venv/bin/python -m uvicorn agentmem.api:app --host 0.0.0.0 --port 8080
```

```bash
.venv/bin/python -m harness.run --limit 150 --top-k 100 --tag baseline
```

Tunables are environment variables, all swept rather than guessed:
`AGENTMEM_RETURN_N`, `AGENTMEM_POOL_N`, `AGENTMEM_FACT_SHARE`.

## Why the offline harness exists

A full evaluation can be submitted **once every three months**. There is no
iterating on the real leaderboard. `harness/` replicates the whole pipeline —
add, search, answer, judge — using the platform's own answer and judge prompts,
copied verbatim from its public evaluation code. Answers and judgements are
cached by prompt hash, so re-scoring after a retrieval change only pays for the
questions whose context actually moved.

The harness runs against LoCoMo, which is the public ancestor of the
leaderboard's `locomo-refined`. It is not the same data, so absolute scores
here do not predict leaderboard position. Relative deltas between our own
configurations are what it is for.

**Coverage is narrower than the leaderboard's.** Of the textual datasets the
platform evaluates, only some can be replicated locally at all:

| dataset | corpus available | in the harness |
|---|---|---|
| LoCoMo | yes | yes |
| LongMemEval-S | yes | yes |
| ScriptMem | **no** — the corpus is withheld | no |
| PersonaMem, BEAM | yes, not yet wired | no |

ScriptMem is not an oversight. Its authors deliberately ship questions and gold
answers without the source conversations, [for copyright
reasons](https://github.com/memorax-ai/ScriptMem) — the `conversation` field
holds a synthetic schema example. There is nothing to write into a memory
system, so no memory system can be scored against it offline.

### What the second dataset changed

LongMemEval-S was worth wiring up because it shares LoCoMo's answer and judge
contract exactly while having the opposite shape: each *question* owns a
haystack of ~54 mostly-irrelevant sessions, and its turns are ShareGPT-length
rather than chat-length. Scoring 58.3 over 60 questions, ten per type:

| question type | score |
|---|---|
| single-session-assistant | 100.0 |
| single-session-user | 90.0 |
| knowledge-update | 80.0 |
| multi-session | 30.0 |
| single-session-preference | 30.0 |
| temporal-reasoning | 20.0 |

Two things came out of it. **Memory governance is a strength, not a gap** —
`knowledge-update` asks whether a later fact correctly supersedes an earlier
one, and inlined dates plus recency-ordered retrieval handle it at 80. And the
"return everything" rule tuned on LoCoMo does not transfer: at a hundred
records LongMemEval's context reaches ~112k characters, and the score is lower
there than in the middle of the range. That is what motivated budgeting
characters rather than records — though at n=60 the difference is two
questions, so the budget is a guard against the failure mode rather than a
measured win.

Reading the numbers above with that in mind: most tuning here happened on one
conversational dataset, and the leaderboard averages over several with
different shapes — multiple-choice, rubric-graded, and preference-graded among
them. Configurations were preferred when they held up across categories and
across both datasets rather than when they won overall by a point.

## Attribution

- Answer and judge contracts in `harness/contracts.py` are copied verbatim from
  the leaderboard's public evaluation code,
  [AML-memory/agent-memory-leaderboard](https://github.com/AML-memory/agent-memory-leaderboard).
- Evaluation corpus: [snap-research/locomo](https://github.com/snap-research/locomo).
- BM25 is textbook Okapi BM25, implemented directly; reciprocal rank fusion
  follows Cormack et al. (2009).

## Layout

```
src/agentmem/     the service — config, llm, schemas, store, ingest, extract, retrieve, api
harness/          offline replica of the platform pipeline
  contracts.py    verbatim platform prompts — do not paraphrase
  adapters/       dataset → Add/Search call streams
tests/            contract and retrieval invariants
runs/             scored experiment output (gitignored)
```
