# Deployment and submission runbook

The evaluation platform calls **our** endpoint, so the service has to survive a
72-hour window under 64 concurrent writers with nobody watching it. This is the
sequence, and the order matters — a full evaluation can only be submitted once
every three months.

## 0. Before anything else: register

Submit at <https://agentmemories.ai/evaluation> to get a Leaderboard Key.
Nothing below can be verified without one, and smoke runs are limited to
**one per hour**, so the key is the critical path.

The form needs:

| field | value |
|---|---|
| system name / version | `AgentMEM` / the commit being submitted |
| track | Academic Methods → Textual |
| repository | the public GitHub URL (academic track requires open source) |
| Add endpoint | `https://<host>/add` |
| Search endpoint | `https://<host>/search` |
| health endpoint | `https://<host>/health` |
| auth scheme | `Bearer` (matches `AGENTMEM_API_KEY`) |
| originality | see README → Attribution |

## 1. Host

Requirements are unglamorous: reachable from the platform's network for 72
hours, and stable under sustained concurrency. Sizing is small — 2 vCPU / 4 GB.
Retrieval is brute-force numpy over a few hundred vectors per query.

The one thing that actually matters is the network path. The evaluator resolves
to `47.112.15.137`, an Alibaba Cloud Shenzhen address inside mainland China,
and it will make roughly 47,000 Add calls at 64-way concurrency. Cross-border
traffic out of the mainland goes through congested international gateways, so
the closer the host sits to Shenzhen the less there is to go wrong.

### Railway

Workable. The caveats are about the route, not the platform:

- **Pick `asia-southeast1` (Singapore).** It is the only Railway region with a
  tolerable path to mainland China; US and EU regions add hundreds of
  milliseconds per call and far more variance.
- **The smoke run is the network test.** It originates from the platform's own
  infrastructure, so it measures the exact path that matters. One per hour is
  enough to find out cheaply, and a pass settles the question.
- **Do not let the service sleep.** Usage-based sleeping mid-run turns into
  failed Adds. `railway.json` pins one replica with restart-on-failure.

Setup: attach the Postgres plugin (it injects `DATABASE_URL`), set the rest of
`.env` as service variables, and deploy from the Dockerfile. `railway.json`
already configures the health check and restart policy, and the image binds
`$PORT` rather than a fixed port.

### Alternative if smoke fails on latency

Alibaba Cloud Shenzhen or Hong Kong, same Docker image, no code changes. Hong
Kong needs no ICP filing and keeps a short path to the evaluator. Worth having
an account ready before submitting, so a failed smoke costs an hour rather than
a day.

Wherever it runs, a laptop on a home connection does not qualify.

## 2. Bring it up

```bash
git clone <repo> && cd AgentMEM && cp .env.example .env
```

Fill in `.env` — at minimum `AGENTMEM_API_KEY` (any long random string; the
same value goes in the registration form), `EMBED_API_KEY`, and
`DATABASE_URL`. Leave `EXTRACT_API_BASE` empty: extraction measured worse than
leaving it off.

```bash
docker compose up -d
```

```bash
docker build -t agentmem . && docker run -d --name agentmem --env-file .env --network host --restart unless-stopped agentmem
```

`--restart unless-stopped` is not optional. The run is 72 hours and unattended.

## 3. Verify before pointing the platform at it

```bash
.venv/bin/python scripts/stress.py --url https://<host> --key "$AGENTMEM_API_KEY" --users 40 --chunks 6
```

This must print `PASS`. It reproduces the platform's load shape — 64
concurrent Adds, 32 concurrent Searches, a replayed 10% to prove retries do not
double-write — and checks the failures that invalidate a submission: an id that
does not echo, a `success` that is not boolean `true`, a missing `data` array,
and any cross-user leakage.

Local reference: p50 2.1s, p99 10.6s, 13.4 adds/s, against a platform timeout
of 1,200s. At that rate the full textual track ingests in about an hour.

Then confirm TLS and auth from outside the host:

```bash
curl -s -m 10 https://<host>/health
```

## 3.5 Never deploy while an evaluation is running

A deploy restarts the container. Anything the platform has in flight dies with
it, and the failure it reports is `ADD_API_CONTRACT_MISMATCH` — "记忆写入未完成"
— which reads like a contract bug in our code and is not one.

This has already cost one smoke attempt. The service had answered 281 Adds
with 200 and zero errors; the log ends with a successful Add followed
immediately by `Stopping Container`.

Before `railway up`, confirm nothing is running:

```bash
railway logs --service agentmem --lines 50 | grep -c "POST /add"
```

Recent Add traffic that is not yours means an evaluation is live. Wait for it.
Smoke is one attempt per hour and full is one per three months, so a deploy
that saves five minutes can cost an hour or a quarter.

## 4. Smoke, then full

Smoke is one per hour. Treat each attempt as expensive and read the whole
error before resubmitting — contract errors (400/422) are not retried by the
platform, so they fail the run immediately rather than degrading it.

Once smoke passes, submit the full evaluation. **Once every three months.**
Before pressing it:

- [ ] `pytest` and `ruff check` clean
- [ ] `scripts/stress.py` prints PASS against the *public* URL, not localhost
- [ ] repository is public, and the submitted commit is pushed
- [ ] `.env` is not in the repository (`git log --all -p | grep -c 'sk-'` is 0)
- [ ] no benchmark answers, questions, or ids in `src/`
- [ ] Search composes no text and consults no expected answer
- [ ] `store.load_user` is still the only read path
- [ ] service has `--restart unless-stopped` and the host will not reboot
- [ ] LLM provider has quota for ~47,000 embedding calls

## 5. During the run

```bash
docker logs -f agentmem 2>&1 | grep -iE 'error|unavailable|429'
```

The one warning worth watching is `embedding unavailable, indexing lexically
only`. It means the provider rate-limited us and those records are lexical-only
— the service keeps serving, and measured impact is small, but a sustained
stream of it means the provider is saturated. In local testing one chunk of
26,345 rows degraded this way.

## 6. Afterwards

Competition rule: evaluation data is used only for that run and deleted within
30 days.

```bash
docker rm -f agentmem && docker compose down -v
```
