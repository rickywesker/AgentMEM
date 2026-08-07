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

### Railway: do not use it for this

Measured, not predicted. `agentmem-production-2ba2.up.railway.app` resolved to
`69.46.46.102`, and from a mainland network that address accepts no TCP
connection at all — four attempts, four 25-second timeouts, `time_connect` zero
throughout. DNS was fine and identical from `1.1.1.1`, `8.8.8.8` and
`223.5.5.5`, so this is address-level blocking, not resolution. `railway.app`
itself still answered; the edge IP serving the app did not. A traceroute left
China Telecom's backbone, crossed PCCW in Hong Kong, and died two hops later.

This cost the second smoke attempt. The platform reported
`ADD_API_CONTRACT_MISMATCH`, which reads like a bug in our Add handler, and the
service logs for that window are empty — not one request arrived. A contract
error is what the platform reports when it cannot deliver the writes, whatever
the reason.

The trap that hid it: the operator's shell had `HTTPS_PROXY` set, so `curl` and
both verification scripts reached the service in 0.5s through an overseas exit
node and reported everything healthy. Whatever verifies the host must take the
route the evaluator takes — `scripts/preflight.py` and `scripts/stress.py` now
build their clients with `trust_env=False` so a shell proxy cannot answer for
the evaluator, and preflight prints a `warn` line when it ignores one.

Any host outside the mainland invites the same failure. It is not specific to
Railway.

### Alibaba Cloud ECS, mainland (current)

`cn-hangzhou`, 4 vCPU / 8 GB, Alibaba Cloud Linux 3. Direct from a mainland
network: 25 ms to connect, 50 ms to a health response.

Two constraints come with a mainland host, and both are load-bearing:

- **No ICP filing means no ports 80/443.** Serve on a high port and address the
  service by IP. Check what the security group actually permits before
  choosing one — deploying behind a closed port repeats the failure above in a
  new costume. Here 80, 443 and 3000 were open and 3000 was taken.
- **Check each LLM provider from the host, not from a laptop.** From here
  `yunwu.ai` and `api.openai.com` both time out, while `api.voyageai.com`
  connects in 0.15 s and `dashscope.aliyuncs.com` in 27 ms. `embed_texts`
  degrades quietly by design, so a mainland host with an unreachable embedding
  provider does not fail — it serves lexical-only results and loses about seven
  points with nothing in the log. Embeddings now run on Voyage
  (`voyage-4-large`, 1024 dimensions); `llm.py` already speaks that shape, so
  it was an env change. Extraction stays off: the rules require gpt-4o-mini and
  no reachable provider serves it.

```bash
ssh root@<host> 'mkdir -p /opt/agentmem'
tar czf - --exclude='.venv' --exclude='.git' --exclude='harness/datasets' \
    --exclude='runs' --exclude='.env' . | ssh root@<host> 'tar xzf - -C /opt/agentmem'
```

Write `/opt/agentmem/.env` (`chmod 600`), setting `PORT` to the open port, then:

```bash
cd /opt/agentmem && docker compose up -d db
```

```bash
docker build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ -t agentmem .
```

The mirror is not a nicety. PyPI resolves from this host at ~20 kB/s, which
turns the dependency layer into a half-hour download that may not finish; the
domestic mirror pulls the same wheels at ~8 MB/s and the whole build takes
under a minute. The build arg defaults to PyPI, so nothing changes elsewhere.

```bash
docker run -d --name agentmem --env-file /opt/agentmem/.env --network host --restart unless-stopped agentmem
```

`--network host` puts the app on the open port directly and lets it reach the
compose Postgres on `localhost:5432`.

**Changing `.env` requires `docker rm -f` and `docker run` again.** `--env-file`
is read once, when the container is created; `docker restart` replays the
environment captured at creation. Editing the file and restarting appears to
work — the service comes back healthy and preflight still prints READY — while
the process keeps the old values. That is exactly how an unreachable
`EMBED_API_BASE` survives a fix: `embed_texts` returns `None` without logging
when the base URL is empty, so nothing anywhere says the embeddings are gone.
Confirm the swap took at the container, not at the file:

```bash
docker inspect agentmem --format '{{range .Config.Env}}{{println .}}{{end}}' | grep EMBED
```

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

Run these from a machine on the same side of the border as the evaluator, and
read the first line of preflight's output: if it warns that it ignored a proxy,
the shell is configured in a way that would have hidden a dead route.

```bash
.venv/bin/python scripts/preflight.py --url http://<host>:<port> --key "$AGENTMEM_API_KEY"
```

This must print `READY`. It checks every contract rule the platform treats as
fatal rather than retryable — a `success` that is not boolean `true`, an id
that does not echo, an absent `data` array, a 5xx where a 4xx belongs — each of
which costs an hour to discover any other way.

```bash
.venv/bin/python scripts/stress.py --url http://<host>:<port> --key "$AGENTMEM_API_KEY" --users 40 --chunks 6
```

This must print `PASS`. It reproduces the platform's load shape — 64
concurrent Adds, 32 concurrent Searches, a replayed 10% to prove retries do not
double-write — and checks the failures that invalidate a submission: an id that
does not echo, a `success` that is not boolean `true`, a missing `data` array,
and any cross-user leakage.

Measured on the Hangzhou host with Voyage embeddings live: p50 1.27s, p99
2.93s, 39.4 adds/s, against a platform timeout of 1,200s. At that rate the full
textual track ingests in about twenty minutes. For reference, the same test
with embeddings off runs at 523 adds/s — the gap is the embedding round trip,
so a suspiciously fast result is the first sign that they are not happening.

Neither script can tell you whether embeddings are being written, because a
service with none is a service that works. Ask the database directly:

```bash
docker compose exec -T db psql -U agentmem -d agentmem </dev/null \
  -c "SELECT count(*), count(embedding), max(octet_length(embedding)) FROM memories;"
```

Every row should carry a vector, and its width should be `EMBED_DIM` × 4 —
4,096 bytes for Voyage's 1024 dimensions. Fewer than all of them means the
provider is rate-limiting; none of them means it was never configured. The
`</dev/null` is not decoration: `docker compose exec` reads stdin, and inside a
heredoc-fed script it will swallow the rest of the script.

Then confirm reachability from outside the host, on the direct path:

```bash
curl -s --noproxy '*' -m 10 http://<host>:<port>/health
```

`--noproxy '*'` is the whole point of the check. Without it this command has
already reported a healthy service that the evaluator could not reach.

## 3.5 Never deploy while an evaluation is running

A deploy restarts the container. Anything the platform has in flight dies with
it, and the failure it reports is `ADD_API_CONTRACT_MISMATCH` — "记忆写入未完成"
— which reads like a contract bug in our code and is not one.

This has already cost one smoke attempt. The service had answered 281 Adds
with 200 and zero errors; the log ends with a successful Add followed
immediately by `Stopping Container`.

Note that the *next* attempt failed with the identical error for an unrelated
reason — an unreachable host, §1. The same code covers both, so never read
`ADD_API_CONTRACT_MISMATCH` as a diagnosis. Check the service log first: Adds
that stop partway mean a restart, and a log with no requests at all means the
platform never got through.

Before redeploying, confirm nothing is running. Ask the database who wrote,
not the log how much:

```bash
docker compose exec -T db psql -U agentmem -d agentmem </dev/null -tAc \
  "SELECT user_id, count(*) FROM memories \
   WHERE user_id NOT LIKE 'eval:preflight:%' AND user_id NOT LIKE 'stress:%' \
   GROUP BY 1;"
```

Any row here was written by the platform: its user_ids look like
`eval:scriptmem:<run-timestamp>_<id>:<conversation>`, and the timestamp says
when the run started.

Counting recent Adds in the log does not work, and has already been tried. A
verification pass leaves hundreds of its own Adds in the same window — 311 in
twenty minutes, of which 268 were the operator's stress run and 43 were a live
smoke run. The totals look identical; only the identity of the writer
separates them. That misread cost a third smoke attempt, on a service that was
otherwise ready.

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
- [ ] `scripts/preflight.py` prints READY and `scripts/stress.py` prints PASS,
      both against the *public* URL and on the evaluator's side of the border
- [ ] the endpoint answers `curl --noproxy '*'` from a mainland network
- [ ] the registered URL matches the deployed host, port and scheme
- [ ] repository is public, and the submitted commit is pushed
- [ ] `.env` is not in the repository (`git log --all -p | grep -c 'sk-'` is 0)
- [ ] no benchmark answers, questions, or ids in `src/`
- [ ] Search composes no text and consults no expected answer
- [ ] `store.load_user` is still the only read path
- [ ] app *and* database have `--restart unless-stopped`, and the host will
      not reboot
- [ ] every row in `memories` carries an embedding of the expected width
- [ ] Voyage has credit for ~47,000 embedding calls

## 5. During the run

```bash
docker logs -f agentmem 2>&1 | grep -iE 'error|unavailable|429'
```

The one warning worth watching is `embedding unavailable, indexing lexically
only`. It means the provider rate-limited us and those records are lexical-only
— the service keeps serving, and measured impact is small, but a sustained
stream of it means the provider is saturated. In local testing one chunk of
26,345 rows degraded this way; Voyage took 4,803 rows at 64-way concurrency
without emitting it once.

Grepping the log only catches a provider that answers badly. A provider that
was never configured is silent, so check coverage in the database partway
through as well, with the query in §3.

## 6. Afterwards

Competition rule: evaluation data is used only for that run and deleted within
30 days.

```bash
docker rm -f agentmem && docker compose down -v
```
