#!/usr/bin/env bash
# Sweep one env knob against the offline harness, one config per server start.
# Usage: scripts/sweep.sh VAR "value value ..." N_QUESTIONS
set -uo pipefail
cd "$(dirname "$0")/.."

VAR="${1:?env var to sweep}"
VALUES="${2:?space-separated values}"
LIMIT="${3:-500}"

pkill -f "uvicorn agentmem.api" 2>/dev/null
python3 -c "import time;time.sleep(1)"

for value in $VALUES; do
  env "$VAR=$value" AGENTMEM_RETURN_N=100 AGENTMEM_POOL_N=140 \
    .venv/bin/python -m uvicorn agentmem.api:app --host 127.0.0.1 --port 8080 --log-level error &
  server=$!
  for _ in $(seq 1 15); do
    curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break
    python3 -c "import time;time.sleep(1)"
  done
  echo "----- $VAR=$value (n=$LIMIT) -----"
  .venv/bin/python -m harness.run --limit "$LIMIT" --concurrency 16 --skip-ingest \
    --top-k 100 --tag "$(echo "$VAR" | tr 'A-Z_' 'a-z-')-$value" 2>&1 | grep -E '"overall"'
  kill $server 2>/dev/null
  wait $server 2>/dev/null
done
echo "SWEEP DONE"
