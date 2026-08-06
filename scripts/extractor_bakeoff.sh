#!/usr/bin/env bash
# Re-ingest and re-score once per extractor model.
#
# Each model needs a fresh index, so this is ingest + eval per model, not a
# cheap sweep. Run it in the background.
#
#   scripts/extractor_bakeoff.sh "gemini-2.5-pro deepseek-chat gpt-5-mini" 500
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS="${1:?space-separated model ids}"
LIMIT="${2:-500}"
DB="postgresql://agentmem:agentmem@localhost:5432/agentmem"
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"

# Credentials come from .env, not the interactive shell profile — this script
# runs under bash and cannot parse a zsh rc file.
set -a
# shellcheck disable=SC1091
source .env
set +a

for model in $MODELS; do
  pkill -f "uvicorn agentmem.api" 2>/dev/null
  python3 -c "import time;time.sleep(1)"
  psql "$DB" -c "TRUNCATE memories" -q

  # "none" is the no-extraction baseline: an empty EXTRACT_API_BASE makes
  # extract() short-circuit. Measured in the same loop so it shares the
  # dataset, index build, and question sample with every model.
  if [ "$model" = "none" ]; then
    extract_base=""
  else
    extract_base="$ANSWER_API_BASE"
  fi

  EXTRACT_API_BASE="$extract_base" \
  EXTRACT_API_KEY="$ANSWER_API_KEY" \
  EXTRACT_MODEL="$model" \
  AGENTMEM_RETURN_N=100 AGENTMEM_POOL_N=140 AGENTMEM_FACT_SHARE=0.5 \
    .venv/bin/python -m uvicorn agentmem.api:app --host 127.0.0.1 --port 8080 --log-level error &
  server=$!
  for _ in $(seq 1 15); do
    curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break
    python3 -c "import time;time.sleep(1)"
  done

  echo "===== extractor=$model ====="
  start=$(python3 -c "import time;print(time.time())")
  .venv/bin/python -m harness.run --limit "$LIMIT" --concurrency 16 --top-k 100 \
    --tag "extractor-$(echo "$model" | tr '/.' '--')" 2>&1 | grep -E '"overall"'
  python3 -c "import time;print(f'  wall {time.time()-$start:.0f}s')"
  psql "$DB" -tAc "SELECT '  facts: '||count(*)||', avg '||round(avg(length(content)))||' chars' FROM memories WHERE kind='fact'"

  kill $server 2>/dev/null
  wait $server 2>/dev/null
done
echo "BAKEOFF DONE"
