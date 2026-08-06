FROM python:3.12-slim

WORKDIR /app

# Dependencies first so source edits do not invalidate the layer.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT','8080')}/health\",timeout=4).status==200 else 1)"

# $PORT, not a literal: Railway, Fly, and Cloud Run all inject the port they
# route to, and a hardcoded one silently fails their health check.
#
# One worker on purpose. The service is IO-bound on the LLM and the database,
# and the in-flight LLM semaphore is per-process — extra workers would multiply
# the upstream concurrency we deliberately cap. Scale with replicas instead.
CMD ["sh", "-c", "exec uvicorn agentmem.api:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 75 --log-level info"]
