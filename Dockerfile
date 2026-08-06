FROM python:3.12-slim

WORKDIR /app

# Dependencies first so source edits do not invalidate the layer.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 \
    AGENTMEM_HOST=0.0.0.0 \
    AGENTMEM_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=4).status==200 else 1)"

# One worker: the service is IO-bound on the LLM and the database, and the
# in-flight LLM semaphore is per-process, so multiple workers would multiply
# the concurrency we deliberately cap. Scale with replicas if ever needed.
CMD ["uvicorn", "agentmem.api:app", "--host", "0.0.0.0", "--port", "8080", \
     "--timeout-keep-alive", "75", "--log-level", "info"]
