# Serving load testing with Locust

Locust 2.46.5 exercises the same FastAPI routes whether they are hosted directly
or mounted by BentoML. It is installed only in the load-generator environment:

```bash
python -m pip install -r loadtest/requirements.txt
```

## Controlled serving-layer benchmark

The controlled target replaces only the `LegalRAGService` dependency with a
deterministic implementation. Requests still pass through the real FastAPI
routing, validation, serialization, error handling, and SSE response code, but
they do not call PostgreSQL, E5, Gemini, or vLLM.

```bash
uvicorn loadtest.controlled_app:app --host 127.0.0.1 --port 8000
```

Run Locust with its web UI:

```bash
locust -f loadtest/locustfile.py --host http://127.0.0.1:8000
```

Or run a modest, repeatable headless development check:

```bash
locust -f loadtest/locustfile.py \
  --host http://127.0.0.1:8000 \
  --headless --users 5 --spawn-rate 1 --run-time 30s
```

These results measure application and HTTP serving behavior. They are not LLM,
embedding, retrieval, or GPU throughput measurements.

## Optional real-provider benchmark

Start the normal FastAPI or BentoML deployment with its intended Gemini or vLLM
configuration, then point the same Locust file at that host. For example, the
BentoML service documented elsewhere defaults to port 3000:

```bash
locust -f loadtest/locustfile.py --host http://127.0.0.1:3000
```

This mode is never run automatically. Gemini quota, internet latency, PostgreSQL,
E5 initialization, and provider availability affect its results. A real vLLM
test additionally depends heavily on GPU type, model, quantization, batching,
context length, and server configuration. Do not compare controlled numbers with
real-provider throughput.

A small real-provider smoke/load run observed intermittent Gemini HTTP 503
responses. The application retries Gemini 429 and 5xx failures with a short,
bounded exponential backoff before returning its existing safe provider error.
Real-provider latency and failure rate must be reported separately; this small
observation is not a production benchmark.

For `/ask/stream`, retries are possible only while obtaining the buffered first
provider chunk, before sources or text are sent to the client. After the first SSE
events are committed, failures are reported as safe `error` events without retry.

## Metrics and interpretation

Locust reports request rate, concurrent users, failure rate, and p50/p95/p99
latencies. `/ask` measures total full-answer latency. Streaming produces three
measurements:

- `/ask/stream headers`: time until the HTTP response headers are available.
- `/ask/stream first event`: elapsed client time until the first non-empty SSE
  line is received. This is HTTP first-event latency, not GPU token-generation
  latency or model TTFT.
- `/ask/stream full response`: elapsed time until the entire SSE response has
  been consumed and validated.

Separate cold-start and warm measurements. The first real request may load the
local E5 model and be substantially slower. Model and cached-service construction
are lock-protected so concurrent cold requests do not duplicate initialization
within one process. Gemini network/provider latency may dominate real requests,
while model and GPU configuration dominate vLLM results.
Always record user count, spawn rate, duration, target service, provider mode,
and whether requests were cold or warm alongside exported Locust statistics.

No real 7B model is launched or benchmarked by this setup.
