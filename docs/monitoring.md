# Prometheus monitoring foundation

The shared FastAPI application exposes `GET /metrics` in Prometheus text format.
BentoML mounts this same application, so it inherits the endpoint and the same
instrumentation without duplicating serving or RAG logic.

## Metrics

| Metric | Type | Meaning |
|---|---|---|
| `legal_rag_http_requests_total` | Counter | Completed requests by HTTP method, registered route template, and status code. |
| `legal_rag_http_request_duration_seconds` | Histogram | End-to-end response time; streaming requests include consumption of the complete SSE body. |
| `legal_rag_http_errors_total` | Counter | Responses with status 400 or greater, by method, route, and status. |
| `legal_rag_ask_requests_total` | Counter | Accepted answer requests by `full` or `stream` mode. |
| `legal_rag_retrieval_duration_seconds` | Histogram | Time spent embedding the query and searching pgvector. |
| `legal_rag_generation_duration_seconds` | Histogram | Full provider generation duration by provider and response mode. |
| `legal_rag_retrieved_sources` | Histogram | Number of legal chunks returned by successful retrieval. |
| `legal_rag_llm_provider_failures_total` | Counter | Generation failures represented by the application's safe provider error abstraction. |
| `legal_rag_llm_tokens_total` | Counter | Authoritative provider-reported input/output/total usage by provider and full/stream mode. |
| `legal_rag_llm_usage_cost_usd_total` | Counter | USD usage cost derived from authoritative input/output counts and explicitly configured rates. |
| `legal_rag_pii_redactions_total` | Counter | Generated PII values redacted by bounded category and response mode. |
| `legal_rag_query_drift_mean_cosine_similarity` | Gauge | Latest current-batch mean similarity to the frozen reference centroid. |
| `legal_rag_query_drift_min_cosine_similarity` | Gauge | Latest current-batch minimum similarity to the reference centroid. |
| `legal_rag_query_drift_p05_cosine_similarity` | Gauge | Latest current-batch fifth-percentile similarity. |
| `legal_rag_query_drift_queries` | Gauge | Number of queries in the latest evaluated batch. |
| `legal_rag_query_drift_similarity_threshold` | Gauge | Optional configured operational mean-similarity threshold. |
| `legal_rag_query_drift_detected` | Gauge | Optional threshold result: `1` below threshold, otherwise `0`. |

Scrape the baseline FastAPI service at:

```text
http://<service-host>:8000/metrics
```

For the default BentoML port, use `http://<service-host>:3000/metrics`. A local
inspection can use:

```bash
curl http://127.0.0.1:8000/metrics
```

Token samples come only from official provider response metadata. Gemini maps
`prompt_token_count`, `candidates_token_count`, and `total_token_count`;
OpenAI-compatible responses map `prompt_tokens`, `completion_tokens`, and
`total_tokens`. Missing values remain unknown and create no sample. Streaming
keeps the last authoritative usage snapshot and records it once after the stream,
so cumulative provider metadata is not double-counted. The Grafana token-rate
query uses only `input|output`; adding `total` would count the same request twice.

Cost is optional and never inferred from a model name. Configure current USD
rates for the deployed provider/model:

```env
LLM_INPUT_COST_PER_1M_TOKENS_USD=
LLM_OUTPUT_COST_PER_1M_TOKENS_USD=
```

When both rates and both authoritative component counts exist, cumulative cost is
`(input_tokens * input_rate + output_tokens * output_rate) / 1_000_000`.
If the provider total contains additional categories (for example Gemini thought
or tool-use tokens) that are not represented by the two configured rates, cost is
not emitted rather than understated. Otherwise no cost series is emitted;
absence means unavailable, not free. Update the rates whenever the provider or
model changes. Grafana calculates configured cost/hour from the
dashboard-selected rate interval of this cumulative counter.

Labels are deliberately low-cardinality. They contain only controlled values
such as registered route templates, HTTP status, response mode, and the bounded
provider set (`gemini`, `vllm`, or `unknown`). Questions, answers, article text,
citations, arbitrary client values, and credentials are never metric labels.

Metric recording is best-effort and isolated from application behavior. An
unexpected Prometheus client error is logged but does not fail the legal request.

This foundation uses the Prometheus client's in-process registry. The current
single-process development services expose it directly; a future multi-worker
deployment must configure and validate Prometheus multiprocess collection rather
than assuming counters are automatically aggregated across workers.

## Local Prometheus and Grafana stack

Prometheus runs in Compose but scrapes the FastAPI process running on the WSL
host. The committed target is `host.docker.internal:8000`, not container
`localhost`. Compose adds the Linux `host-gateway` mapping for that name. Bind
Uvicorn to all host interfaces so the container can reach it:

```bash
uvicorn legal_rag.api.app:app --host 0.0.0.0 --port 8000
```

Start only the two monitoring services:

```bash
docker compose up -d prometheus grafana
docker compose ps prometheus grafana
```

Open:

- Prometheus: `http://localhost:9090`
- Prometheus targets: `http://localhost:9090/targets`
- Grafana: `http://localhost:3001`
- Provisioned dashboard: folder **Legal RAG**, dashboard
  **Arabic Legal RAG Observability**

Grafana defaults to `admin` / `admin` only when no local overrides are supplied.
For local use, set `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` in the
ignored `.env`; `.env.example` contains a non-secret placeholder. Change the
default before using the stack outside an isolated development machine.

Verify the application endpoint and Prometheus target without exposing secrets:

```bash
curl --fail http://localhost:8000/metrics

curl --silent --fail http://localhost:9090/api/v1/targets \
  | python -c "import json,sys; data=json.load(sys.stdin); print([(item['labels']['job'], item['health'], item.get('lastError', '')) for item in data['data']['activeTargets']])"
```

The `legal-rag-api` target should report `up`. If it is down, confirm Uvicorn is
bound to `0.0.0.0:8000`; binding only to `127.0.0.1` prevents a container from
reaching it.

Generate safe HTTP traffic:

```bash
for request in 1 2 3 4 5; do curl --silent --output /dev/null http://localhost:8000/health; done
```

Generate RAG traffic only when PostgreSQL and the selected provider are
intentionally configured:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"متى يكون الشخص مسؤولاً عن التعويض؟","top_k":5}'

curl --no-buffer -X POST http://localhost:8000/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"ما هي أحكام الإيجار؟","top_k":5}'
```

Health traffic populates HTTP panels. Successful answer traffic additionally
populates retrieval, generation, source-count, and provider panels. The token
panel is populated only when the selected provider returns authoritative usage
metadata. The configured-cost panel additionally requires explicit current
input/output rates; an empty panel means evidence is unavailable, not zero cost.

Stop the monitoring services without affecting PostgreSQL or MLflow:

```bash
docker compose stop grafana prometheus
docker compose rm -f grafana prometheus
```

Named volumes preserve seven days of Prometheus data and Grafana state across
container recreation. Use `docker volume rm` only when deliberately resetting
local monitoring history.

## Query embedding drift

Query embedding drift asks whether a recent query population occupies a
different semantic region from the project's frozen benchmark questions. It is
a distribution signal only: it does not determine whether retrieval was correct,
whether an answer was faithful, or whether a legal conclusion was valid.

The reference is built directly from the 50 questions in the DVC-managed
`data/evaluation/legal_rag_eval_v1.json`. Questions are not copied into a second
dataset. The existing `SentenceTransformerEmbedder` batch-encodes them with the
E5 `query:` prefix, normalized embeddings, and the configured
`intfloat/multilingual-e5-small` model. For reference vectors
`r_1, ..., r_n`, the implementation calculates the arithmetic centroid and then
L2-normalizes it:

```text
c = normalize((1 / n) * sum(r_i))
```

For each current query embedding `q_j`, it calculates:

```text
s_j = cosine(q_j, c) = (q_j · c) / (||q_j|| * ||c||)
```

The report contains minimum, linearly interpolated p05, median, mean, p95, and
maximum similarity. It also reports `current_mean - reference_mean`; lower values
mean the current batch is less aligned with the reference centroid. These are raw
statistics, not a claim that semantic or legal quality degraded.

Inspect the reference without starting FastAPI:

```bash
python -m legal_rag.monitoring.drift
```

Evaluate a JSON batch containing a list of strings, a `{"queries": [...]}`
object, or a JSONL file whose lines are strings or `{"query": "..."}` objects:

```bash
python -m legal_rag.monitoring.drift --input current_queries.json
python -m legal_rag.monitoring.drift \
  --input current_queries.jsonl \
  --threshold 0.60 \
  --compact
```

The threshold is optional and must be between `-1` and `1`. It classifies drift
when the current batch mean is below the supplied value. `0.60` above is syntax
illustration only, not a recommended or empirically validated threshold. Choose
an operational threshold only after observing representative historical batches
and expected variation.

The calculation is independent of Prometheus. A long-running process can call
`record_query_drift_metrics(report, metrics)` to publish the latest report into
its in-process registry. The standalone CLI prints JSON and exits, so it does not
magically update the separate FastAPI process's registry. Scheduled execution and
durable handoff of drift reports are intentionally deferred until an operational
collection path is selected. Metric labels contain only fixed `population` or
`policy` values and never contain query text.

## Langfuse RAG tracing

Prometheus and Grafana answer aggregate operational questions such as request
rate, latency, and failure rate. Langfuse provides request-level RAG lineage:

```text
legal-rag-request (chain)
  ├── retrieval (retriever)
  └── generation (generation)
```

The integration uses the OpenTelemetry-based Langfuse Python SDK v4 observation
API (`get_client()` and `start_observation()`), not the legacy trace/span/
generation APIs. Manual v4 observations are used deliberately because a
streaming response may be consumed after the route function returns and possibly
on a different execution context.

Tracing is opt-in and disabled safely when credentials are absent:

```bash
export LANGFUSE_ENABLED=true
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

`LANGFUSE_BASE_URL` can instead point to a compatible, separately operated
self-hosted installation. This project does not add ClickHouse, Redis, object
storage, or the other services required to self-host Langfuse locally.

The root observation records only controlled metadata: endpoint/mode, requested
`top_k`, provider, embedding model, and status. Retrieval records source count,
article numbers, and deterministic chunk IDs, but not full legal text or query
content. Generation records provider, configured model, mode, and status, but
not the question, grounded prompt, or answer. No function decorator is used, so
arguments and return values are not captured automatically. Authoritative token
usage is recorded in Prometheus when supplied by a provider; it is intentionally
not added to Langfuse observations because the tracing abstraction does not
require usage fields and content privacy remains the priority.

For `/ask/stream`, retrieval is still performed exactly once. The root and
generation observations stay open while the answer iterator is consumed and are
ended only on completion, provider failure, or early stream closure. Existing
SSE ordering and retry behavior remain unchanged.

All client creation and observation operations are best-effort. Disabled,
misconfigured, unreachable, or failing Langfuse ingestion cannot turn a legal
RAG request into an application error. SDK warning messages never include keys,
questions, answers, prompts, or provider exception details.

## RAGAS quality evaluation

Prometheus measures aggregate operational behavior, Grafana visualizes it, and
Langfuse traces individual online requests. RAGAS evaluates answer and retrieval
quality offline against the existing frozen benchmark. It reuses all 50 grounded
questions in `data/evaluation/legal_rag_eval_v1.json`; no second monitoring
dataset is maintained.

With PostgreSQL, MLflow, the local E5 model, and Gemini configuration available,
run the complete benchmark manually:

```bash
python -m legal_rag.evaluation.end_to_end \
  --output artifacts/evaluation/ragas_50_case_report.json \
  --case-delay-seconds 1
```

The delay is an optional request-pacing control, not a retry or quality setting.
Use `--case-delay-seconds 0` to disable it. Add `--no-mlflow` when only the local
machine-readable report is wanted. Omitting `--limit` is what evaluates all 50
cases; `--limit` remains available only for controlled smoke diagnostics.

Each case performs retrieval once, grounded generation once, and then invokes
the four configured RAGAS metrics sequentially:

- faithfulness;
- answer relevancy;
- context recall;
- context precision.

RAGAS uses `EVALUATION_MODEL` as a separate Gemini judge. Thus a complete run has
50 answer-generation calls and up to 200 metric evaluations. The exact number
of Gemini HTTP calls is not fixed because an individual RAGAS metric may use
multiple judge prompts and provider/client retries.

The JSON report records dataset and corpus provenance, timestamp, Git state,
generation and judge models, installed RAGAS version, all 50 case records,
successful/failed counts, per-case results, and aggregate metrics. Provider or
pipeline failures are represented with `status: "failed"`, a failure stage and
exception type, and a numeric provider status such as 429/503 when discoverable.
Failed metric values remain `null`; no zero, default, or fabricated score enters
the aggregates. RAGAS aggregates use only cases that returned all four valid
metrics. Retrieval aggregates use only cases whose retrieval completed.
After persisting and printing a partial report, the CLI exits with status `2` so
automation cannot mistake a provider-blocked run for a fully successful one.

The local report is written before MLflow logging, so MLflow availability cannot
erase completed evidence. The same report is logged as
`evaluation/end_to_end_report.json` when MLflow is enabled. A run with failures
is evidence of a partial run, not a completed 50-case quality result; do not
claim RAGAS scores unless the report says `successful_cases: 50` and
`failed_cases: 0`.

## Generated-answer PII guardrail

Every generated `/ask` and `/ask/stream` answer passes through a deterministic
output guardrail before it becomes client-visible. It recognizes deliberately
narrow, high-confidence patterns for:

- email addresses;
- Egyptian mobile numbers using the `010`, `011`, `012`, or `015` prefixes,
  including common `+20`/`0020` and space/hyphen formatting;
- contiguous 14-digit Egyptian national IDs whose encoded birth date is valid
  and whose governorate code is in the documented numeric range.

Detected values are replaced with stable markers:

```text
[REDACTED_EMAIL]
[REDACTED_PHONE]
[REDACTED_NATIONAL_ID]
```

Normal article numbers, citations, monetary values, and ordinary separated dates
are not intentionally targeted. Retrieved corpus chunks and source metadata are
not rewritten; the control applies only to generated user-facing answer text.

For streaming responses, the shared RAG service retains only a suffix that could
still become a supported PII value. Proven-safe text is emitted incrementally;
the maximum pending suffix is 254 characters for a standards-length email, while
phone and national-ID candidates are shorter. At EOF the remaining suffix is
redacted and flushed. On provider failure it is discarded, so a value split
across chunks cannot leak. The successful SSE contract remains
`sources -> token... -> done`.

Prometheus records only category (`email`, `phone`, or `national_id`), mode
(`full` or `stream`), and count. Langfuse records only whether the guardrail was
triggered, total count, and the bounded category list on the generation
observation. Neither system receives the detected value.

This is deterministic pattern protection, not universal PII detection. Unusual
formatting, other identifiers, names, addresses, and non-Egyptian phone numbers
can be missed; any pattern-based approach can also produce false positives. A
broader privacy policy would require separately reviewed detectors and data
handling controls rather than silently expanding these expressions.

## Session 4 evidence status

Implemented and covered by automated tests: Prometheus metrics, provisioned
Grafana panels, cosine query-distribution drift, request-level Langfuse tracing,
the 50-case RAGAS runner with explicit partial failures, generated-answer PII
redaction, authoritative provider usage extraction, and configured-rate cost.

Live verified: Prometheus scraping and Grafana application/process panels;
Langfuse Cloud's `legal-rag-request -> retrieval + generation` hierarchy; and an
offline drift comparison whose frozen-reference mean was approximately `0.89856`
versus approximately `0.76144` for a deliberately out-of-domain batch (delta
approximately `-0.13711`). Earlier live `/ask` requests also completed against
Gemini.

Provider blocked: a completed 50-case RAGAS score report and live token/cost
samples. Recent Gemini attempts exhausted bounded retries with 503 and then 429
responses. These are external provider availability/quota failures, not evidence
of application failure. No RAGAS score, token sample, or cost value is claimed
until a real request returns the required authoritative data.
