# HTTP answer streaming

The service supports two answer modes with identical retrieval and grounding:

- `POST /ask` waits for generation to finish and returns one JSON response.
- `POST /ask/stream` retrieves once and returns the guarded answer using
  Server-Sent Events (SSE).

Both routes accept the same request:

```json
{"question": "متى يكون الشخص مسؤولاً عن التعويض؟", "top_k": 5}
```

The streaming event sequence is:

```text
event: sources
data: {"question":"...","sources":[...]}

event: token
data: {"text":"sanitized answer fragment"}

event: token
data: {"text":"next sanitized fragment"}

event: done
data: {}
```

The guarded implementation emits safe answer fragments incrementally. If
generation fails before safe output exists, the service emits a sanitized
`error` event without sources or `done`. If it fails after safe output was
emitted, those events remain visible, the unresolved suffix is discarded, and a
final sanitized `error` event replaces `done`.

The service retains only the earliest suffix that can still become a supported
PII pattern. Text before that suffix is redacted and emitted immediately. A
possible email local/domain portion is bounded by the standards-length 64/local
and 254/total limits, so the maximum retained suffix is 254 characters. Egyptian
mobile candidates require at most 23 characters and national IDs 14. Ordinary
text that cannot become one of these patterns is released on the current
provider chunk. At successful EOF the suffix is redacted and flushed.

This rolling buffer protects values split at arbitrary provider boundaries
without buffering the whole answer. It may delay an ambiguous ASCII token until
a delimiter arrives; it does not claim exact model-token timing.

Example client:

```bash
curl --no-buffer -X POST http://localhost:8000/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"متى يكون الشخص مسؤولاً عن التعويض؟","top_k":5}'
```

Gemini uses its supported streaming generation call. The OpenAI-compatible
provider sends `"stream": true` to vLLM and parses its `data:` events until
`data: [DONE]`. BentoML mounts the same FastAPI application, so it exposes this
route without a second implementation.

This remains an SSE response transport for one HTTP LLM request. It is not Kafka, Redis, RabbitMQ,
or another event-streaming architecture. No real 7B model was run or benchmarked
on the current CPU-only development machine.
