# HTTP answer streaming

The service supports two answer modes with identical retrieval and grounding:

- `POST /ask` waits for generation to finish and returns one JSON response.
- `POST /ask/stream` retrieves once and progressively returns answer chunks as
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
data: {"text":"answer fragment"}

event: done
data: {}
```

There may be multiple `token` events. If generation fails after the HTTP stream
has begun, the service emits a sanitized `error` event instead of internal
provider details, and does not emit `done`.

Before committing any SSE event to the client, the service starts generation and
buffers exactly the first non-empty provider chunk. Transient Gemini failures may
therefore use the bounded retry policy before sources become client-visible. Once
that first chunk exists, the service emits sources once, emits the buffered token,
and continues normally. Failures after this commit point are never retried because
that could duplicate client-visible text; they produce the sanitized `error`
event instead.

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

This is streaming for one HTTP LLM response. It is not Kafka, Redis, RabbitMQ,
or another event-streaming architecture. No real 7B model was run or benchmarked
on the current CPU-only development machine.
