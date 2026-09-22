# BentoML serving alternative

FastAPI remains the serving baseline and its `/health` and `/ask` contracts are
unchanged. BentoML 1.4.39 provides an alternative production-oriented service
lifecycle by mounting that same FastAPI ASGI application rather than recreating
its routes or RAG logic.

This project does not produce a newly trained sklearn or PyTorch artifact, so the
BentoML service does not use a Bento model store. It wraps the application-level
`LegalRAGService`, whose existing dependency chain constructs the pgvector
repository, local E5 embedder, retriever, and Gemini generator lazily on the first
`/ask` request. `/health` remains lightweight and initializes none of them.

Install BentoML separately from the baseline runtime and start the service:

```bash
python -m pip install -e .
python -m pip install -r serving/requirements.txt
bentoml serve legal_rag.serving.service:LegalRAGBentoService \
  --host 0.0.0.0 --port 3000
```

The mounted endpoints are:

- `GET /health` — returns `status`, application name, and version.
- `POST /ask` — accepts `question` and optional positive `top_k`, then returns the
  grounded answer and concise legal source metadata.
- `POST /ask/stream` — accepts the same request, buffers and redacts the generated
  answer, then emits source, sanitized-answer, and completion events using SSE.

Environment-based PostgreSQL and Gemini configuration is unchanged. No secrets
are packaged in the service.

Gemini requests use a small bounded retry policy for transient 429 and 5xx
responses. Authentication and other non-transient failures are not retried. Lazy
E5 model initialization is protected against duplicate concurrent loading within
each service process.

BentoML owns the application/service process boundary; it does not replace the
retriever or generator and does not imply better performance than FastAPI.
Locust provides controlled application-layer measurements and optional
real-provider scenarios, but those results must not be treated as proof that
BentoML is faster. vLLM may own generative-model inference while BentoML
continues to own the surrounding RAG application and service lifecycle.
