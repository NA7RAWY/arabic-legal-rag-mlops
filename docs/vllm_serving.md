# vLLM generation backend

The project separates the RAG application from generative-model inference:

- FastAPI remains the baseline HTTP application.
- BentoML can own the production-oriented RAG application/service lifecycle.
- vLLM can serve an open generative model through its OpenAI-compatible API.
- `LegalRAGService` remains responsible for coordinating retrieval and generation.

The application talks to vLLM over `POST /v1/chat/completions`; it does not import
or embed the vLLM server package. Both Gemini and vLLM receive the same grounded
system instruction and user/context prompt. Gemini remains the default and is a
practical development option when the host lacks a suitable GPU.

In a GPU deployment, an example configurable vLLM server is:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192
```

The model above is an example deployment choice, not a model bundled with this
application. No local 7B model or performance benchmark was run on the current
CPU-only development machine.

Select the backend through environment variables:

```bash
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
VLLM_API_KEY=EMPTY
```

For a remote or authenticated OpenAI-compatible endpoint, replace the base URL,
model, and API key through the deployment environment. Do not commit credentials.
Switch back to Gemini with `LLM_PROVIDER=gemini` and the existing Gemini settings.

For progressive responses, `POST /ask/stream` sends `"stream": true` to the
same chat-completions endpoint and relays text deltas as SSE. `POST /ask` retains
the existing complete-response behavior.
