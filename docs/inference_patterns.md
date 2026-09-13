# Inference and orchestration patterns

## Primary inference pattern: online web service

The Arabic Legal RAG system serves interactive legal questions through an online
request/response flow:

```text
user legal question
  -> retrieve relevant Egyptian Civil Code articles
  -> generate a grounded answer
  -> return the answer with legal sources
```

This pattern fits the product because a user expects an answer for their current
question, with citations, at request time. FastAPI owns the HTTP boundary while
the existing retriever and generator retain the RAG responsibilities.

Batch scoring is useful for offline corpus indexing and evaluation, but it is not
the primary inference pattern: interactive questions arrive individually and
cannot wait for a scheduled batch. Event-stream inference through Kafka, Redis,
or a similar broker is also unnecessary today because there is no continuous
event source or asynchronous consumer workflow in the current use case.

Token streaming is a separate concern. The optional `/ask/stream` route
incrementally sends one answer's model chunks over an existing HTTP connection
to improve perceived latency; it does not introduce an event broker or turn
inference into an event-stream architecture. The regular `/ask` route remains a
normal non-streaming request/response service.

## Airflow's role

Airflow coordinates repeatable maintenance and evaluation operations in this
order:

```text
validate corpus
  -> validate evaluation dataset
  -> rebuild vector index
  -> run retrieval evaluation
  -> run RAG evaluation
```

The DAG in `orchestration/dags/legal_rag_pipeline.py` is manually triggered and
calls the project's existing Python modules. Airflow does not parse articles,
create embeddings, retrieve chunks, generate answers, or calculate metrics; the
application modules continue to own that logic.

Airflow is deliberately excluded from the main application dependencies. Its
environment installs `orchestration/requirements.txt` alongside this project.

The final RAG evaluation command is disabled by default. It runs only when the
Airflow environment explicitly sets `ENABLE_LIVE_RAG_EVALUATION=true`, preventing
an ordinary manual DAG run from consuming Gemini judge quota.

This is not a classical `extract -> train -> evaluate -> register` ML pipeline.
The system has no trained project-specific model to register: it validates a
canonical legal corpus, rebuilds a derived vector index, and evaluates retrieval
and provider-backed generation. Model training and a model registry would only
belong here if the project later introduced a trainable model lifecycle.
