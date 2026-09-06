# Arabic Legal RAG

A portfolio-oriented retrieval-augmented generation service for asking questions about the Egyptian Civil Code in Arabic. The project combines a validated legal corpus, local multilingual embeddings, PostgreSQL/pgvector semantic search, grounded Gemini generation, and a FastAPI interface.

Module 1 is a working baseline, not a claim of production or legal-advice readiness.

## Problem statement

Legal questions require traceable answers grounded in authoritative text. Generic language models can hallucinate provisions or citations, while keyword search can miss semantically related Arabic phrasing. This project retrieves relevant Civil Code articles first and supplies only that context to the answer generator.

The canonical JSON corpus is the source of truth. PostgreSQL is a rebuildable vector index, not the authoritative corpus.

## Module 1 scope

- Typed corpus loading and validation
- One legal article = one chunk baseline
- Arabic text with English fallback where Arabic is unavailable
- Local `intfloat/multilingual-e5-small` embeddings
- PostgreSQL 16 with pgvector and idempotent chunk upserts
- Cosine-similarity retrieval with configurable `top_k`
- Provider-independent generation interface, currently implemented with Gemini
- FastAPI `/health` and `/ask` endpoints
- Unit/API tests and a local Docker Compose stack

Authentication, user data, conversations, evaluation pipelines, monitoring, deployment automation, and a frontend are outside Module 1.

## Architecture

```text
Egyptian Civil Code JSON
        |
        v
      Loader
        |
        v
Article-level Chunker
        |
        v
multilingual-e5-small (local embeddings)
        |
        v
PostgreSQL + pgvector
        |
        v
     Retriever
        |
        v
Gemini via LLM provider abstraction
        |
        v
      FastAPI
```

Retrieved article number, citation, language, and text are passed to Gemini in a deterministic context format. The prompt restricts generation to that context and asks for cited, concise, informational answers.

## Tech stack

- Python 3.12 and setuptools with a `src` layout
- FastAPI and Uvicorn
- Sentence Transformers with multilingual E5
- PostgreSQL 16 and pgvector
- Psycopg 3
- Google Gen AI SDK
- Pytest
- Docker and Docker Compose

## Project structure

```text
.
├── data/processed/                     # Canonical corpus and Module 0 artifacts
├── docker/postgres/init.sql            # pgvector extension and legal_chunks schema
├── scripts/extract_civil_code.py       # Module 0 extraction workflow
├── src/legal_rag/
│   ├── api/                            # FastAPI app, schemas, dependency wiring
│   ├── ingestion/                      # Loader, chunker, indexing orchestration
│   ├── rag/                            # Embedder, retriever, generator, RAG service
│   ├── storage/                        # Parameterized PostgreSQL repository
│   ├── config.py
│   └── logging_conf.py
├── tests/
├── compose.yaml
├── Dockerfile
└── pyproject.toml
```

## Setup

Requirements: Python 3.12+, Docker with Compose, and a Gemini API key for answer generation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

Set a real `GEMINI_API_KEY` only in the ignored local `.env` file or your shell. Never commit API keys or production credentials.
Docker Compose reads `.env` automatically. For host-based commands, export the file into the current shell first (for example, `set -a; source .env; set +a`).

### Environment variables

| Variable | Development default/purpose |
|---|---|
| `POSTGRES_DB` | `legal_rag` |
| `POSTGRES_USER` | `legal_rag` |
| `POSTGRES_PASSWORD` | Local development credential; replace outside local use |
| `POSTGRES_HOST` | `localhost` on the host; Compose overrides it to `postgres` |
| `POSTGRES_PORT` | `5432` |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` |
| `GEMINI_API_KEY` | Required for `/ask`; no real default |
| `GEMINI_MODEL` | Configurable Gemini model name |

## PostgreSQL and corpus indexing

Start PostgreSQL and wait for it to become healthy:

```bash
docker compose up -d postgres
docker compose ps
```

The initialization script enables pgvector and creates `legal_chunks` with a `vector(384)` embedding column. It only runs when PostgreSQL initializes an empty data volume.

Indexing is intentionally explicit and idempotent; API startup never re-indexes automatically. With the host virtual environment and `POSTGRES_HOST=localhost`:

```bash
python -c "from legal_rag.config import get_config; from legal_rag.ingestion import index_corpus; from legal_rag.rag import SentenceTransformerEmbedder; from legal_rag.storage import PostgresChunkRepository; c=get_config(); print(index_corpus(PostgresChunkRepository(c), SentenceTransformerEmbedder(c.embedding_model), c))"
```

The same operation can be run explicitly in a one-off application container
after restoring the DVC data and starting PostgreSQL. The corpus is mounted at
runtime rather than baked into the production image:

```bash
docker compose run --rm -v "$PWD/data:/app/data:ro" api python -c "from legal_rag.config import get_config; from legal_rag.ingestion import index_corpus; from legal_rag.rag import SentenceTransformerEmbedder; from legal_rag.storage import PostgresChunkRepository; c=get_config(); print(index_corpus(PostgresChunkRepository(c), SentenceTransformerEmbedder(c.embedding_model), c))"
```

Re-indexing upserts deterministic `article-N` chunk IDs, so it rebuilds/updates rows rather than duplicating them.

## Running locally

Start PostgreSQL, then run the API from the host:

```bash
docker compose up -d postgres
uvicorn legal_rag.api.app:app --host 127.0.0.1 --port 8000
```

For an already cached embedding model, optional offline mode is:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uvicorn legal_rag.api.app:app --host 127.0.0.1 --port 8000
```

## Running with Docker Compose

```bash
docker compose build
docker compose up -d
docker compose ps
```

Compose runs the API and PostgreSQL on one network, persists database data in `postgres_data`, and persists downloaded Hugging Face model files in `huggingface_cache`. The E5 model is downloaded on the first retrieval request if the cache is empty.

## MLflow tracking

The local MLflow server uses an isolated SQLite backend and filesystem artifact store in the `mlflow_data` Docker volume. It does not create or modify tables in the legal RAG PostgreSQL database.

Start the stack and open the MLflow UI at [http://localhost:5000](http://localhost:5000):

```bash
docker compose up -d
docker compose ps
curl http://localhost:5000/health
```

Verify the tracking API from the host environment:

```bash
python -c "import mlflow; mlflow.set_tracking_uri('http://localhost:5000'); mlflow.set_experiment('arabic-legal-rag-dev'); print([item.name for item in mlflow.search_experiments()])"
```

Experiment metadata and artifacts persist across container restarts. Evaluation metric names are reserved in the tracking abstraction, but Module 2 does not calculate or invent evaluation values yet.

Create a metadata-only baseline run without invoking retrieval or Gemini:

```bash
python -m legal_rag.tracking.baseline
```

## API

Interactive Swagger documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### `GET /health`

Lightweight process health check. It does not initialize embeddings or call PostgreSQL or Gemini.

```bash
curl http://localhost:8000/health
```

```json
{"status":"ok","app":"Arabic Legal RAG","version":"0.1.0"}
```

### `POST /ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"متى يكون الشخص مسؤولاً عن التعويض؟","top_k":5}'
```

Example response shape (answer and ranked articles depend on retrieval and the provider response):

```json
{
  "question": "متى يكون الشخص مسؤولاً عن التعويض؟",
  "answer": "إجابة عربية مستندة إلى المواد القانونية المسترجعة...",
  "sources": [
    {
      "chunk_id": "article-N",
      "article_number": 0,
      "citation": "Egyptian Civil Code, Article N",
      "language": "ar",
      "similarity": 0.82
    }
  ]
}
```

The placeholder article above documents the response contract without claiming a fixed retrieval result.

## Testing

Tests use fakes and dependency overrides, so the normal suite does not require Docker, PostgreSQL, Hugging Face downloads, or Gemini calls.

### Evaluation dataset

`data/evaluation/legal_rag_eval_v1.json` contains 50 manually curated Arabic
cases grounded in exact excerpts from the canonical Egyptian Civil Code corpus.
The loader validates the exact schema, unique IDs and questions, article
existence, Arabic source availability, and article/excerpt correspondence.
Inspect its deterministic summary with:

```bash
python -m legal_rag.evaluation.summary
```

The 50-case benchmark is the first broad Module 2 evaluation set and can be
expanded toward 100 cases as additional corpus-grounded coverage is curated.
The end-to-end runner supports RAGAS metrics, while live scoring remains subject
to the configured judge provider's availability; metric values are never invented.

### Data versioning with DVC

DVC manages the canonical corpus and the v1 evaluation dataset while Git tracks
their small `.dvc` pointer files. The JSON corpus remains the source of truth;
the PostgreSQL vector index can be rebuilt from it.

```bash
dvc pull       # restore DVC-managed files when a remote becomes available
dvc status     # compare the workspace with DVC metadata
dvc repro      # validate the evaluation dataset against the corpus
```

The `validate_evaluation` stage performs schema and corpus-grounding validation
and prints a deterministic dataset summary. No DVC remote is configured yet;
local cache storage is used for Module 2, and a cloud remote can be added later.

## Continuous integration

GitHub Actions runs on pull requests and pushes to `main`. The quality job checks
Ruff linting and formatting, runs pre-commit, and enforces at least 80% test
coverage. Tests use fakes and make no live PostgreSQL, MLflow, Hugging Face, or
Gemini calls.

Because no DVC remote exists yet, a fresh GitHub runner cannot restore the
canonical corpus or evaluation dataset. Only the three integration checks that
require those exact files are skipped when DVC outputs are absent; their loader,
schema, and orchestration behavior remains covered with committed synthetic
fixtures. Full data validation requires restored outputs and `dvc repro` until a
remote is configured.

A separate job builds the production image as `legal-rag:<git-sha>`. The image
does not contain the canonical corpus: runtime serving uses the rebuildable
PostgreSQL index, while explicit indexing mounts restored data. CI validates the
image but does not push it; registry selection and credentials belong to the
future deployment layer.

```bash
pytest -q
```

### Experimental long-article chunking

The Module 2 comparison keeps the production one-article-per-chunk baseline and
indexes `split_long_articles` into a separate PostgreSQL table. Based on the
canonical corpus character-length distribution, the initial experimental
configuration splits only articles longer than 600 characters into 500-character
windows with 75-character overlap. This targets the long tail (30 of 1,149
articles) without fragmenting typical provisions. Retrieval metrics are computed
at the article level after removing duplicate article results.

## Current limitations

- Article-level chunks are a baseline and may be coarse for long provisions.
- Retrieval has no reranker, hybrid keyword search, or formal quality evaluation yet.
- The API is synchronous and has no authentication, rate limiting, or request persistence.
- First containerized retrieval is slower while the embedding model downloads.
- Gemini availability and quotas are external dependencies.
- Answers are informational and are not professional legal advice.

## Future MLOps roadmap

- Versioned ingestion and reproducible index builds
- Retrieval and grounded-answer evaluation datasets and metrics
- Hybrid retrieval, reranking, and chunking experiments
- CI quality gates, container scanning, and automated deployment
- Observability for latency, failures, retrieval quality, and model usage
- Model/prompt versioning and controlled provider comparison
- Production secrets management, authentication, rate limiting, and resilience
