FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/legal-rag/.cache/huggingface

WORKDIR /app

RUN groupadd --system legal-rag \
    && useradd --system --gid legal-rag --create-home legal-rag

# Install the CPU-only PyTorch wheel first to avoid CUDA runtime packages.
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install \
        "anyio>=4.8,<4.10" \
        "fastapi>=0.115,<0.116" \
        google-genai \
        "psycopg[binary]" \
        sentence-transformers \
        uvicorn

COPY pyproject.toml ./
COPY src ./src
COPY data/processed/civil_code_articles_clean_v2.json \
    ./data/processed/civil_code_articles_clean_v2.json

RUN python -m pip install --no-deps . \
    && mkdir -p /home/legal-rag/.cache/huggingface \
    && chown -R legal-rag:legal-rag /home/legal-rag

USER legal-rag

EXPOSE 8000

CMD ["uvicorn", "legal_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
