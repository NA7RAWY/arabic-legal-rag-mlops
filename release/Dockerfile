FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/legal-rag/.cache/huggingface

WORKDIR /app

RUN groupadd --system legal-rag \
    && useradd --system --gid legal-rag --create-home legal-rag

COPY pyproject.toml ./
COPY serving/requirements.txt ./serving/requirements.txt
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install . -r serving/requirements.txt \
    && mkdir -p "$HF_HOME" \
    && chown -R legal-rag:legal-rag /app /home/legal-rag

USER legal-rag

EXPOSE 3000

CMD ["bentoml", "serve", "legal_rag.serving.service:LegalRAGBentoService", "--host", "0.0.0.0", "--port", "3000"]
