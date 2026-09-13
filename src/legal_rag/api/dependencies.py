"""Lazy dependency construction for the API layer."""

from threading import Lock

from legal_rag.config import get_config
from legal_rag.rag import (
    LegalRAGService,
    LegalRetriever,
    SentenceTransformerEmbedder,
    build_generator,
)
from legal_rag.storage import PostgresChunkRepository


class RAGConfigurationError(RuntimeError):
    """Raised when required RAG service configuration is unavailable."""


def _build_rag_service() -> LegalRAGService:
    config = get_config()
    repository = PostgresChunkRepository(config)
    embedder = SentenceTransformerEmbedder(config.embedding_model)
    retriever = LegalRetriever(embedder, repository, config)
    try:
        generator = build_generator(config)
    except ValueError as exc:
        raise RAGConfigurationError("RAG generation service is not configured") from exc
    return LegalRAGService(retriever, generator)


_service: LegalRAGService | None = None
_service_lock = Lock()


def get_rag_service() -> LegalRAGService:
    """Build one cached RAG service without duplicate concurrent cold starts."""

    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = _build_rag_service()
    return _service
