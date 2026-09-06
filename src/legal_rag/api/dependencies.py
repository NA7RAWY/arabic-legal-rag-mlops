"""Lazy dependency construction for the API layer."""

from functools import lru_cache

from legal_rag.config import get_config
from legal_rag.rag import (
    GeminiGenerator,
    LegalRAGService,
    LegalRetriever,
    SentenceTransformerEmbedder,
)
from legal_rag.storage import PostgresChunkRepository


class RAGConfigurationError(RuntimeError):
    """Raised when required RAG service configuration is unavailable."""


@lru_cache(maxsize=1)
def get_rag_service() -> LegalRAGService:
    """Build and cache the RAG service without eagerly loading its model."""

    config = get_config()
    repository = PostgresChunkRepository(config)
    embedder = SentenceTransformerEmbedder(config.embedding_model)
    retriever = LegalRetriever(embedder, repository, config)
    try:
        generator = GeminiGenerator(config)
    except ValueError as exc:
        raise RAGConfigurationError("RAG generation service is not configured") from exc
    return LegalRAGService(retriever, generator)
