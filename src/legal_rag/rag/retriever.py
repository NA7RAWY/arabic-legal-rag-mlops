"""Semantic retrieval orchestration for legal chunks."""

from typing import Protocol

from legal_rag.config import AppConfig, get_config
from legal_rag.storage import RetrievalResult


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class SemanticSearchRepository(Protocol):
    def semantic_search(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]: ...


class LegalRetriever:
    """Embed user queries and delegate nearest-neighbor search to storage."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        repository: SemanticSearchRepository,
        config: AppConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.repository = repository
        self.config = config if config is not None else get_config()

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return the most semantically similar legal chunks."""

        if not query.strip():
            raise ValueError("Query text must not be empty or whitespace-only")

        result_limit = self.config.retrieval_top_k if top_k is None else top_k
        if result_limit <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self.embedder.embed_query(query)
        return self.repository.semantic_search(query_embedding, result_limit)
