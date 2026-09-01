"""Persistence adapters for legal RAG data."""

from legal_rag.storage.postgres import PostgresChunkRepository, RetrievalResult

__all__ = ["PostgresChunkRepository", "RetrievalResult"]
