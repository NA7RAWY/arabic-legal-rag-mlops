"""Orchestrate corpus loading, chunking, embedding, and persistence."""

import logging
from typing import Protocol

from legal_rag.config import AppConfig, get_config
from legal_rag.ingestion.chunker import LegalChunk, chunk_articles
from legal_rag.ingestion.loader import load_corpus

logger = logging.getLogger(__name__)


class ChunkEmbedder(Protocol):
    def embed_chunks(self, chunks: list[LegalChunk]) -> list[list[float]]: ...


class ChunkRepository(Protocol):
    def upsert_chunks(
        self,
        chunks: list[LegalChunk],
        embeddings: list[list[float]],
    ) -> int: ...


def index_corpus(
    repository: ChunkRepository,
    embedder: ChunkEmbedder,
    config: AppConfig | None = None,
) -> int:
    """Load, chunk, embed, and upsert the configured legal corpus."""

    active_config = config if config is not None else get_config()
    logger.info("Starting corpus indexing from %s", active_config.corpus_path)

    articles = load_corpus(active_config)
    chunks = chunk_articles(articles)
    embeddings = embedder.embed_chunks(chunks)
    if len(embeddings) != len(chunks):
        raise ValueError(
            "Chunk and embedding counts must match: "
            f"{len(chunks)} chunks, {len(embeddings)} embeddings"
        )

    indexed_count = repository.upsert_chunks(chunks, embeddings)
    logger.info("Successfully indexed %d legal chunks", indexed_count)
    return indexed_count
