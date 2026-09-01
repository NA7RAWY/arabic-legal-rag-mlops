"""Document ingestion package."""

from legal_rag.ingestion.chunker import LegalChunk, article_to_chunk, chunk_articles
from legal_rag.ingestion.indexer import index_corpus
from legal_rag.ingestion.loader import LegalArticle, load_articles, load_corpus

__all__ = [
    "LegalArticle",
    "LegalChunk",
    "article_to_chunk",
    "chunk_articles",
    "index_corpus",
    "load_articles",
    "load_corpus",
]
