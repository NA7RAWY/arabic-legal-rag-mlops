"""Retrieval-augmented generation package."""

from legal_rag.rag.embedder import SentenceTransformerEmbedder
from legal_rag.rag.generator import (
    GeminiGenerator,
    GenerationError,
    LLMGenerator,
    build_legal_context,
)
from legal_rag.rag.retriever import LegalRetriever
from legal_rag.rag.service import (
    LegalRAGResult,
    LegalRAGService,
    NoRetrievedContextError,
)
from legal_rag.storage import RetrievalResult

__all__ = [
    "GeminiGenerator",
    "GenerationError",
    "LLMGenerator",
    "LegalRAGResult",
    "LegalRAGService",
    "LegalRetriever",
    "NoRetrievedContextError",
    "RetrievalResult",
    "SentenceTransformerEmbedder",
    "build_legal_context",
]
