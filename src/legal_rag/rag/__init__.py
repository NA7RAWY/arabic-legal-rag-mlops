"""Retrieval-augmented generation package."""

from legal_rag.rag.embedder import SentenceTransformerEmbedder
from legal_rag.rag.generator import (
    GeminiGenerator,
    GenerationError,
    LLMGenerator,
    OpenAICompatibleGenerator,
    build_generation_prompt,
    build_generator,
    build_legal_context,
)
from legal_rag.rag.retriever import LegalRetriever
from legal_rag.rag.service import (
    LegalRAGResult,
    LegalRAGService,
    LegalRAGStreamResult,
    NoRetrievedContextError,
)
from legal_rag.storage import RetrievalResult

__all__ = [
    "GeminiGenerator",
    "GenerationError",
    "LLMGenerator",
    "LegalRAGResult",
    "LegalRAGService",
    "LegalRAGStreamResult",
    "LegalRetriever",
    "NoRetrievedContextError",
    "OpenAICompatibleGenerator",
    "RetrievalResult",
    "SentenceTransformerEmbedder",
    "build_generation_prompt",
    "build_generator",
    "build_legal_context",
]
