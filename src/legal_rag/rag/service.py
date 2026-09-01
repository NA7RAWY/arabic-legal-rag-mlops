"""End-to-end retrieval-augmented generation orchestration."""

from dataclasses import dataclass
from typing import Protocol

from legal_rag.rag.generator import LLMGenerator, build_legal_context
from legal_rag.storage import RetrievalResult


class LegalSearch(Protocol):
    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]: ...


class NoRetrievedContextError(RuntimeError):
    """Raised when retrieval provides no legal sources for generation."""


@dataclass(frozen=True, slots=True)
class LegalRAGResult:
    """A grounded answer and the legal sources used to produce it."""

    question: str
    answer: str
    retrieved_sources: tuple[RetrievalResult, ...]


class LegalRAGService:
    """Coordinate legal retrieval, context construction, and generation."""

    def __init__(self, retriever: LegalSearch, generator: LLMGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(
        self,
        question: str,
        top_k: int | None = None,
    ) -> LegalRAGResult:
        """Answer a legal question using retrieved context only."""

        if not question.strip():
            raise ValueError("Question must not be empty or whitespace-only")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        sources = self.retriever.search(question, top_k=top_k)
        if not sources:
            raise NoRetrievedContextError(
                "No legal context was retrieved for the question"
            )

        context = build_legal_context(sources)
        answer = self.generator.generate(question, context)
        return LegalRAGResult(
            question=question,
            answer=answer,
            retrieved_sources=tuple(sources),
        )
