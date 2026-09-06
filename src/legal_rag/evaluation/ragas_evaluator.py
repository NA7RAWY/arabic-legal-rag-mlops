"""RAGAS 0.4 metric adapter for end-to-end RAG evaluation."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Protocol

from legal_rag.rag.embedder import SentenceTransformerEmbedder


class RagasEvaluationError(RuntimeError):
    """Raised when a judge metric cannot produce a usable score."""


@dataclass(frozen=True, slots=True)
class RagasScores:
    """Normalized RAGAS scores for one evaluation case."""

    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float


class RagasEvaluator(Protocol):
    """Provider-independent contract consumed by the evaluation runner."""

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        reference_answer: str,
    ) -> RagasScores: ...


def normalize_metric_value(value: Any, metric_name: str) -> float:
    """Extract and validate a numeric metric from RAGAS or a test double."""

    raw_value = getattr(value, "value", value)
    if isinstance(raw_value, bool):
        raise RagasEvaluationError(f"{metric_name} returned a non-numeric value")
    try:
        score = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise RagasEvaluationError(
            f"{metric_name} returned a non-numeric value"
        ) from exc
    if not math.isfinite(score):
        raise RagasEvaluationError(f"{metric_name} returned a non-finite value")
    if not 0.0 <= score <= 1.0:
        raise RagasEvaluationError(
            f"{metric_name} returned a value outside [0, 1]: {score}"
        )
    return score


def build_local_ragas_embeddings(embedder: SentenceTransformerEmbedder) -> Any:
    """Build a modern BaseRagasEmbedding using the already configured model."""

    from ragas.embeddings.base import BaseRagasEmbedding

    class LocalE5Embeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return embedder.embed_query(text)

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(embedder.embed_query, text)

    return LocalE5Embeddings()


class GeminiRagasEvaluator:
    """Evaluate generated responses with RAGAS 0.4 and Gemini as judge."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        embedder: SentenceTransformerEmbedder,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Gemini API key is required for evaluation; set GEMINI_API_KEY"
            )
        self.model_name = model_name
        self._metrics = metrics or self._build_metrics(
            api_key=api_key,
            model_name=model_name,
            embedder=embedder,
        )

    @staticmethod
    def _build_metrics(
        *,
        api_key: str,
        model_name: str,
        embedder: SentenceTransformerEmbedder,
    ) -> dict[str, Any]:
        import instructor
        from google import genai
        from ragas.llms.base import InstructorLLM, InstructorModelArgs
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        # RAGAS 0.4.3's Google factory creates a synchronous instructor client,
        # while collections metrics expose async ``ascore`` methods. Construct
        # the same supported InstructorLLM explicitly with Instructor's async
        # Google mode to bridge that upstream mismatch.
        google_client = genai.Client(api_key=api_key)
        patched_client = instructor.from_genai(
            google_client,
            use_async=True,
            model=model_name,
        )
        judge = InstructorLLM(
            client=patched_client,
            model=model_name,
            provider="google",
            model_args=InstructorModelArgs(),
        )
        embeddings = build_local_ragas_embeddings(embedder)
        return {
            "faithfulness": Faithfulness(llm=judge),
            "answer_relevancy": AnswerRelevancy(
                llm=judge,
                embeddings=embeddings,
            ),
            "context_recall": ContextRecall(llm=judge),
            "context_precision": ContextPrecision(llm=judge),
        }

    async def _evaluate_async(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        reference_answer: str,
    ) -> RagasScores:
        # Run sequentially to keep request pressure predictable for a small
        # local benchmark and avoid multiplying provider rate-limit bursts.
        try:
            faithfulness = await self._metrics["faithfulness"].ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
            answer_relevancy = await self._metrics["answer_relevancy"].ascore(
                user_input=question,
                response=answer,
            )
            context_recall = await self._metrics["context_recall"].ascore(
                user_input=question,
                retrieved_contexts=contexts,
                reference=reference_answer,
            )
            context_precision = await self._metrics["context_precision"].ascore(
                user_input=question,
                reference=reference_answer,
                retrieved_contexts=contexts,
            )
        except Exception as exc:
            raise RagasEvaluationError("RAGAS judge evaluation failed") from exc

        return RagasScores(
            faithfulness=normalize_metric_value(faithfulness, "faithfulness"),
            answer_relevancy=normalize_metric_value(
                answer_relevancy, "answer_relevancy"
            ),
            context_recall=normalize_metric_value(context_recall, "context_recall"),
            context_precision=normalize_metric_value(
                context_precision, "context_precision"
            ),
        )

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        reference_answer: str,
    ) -> RagasScores:
        """Run all four judge metrics exactly once for one generated answer."""

        return asyncio.run(
            self._evaluate_async(
                question=question,
                answer=answer,
                contexts=contexts,
                reference_answer=reference_answer,
            )
        )
