"""End-to-end retrieval-augmented generation orchestration."""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from legal_rag.guardrails import (
    PIICategory,
    PIIGuardrail,
    PIIRedactionResult,
    StreamingPIIRedactor,
)
from legal_rag.monitoring import (
    NoOpTracer,
    PrometheusMetrics,
    RAGTrace,
    RAGTracer,
    get_metrics,
)
from legal_rag.monitoring.tracing import retrieval_metadata
from legal_rag.rag.generator import (
    GenerationChunk,
    GenerationError,
    GenerationResult,
    LLMGenerator,
    LLMUsage,
    build_legal_context,
)
from legal_rag.storage import RetrievalResult

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class LegalRAGStreamResult:
    """A grounded answer stream and the legal sources used to produce it."""

    question: str
    chunks: Iterator[str]
    retrieved_sources: tuple[RetrievalResult, ...]


class LegalRAGService:
    """Coordinate legal retrieval, context construction, and generation."""

    def __init__(
        self,
        retriever: LegalSearch,
        generator: LLMGenerator,
        *,
        metrics: PrometheusMetrics | None = None,
        tracer: RAGTracer | None = None,
        pii_guardrail: PIIGuardrail | None = None,
        provider: str = "unknown",
        embedding_model: str = "unknown",
        generator_model: str = "unknown",
        input_cost_per_million_tokens_usd: float | None = None,
        output_cost_per_million_tokens_usd: float | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.metrics = metrics if metrics is not None else get_metrics()
        self.tracer = tracer if tracer is not None else NoOpTracer()
        self.pii_guardrail = (
            pii_guardrail if pii_guardrail is not None else PIIGuardrail()
        )
        self.provider = provider
        self.embedding_model = embedding_model
        self.generator_model = generator_model
        self.input_cost_per_million_tokens_usd = input_cost_per_million_tokens_usd
        self.output_cost_per_million_tokens_usd = output_cost_per_million_tokens_usd
        self.clock = clock

    def _generate(self, question: str, context: str) -> GenerationResult:
        method = getattr(self.generator, "generate_with_usage", None)
        if callable(method):
            return method(question, context)
        return GenerationResult(self.generator.generate(question, context))

    def _stream_generate(
        self, question: str, context: str
    ) -> Iterator[GenerationChunk]:
        method = getattr(self.generator, "stream_generate_with_usage", None)
        if callable(method):
            return iter(method(question, context))
        chunks = self.generator.stream_generate(question, context)
        return (GenerationChunk(text=text) for text in chunks)

    def _record_usage(self, usage: LLMUsage | None, mode: str) -> None:
        if usage is None:
            return
        self.metrics.record_token_usage(
            self.provider,
            mode=mode,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            input_cost_per_million_tokens_usd=(self.input_cost_per_million_tokens_usd),
            output_cost_per_million_tokens_usd=(
                self.output_cost_per_million_tokens_usd
            ),
        )

    def _redact_answer(
        self,
        answer: str,
        mode: str,
    ) -> tuple[str, dict[str, object]]:
        result: PIIRedactionResult = self.pii_guardrail.redact(answer)
        self._record_redactions(result, mode)
        return result.text, self._guardrail_metadata(result.counts)

    def _record_redactions(self, result: PIIRedactionResult, mode: str) -> None:
        for category, count in result.counts:
            self.metrics.record_pii_redactions(category.value, mode, count)

    @staticmethod
    def _guardrail_metadata(
        counts: tuple[tuple[PIICategory, int], ...],
    ) -> dict[str, object]:
        active = [(category, count) for category, count in counts if count]
        return {
            "pii_guardrail_triggered": bool(active),
            "pii_redaction_count": sum(count for _, count in active),
            "pii_categories": [category.value for category, _ in active],
        }

    def _start_trace(self, mode: str, top_k: int | None) -> RAGTrace:
        try:
            return self.tracer.start_request(
                mode=mode,
                top_k=top_k,
                provider=self.provider,
                embedding_model=self.embedding_model,
                generator_model=self.generator_model,
            )
        except Exception:
            logger.warning("RAG tracing failed to start; request will continue")
            return NoOpTracer().start_request(
                mode=mode,
                top_k=top_k,
                provider=self.provider,
                embedding_model=self.embedding_model,
                generator_model=self.generator_model,
            )

    def _retrieve(
        self,
        question: str,
        top_k: int | None,
        trace: RAGTrace,
    ) -> list[RetrievalResult]:
        started = self.clock()
        sources: list[RetrievalResult] | None = None
        observation = trace.start_retrieval()
        try:
            sources = self.retriever.search(question, top_k=top_k)
            observation.succeed(retrieval_metadata(sources))
            return sources
        except Exception:
            observation.fail("retrieval_failed")
            raise
        finally:
            observation.end()
            self.metrics.record_retrieval(
                self.clock() - started,
                None if sources is None else len(sources),
            )

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

        trace = self._start_trace("full", top_k)
        try:
            sources = self._retrieve(question, top_k, trace)
            if not sources:
                raise NoRetrievedContextError(
                    "No legal context was retrieved for the question"
                )

            context = build_legal_context(sources)
            started = self.clock()
            generation = trace.start_generation()
            try:
                generated = self._generate(question, context)
                generated_answer = generated.text
                self._record_usage(generated.usage, "full")
                answer, guardrail_metadata = self._redact_answer(
                    generated_answer, "full"
                )
                generation.succeed(guardrail_metadata)
            except GenerationError:
                generation.fail("generation_failed")
                self.metrics.record_provider_failure(self.provider, "full")
                raise
            except Exception:
                generation.fail("generation_failed")
                raise
            finally:
                generation.end()
                self.metrics.record_generation(
                    self.provider,
                    "full",
                    self.clock() - started,
                )
            result = LegalRAGResult(
                question=question,
                answer=answer,
                retrieved_sources=tuple(sources),
            )
            trace.succeed()
            return result
        except Exception:
            trace.fail("rag_request_failed")
            raise
        finally:
            trace.end()

    def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
    ) -> LegalRAGStreamResult:
        """Retrieve once and return a provider-backed grounded answer stream."""

        if not question.strip():
            raise ValueError("Question must not be empty or whitespace-only")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        trace = self._start_trace("stream", top_k)
        try:
            sources = self._retrieve(question, top_k, trace)
            if not sources:
                raise NoRetrievedContextError(
                    "No legal context was retrieved for the question"
                )
        except Exception:
            trace.fail("rag_request_failed")
            trace.end()
            raise

        context = build_legal_context(sources)
        generation = trace.start_generation()
        try:
            chunks = self._stream_generate(question, context)
        except GenerationError:
            generation.fail("generation_failed")
            generation.end()
            self.metrics.record_provider_failure(self.provider, "stream")
            trace.fail("rag_request_failed")
            trace.end()
            raise
        except Exception:
            generation.fail("generation_failed")
            generation.end()
            trace.fail("rag_request_failed")
            trace.end()
            raise

        def measured_chunks() -> Iterator[str]:
            started = self.clock()
            completed = False
            failed = False
            redactor = StreamingPIIRedactor(self.pii_guardrail)
            try:
                usage: LLMUsage | None = None
                for chunk in chunks:
                    if chunk.text:
                        redacted = redactor.feed(chunk.text)
                        self._record_redactions(redacted, "stream")
                        if redacted.text:
                            yield redacted.text
                    if chunk.usage is not None:
                        usage = chunk.usage
                self._record_usage(usage, "stream")
                tail = redactor.finish()
                self._record_redactions(tail, "stream")
                if tail.text:
                    yield tail.text
            except GenerationError:
                failed = True
                redactor.discard()
                generation.fail("generation_failed")
                trace.fail("rag_request_failed")
                self.metrics.record_provider_failure(self.provider, "stream")
                raise
            except Exception:
                failed = True
                redactor.discard()
                generation.fail("generation_failed")
                trace.fail("rag_request_failed")
                raise
            else:
                completed = True
                generation.succeed(self._guardrail_metadata(redactor.counts))
                trace.succeed()
            finally:
                if not completed and not failed:
                    redactor.discard()
                    generation.fail("stream_incomplete")
                    trace.fail("stream_incomplete")
                generation.end()
                trace.end()
                self.metrics.record_generation(
                    self.provider,
                    "stream",
                    self.clock() - started,
                )

        return LegalRAGStreamResult(
            question=question,
            chunks=measured_chunks(),
            retrieved_sources=tuple(sources),
        )
