"""Low-cardinality Prometheus metrics and HTTP instrumentation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from math import isfinite
from time import perf_counter
from typing import Any

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

_KNOWN_PROVIDERS = frozenset({"gemini", "vllm"})
_PII_CATEGORIES = frozenset({"email", "phone", "national_id"})
_RESPONSE_MODES = frozenset({"full", "stream"})


def _provider_label(provider: str) -> str:
    normalized = provider.strip().lower()
    return normalized if normalized in _KNOWN_PROVIDERS else "unknown"


class PrometheusMetrics:
    """Own application metrics and make recording failures non-fatal."""

    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        self.registry = registry
        self.http_requests = Counter(
            "legal_rag_http_requests_total",
            "Completed HTTP requests by method, registered route, and status.",
            ("method", "route", "status"),
            registry=registry,
        )
        self.http_request_latency = Histogram(
            "legal_rag_http_request_duration_seconds",
            "End-to-end HTTP response duration, including streamed response bodies.",
            ("method", "route"),
            registry=registry,
        )
        self.http_errors = Counter(
            "legal_rag_http_errors_total",
            "Completed HTTP responses with status 400 or higher.",
            ("method", "route", "status"),
            registry=registry,
        )
        self.ask_requests = Counter(
            "legal_rag_ask_requests_total",
            "Legal answer requests by full-response or streaming mode.",
            ("mode",),
            registry=registry,
        )
        self.retrieval_latency = Histogram(
            "legal_rag_retrieval_duration_seconds",
            "Time spent retrieving legal context for one RAG request.",
            registry=registry,
        )
        self.generation_latency = Histogram(
            "legal_rag_generation_duration_seconds",
            "Provider generation duration by provider and response mode.",
            ("provider", "mode"),
            registry=registry,
        )
        self.retrieved_sources = Histogram(
            "legal_rag_retrieved_sources",
            "Number of legal sources returned by retrieval per RAG request.",
            buckets=(0, 1, 2, 3, 5, 8, 10, 20),
            registry=registry,
        )
        self.provider_failures = Counter(
            "legal_rag_llm_provider_failures_total",
            "LLM provider generation failures by provider and response mode.",
            ("provider", "mode"),
            registry=registry,
        )
        self.llm_tokens = Counter(
            "legal_rag_llm_tokens_total",
            "Authoritative provider-reported tokens by type and response mode.",
            ("provider", "token_type", "mode"),
            registry=registry,
        )
        self.llm_usage_cost_usd = Counter(
            "legal_rag_llm_usage_cost_usd_total",
            "Usage cost derived from authoritative tokens and configured USD rates.",
            ("provider", "mode"),
            registry=registry,
        )
        self.pii_redactions = Counter(
            "legal_rag_pii_redactions_total",
            "Generated-answer PII values redacted by category and response mode.",
            ("category", "mode"),
            registry=registry,
        )
        self.query_drift_mean_similarity = Gauge(
            "legal_rag_query_drift_mean_cosine_similarity",
            "Latest current-batch mean cosine similarity to the reference centroid.",
            ("population",),
            registry=registry,
        )
        self.query_drift_min_similarity = Gauge(
            "legal_rag_query_drift_min_cosine_similarity",
            "Latest current-batch minimum cosine similarity to the reference centroid.",
            ("population",),
            registry=registry,
        )
        self.query_drift_p05_similarity = Gauge(
            "legal_rag_query_drift_p05_cosine_similarity",
            "Latest current-batch fifth-percentile cosine similarity.",
            ("population",),
            registry=registry,
        )
        self.query_drift_queries = Gauge(
            "legal_rag_query_drift_queries",
            "Number of queries in the latest evaluated current batch.",
            ("population",),
            registry=registry,
        )
        self.query_drift_threshold = Gauge(
            "legal_rag_query_drift_similarity_threshold",
            "Configured operational mean-similarity threshold.",
            ("policy",),
            registry=registry,
        )
        self.query_drift_detected = Gauge(
            "legal_rag_query_drift_detected",
            "Whether the latest mean similarity is below the configured threshold.",
            ("policy",),
            registry=registry,
        )

    @staticmethod
    def _safe(operation: Callable[[], None]) -> None:
        try:
            operation()
        except Exception:
            logger.exception("Prometheus metric recording failed")

    def record_http(
        self,
        method: str,
        route: str,
        status: int,
        duration_seconds: float,
    ) -> None:
        """Record one completed HTTP response without content-derived labels."""

        status_label = str(status)

        def operation() -> None:
            self.http_requests.labels(method, route, status_label).inc()
            self.http_request_latency.labels(method, route).observe(duration_seconds)
            if status >= 400:
                self.http_errors.labels(method, route, status_label).inc()

        self._safe(operation)

    def record_ask(self, mode: str) -> None:
        self._safe(lambda: self.ask_requests.labels(mode).inc())

    def record_retrieval(
        self,
        duration_seconds: float,
        source_count: int | None,
    ) -> None:
        def operation() -> None:
            self.retrieval_latency.observe(duration_seconds)
            if source_count is not None:
                self.retrieved_sources.observe(source_count)

        self._safe(operation)

    def record_generation(
        self, provider: str, mode: str, duration_seconds: float
    ) -> None:
        self._safe(
            lambda: self.generation_latency.labels(
                _provider_label(provider), mode
            ).observe(duration_seconds)
        )

    def record_provider_failure(self, provider: str, mode: str) -> None:
        self._safe(
            lambda: self.provider_failures.labels(_provider_label(provider), mode).inc()
        )

    def record_token_usage(
        self,
        provider: str,
        *,
        mode: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        input_cost_per_million_tokens_usd: float | None = None,
        output_cost_per_million_tokens_usd: float | None = None,
    ) -> None:
        """Record trustworthy provider-reported usage when adapters expose it."""

        values = (input_tokens, output_tokens, total_tokens)
        if mode not in _RESPONSE_MODES or any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in values
        ):
            return

        def operation() -> None:
            label = _provider_label(provider)
            for token_type, value in zip(
                ("input", "output", "total"), values, strict=True
            ):
                if value is not None:
                    self.llm_tokens.labels(label, token_type, mode).inc(value)
            if (
                input_tokens is not None
                and output_tokens is not None
                and input_cost_per_million_tokens_usd is not None
                and output_cost_per_million_tokens_usd is not None
                and isfinite(input_cost_per_million_tokens_usd)
                and isfinite(output_cost_per_million_tokens_usd)
                and input_cost_per_million_tokens_usd >= 0
                and output_cost_per_million_tokens_usd >= 0
                and (
                    total_tokens is None or total_tokens == input_tokens + output_tokens
                )
            ):
                cost = (
                    input_tokens * input_cost_per_million_tokens_usd
                    + output_tokens * output_cost_per_million_tokens_usd
                ) / 1_000_000
                self.llm_usage_cost_usd.labels(label, mode).inc(cost)

        self._safe(operation)

    def record_pii_redactions(self, category: str, mode: str, count: int) -> None:
        """Record bounded guardrail categories without retaining detected values."""

        if category not in _PII_CATEGORIES or mode not in _RESPONSE_MODES or count <= 0:
            return
        self._safe(lambda: self.pii_redactions.labels(category, mode).inc(count))

    def record_query_drift(
        self,
        *,
        mean_similarity: float,
        minimum_similarity: float,
        p05_similarity: float,
        query_count: int,
        threshold: float | None = None,
        drift_detected: bool | None = None,
    ) -> None:
        """Record one calculated drift report without query-derived labels."""

        if query_count <= 0:
            return
        if any(
            not -1 <= value <= 1
            for value in (mean_similarity, minimum_similarity, p05_similarity)
        ):
            return
        if (threshold is None) != (drift_detected is None):
            return

        def operation() -> None:
            population = "current_batch"
            self.query_drift_mean_similarity.labels(population).set(mean_similarity)
            self.query_drift_min_similarity.labels(population).set(minimum_similarity)
            self.query_drift_p05_similarity.labels(population).set(p05_similarity)
            self.query_drift_queries.labels(population).set(query_count)
            if threshold is not None and drift_detected is not None:
                policy = "configured_mean_similarity"
                self.query_drift_threshold.labels(policy).set(threshold)
                self.query_drift_detected.labels(policy).set(1 if drift_detected else 0)

        self._safe(operation)


_metrics = PrometheusMetrics()


def get_metrics() -> PrometheusMetrics:
    """Return the process-wide metrics recorder used by all serving adapters."""

    return _metrics


class PrometheusMiddleware:
    """Measure complete ASGI HTTP responses, including SSE streams."""

    def __init__(
        self,
        app: Any,
        metrics: PrometheusMetrics,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.app = app
        self.metrics = metrics
        self.clock = clock

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = self.clock()
        status = 500
        recorded = False

        def route_label() -> str:
            route = scope.get("route")
            path = getattr(route, "path", None)
            return path if isinstance(path, str) else "unmatched"

        def record() -> None:
            nonlocal recorded
            if recorded:
                return
            recorded = True
            self.metrics.record_http(
                scope.get("method", "UNKNOWN"),
                route_label(),
                status,
                self.clock() - started,
            )

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                record()

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            record()
            raise
        finally:
            if not recorded:
                record()
