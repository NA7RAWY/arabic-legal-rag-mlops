"""Deterministic retrieval metrics for corpus-grounded evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CaseRetrievalMetrics:
    """Retrieval metrics for one evaluation case at a fixed cutoff."""

    hit_rate_at_k: float
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class AggregateRetrievalMetrics:
    """Mean retrieval metrics across an evaluation dataset."""

    hit_rate_at_k: float
    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float


def calculate_case_metrics(
    retrieved_article_numbers: list[int] | tuple[int, ...],
    relevant_article_numbers: list[int] | tuple[int, ...],
    *,
    top_k: int,
) -> CaseRetrievalMetrics:
    """Calculate hit, recall, precision, and first-hit reciprocal rank."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not relevant_article_numbers:
        raise ValueError("relevant_article_numbers must not be empty")

    retrieved_at_k = tuple(retrieved_article_numbers[:top_k])
    relevant = set(relevant_article_numbers)
    retrieved_relevant = relevant.intersection(retrieved_at_k)
    first_relevant_rank = next(
        (
            rank
            for rank, article_number in enumerate(retrieved_at_k, start=1)
            if article_number in relevant
        ),
        None,
    )

    return CaseRetrievalMetrics(
        hit_rate_at_k=1.0 if retrieved_relevant else 0.0,
        recall_at_k=len(retrieved_relevant) / len(relevant),
        precision_at_k=len(retrieved_relevant) / top_k,
        reciprocal_rank=(
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        ),
    )


def aggregate_metrics(
    case_metrics: list[CaseRetrievalMetrics] | tuple[CaseRetrievalMetrics, ...],
) -> AggregateRetrievalMetrics:
    """Return the arithmetic mean of each per-case retrieval metric."""

    if not case_metrics:
        raise ValueError("case_metrics must not be empty")
    count = len(case_metrics)
    return AggregateRetrievalMetrics(
        hit_rate_at_k=sum(metric.hit_rate_at_k for metric in case_metrics) / count,
        recall_at_k=sum(metric.recall_at_k for metric in case_metrics) / count,
        precision_at_k=(sum(metric.precision_at_k for metric in case_metrics) / count),
        mean_reciprocal_rank=(
            sum(metric.reciprocal_rank for metric in case_metrics) / count
        ),
    )
