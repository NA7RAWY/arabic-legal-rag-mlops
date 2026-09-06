"""Evaluation dataset models and validation."""

from legal_rag.evaluation.dataset import (
    EvaluationCase,
    EvaluationDataset,
    SupportingExcerpt,
    load_evaluation_dataset,
)
from legal_rag.evaluation.retrieval_metrics import (
    AggregateRetrievalMetrics,
    CaseRetrievalMetrics,
    aggregate_metrics,
    calculate_case_metrics,
)

__all__ = [
    "EvaluationCase",
    "EvaluationDataset",
    "SupportingExcerpt",
    "AggregateRetrievalMetrics",
    "CaseRetrievalMetrics",
    "aggregate_metrics",
    "calculate_case_metrics",
    "load_evaluation_dataset",
]
