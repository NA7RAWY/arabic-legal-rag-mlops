"""Tests for deterministic retrieval metrics and evaluation orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import legal_rag.evaluation.runner as runner_module
from legal_rag.config import AppConfig
from legal_rag.evaluation.dataset import EvaluationCase, EvaluationDataset
from legal_rag.evaluation.retrieval_metrics import (
    CaseRetrievalMetrics,
    aggregate_metrics,
    calculate_case_metrics,
)
from legal_rag.evaluation.runner import evaluate_retrieval, run_retrieval_evaluation
from legal_rag.storage import RetrievalResult
from legal_rag.tracking import RAGEvaluationMetric, RAGExperimentConfig


def retrieval_result(article_number: int, similarity: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"article-{article_number}",
        article_number=article_number,
        text=f"Article {article_number}",
        language="ar",
        book=None,
        chapter=None,
        section=None,
        topic=None,
        is_repealed=False,
        source_page=1,
        citation=f"Article {article_number}",
        similarity=similarity,
    )


def evaluation_case(
    case_id: str,
    question: str,
    relevant: tuple[int, ...],
) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        question=question,
        reference_answer="Reference",
        relevant_article_numbers=relevant,
        category="test",
        language="ar",
        source_rationale="Test rationale",
        supporting_excerpts=(),
    )


class FakeRetriever:
    def __init__(self, results: Mapping[str, list[RetrievalResult]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int | None]] = []

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.results[query]


def test_perfect_retrieval_metrics() -> None:
    metrics = calculate_case_metrics([10, 20], [10, 20], top_k=2)

    assert metrics == CaseRetrievalMetrics(1.0, 1.0, 1.0, 1.0)


def test_partial_multi_article_retrieval_metrics() -> None:
    metrics = calculate_case_metrics([99, 20, 30], [10, 20], top_k=3)

    assert metrics.hit_rate_at_k == 1.0
    assert metrics.recall_at_k == 0.5
    assert metrics.precision_at_k == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank == 0.5


def test_no_relevant_retrieval_metrics() -> None:
    metrics = calculate_case_metrics([1, 2, 3], [10], top_k=3)

    assert metrics == CaseRetrievalMetrics(0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("retrieved", "expected"),
    [
        ([10, 20, 30], 1.0),
        ([1, 10, 30], 0.5),
        ([1, 2, 10], 1 / 3),
        ([1, 2, 3], 0.0),
    ],
)
def test_reciprocal_rank_uses_first_relevant_result(
    retrieved: list[int],
    expected: float,
) -> None:
    metrics = calculate_case_metrics(retrieved, [10, 20], top_k=3)

    assert metrics.reciprocal_rank == pytest.approx(expected)


def test_metric_aggregation_is_arithmetic_mean() -> None:
    aggregate = aggregate_metrics(
        [
            CaseRetrievalMetrics(1.0, 1.0, 0.5, 1.0),
            CaseRetrievalMetrics(0.0, 0.0, 0.0, 0.0),
        ]
    )

    assert aggregate.hit_rate_at_k == 0.5
    assert aggregate.recall_at_k == 0.5
    assert aggregate.precision_at_k == 0.25
    assert aggregate.mean_reciprocal_rank == 0.5


def test_evaluation_preserves_case_and_result_order() -> None:
    dataset = EvaluationDataset(
        version="test-v1",
        cases=(
            evaluation_case("case-b", "question-b", (20,)),
            evaluation_case("case-a", "question-a", (10,)),
        ),
    )
    retriever = FakeRetriever(
        {
            "question-b": [retrieval_result(99), retrieval_result(20, 0.8)],
            "question-a": [retrieval_result(10), retrieval_result(30, 0.7)],
        }
    )

    report = evaluate_retrieval(
        dataset,
        retriever,
        dataset_path=Path("evaluation.json"),
        top_k=2,
    )

    assert [case.case_id for case in report.cases] == ["case-b", "case-a"]
    assert report.cases[0].retrieved_article_numbers == (99, 20)
    assert [item.rank for item in report.cases[0].retrieved] == [1, 2]
    assert retriever.calls == [("question-b", 2), ("question-a", 2)]


@dataclass
class FakeRunInfo:
    run_id: str


@dataclass
class FakeRun:
    info: FakeRunInfo


class FakeTracker:
    def __init__(self) -> None:
        self.run_name: str | None = None
        self.experiment_config: RAGExperimentConfig | None = None
        self.metrics: dict[str, float] = {}
        self.artifacts: dict[str, Mapping[str, Any]] = {}
        self.prompt: str | None = None

    def start_run(self, run_name: str | None = None) -> nullcontext[FakeRun]:
        self.run_name = run_name
        return nullcontext(FakeRun(FakeRunInfo("retrieval-run-id")))

    def log_experiment_config(self, experiment: RAGExperimentConfig) -> None:
        self.experiment_config = experiment

    def log_evaluation_metrics(
        self,
        metrics: Mapping[RAGEvaluationMetric | str, float],
        *,
        step: int | None = None,
    ) -> None:
        self.metrics = {str(name): value for name, value in metrics.items()}

    def log_config_bundle(
        self,
        bundle: Mapping[str, Any],
        artifact_file: str = "config/rag_config.json",
    ) -> None:
        self.artifacts[artifact_file] = bundle

    def log_prompt(
        self,
        prompt: str,
        artifact_file: str = "prompts/system_prompt.txt",
    ) -> None:
        self.prompt = prompt


def test_mlflow_orchestration_logs_metrics_params_tags_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = EvaluationDataset(
        version="eval-test-v1",
        cases=(evaluation_case("case-1", "question", (10,)),),
    )
    monkeypatch.setattr(
        runner_module,
        "load_evaluation_dataset",
        lambda path, *, corpus_path: dataset,
    )
    monkeypatch.setattr(runner_module, "corpus_sha256", lambda path: "deadbeef")
    retriever = FakeRetriever({"question": [retrieval_result(10)]})
    tracker = FakeTracker()
    config = AppConfig(retrieval_top_k=1)

    outcome = run_retrieval_evaluation(
        config,
        retriever=retriever,
        tracker=tracker,
        git_commit="abc123",
        git_dirty=True,
    )

    assert outcome.run_id == "retrieval-run-id"
    assert tracker.run_name == "module2-retrieval-eval"
    assert tracker.experiment_config is not None
    assert tracker.experiment_config.eval_dataset == "eval-test-v1"
    assert tracker.experiment_config.run_purpose == "module2-retrieval-eval"
    assert tracker.experiment_config.git_dirty is True
    assert tracker.metrics == {
        "retrieval_hit_rate_at_k": 1.0,
        "retrieval_recall_at_k": 1.0,
        "retrieval_precision_at_k": 1.0,
        "retrieval_mrr": 1.0,
    }
    assert set(tracker.artifacts) == {
        "evaluation/retrieval_report.json",
        "config/rag_config.json",
    }
    assert tracker.prompt is not None
