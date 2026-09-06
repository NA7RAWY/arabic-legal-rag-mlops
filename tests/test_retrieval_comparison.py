"""Tests for deterministic retrieval top-k experiment comparison."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import legal_rag.evaluation.comparison as module
from legal_rag.config import AppConfig
from legal_rag.evaluation.comparison import (
    run_retrieval_comparison,
    summarize_comparison,
)
from legal_rag.evaluation.dataset import EvaluationCase, EvaluationDataset
from legal_rag.evaluation.retrieval_metrics import AggregateRetrievalMetrics
from legal_rag.evaluation.runner import RetrievalEvaluationReport
from legal_rag.storage import RetrievalResult
from legal_rag.tracking import RAGEvaluationMetric, RAGExperimentConfig


def _case(case_id: str, question: str, relevant: int) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        question=question,
        reference_answer="reference",
        relevant_article_numbers=(relevant,),
        category="test",
        language="ar",
        source_rationale="test",
        supporting_excerpts=(),
    )


def _result(article_number: int) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"article-{article_number}",
        article_number=article_number,
        text="text",
        language="ar",
        book=None,
        chapter=None,
        section=None,
        topic=None,
        is_repealed=False,
        source_page=1,
        citation="citation",
        similarity=0.9,
    )


class FakeRetriever:
    def __init__(self, ranked: dict[str, list[RetrievalResult]]) -> None:
        self.ranked = ranked
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        assert top_k is not None
        return self.ranked[query][:top_k]


@dataclass
class FakeRunInfo:
    run_id: str


class FakeTracker:
    def __init__(self) -> None:
        self.run_names: list[str] = []
        self.configs: list[RAGExperimentConfig] = []
        self.metrics: list[dict[str, float]] = []
        self.artifacts: list[tuple[str, Mapping[str, Any]]] = []

    def start_run(self, run_name: str | None = None) -> nullcontext[Any]:
        self.run_names.append(str(run_name))
        run_id = f"run-{len(self.run_names)}"
        return nullcontext(type("Run", (), {"info": FakeRunInfo(run_id)})())

    def log_experiment_config(self, experiment: RAGExperimentConfig) -> None:
        self.configs.append(experiment)

    def log_evaluation_metrics(
        self,
        metrics: Mapping[RAGEvaluationMetric | str, float],
        *,
        step: int | None = None,
    ) -> None:
        self.metrics.append({str(key): value for key, value in metrics.items()})

    def log_config_bundle(
        self,
        bundle: Mapping[str, Any],
        artifact_file: str = "config/rag_config.json",
    ) -> None:
        self.artifacts.append((artifact_file, bundle))


def _report(top_k: int, values: tuple[float, float, float, float]) -> Any:
    return RetrievalEvaluationReport(
        dataset_version="v1",
        dataset_path="eval.json",
        number_of_cases=1,
        retrieval_top_k=top_k,
        aggregate_metrics=AggregateRetrievalMetrics(*values),
        cases=(),
    )


def test_summary_selects_metric_leaders_and_smallest_tie() -> None:
    summary = summarize_comparison(
        (
            _report(3, (0.8, 0.6, 0.4, 0.7)),
            _report(5, (1.0, 0.8, 0.3, 0.9)),
            _report(8, (1.0, 1.0, 0.2, 0.9)),
        )
    )

    assert summary.top_k_values == (3, 5, 8)
    assert summary.best_hit_rate_top_k == 5
    assert summary.best_recall_top_k == 8
    assert summary.best_precision_top_k == 3
    assert summary.best_mrr_top_k == 5


@pytest.mark.parametrize("reports", [(), (_report(3, (1, 1, 1, 1)),) * 2])
def test_summary_rejects_empty_or_duplicate_configurations(
    reports: tuple[Any, ...],
) -> None:
    with pytest.raises(ValueError):
        summarize_comparison(reports)


def test_comparison_logs_three_ordered_runs_with_fixed_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = EvaluationDataset(
        version="eval-v1",
        cases=(
            _case("one", "q1", 10),
            _case("two", "q2", 20),
        ),
    )
    monkeypatch.setattr(module, "load_evaluation_dataset", lambda *a, **k: dataset)
    monkeypatch.setattr(
        module,
        "corpus_sha256",
        lambda path: "corpus-hash" if "corpus" in str(path) else "eval-hash",
    )
    retriever = FakeRetriever(
        {
            "q1": [_result(10), _result(99), _result(98)],
            "q2": [_result(99), _result(98), _result(20)],
        }
    )
    tracker = FakeTracker()
    config = AppConfig(corpus_path=Path("corpus.json"))

    result = run_retrieval_comparison(
        config,
        dataset_path=Path("evaluation.json"),
        top_k_values=(1, 2, 3),
        retriever=retriever,
        tracker=tracker,
        git_commit="abc",
        git_dirty=True,
    )

    assert [run.top_k for run in result.runs] == [1, 2, 3]
    assert [run.run_id for run in result.runs] == ["run-1", "run-2", "run-3"]
    assert [config.retrieval_top_k for config in tracker.configs] == [1, 2, 3]
    assert {config.corpus_version for config in tracker.configs} == {
        "sha256:corpus-hash"
    }
    assert {config.eval_dataset_hash for config in tracker.configs} == {
        "sha256:eval-hash"
    }
    assert {config.eval_dataset_sha256 for config in tracker.configs} == {"eval-hash"}
    assert {config.eval_cases for config in tracker.configs} == {2}
    assert all(config.run_purpose == module.RUN_PURPOSE for config in tracker.configs)
    assert all(
        set(metrics)
        == {
            "hit_rate_at_k",
            "mean_recall_at_k",
            "mean_precision_at_k",
            "mrr",
        }
        for metrics in tracker.metrics
    )
    assert [call[1] for call in retriever.calls] == [1, 1, 2, 2, 3, 3]
    assert len(tracker.artifacts) == 9
