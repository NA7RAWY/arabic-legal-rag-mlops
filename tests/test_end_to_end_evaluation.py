"""Unit tests for generation and RAGAS evaluation orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import legal_rag.evaluation.end_to_end as module
from legal_rag.config import AppConfig
from legal_rag.evaluation.dataset import EvaluationCase, EvaluationDataset
from legal_rag.evaluation.end_to_end import (
    evaluate_end_to_end,
    run_end_to_end_evaluation,
)
from legal_rag.evaluation.ragas_evaluator import (
    GeminiRagasEvaluator,
    RagasEvaluationError,
    RagasScores,
    normalize_metric_value,
)
from legal_rag.storage import RetrievalResult
from legal_rag.tracking import RAGEvaluationMetric, RAGExperimentConfig


def _case(case_id: str, question: str, relevant: tuple[int, ...]) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        question=question,
        reference_answer=f"reference-{case_id}",
        relevant_article_numbers=relevant,
        category="test",
        language="ar",
        source_rationale="test",
        supporting_excerpts=(),
    )


def _result(article: int, text: str, similarity: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"article-{article}",
        article_number=article,
        text=text,
        language="ar",
        book=None,
        chapter=None,
        section=None,
        topic=None,
        is_repealed=False,
        source_page=1,
        citation=f"المادة {article}",
        similarity=similarity,
    )


class FakeRetriever:
    def __init__(self, results: dict[str, list[RetrievalResult]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.results[query]


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(self, question: str, context: str) -> str:
        self.calls.append((question, context))
        return f"answer-{question}"


class FakeEvaluator:
    def __init__(self, scores: RagasScores | None = None) -> None:
        self.scores = scores or RagasScores(0.9, 0.8, 0.7, 0.6)
        self.calls: list[dict[str, Any]] = []

    def evaluate(self, **kwargs: Any) -> RagasScores:
        self.calls.append(kwargs)
        return self.scores


def test_end_to_end_orchestration_preserves_order_and_context() -> None:
    dataset = EvaluationDataset(
        "test-v1", (_case("b", "q-b", (20,)), _case("a", "q-a", (10,)))
    )
    retriever = FakeRetriever(
        {
            "q-b": [_result(99, "ctx-99", 0.9), _result(20, "ctx-20", 0.8)],
            "q-a": [_result(10, "ctx-10", 0.95)],
        }
    )
    generator = FakeGenerator()
    evaluator = FakeEvaluator()

    report = evaluate_end_to_end(
        dataset,
        retriever=retriever,
        generator=generator,
        evaluator=evaluator,
        dataset_path=Path("eval.json"),
        top_k=2,
        corpus_version="sha256:x",
        git_commit="abc",
        git_dirty=True,
        generation_model="generator",
        evaluation_model="judge",
    )

    assert [case.case_id for case in report.cases] == ["b", "a"]
    assert report.cases[0].retrieved_article_numbers == (99, 20)
    assert evaluator.calls[0]["contexts"] == ["ctx-99", "ctx-20"]
    assert "Article number: 99" in generator.calls[0][1]
    assert "Citation: المادة 20" in generator.calls[0][1]
    assert len(generator.calls) == 2
    assert report.aggregate_retrieval_metrics.hit_rate_at_k == 1.0
    assert report.aggregate_ragas_metrics.faithfulness == 0.9


@pytest.mark.parametrize("value", [None, "bad", float("nan"), -0.1, 1.1, True])
def test_metric_normalization_rejects_invalid_values(value: Any) -> None:
    with pytest.raises(RagasEvaluationError):
        normalize_metric_value(value, "faithfulness")


def test_metric_normalization_accepts_ragas_result_shape() -> None:
    assert normalize_metric_value(SimpleNamespace(value=0.75), "metric") == 0.75


class AsyncMetric:
    def __init__(self, result: Any = 0.5, error: Exception | None = None) -> None:
        self.result = result
        self.error = error

    async def ascore(self, **kwargs: Any) -> Any:
        if self.error:
            raise self.error
        return self.result


def test_provider_error_is_wrapped() -> None:
    metrics = {
        "faithfulness": AsyncMetric(error=RuntimeError("provider detail")),
        "answer_relevancy": AsyncMetric(),
        "context_recall": AsyncMetric(),
        "context_precision": AsyncMetric(),
    }
    evaluator = GeminiRagasEvaluator(
        api_key="test-key", model_name="judge", embedder=object(), metrics=metrics
    )
    with pytest.raises(RagasEvaluationError, match="judge evaluation failed"):
        evaluator.evaluate(
            question="q", answer="a", contexts=["c"], reference_answer="r"
        )


@dataclass
class FakeRunInfo:
    run_id: str


class FakeTracker:
    def __init__(self) -> None:
        self.experiment: RAGExperimentConfig | None = None
        self.metrics: dict[str, float] = {}
        self.artifacts: set[str] = set()
        self.prompt = ""

    def start_run(self, run_name: str | None = None) -> nullcontext[Any]:
        return nullcontext(SimpleNamespace(info=FakeRunInfo("e2e-run")))

    def log_experiment_config(self, experiment: RAGExperimentConfig) -> None:
        self.experiment = experiment

    def log_evaluation_metrics(
        self,
        metrics: Mapping[RAGEvaluationMetric | str, float],
        *,
        step: int | None = None,
    ) -> None:
        self.metrics = {str(key): value for key, value in metrics.items()}

    def log_config_bundle(
        self, bundle: Mapping[str, Any], artifact_file: str = "config/rag_config.json"
    ) -> None:
        self.artifacts.add(artifact_file)

    def log_prompt(
        self, prompt: str, artifact_file: str = "prompts/system_prompt.txt"
    ) -> None:
        self.prompt = prompt
        self.artifacts.add(artifact_file)


def test_mlflow_integration_logs_namespaced_metrics_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = EvaluationDataset("eval-v1", (_case("1", "q", (10,)),))
    monkeypatch.setattr(module, "load_evaluation_dataset", lambda *a, **k: dataset)
    monkeypatch.setattr(module, "corpus_sha256", lambda path: "hash")
    tracker = FakeTracker()
    outcome = run_end_to_end_evaluation(
        AppConfig(retrieval_top_k=1, evaluation_model="judge"),
        retriever=FakeRetriever({"q": [_result(10, "ctx", 0.9)]}),
        generator=FakeGenerator(),
        evaluator=FakeEvaluator(),
        tracker=tracker,
        git_commit="abc",
        git_dirty=True,
    )

    assert outcome.run_id == "e2e-run"
    assert tracker.experiment is not None
    assert tracker.experiment.evaluator_model == "judge"
    assert tracker.experiment.run_purpose == "module2-end-to-end-eval"
    assert set(tracker.metrics) == {
        "retrieval_hit_rate_at_k",
        "retrieval_recall_at_k",
        "retrieval_precision_at_k",
        "retrieval_mrr",
        "ragas_faithfulness",
        "ragas_answer_relevancy",
        "ragas_context_recall",
        "ragas_context_precision",
    }
    assert tracker.artifacts == {
        "evaluation/end_to_end_report.json",
        "config/rag_config.json",
        "config/evaluation_config.json",
        "prompts/system_prompt.txt",
    }
