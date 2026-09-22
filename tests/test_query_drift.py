"""Deterministic query-embedding drift tests without model downloads."""

import json
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry

from legal_rag.evaluation import EvaluationCase, EvaluationDataset
from legal_rag.monitoring import PrometheusMetrics
from legal_rag.monitoring.drift import (
    build_query_reference,
    cosine_similarity,
    evaluate_query_drift,
    load_query_batch,
    record_query_drift_metrics,
)


def _case(case_id: str, question: str) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        question=question,
        reference_answer="answer",
        relevant_article_numbers=(1,),
        category="category",
        language="ar",
        source_rationale="rationale",
        supporting_excerpts=(),
    )


def _dataset(*questions: str) -> EvaluationDataset:
    return EvaluationDataset(
        version="test-v1",
        cases=tuple(
            _case(f"case-{index}", question)
            for index, question in enumerate(questions, start=1)
        ),
    )


class FakeEmbedder:
    model_name = "fake-normalized-model"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.vectors[text] for text in texts]


def test_cosine_similarity_is_deterministic() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_reference_is_dataset_ordered_normalized_centroid_and_summary() -> None:
    dataset = _dataset("q1", "q2")
    embedder = FakeEmbedder({"q1": [1.0, 0.0], "q2": [0.0, 1.0]})

    reference = build_query_reference(dataset, embedder, expected_dimension=2)

    expected = 2**-0.5
    assert embedder.calls == [["q1", "q2"]]
    assert reference.query_count == 2
    assert reference.centroid == pytest.approx((expected, expected))
    assert reference.similarity.minimum == pytest.approx(expected)
    assert reference.similarity.mean == pytest.approx(expected)
    assert reference.summary()["representation"] == "l2_normalized_arithmetic_centroid"


def test_identical_population_matches_reference_distribution() -> None:
    dataset = _dataset("q1", "q2")
    embedder = FakeEmbedder({"q1": [1.0, 0.0], "q2": [0.0, 1.0]})
    reference = build_query_reference(dataset, embedder, expected_dimension=2)

    report = evaluate_query_drift(["q1", "q2"], reference, embedder)

    assert report.similarity == reference.similarity
    assert report.mean_similarity_delta == pytest.approx(0.0)
    assert report.threshold is None
    assert report.drift_detected is None


def test_different_population_has_lower_similarity_and_optional_status() -> None:
    dataset = _dataset("reference-1", "reference-2")
    embedder = FakeEmbedder(
        {
            "reference-1": [1.0, 0.0],
            "reference-2": [1.0, 0.0],
            "different": [-1.0, 0.0],
        }
    )
    reference = build_query_reference(dataset, embedder, expected_dimension=2)

    report = evaluate_query_drift(
        ["different"],
        reference,
        embedder,
        threshold=0.5,
    )

    assert report.similarity.mean == pytest.approx(-1.0)
    assert report.mean_similarity_delta == pytest.approx(-2.0)
    assert report.drift_detected is True


@pytest.mark.parametrize("queries", [[], [""], ["   "]])
def test_empty_or_invalid_current_batch_is_rejected(queries: list[str]) -> None:
    embedder = FakeEmbedder({"reference": [1.0, 0.0]})
    reference = build_query_reference(
        _dataset("reference"),
        embedder,
        expected_dimension=2,
    )

    with pytest.raises(ValueError, match="must not be empty|non-empty strings"):
        evaluate_query_drift(queries, reference, embedder)


def test_embedding_dimension_mismatch_is_rejected() -> None:
    embedder = FakeEmbedder({"reference": [1.0, 0.0], "current": [1.0]})
    reference = build_query_reference(
        _dataset("reference"),
        embedder,
        expected_dimension=2,
    )

    with pytest.raises(ValueError, match="dimension 1; expected 2"):
        evaluate_query_drift(["current"], reference, embedder)


def test_json_and_jsonl_query_batches_are_loaded(tmp_path: Path) -> None:
    json_path = tmp_path / "queries.json"
    json_path.write_text(
        json.dumps({"queries": ["first", {"query": "second"}]}),
        encoding="utf-8",
    )
    jsonl_path = tmp_path / "queries.jsonl"
    jsonl_path.write_text('"third"\n{"query": "fourth"}\n', encoding="utf-8")

    assert load_query_batch(json_path) == ["first", "second"]
    assert load_query_batch(jsonl_path) == ["third", "fourth"]


def test_prometheus_records_latest_report_without_query_labels() -> None:
    metrics = PrometheusMetrics(CollectorRegistry())
    embedder = FakeEmbedder({"reference": [1.0, 0.0], "current": [0.8, 0.6]})
    reference = build_query_reference(
        _dataset("reference"),
        embedder,
        expected_dimension=2,
    )
    report = evaluate_query_drift(["current"], reference, embedder, threshold=0.6)

    record_query_drift_metrics(report, metrics)

    labels = {"population": "current_batch"}
    assert metrics.registry.get_sample_value(
        "legal_rag_query_drift_mean_cosine_similarity", labels
    ) == pytest.approx(0.8)
    assert metrics.registry.get_sample_value(
        "legal_rag_query_drift_min_cosine_similarity", labels
    ) == pytest.approx(0.8)
    assert metrics.registry.get_sample_value(
        "legal_rag_query_drift_p05_cosine_similarity", labels
    ) == pytest.approx(0.8)
    assert metrics.registry.get_sample_value(
        "legal_rag_query_drift_queries", labels
    ) == pytest.approx(1)
    policy = {"policy": "configured_mean_similarity"}
    assert metrics.registry.get_sample_value(
        "legal_rag_query_drift_similarity_threshold", policy
    ) == pytest.approx(0.6)
    assert metrics.registry.get_sample_value(
        "legal_rag_query_drift_detected", policy
    ) == pytest.approx(0)
