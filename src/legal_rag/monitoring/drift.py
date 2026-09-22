"""Offline query-embedding drift analysis against the frozen evaluation set."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from legal_rag.config import get_config
from legal_rag.evaluation.dataset import (
    DEFAULT_EVALUATION_PATH,
    EvaluationDataset,
    load_evaluation_dataset,
)
from legal_rag.rag import SentenceTransformerEmbedder

if TYPE_CHECKING:
    from legal_rag.monitoring.metrics import PrometheusMetrics

DEFAULT_EMBEDDING_DIMENSION = 384


class QueryBatchEmbedder(Protocol):
    """Minimal normalized query-batch embedding contract."""

    model_name: str

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class SimilaritySummary:
    """Raw distribution summary for cosine similarities."""

    minimum: float
    p05: float
    median: float
    mean: float
    p95: float
    maximum: float


@dataclass(frozen=True, slots=True)
class QueryReference:
    """Reference centroid and its within-reference similarity distribution."""

    dataset_version: str
    query_count: int
    embedding_model: str
    embedding_dimension: int
    centroid: tuple[float, ...]
    similarity: SimilaritySummary

    def summary(self) -> dict[str, Any]:
        """Return inspectable metadata without serializing the full centroid."""

        return {
            "dataset_version": self.dataset_version,
            "query_count": self.query_count,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "representation": "l2_normalized_arithmetic_centroid",
            "similarity_to_centroid": asdict(self.similarity),
        }


@dataclass(frozen=True, slots=True)
class QueryDriftReport:
    """Similarity report for one current query population."""

    reference: QueryReference
    query_count: int
    similarity: SimilaritySummary
    mean_similarity_delta: float
    threshold: float | None
    drift_detected: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference.summary(),
            "current_batch": {
                "query_count": self.query_count,
                "similarity_to_reference_centroid": asdict(self.similarity),
                "mean_similarity_delta": self.mean_similarity_delta,
                "operational_threshold": self.threshold,
                "drift_detected": self.drift_detected,
            },
        }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity for equal, finite, non-zero vectors."""

    if not left or not right:
        raise ValueError("Cosine similarity requires non-empty vectors")
    if len(left) != len(right):
        raise ValueError("Cosine similarity vectors must have equal dimensions")
    if any(not math.isfinite(value) for value in (*left, *right)):
        raise ValueError("Cosine similarity vectors must contain finite values")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_similarities(values: list[float]) -> SimilaritySummary:
    """Return deterministic linearly interpolated distribution statistics."""

    if not values:
        raise ValueError("Similarity values must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Similarity values must be finite")
    ordered = sorted(values)
    return SimilaritySummary(
        minimum=ordered[0],
        p05=_percentile(ordered, 0.05),
        median=_percentile(ordered, 0.5),
        mean=sum(ordered) / len(ordered),
        p95=_percentile(ordered, 0.95),
        maximum=ordered[-1],
    )


def _validate_embeddings(
    embeddings: list[list[float]],
    expected_count: int,
    expected_dimension: int,
) -> None:
    if len(embeddings) != expected_count:
        raise ValueError(
            f"Expected {expected_count} embeddings, received {len(embeddings)}"
        )
    if expected_dimension <= 0:
        raise ValueError("Expected embedding dimension must be positive")
    for index, embedding in enumerate(embeddings):
        if len(embedding) != expected_dimension:
            raise ValueError(
                f"Embedding {index} has dimension {len(embedding)}; "
                f"expected {expected_dimension}"
            )
        if any(not math.isfinite(value) for value in embedding):
            raise ValueError(f"Embedding {index} contains a non-finite value")


def _normalized_centroid(embeddings: list[list[float]]) -> list[float]:
    dimension = len(embeddings[0])
    centroid = [
        sum(embedding[index] for embedding in embeddings) / len(embeddings)
        for index in range(dimension)
    ]
    norm = math.sqrt(sum(value * value for value in centroid))
    if norm == 0:
        raise ValueError("Reference embedding centroid must not be a zero vector")
    return [value / norm for value in centroid]


def build_query_reference(
    dataset: EvaluationDataset,
    embedder: QueryBatchEmbedder,
    *,
    expected_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
) -> QueryReference:
    """Build the reference directly from evaluation questions in dataset order."""

    questions = [case.question for case in dataset.cases]
    if not questions:
        raise ValueError("Reference evaluation dataset must contain questions")
    embeddings = embedder.embed_queries(questions)
    _validate_embeddings(embeddings, len(questions), expected_dimension)
    centroid = _normalized_centroid(embeddings)
    similarities = [cosine_similarity(embedding, centroid) for embedding in embeddings]
    return QueryReference(
        dataset_version=dataset.version,
        query_count=len(questions),
        embedding_model=embedder.model_name,
        embedding_dimension=expected_dimension,
        centroid=tuple(centroid),
        similarity=summarize_similarities(similarities),
    )


def evaluate_query_drift(
    queries: list[str],
    reference: QueryReference,
    embedder: QueryBatchEmbedder,
    *,
    threshold: float | None = None,
) -> QueryDriftReport:
    """Compare a current query batch with the frozen reference centroid."""

    if not queries:
        raise ValueError("Current query batch must not be empty")
    if any(not isinstance(query, str) or not query.strip() for query in queries):
        raise ValueError("Current queries must be non-empty strings")
    if threshold is not None and not -1 <= threshold <= 1:
        raise ValueError("Cosine similarity threshold must be between -1 and 1")

    embeddings = embedder.embed_queries([query.strip() for query in queries])
    _validate_embeddings(
        embeddings,
        len(queries),
        reference.embedding_dimension,
    )
    centroid = list(reference.centroid)
    similarities = [cosine_similarity(embedding, centroid) for embedding in embeddings]
    summary = summarize_similarities(similarities)
    return QueryDriftReport(
        reference=reference,
        query_count=len(queries),
        similarity=summary,
        mean_similarity_delta=summary.mean - reference.similarity.mean,
        threshold=threshold,
        drift_detected=None if threshold is None else summary.mean < threshold,
    )


def record_query_drift_metrics(
    report: QueryDriftReport,
    metrics: PrometheusMetrics | None = None,
) -> None:
    """Copy a calculated report into a long-running Prometheus registry."""

    from legal_rag.monitoring.metrics import get_metrics

    active_metrics = metrics if metrics is not None else get_metrics()
    active_metrics.record_query_drift(
        mean_similarity=report.similarity.mean,
        minimum_similarity=report.similarity.minimum,
        p05_similarity=report.similarity.p05,
        query_count=report.query_count,
        threshold=report.threshold,
        drift_detected=report.drift_detected,
    )


def _parse_query(value: Any, location: str) -> str:
    if isinstance(value, str):
        query = value
    elif isinstance(value, dict) and set(value) == {"query"}:
        query = value["query"]
    else:
        raise ValueError(f"{location} must be a string or an object with only 'query'")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{location} query must be a non-empty string")
    return query.strip()


def load_query_batch(path: Path) -> list[str]:
    """Load JSON array/object or JSONL query records from disk."""

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Current query batch file must not be empty")

    if path.suffix.casefold() == ".jsonl":
        values = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL query at line {line_number}: {exc}"
                ) from exc
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid query batch JSON in {path}: {exc}") from exc
        if isinstance(payload, dict) and set(payload) == {"queries"}:
            values = payload["queries"]
        else:
            values = payload

    if not isinstance(values, list) or not values:
        raise ValueError("Current query batch must be a non-empty list")
    return [
        _parse_query(value, f"queries[{index}]") for index, value in enumerate(values)
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Current query JSON/JSONL batch")
    parser.add_argument(
        "--evaluation-path",
        type=Path,
        default=DEFAULT_EVALUATION_PATH,
    )
    parser.add_argument("--corpus-path", type=Path)
    parser.add_argument("--model", help="Sentence Transformer model override")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional operational threshold for current mean cosine similarity",
    )
    parser.add_argument("--compact", action="store_true", help="Print compact JSON")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Inspect the frozen reference or evaluate a current query batch."""

    args = _parser().parse_args(argv)
    config = get_config()
    corpus_path = args.corpus_path or config.corpus_path
    dataset = load_evaluation_dataset(
        args.evaluation_path,
        corpus_path=corpus_path,
    )
    embedder = SentenceTransformerEmbedder(args.model or config.embedding_model)
    reference = build_query_reference(dataset, embedder)
    output: dict[str, Any]
    if args.input is None:
        output = {"reference": reference.summary()}
    else:
        output = evaluate_query_drift(
            load_query_batch(args.input),
            reference,
            embedder,
            threshold=args.threshold,
        ).as_dict()
    print(json.dumps(output, ensure_ascii=False, indent=None if args.compact else 2))


if __name__ == "__main__":
    main()
