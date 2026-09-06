"""Tests for the controlled long-article chunking experiment."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from legal_rag.evaluation.chunking_comparison import (
    ArticleDeduplicatingRetriever,
    ChunkStructure,
    _log_strategy,
    metric_winners,
)
from legal_rag.evaluation.retrieval_metrics import AggregateRetrievalMetrics
from legal_rag.evaluation.runner import RetrievalEvaluationReport
from legal_rag.ingestion import LegalArticle
from legal_rag.ingestion.experimental_chunker import (
    CorpusLengthAnalysis,
    SplitChunkConfig,
    split_long_article,
)
from legal_rag.storage import RetrievalResult
from legal_rag.tracking import RAGExperimentConfig


def _article(text: str, number: int = 7) -> LegalArticle:
    return LegalArticle(
        article_number=number,
        book="book",
        chapter="chapter",
        section="section",
        topic="topic",
        text_ar=text,
        text_en="English",
        is_repealed=True,
        source_page=9,
        citation="citation",
    )


def test_short_article_remains_one_production_compatible_chunk() -> None:
    chunks = split_long_article(_article("قصير"), SplitChunkConfig())

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "article-7"
    assert chunks[0].text == "قصير"


def test_long_article_split_is_deterministic_overlapping_and_traceable() -> None:
    text = "".join(str(index % 10) for index in range(1100))
    article = _article(text)
    config = SplitChunkConfig(split_threshold=600, chunk_size=500, chunk_overlap=75)

    first = split_long_article(article, config)
    second = split_long_article(article, config)

    assert first == second
    assert [chunk.chunk_id for chunk in first] == [
        "article-7-part-1",
        "article-7-part-2",
        "article-7-part-3",
    ]
    assert first[0].text[-75:] == first[1].text[:75]
    assert all(chunk.article_number == article.article_number for chunk in first)
    assert all(chunk.book == article.book for chunk in first)
    assert all(chunk.is_repealed is True for chunk in first)
    assert all(chunk.citation == article.citation for chunk in first)


def _retrieval(article: int, chunk: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk,
        article_number=article,
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


class RankedRetriever:
    def __init__(self) -> None:
        self.requested_top_k: int | None = None

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        self.requested_top_k = top_k
        return [
            _retrieval(10, "article-10-part-1"),
            _retrieval(10, "article-10-part-2"),
            _retrieval(20, "article-20"),
            _retrieval(30, "article-30"),
        ]


def test_article_level_retrieval_deduplicates_chunks_in_rank_order() -> None:
    ranked = RankedRetriever()
    retriever = ArticleDeduplicatingRetriever(ranked, max_chunks_per_article=3)

    results = retriever.search("question", top_k=3)

    assert [result.article_number for result in results] == [10, 20, 30]
    assert retriever.duplicates_removed == 1
    assert ranked.requested_top_k == 9


def _report(
    top_k: int, metrics: AggregateRetrievalMetrics
) -> RetrievalEvaluationReport:
    return RetrievalEvaluationReport("v1", "eval.json", 1, top_k, metrics, ())


def test_metric_winners_preserve_ties() -> None:
    winners = metric_winners(
        _report(5, AggregateRetrievalMetrics(1.0, 0.8, 0.2, 0.9)),
        _report(5, AggregateRetrievalMetrics(1.0, 0.9, 0.1, 0.8)),
    )

    assert winners == {
        "hit_rate": "tie",
        "recall": "split_long_articles",
        "precision": "one_article_per_chunk",
        "mrr": "one_article_per_chunk",
    }


@dataclass
class _RunInfo:
    run_id: str


class FakeTracker:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.artifacts: list[str] = []

    def start_run(self, run_name: str | None = None) -> nullcontext[Any]:
        return nullcontext(type("Run", (), {"info": _RunInfo("run-id")})())

    def log_experiment_config(self, experiment: RAGExperimentConfig) -> None:
        self.experiment = experiment

    def log_evaluation_metrics(self, metrics: Any) -> None:
        self.metrics = {str(key): value for key, value in metrics.items()}

    def log_config_bundle(self, bundle: Any, artifact_file: str) -> None:
        self.artifacts.append(artifact_file)


def test_strategy_logging_records_quality_structure_and_artifacts() -> None:
    tracker = FakeTracker()
    config = RAGExperimentConfig(
        "split_long_articles",
        500,
        75,
        5,
        "e5",
        "info-only",
        "v1",
        "commit",
        "sha256:corpus",
        split_threshold=600,
    )
    report = _report(5, AggregateRetrievalMetrics(1.0, 0.8, 0.2, 0.9))
    structure = ChunkStructure(1181, 30, 1181 / 1149, 3)
    analysis = CorpusLengthAnalysis(1149, 27, 205, 303, 424, 513, 1408, 30, 2.61)

    run_id = _log_strategy(
        tracker=tracker,
        experiment_config=config,
        report=report,
        structure=structure,
        analysis=analysis,
        split_config=SplitChunkConfig(),
        comparison={"winner": "baseline"},
    )

    assert run_id == "run-id"
    assert tracker.metrics["total_chunks"] == 1181
    assert tracker.metrics["articles_split"] == 30
    assert tracker.artifacts == [
        "config/chunking.json",
        "evaluation/retrieval_report.json",
        "analysis/corpus_lengths.json",
        "evaluation/comparison_summary.json",
    ]
