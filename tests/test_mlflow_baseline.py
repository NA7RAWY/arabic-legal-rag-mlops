"""Tests for the configuration-only MLflow baseline runner."""

import hashlib
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legal_rag.config import AppConfig
from legal_rag.rag.generator import SYSTEM_INSTRUCTION
from legal_rag.tracking.baseline import (
    PROMPT_VERSION,
    RUN_NAME,
    build_baseline_config,
    corpus_sha256,
    run_baseline_experiment,
)
from legal_rag.tracking.mlflow_tracker import RAGExperimentConfig


@dataclass
class FakeRunInfo:
    run_id: str


@dataclass
class FakeRun:
    info: FakeRunInfo


class FakeBaselineTracker:
    def __init__(self) -> None:
        self.run_name: str | None = None
        self.experiment: RAGExperimentConfig | None = None
        self.bundle: Mapping[str, Any] | None = None
        self.prompt: str | None = None

    def start_run(self, run_name: str | None = None) -> nullcontext[FakeRun]:
        self.run_name = run_name
        return nullcontext(FakeRun(FakeRunInfo("baseline-run-id")))

    def log_experiment_config(self, experiment: RAGExperimentConfig) -> None:
        self.experiment = experiment

    def log_config_bundle(
        self,
        bundle: Mapping[str, Any],
        artifact_file: str = "config/rag_config.json",
    ) -> None:
        self.bundle = bundle

    def log_prompt(
        self,
        prompt: str,
        artifact_file: str = "prompts/system_prompt.txt",
    ) -> None:
        self.prompt = prompt


def test_corpus_sha256_hashes_file_content(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_bytes(b'[{"article_number": 1}]')

    assert corpus_sha256(corpus) == hashlib.sha256(corpus.read_bytes()).hexdigest()


def test_build_baseline_config_uses_real_application_values() -> None:
    config = AppConfig(
        retrieval_top_k=7,
        embedding_model="test-embedder",
        gemini_model="test-generator",
    )

    baseline = build_baseline_config(
        config,
        git_commit="abc123",
        git_dirty=True,
        corpus_version="deadbeef",
    )

    assert baseline.chunk_strategy == "one_article_per_chunk"
    assert baseline.chunk_size == "article_level_variable_length"
    assert baseline.chunk_overlap == 0
    assert baseline.retrieval_top_k == 7
    assert baseline.embedding_model == "test-embedder"
    assert baseline.generator_model == "test-generator"
    assert baseline.prompt_version == PROMPT_VERSION
    assert baseline.git_commit == "abc123"
    assert baseline.git_dirty is True
    assert baseline.corpus_version == "sha256:deadbeef"
    assert baseline.run_purpose == "module2-baseline"


def test_runner_logs_configuration_and_prompt_without_metrics(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text("[]", encoding="utf-8")
    config = AppConfig(
        corpus_path=corpus,
        mlflow_experiment_name="arabic-legal-rag-dev",
    )
    tracker = FakeBaselineTracker()

    run_id = run_baseline_experiment(
        config,
        tracker=tracker,
        git_commit="abc123",
        git_dirty=True,
    )

    assert run_id == "baseline-run-id"
    assert tracker.run_name == RUN_NAME
    assert tracker.experiment is not None
    assert tracker.experiment.git_dirty is True
    assert tracker.experiment.corpus_version == (
        f"sha256:{hashlib.sha256(b'[]').hexdigest()}"
    )
    assert tracker.bundle is not None
    assert tracker.bundle["corpus_path"] == str(corpus)
    assert tracker.bundle["mlflow_experiment_name"] == "arabic-legal-rag-dev"
    assert tracker.prompt == SYSTEM_INSTRUCTION
