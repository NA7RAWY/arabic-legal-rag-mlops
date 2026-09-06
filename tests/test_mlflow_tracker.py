"""Unit tests for the MLflow tracking abstraction."""

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import pytest

from legal_rag.config import AppConfig
from legal_rag.tracking import (
    MLflowTracker,
    RAGEvaluationMetric,
    RAGExperimentConfig,
)


class FakeMLflow:
    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.run_name: str | None = None
        self.params: dict[str, Any] = {}
        self.tags: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.metric_step: int | None = None
        self.dict_artifact: tuple[Mapping[str, Any], str] | None = None
        self.text_artifact: tuple[str, str] | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def start_run(self, *, run_name: str | None = None) -> nullcontext[str]:
        self.run_name = run_name
        return nullcontext("run")

    def log_params(self, params: Mapping[str, Any]) -> None:
        self.params = dict(params)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self.tags = dict(tags)

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        self.metrics = dict(metrics)
        self.metric_step = step

    def log_dict(
        self,
        dictionary: Mapping[str, Any],
        artifact_file: str,
    ) -> None:
        self.dict_artifact = (dictionary, artifact_file)

    def log_text(self, text: str, artifact_file: str) -> None:
        self.text_artifact = (text, artifact_file)


def experiment_config() -> RAGExperimentConfig:
    return RAGExperimentConfig(
        chunk_strategy="one-article-per-chunk",
        chunk_size="article",
        chunk_overlap=0,
        retrieval_top_k=5,
        embedding_model="intfloat/multilingual-e5-small",
        generator_model="gemini-test",
        prompt_version="v1",
        git_commit="abc1234",
        corpus_version="civil-code-clean-v2",
    )


def test_mlflow_config_reads_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.test:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-rag")
    monkeypatch.setenv("MLFLOW_BACKEND_STORE_URI", "sqlite:////tmp/test.db")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", "/tmp/test-artifacts")

    config = AppConfig()

    assert config.mlflow_tracking_uri == "http://mlflow.test:5000"
    assert config.mlflow_experiment_name == "test-rag"
    assert config.mlflow_backend_store_uri == "sqlite:////tmp/test.db"
    assert config.mlflow_artifact_root == "/tmp/test-artifacts"


def test_start_run_configures_tracking_and_experiment() -> None:
    fake = FakeMLflow()
    config = AppConfig(
        mlflow_tracking_uri="http://tracker:5000",
        mlflow_experiment_name="test-experiment",
    )
    tracker = MLflowTracker(config, mlflow_module=fake)

    with tracker.start_run("baseline") as run:
        assert run == "run"

    assert fake.tracking_uri == "http://tracker:5000"
    assert fake.experiment_name == "test-experiment"
    assert fake.run_name == "baseline"


def test_log_experiment_config_separates_params_and_tags() -> None:
    fake = FakeMLflow()
    tracker = MLflowTracker(mlflow_module=fake)

    tracker.log_experiment_config(experiment_config())

    assert fake.params == {
        "chunk_strategy": "one-article-per-chunk",
        "chunk_size": "article",
        "chunk_overlap": 0,
        "retrieval_top_k": 5,
        "embedding_model": "intfloat/multilingual-e5-small",
        "generator_model": "gemini-test",
        "prompt_version": "v1",
        "corpus_version": "civil-code-clean-v2",
    }
    assert fake.tags == {
        "git_commit": "abc1234",
        "corpus_version": "civil-code-clean-v2",
        "framework": "custom-rag",
        "project_type": "arabic-legal-rag",
        "git_dirty": "false",
    }


def test_log_supported_evaluation_metrics() -> None:
    fake = FakeMLflow()
    tracker = MLflowTracker(mlflow_module=fake)

    tracker.log_evaluation_metrics(
        {
            RAGEvaluationMetric.FAITHFULNESS: 0.8,
            RAGEvaluationMetric.CONTEXT_RECALL: 0.75,
        },
        step=2,
    )

    assert fake.metrics == {"faithfulness": 0.8, "context_recall": 0.75}
    assert fake.metric_step == 2


@pytest.mark.parametrize("metric", ["made_up_metric", "accuracy"])
def test_rejects_unsupported_evaluation_metric(metric: str) -> None:
    tracker = MLflowTracker(mlflow_module=FakeMLflow())

    with pytest.raises(ValueError, match="Unsupported RAG evaluation metric"):
        tracker.log_evaluation_metrics({metric: 1.0})


def test_rejects_non_finite_metric() -> None:
    tracker = MLflowTracker(mlflow_module=FakeMLflow())

    with pytest.raises(ValueError, match="must be finite"):
        tracker.log_evaluation_metrics({"answer_relevancy": float("nan")})


def test_logs_config_and_prompt_artifacts() -> None:
    fake = FakeMLflow()
    tracker = MLflowTracker(mlflow_module=fake)
    bundle = {"retrieval_top_k": 5, "prompt_version": "v1"}

    tracker.log_config_bundle(bundle)
    tracker.log_prompt("Use only retrieved context.")

    assert fake.dict_artifact == (bundle, "config/rag_config.json")
    assert fake.text_artifact == (
        "Use only retrieved context.",
        "prompts/system_prompt.txt",
    )


def test_rejects_blank_prompt_artifact() -> None:
    tracker = MLflowTracker(mlflow_module=FakeMLflow())

    with pytest.raises(ValueError, match="Prompt text must not be empty"):
        tracker.log_prompt("   ")
