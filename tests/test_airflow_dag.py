"""Static tests for the Airflow DAG without requiring Airflow installation."""

import ast
from pathlib import Path

DAG_PATH = Path("orchestration/dags/legal_rag_pipeline.py")
EXPECTED_TASK_IDS = (
    "validate_corpus",
    "validate_evaluation_dataset",
    "rebuild_vector_index",
    "run_retrieval_evaluation",
    "run_rag_evaluation",
)


def _source() -> str:
    return DAG_PATH.read_text(encoding="utf-8")


def test_airflow_dag_is_valid_python_with_expected_tasks() -> None:
    source = _source()

    ast.parse(source)
    for task_id in EXPECTED_TASK_IDS:
        assert f'task_id="{task_id}"' in source


def test_airflow_dag_is_manual_and_has_safe_retry_policy() -> None:
    source = _source()

    assert "schedule=None" in source
    assert "catchup=False" in source
    assert '"retries": 1' in source
    assert "timedelta(minutes=5)" in source


def test_airflow_dag_calls_existing_project_entrypoints() -> None:
    source = _source()

    assert "python -m legal_rag.evaluation.summary" in source
    assert "python -m legal_rag.evaluation.runner" in source
    assert "python -m legal_rag.evaluation.end_to_end" in source
    assert "index_corpus" in source


def test_python_logic_is_not_embedded_in_bash_commands() -> None:
    source = _source()

    assert "python -c" not in source
    assert "validate_corpus = PythonOperator(" in source
    assert "python_callable=validate_corpus_callable" in source
    assert "rebuild_vector_index = PythonOperator(" in source
    assert "python_callable=rebuild_vector_index_callable" in source
    assert "validate_evaluation_dataset = BashOperator(" in source
    assert "run_retrieval_evaluation = BashOperator(" in source
    assert "run_rag_evaluation = BashOperator(" in source


def test_project_root_is_derived_from_dag_location() -> None:
    source = _source()

    assert "PROJECT_ROOT = Path(__file__).resolve().parents[2]" in source
    assert "/home/" not in source
    assert "/Users/" not in source


def test_python_tasks_receive_root_and_resolve_corpus_path() -> None:
    source = _source()

    assert "def resolve_project_path(" in source
    assert "path if path.is_absolute()" in source
    assert "resolve_project_path(config.corpus_path, Path(project_root))" in source
    assert source.count('op_kwargs={"project_root": str(PROJECT_ROOT)}') == 2


def test_every_bash_task_uses_project_root_as_working_directory() -> None:
    source = _source()

    assert source.count("cwd=str(PROJECT_ROOT)") == 3


def test_live_rag_evaluation_is_opt_in_and_dependency_order_is_explicit() -> None:
    source = _source()

    assert "ENABLE_LIVE_RAG_EVALUATION:-false" in source
    assert "validate_corpus\n        >> validate_evaluation_dataset" in source
    assert "validate_evaluation_dataset\n        >> rebuild_vector_index" in source
    assert "rebuild_vector_index\n        >> run_retrieval_evaluation" in source
    assert "run_retrieval_evaluation\n        >> run_rag_evaluation" in source
