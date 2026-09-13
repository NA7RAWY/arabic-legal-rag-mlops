"""Tests for dependency-free controlled load-test support."""

from fastapi.testclient import TestClient
from loadtest.controlled_app import app
from loadtest.helpers import SSESummary, summarize_sse_lines


def test_sse_summary_recognizes_complete_stream() -> None:
    summary = summarize_sse_lines(
        [
            "event: sources",
            'data: {"sources": []}',
            "event: token",
            'data: {"text": "answer"}',
            "event: done",
            "data: {}",
        ]
    )

    assert summary == SSESummary(("sources", "token", "done"))
    assert summary.is_complete


def test_sse_summary_rejects_error_or_incomplete_stream() -> None:
    assert not summarize_sse_lines(["event: sources", "event: error"]).is_complete
    assert not summarize_sse_lines(["event: sources", "event: token"]).is_complete


def test_controlled_app_exercises_real_ask_contract_without_external_services() -> None:
    response = TestClient(app).post(
        "/ask",
        json={"question": "متى يجب التعويض؟", "top_k": 5},
    )

    assert response.status_code == 200
    assert response.json()["answer"]
    assert response.json()["sources"][0]["article_number"] == 164


def test_controlled_app_exercises_real_streaming_contract() -> None:
    response = TestClient(app).post(
        "/ask/stream",
        json={"question": "متى يجب التعويض؟", "top_k": 5},
    )

    assert response.status_code == 200
    summary = summarize_sse_lines(response.text.splitlines())
    assert summary.is_complete
    assert response.text.count("event: token") == 3
