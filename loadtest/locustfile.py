"""Locust scenarios for controlled or explicitly selected real deployments."""

from __future__ import annotations

from itertools import cycle
from time import perf_counter

from locust import HttpUser, between, events, task

from loadtest.helpers import summarize_sse_lines

QUESTIONS = (
    "متى يكون الشخص مسؤولاً عن التعويض؟",
    "ما هي شروط صحة العقد؟",
    "ما هي أحكام الإيجار؟",
)
_questions = cycle(QUESTIONS)


class LegalRAGUser(HttpUser):
    """Exercise health, complete-answer, and streaming API paths."""

    wait_time = between(1, 3)

    @task(1)
    def health(self) -> None:
        with self.client.get(
            "/health", name="/health", catch_response=True
        ) as response:
            if response.status_code != 200 or response.json().get("status") != "ok":
                response.failure("unexpected health response")

    @task(2)
    def ask(self) -> None:
        payload = {"question": next(_questions), "top_k": 5}
        with self.client.post(
            "/ask", json=payload, name="/ask", catch_response=True
        ) as response:
            body = response.json()
            if response.status_code != 200:
                response.failure(f"unexpected status {response.status_code}")
            elif not body.get("answer") or not body.get("sources"):
                response.failure("answer or sources missing")

    @task(2)
    def ask_stream(self) -> None:
        payload = {"question": next(_questions), "top_k": 5}
        started = perf_counter()
        first_event_ms: float | None = None
        lines: list[str] = []
        error: Exception | None = None

        with self.client.post(
            "/ask/stream",
            json=payload,
            name="/ask/stream headers",
            stream=True,
            catch_response=True,
        ) as response:
            try:
                if response.status_code != 200:
                    raise ValueError(f"unexpected status {response.status_code}")
                for line in response.iter_lines(decode_unicode=True):
                    if line and first_event_ms is None:
                        first_event_ms = (perf_counter() - started) * 1000
                    if line:
                        lines.append(line)
                summary = summarize_sse_lines(lines)
                if not summary.is_complete:
                    raise ValueError(f"incomplete SSE sequence: {summary.events}")
                response.success()
            except Exception as exc:
                error = exc
                response.failure(str(exc))

        completed_ms = (perf_counter() - started) * 1000
        if first_event_ms is not None:
            events.request.fire(
                request_type="SSE",
                name="/ask/stream first event",
                response_time=first_event_ms,
                response_length=0,
                exception=error,
                context={},
            )
        events.request.fire(
            request_type="SSE",
            name="/ask/stream full response",
            response_time=completed_ms,
            response_length=sum(len(line.encode("utf-8")) for line in lines),
            exception=error,
            context={},
        )
