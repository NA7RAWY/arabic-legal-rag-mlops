"""Dependency-free validation helpers for streamed load-test responses."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SSESummary:
    """Relevant event names observed in one complete SSE response."""

    events: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return (
            bool(self.events)
            and self.events[0] == "sources"
            and "token" in self.events
            and self.events[-1] == "done"
            and "error" not in self.events
        )


def summarize_sse_lines(lines: list[str]) -> SSESummary:
    """Extract SSE event names while rejecting malformed event declarations."""

    events: list[str] = []
    for line in lines:
        if not line.startswith("event:"):
            continue
        event = line.removeprefix("event:").strip()
        if not event:
            raise ValueError("SSE event name must not be empty")
        events.append(event)
    return SSESummary(tuple(events))
