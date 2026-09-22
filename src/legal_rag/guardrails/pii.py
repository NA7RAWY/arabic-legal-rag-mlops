"""High-confidence deterministic PII redaction for generated answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class PIICategory(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    NATIONAL_ID = "national_id"


_MARKERS = {
    PIICategory.EMAIL: "[REDACTED_EMAIL]",
    PIICategory.PHONE: "[REDACTED_PHONE]",
    PIICategory.NATIONAL_ID: "[REDACTED_NATIONAL_ID]",
}

_EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}"
    r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+(?![\w.-])"
)
_EGYPTIAN_MOBILE_PATTERN = re.compile(
    r"(?<!\d)(?:(?:\+20|0020)[ -]?1[0125]|01[0125])(?:[ -]?\d){8}(?!\d)"
)
_NATIONAL_ID_CANDIDATE = re.compile(r"(?<!\d)[23]\d{13}(?!\d)")
_EGYPTIAN_GOVERNORATE_CODES = frozenset(range(1, 36)) | {88}
_EMAIL_LOCAL_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.!#$%&'*+/=?^_`{|}~-"
)
_EMAIL_DOMAIN_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-."
)
_MAX_EMAIL_LENGTH = 254


@dataclass(frozen=True, slots=True)
class PIIRedactionResult:
    """Sanitized text and bounded category counts only."""

    text: str
    counts: tuple[tuple[PIICategory, int], ...]

    @property
    def total(self) -> int:
        return sum(count for _, count in self.counts)

    @property
    def categories(self) -> tuple[PIICategory, ...]:
        return tuple(category for category, count in self.counts if count)

    @property
    def triggered(self) -> bool:
        return self.total > 0


def _is_valid_national_id(candidate: str) -> bool:
    century = 1900 if candidate[0] == "2" else 2000
    try:
        date(
            century + int(candidate[1:3]),
            int(candidate[3:5]),
            int(candidate[5:7]),
        )
    except ValueError:
        return False
    return int(candidate[7:9]) in _EGYPTIAN_GOVERNORATE_CODES


class PIIGuardrail:
    """Redact selected high-confidence PII from generated answer text."""

    def redact(self, text: str) -> PIIRedactionResult:
        if not isinstance(text, str):
            raise TypeError("PII guardrail input must be text")

        counts = {category: 0 for category in PIICategory}

        def replace_email(match: re.Match[str]) -> str:
            if len(match.group(0)) > _MAX_EMAIL_LENGTH:
                return match.group(0)
            counts[PIICategory.EMAIL] += 1
            return _MARKERS[PIICategory.EMAIL]

        def replace_phone(match: re.Match[str]) -> str:
            counts[PIICategory.PHONE] += 1
            return _MARKERS[PIICategory.PHONE]

        def replace_national_id(match: re.Match[str]) -> str:
            candidate = match.group(0)
            if not _is_valid_national_id(candidate):
                return candidate
            counts[PIICategory.NATIONAL_ID] += 1
            return _MARKERS[PIICategory.NATIONAL_ID]

        sanitized = _EMAIL_PATTERN.sub(replace_email, text)
        sanitized = _EGYPTIAN_MOBILE_PATTERN.sub(replace_phone, sanitized)
        sanitized = _NATIONAL_ID_CANDIDATE.sub(replace_national_id, sanitized)
        return PIIRedactionResult(
            text=sanitized,
            counts=tuple((category, counts[category]) for category in PIICategory),
        )


def _email_prefix(value: str) -> bool:
    if not value or len(value) > _MAX_EMAIL_LENGTH or value.count("@") > 1:
        return False
    if "@" not in value:
        return len(value) <= 64 and all(char in _EMAIL_LOCAL_CHARS for char in value)
    local, domain = value.split("@", 1)
    if not 1 <= len(local) <= 64 or any(
        char not in _EMAIL_LOCAL_CHARS for char in local
    ):
        return False
    if not domain:
        return True
    if any(char not in _EMAIL_DOMAIN_CHARS for char in domain):
        return False
    labels = domain.split(".")
    if any(not label or len(label) > 63 for label in labels[:-1]):
        return False
    if any(not label[0].isalnum() or not label[-1].isalnum() for label in labels[:-1]):
        return False
    current = labels[-1]
    return not current or (
        len(current) <= 63
        and current[0].isalnum()
        and all(char.isascii() and (char.isalnum() or char == "-") for char in current)
    )


def _phone_tail_prefix(value: str) -> bool:
    digits = 0
    awaiting_digit = False
    for char in value:
        if char in " -":
            if awaiting_digit or digits >= 8:
                return False
            awaiting_digit = True
            continue
        if not char.isascii() or not char.isdigit() or digits >= 8:
            return False
        digits += 1
        awaiting_digit = False
    return digits < 8 or (digits == 8 and not awaiting_digit)


def _phone_prefix(value: str) -> bool:
    fixed_prefixes = tuple(
        f"{country}{separator}1{carrier}"
        for country in ("+20", "0020")
        for separator in ("", " ", "-")
        for carrier in "0125"
    ) + tuple(f"01{carrier}" for carrier in "0125")
    for fixed in fixed_prefixes:
        if fixed.startswith(value):
            return True
        if value.startswith(fixed) and _phone_tail_prefix(value[len(fixed) :]):
            return True
    return False


def _national_id_prefix(value: str) -> bool:
    if not value or value[0] not in "23" or not value.isascii() or not value.isdigit():
        return False
    if len(value) < 14:
        return True
    return len(value) == 14 and _is_valid_national_id(value)


def _could_be_pii_prefix(value: str, preceding: str) -> bool:
    email_boundary = not preceding or not re.match(r"[\w.+-]", preceding)
    digit_boundary = not preceding or not preceding.isdigit()
    return (
        (email_boundary and _email_prefix(value))
        or (digit_boundary and _phone_prefix(value))
        or (digit_boundary and _national_id_prefix(value))
    )


class StreamingPIIRedactor:
    """Incrementally redact PII while retaining only a possible PII suffix."""

    max_buffered_characters = _MAX_EMAIL_LENGTH

    def __init__(self, guardrail: PIIGuardrail | None = None) -> None:
        self.guardrail = guardrail if guardrail is not None else PIIGuardrail()
        self._buffer = ""
        self._preceding = ""
        self._counts = {category: 0 for category in PIICategory}

    @property
    def counts(self) -> tuple[tuple[PIICategory, int], ...]:
        return tuple((category, self._counts[category]) for category in PIICategory)

    def _redact(self, text: str) -> PIIRedactionResult:
        result = self.guardrail.redact(text)
        for category, count in result.counts:
            self._counts[category] += count
        return result

    def feed(self, text: str) -> PIIRedactionResult:
        """Return text proven safe without assuming the stream has ended."""

        if not isinstance(text, str):
            raise TypeError("PII stream chunk must be text")
        self._buffer += text
        pending_start = len(self._buffer)
        lower_bound = max(0, len(self._buffer) - self.max_buffered_characters)
        for index in range(lower_bound, len(self._buffer)):
            preceding = self._buffer[index - 1] if index else self._preceding
            if _could_be_pii_prefix(self._buffer[index:], preceding):
                pending_start = index
                break
        safe = self._buffer[:pending_start]
        self._buffer = self._buffer[pending_start:]
        if safe:
            self._preceding = safe[-1]
        return self._redact(safe)

    def finish(self) -> PIIRedactionResult:
        """Redact and release the remaining suffix at successful end-of-stream."""

        buffered = self._buffer
        self._buffer = ""
        if buffered:
            self._preceding = buffered[-1]
        return self._redact(buffered)

    def discard(self) -> None:
        """Drop an unresolved suffix after provider failure without exposing it."""

        self._buffer = ""
