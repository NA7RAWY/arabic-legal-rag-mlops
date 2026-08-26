"""Extract and validate the full bilingual Egyptian Civil Code corpus."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data/raw/egyptian_civil_code.pdf"
OUTPUT_PATH = ROOT / "data/processed/civil_code_articles.json"
REPORT_PATH = ROOT / "data/processed/extraction_report.json"
EXPECTED_MISSING_ARABIC_TEXT = {1022}

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
ENGLISH_RE = re.compile(r"[A-Za-z]")
# Accept verified PDF defects: a missing A, no space before the number, or body
# text continuing on the marker line. Lowercase continuations and punctuation
# are excluded so cross-references such as "Article 901 has..." do not match.
ENGLISH_MARKER = re.compile(r"(?m)^A?rticle\s*(\d+)(?:\s*$|\s+(?=[A-Z]))")
ARTICLE_LINE = re.compile(r"^A?rticle\s*(\d+)(?:\s*$|\s+(?=[A-Z]))")
ARABIC_MARKER_AT_START = re.compile(
    r"^\s*\(?\s*م\s*ادة\s*\n?\s*\(?\s*[٠-٩۰-۹]+\s*\(?",
    re.DOTALL,
)
REPEAL_RANGE_RE = re.compile(
    r"\*?\s*Articles\s+(\d+)\s*-\s*(\d+)\s+"
    r"(?:(?:have been )?repealed(?: by Presidential\s+Decree\.)?)",
    re.IGNORECASE,
)
ARABIC_PARAGRAPH = re.compile(r"[()]\s*([٠-٩۰-۹]+)\s*[()]")
ARABIC_LETTER_PARAGRAPH = re.compile(r"[()]\s*([أبجدهـ])\s*[()]")

SAFE_ARABIC_REPAIRS = {
    "فحواه.ا": "فحواها.",
    "ال ميالدي": "الميالدي",
    "به ا": "بها",
    "وق ت": "وقت",
    "إدار ته ا": "إدارتها",
    "الز وجي.ن": "الزوجين.",
    "المدين ا به": "المدين بها",
    "ي سري": "يسري",
    "يرا د": "يراد",
    "قانو نه ما": "قانونهما",
}


@dataclass(frozen=True)
class Position:
    page_index: int
    y: float


@dataclass(frozen=True)
class ArticleStart:
    number: int
    position: Position


def article_number(value: str) -> int:
    """Normalize digits, correcting this PDF's RTL glyph order when needed."""
    if re.search(r"[٠-٩۰-۹]", value):
        value = value[::-1]
    return int(value.translate(ARABIC_DIGITS))


def compact_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def page_lines(page: pymupdf.Page) -> Iterator[tuple[float, float, str, bool]]:
    lines = []
    for block in page.get_text("dict", sort=False)["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            if compact_line(text):
                x0, y0, _, _ = line["bbox"]
                is_bold = any(span["flags"] & 16 for span in line["spans"])
                lines.append((y0, x0, text, is_bold))
    yield from sorted(lines, key=lambda item: (round(item[0], 1), item[1]))


def find_article_starts(document: pymupdf.Document) -> list[ArticleStart]:
    starts: list[ArticleStart] = []
    for page_index, page in enumerate(document):
        middle = page.rect.width / 2
        for y, x, raw_line, _ in page_lines(page):
            if x >= middle:
                continue
            match = ARTICLE_LINE.match(compact_line(raw_line))
            if match:
                starts.append(
                    ArticleStart(int(match.group(1)), Position(page_index, y))
                )
    starts.sort(key=lambda item: (item.position.page_index, item.position.y))
    return starts


def classify_explicit_heading(
    text: str, explicit_book_seen: bool
) -> tuple[str, str, bool] | None:
    """Classify only explicit or structurally unambiguous PDF headings."""
    line = compact_line(text).rstrip(":")
    if not explicit_book_seen and line in {"باب تمهيدي", "أحكام عامة"}:
        return "book", line, False
    if re.fullmatch(r"BOOK\s+[IVXLCDM]+(?:\s+.+)?", line, re.IGNORECASE):
        return "book", line, True
    if re.fullmatch(r"CHAPTER\s+[IVXLCDM]+(?:\s+.+)?", line, re.IGNORECASE):
        return "chapter", line, False
    if re.fullmatch(r"SECTION\s+[IVXLCDM]+(?:\s+.+)?", line, re.IGNORECASE):
        # Preserve the accepted preliminary hierarchy before BOOK I appears.
        field = "section" if explicit_book_seen else "chapter"
        return field, line, False
    if line == "Laws and their Applications":
        return "chapter", line, False
    if re.fullmatch(r"Conflicts of law as to .+", line):
        return "topic", line, False
    return None


def update_hierarchy(
    state: dict[str, str | None], field: str, value: str, append: bool = False
) -> None:
    levels = ("book", "chapter", "section", "topic")
    if append and state[field]:
        state[field] = f"{state[field]} — {value}"
    else:
        state[field] = value
    index = levels.index(field)
    for lower_field in levels[index + 1 :]:
        state[lower_field] = None


def next_article_on_page(
    starts: list[ArticleStart], page_index: int, y: float
) -> ArticleStart | None:
    return next(
        (
            start
            for start in starts
            if start.position.page_index == page_index and start.position.y > y
        ),
        None,
    )


def detect_hierarchy(
    document: pymupdf.Document, starts: list[ArticleStart]
) -> tuple[dict[int, dict[str, str | None]], list[dict[str, object]], list[Position]]:
    state = {"book": None, "chapter": None, "section": None, "topic": None}
    snapshots: dict[int, dict[str, str | None]] = {}
    transitions: list[dict[str, object]] = []
    heading_positions: list[Position] = []
    explicit_book_seen = False
    pending: tuple[str, int, float] | None = None

    starts_by_position = {
        (start.position.page_index, round(start.position.y, 1)): start.number
        for start in starts
    }

    for page_index, page in enumerate(document):
        middle = page.rect.width / 2
        for y, x, raw_line, is_bold in page_lines(page):
            line = compact_line(raw_line)
            number = starts_by_position.get((page_index, round(y, 1)))
            if number is not None and x < middle:
                snapshots[number] = state.copy()
                pending = None
                continue

            heading = classify_explicit_heading(line, explicit_book_seen)
            if heading:
                field, value, is_explicit_book = heading
                append = state[field] is not None and (
                    line == "أحكام عامة" or line == "Laws and their Applications"
                )
                update_hierarchy(state, field, value, append=append)
                explicit_book_seen = explicit_book_seen or is_explicit_book
                transitions.append(
                    {"page": page_index + 1, "field": field, "value": state[field]}
                )
                heading_positions.append(Position(page_index, y))
                pending = (
                    (field, page_index, y)
                    if re.fullmatch(
                        r"(?:BOOK|CHAPTER|SECTION)\s+[IVXLCDM]+",
                        line,
                        re.IGNORECASE,
                    )
                    else None
                )
                continue

            if (
                pending
                and page_index == pending[1]
                and x < middle
                and 0 < y - pending[2] <= 22
                and ENGLISH_RE.search(line)
                and is_bold
                and not ARTICLE_LINE.match(line)
                and not re.match(r"(?:BOOK|CHAPTER|SECTION)\b", line, re.IGNORECASE)
                and not re.fullmatch(r"\d+\s*[.-]\s*.+", line)
            ):
                field = pending[0]
                update_hierarchy(state, field, line, append=True)
                transitions.append(
                    {"page": page_index + 1, "field": field, "value": state[field]}
                )
                heading_positions.append(Position(page_index, y))
                pending = (field, page_index, y)
                continue

            numbered = re.fullmatch(r"\d+\s*[.-]\s*.+", line)
            following = next_article_on_page(starts, page_index, y)
            if (
                numbered
                and is_bold
                and x < middle
                and following
                and 0 < following.position.y - y <= 35
            ):
                field = "topic" if explicit_book_seen else "section"
                update_hierarchy(state, field, line)
                transitions.append(
                    {"page": page_index + 1, "field": field, "value": state[field]}
                )
                heading_positions.append(Position(page_index, y))
                pending = (field, page_index, y)

    return snapshots, transitions, heading_positions


def position_key(position: Position) -> tuple[int, float]:
    return position.page_index, position.y


def first_heading_between(
    start: Position, end: Position, headings: list[Position]
) -> Position | None:
    candidates = [
        heading
        for heading in headings
        if position_key(start) < position_key(heading) < position_key(end)
    ]
    return min(candidates, key=position_key) if candidates else None


def extract_between(
    document: pymupdf.Document,
    start: Position,
    end: Position,
    language: str,
) -> str:
    parts: list[str] = []
    for page_index in range(start.page_index, end.page_index + 1):
        page = document[page_index]
        middle = page.rect.width / 2
        x0, x1 = (0, middle) if language == "en" else (middle, page.rect.width)
        y0 = start.y if page_index == start.page_index else 0
        y1 = end.y if page_index == end.page_index else page.rect.height
        if y1 > y0:
            parts.append(page.get_text(clip=pymupdf.Rect(x0, y0, x1, y1), sort=False))
    return "\n".join(parts)


def is_body_heading(text: str) -> bool:
    line = compact_line(text).rstrip(":")
    return (
        classify_explicit_heading(line, True) is not None
        or re.fullmatch(r"[٠-٩۰-۹]+\s*-\s*(?:.+)?", line) is not None
        or line == "تطبيق القانون"
        or re.fullmatch(r"تنازع القوانين من حيث .+", line) is not None
        or REPEAL_RANGE_RE.fullmatch(line) is not None
        or re.fullmatch(r"المواد من .+ ملغاة", line) is not None
    )


def clean_english(text: str) -> str:
    text = ENGLISH_MARKER.sub("", text, count=1)
    lines = [compact_line(line) for line in text.splitlines()]
    body = " ".join(line for line in lines if line and not is_body_heading(line))
    return compact_line(ARABIC_RE.sub("", body))


def normalize_paragraph_label(match: re.Match[str]) -> str:
    return f"({article_number(match.group(1))})"


def clean_arabic(text: str) -> str:
    text = ARABIC_MARKER_AT_START.sub("", text, count=1)
    lines = [compact_line(line) for line in text.splitlines()]
    body = " ".join(line for line in lines if line and not is_body_heading(line))
    for artifact, repair in SAFE_ARABIC_REPAIRS.items():
        body = body.replace(artifact, repair)
    body = re.sub(r"(^|\s)\)\s*(?=\([٠-٩۰-۹])", r"\1", body)
    body = ARABIC_PARAGRAPH.sub(normalize_paragraph_label, body)
    body = ARABIC_LETTER_PARAGRAPH.sub(lambda match: f"({match.group(1)})", body)
    body = re.sub(r"\s+([،؛:.])", r"\1", body)
    body = re.sub(r"([،؛:])(?=\S)", r"\1 ", body)
    body = ENGLISH_RE.sub("", body)
    return compact_line(body)


def detect_repeal_ranges(
    document: pymupdf.Document,
) -> list[dict[str, object]]:
    detected: list[dict[str, object]] = []
    for page_index, page in enumerate(document):
        text = page.get_text(sort=False)
        for match in REPEAL_RANGE_RE.finditer(text):
            start, end = map(int, match.group(1, 2))
            english_notice = compact_line(match.group(0).lstrip("* "))
            right = page.get_text(
                clip=pymupdf.Rect(
                    page.rect.width / 2, 0, page.rect.width, page.rect.height
                ),
                sort=False,
            )
            arabic_lines = [compact_line(line) for line in right.splitlines()]
            notice_index = next(
                (i for i, line in enumerate(arabic_lines) if "المواد من" in line),
                None,
            )
            arabic_notice = ""
            if notice_index is not None:
                selected = []
                for line in arabic_lines[notice_index : notice_index + 8]:
                    if line.startswith(("الفصل", "الكتاب", "الباب")) and selected:
                        break
                    if line:
                        selected.append(line)
                arabic_notice = compact_line(" ".join(selected))
            detected.append(
                {
                    "start": start,
                    "end": end,
                    "source_page": page_index + 1,
                    "text_ar": arabic_notice,
                    "text_en": english_notice,
                }
            )
    return detected


def extract_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    with pymupdf.open(PDF_PATH) as document:
        starts = find_article_starts(document)
        numbers = [start.number for start in starts]
        duplicates = sorted(
            number for number, count in Counter(numbers).items() if count > 1
        )
        if duplicates:
            raise RuntimeError(f"Duplicate article markers: {duplicates}")
        if numbers != sorted(numbers):
            raise RuntimeError("Article markers are not in ascending document order")

        hierarchy, transitions, heading_positions = detect_hierarchy(document, starts)
        normal_records: dict[int, dict[str, object]] = {}
        document_end = Position(document.page_count - 1, document[-1].rect.height)

        for index, start in enumerate(starts):
            next_position = (
                starts[index + 1].position if index + 1 < len(starts) else document_end
            )
            heading_end = first_heading_between(
                start.position, next_position, heading_positions
            )
            end = heading_end or next_position
            text_en = clean_english(
                extract_between(document, start.position, end, "en")
            )
            text_ar = clean_arabic(extract_between(document, start.position, end, "ar"))
            normal_records[start.number] = {
                "article_number": start.number,
                **hierarchy.get(
                    start.number,
                    {"book": None, "chapter": None, "section": None, "topic": None},
                ),
                "text_ar": text_ar or None,
                "text_en": text_en or None,
                "is_repealed": False,
                "source_page": start.position.page_index + 1,
                "citation": f"Egyptian Civil Code, Article {start.number}",
            }

        repeal_ranges = detect_repeal_ranges(document)
        for repeal in repeal_ranges:
            range_numbers = range(int(repeal["start"]), int(repeal["end"]) + 1)
            anchor = normal_records.get(int(repeal["start"]))
            if anchor:
                notice_ar = anchor["text_ar"] or repeal["text_ar"]
                notice_en = anchor["text_en"] or repeal["text_en"]
                range_hierarchy = {
                    field: anchor[field]
                    for field in ("book", "chapter", "section", "topic")
                }
            else:
                previous = normal_records[
                    max(n for n in normal_records if n < repeal["start"])
                ]
                notice_ar = repeal["text_ar"]
                notice_en = repeal["text_en"]
                range_hierarchy = {
                    field: previous[field]
                    for field in ("book", "chapter", "section", "topic")
                }
            for number in range_numbers:
                normal_records[number] = {
                    "article_number": number,
                    **range_hierarchy,
                    "text_ar": notice_ar,
                    "text_en": notice_en,
                    "is_repealed": True,
                    "source_page": repeal["source_page"],
                    "citation": f"Egyptian Civil Code, Article {number}",
                }

        # Domain requirement: the entire explicitly repealed 54-80 range is retained.
        for number in range(54, 81):
            if (
                number not in normal_records
                or not normal_records[number]["is_repealed"]
            ):
                raise RuntimeError(
                    f"Required repealed Article {number} was not preserved"
                )

        records = [normal_records[number] for number in sorted(normal_records)]
        return records, transitions


def hierarchy_transitions(records: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = ("book", "chapter", "section", "topic")
    transitions: list[dict[str, object]] = []
    previous = {field: None for field in fields}
    for record in records:
        changes = {
            field: record[field] for field in fields if record[field] != previous[field]
        }
        if changes:
            transitions.append(
                {"article_number": record["article_number"], "changes": changes}
            )
        previous = {field: record[field] for field in fields}
    return transitions


def validate(records: list[dict[str, object]]) -> dict[str, object]:
    numbers = [int(record["article_number"]) for record in records]
    counts = Counter(numbers)
    duplicates = sorted(number for number, count in counts.items() if count > 1)
    if duplicates:
        raise RuntimeError(f"Duplicate article numbers: {duplicates}")
    if numbers != sorted(numbers):
        raise RuntimeError("Article numbers are not sorted ascending")

    gaps = (
        [
            number
            for number in range(numbers[0], numbers[-1] + 1)
            if number not in counts
        ]
        if numbers
        else []
    )
    missing_arabic = [
        record["article_number"]
        for record in records
        if not record["text_ar"] or not ARABIC_RE.search(str(record["text_ar"]))
    ]
    expected_missing_arabic = sorted(
        number for number in missing_arabic if number in EXPECTED_MISSING_ARABIC_TEXT
    )
    unexpected_missing_arabic = sorted(
        number
        for number in missing_arabic
        if number not in EXPECTED_MISSING_ARABIC_TEXT
    )
    missing_english = [
        record["article_number"]
        for record in records
        if not record["text_en"] or not ENGLISH_RE.search(str(record["text_en"]))
    ]
    pages = [int(record["source_page"]) for record in records]
    repealed = [record["article_number"] for record in records if record["is_repealed"]]
    expected_schema = [
        "article_number",
        "book",
        "chapter",
        "section",
        "topic",
        "text_ar",
        "text_en",
        "is_repealed",
        "source_page",
        "citation",
    ]
    if any(list(record) != expected_schema for record in records):
        raise RuntimeError("One or more records do not match the required schema")

    return {
        "total_extracted_articles": len(records),
        "article_number_range": [numbers[0], numbers[-1]] if numbers else [],
        "duplicate_article_numbers": duplicates,
        "missing_article_numbers": gaps,
        "expected_missing_arabic_text": expected_missing_arabic,
        "unexpected_missing_arabic_text": unexpected_missing_arabic,
        "articles_missing_english_text": missing_english,
        "source_page_range": [min(pages), max(pages)] if pages else [],
        "hierarchy_transitions": hierarchy_transitions(records),
        "repealed_article_numbers_detected": repealed,
    }


def print_report(report: dict[str, object]) -> None:
    print(f"Total extracted articles: {report['total_extracted_articles']}")
    print(f"Article number range: {report['article_number_range']}")
    print(f"Source page range: {report['source_page_range']}")
    print(f"Missing article numbers: {report['missing_article_numbers']}")
    print(f"Duplicate article numbers: {report['duplicate_article_numbers']}")
    print(f"Expected missing Arabic text: {report['expected_missing_arabic_text']}")
    print(f"Unexpected missing Arabic text: {report['unexpected_missing_arabic_text']}")
    print(f"Articles missing English text: {report['articles_missing_english_text']}")
    print(
        "Repealed article numbers detected: "
        f"{report['repealed_article_numbers_detected']}"
    )
    print(f"Hierarchy transitions detected: {len(report['hierarchy_transitions'])}")
    for transition in report["hierarchy_transitions"]:
        print(f"  - Article {transition['article_number']}: {transition['changes']}")


def main() -> None:
    records, _ = extract_records()
    report = validate(records)  # Validation must succeed before either write.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print_report(report)


if __name__ == "__main__":
    main()
