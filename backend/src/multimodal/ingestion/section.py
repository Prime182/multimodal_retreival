from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .utils import PageText


_HEADING_NUMBER_RE = re.compile(r"^\s*((\d+(\.\d+)*)\.?|[IVXLC]+\.?)\s+[A-Z]")
_KNOWN_SECTION_NAMES = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "appendix",
}
_CAPTION_PREFIXES = {"figure", "fig", "table", "scheme"}
_MATH_SYMBOLS = frozenset("=<>±∑∫√≈≠≤≥∞∂∆∇λμσπθβα^")


@dataclass(slots=True, frozen=True)
class SectionSpan:
    page_number: int
    line_start: int
    section_name: str


def build_section_spans(pages: Sequence[PageText]) -> list[SectionSpan]:
    if not pages:
        return []

    spans = [SectionSpan(pages[0].page_number, 0, "Document")]
    current_section = "Document"

    for page in pages:
        for line_index, raw_line in enumerate(page.text.splitlines()):
            heading = detect_heading(raw_line.strip())
            if heading and heading != current_section:
                current_section = heading
                spans.append(SectionSpan(page.page_number, line_index, current_section))

    return spans


def resolve_section(
    page_number: int,
    line_index: int,
    spans: Sequence[SectionSpan],
) -> str | None:
    result: str | None = None
    for span in spans:
        if span.page_number > page_number:
            break
        if span.page_number == page_number and span.line_start > line_index:
            break
        result = span.section_name
    return result


def detect_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return None

    words = stripped.split()
    if len(words) == 0 or len(words) > 10:
        return None

    if words[0].lower().rstrip(".") in _CAPTION_PREFIXES:
        return None
    if any(char in _MATH_SYMBOLS for char in stripped):
        return None

    normalized = re.sub(r"[:.]+$", "", stripped).strip()
    lowered = normalized.lower()
    if lowered in _KNOWN_SECTION_NAMES:
        return normalized.title()
    compact = normalized.replace(" ", "").lower()
    if compact in _KNOWN_SECTION_NAMES and all(len(word) == 1 for word in normalized.split()):
        return compact.title()
    if _HEADING_NUMBER_RE.match(normalized):
        return normalized

    non_space = [char for char in stripped if not char.isspace()]
    if non_space and sum(char.isdigit() for char in non_space) / len(non_space) > 0.35:
        return None
    if stripped[:-1] and re.search(r"[.?!]", stripped[:-1]):
        return None

    if normalized.isupper() and 3 <= len(normalized) <= 80 and len(words) <= 6:
        return normalized.title()

    return None
