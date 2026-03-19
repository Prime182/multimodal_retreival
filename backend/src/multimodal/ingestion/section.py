"""
backend/src/multimodal/ingestion/section.py

Section span detection.

Since pymupdf4llm does not populate page_boxes by default, blocks are [].
build_section_spans_from_blocks falls back to scanning page.text for
heading lines using detect_heading(), which is the same heuristic used
by the text chunker.

resolve_section()         — line-index based  (used by text & equation)
resolve_section_spatial() — bbox based        (used by table & image)
Both are kept so callers don't need changing.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..types import PageBlocks, SectionSpan


_HEADING_NUMBER_RE = re.compile(r"^\s*((\d+(\.\d+)*)\.?|[IVXLC]+\.?)\s+[A-Z]")
_KNOWN_SECTION_NAMES = {
    "abstract", "introduction", "background",
    "methods", "materials and methods",
    "results", "discussion",
    "conclusion", "conclusions",
    "references", "appendix",
}
_CAPTION_PREFIXES = {"figure", "fig", "table", "scheme"}
_MATH_SYMBOLS = frozenset("=<>±∑∫√≈≠≤≥∞∂∆∇λμσπθβα^")


# ── Span building ──────────────────────────────────────────────────────────────

def build_section_spans_from_blocks(pages: list[PageBlocks]) -> list[SectionSpan]:
    """
    Build SectionSpan list from page data.

    Primary:  reads block["type"] == "section_header" / "header" from page.blocks
              (only available if pymupdf4llm is configured to return page_boxes).
    Fallback: scans page.text line by line with detect_heading()
              (always available — this is the usual path).
    """
    spans: list[SectionSpan] = []

    if pages:
        spans.append(SectionSpan(
            page_number=pages[0].page_number,
            bbox=(0.0, 0.0, 0.0, 0.0),
            line_start=0,
            section_name="Document",
            column=0,
        ))

    for page in pages:
        # ── Try block-based detection first ───────────────────────────────────
        block_spans_found = False
        for block in page.blocks:
            if not isinstance(block, dict) or "type" not in block:
                continue
            if block["type"] not in ("section_header", "header"):
                continue
            if "bbox" not in block:
                continue
            x0, y0, x1, y1 = block["bbox"]
            section_name = block.get("text", "").strip()
            if not section_name:
                continue
            column = 1 if page.width > 0 and x0 > page.width / 2 else 0
            spans.append(SectionSpan(
                page_number=page.page_number,
                bbox=tuple(block["bbox"]),
                line_start=int(y0),
                section_name=section_name,
                column=column,
            ))
            block_spans_found = True

        # ── Fallback: scan page.text for heading lines ────────────────────────
        if not block_spans_found and page.text:
            for line_index, raw_line in enumerate(page.text.splitlines()):
                heading = detect_heading(raw_line.strip())
                if heading:
                    spans.append(SectionSpan(
                        page_number=page.page_number,
                        bbox=(0.0, float(line_index * 10), page.width, float(line_index * 10 + 10)),
                        line_start=line_index,
                        section_name=heading,
                        column=0,
                    ))

    return spans


# ── Resolution helpers ─────────────────────────────────────────────────────────

def resolve_section(
    page_number: int,
    line_index: int,
    spans: Sequence[SectionSpan],
) -> str | None:
    """
    Return the active section name at (page_number, line_index).
    Uses integer line_start comparison — safe to call from text & equation paths.
    """
    result: str | None = None
    for span in spans:
        if span.page_number > page_number:
            break
        if span.page_number == page_number and span.line_start > line_index:
            break
        result = span.section_name
    return result


def resolve_section_spatial(
    page_number: int,
    bbox: tuple[float, float, float, float],
    spans: Sequence[SectionSpan],
) -> str | None:
    """
    Return the active section name at (page_number, bbox).
    bbox = (x0, y0, x1, y1).  Uses y0 for ordering within a page.
    Safe to call from table & image paths.
    """
    _x0, y0, _x1, _y1 = bbox
    result: str | None = None
    for span in spans:
        if span.page_number > page_number:
            break
        if span.page_number == page_number and span.line_start > y0:
            break
        result = span.section_name
    return result


# ── Heading detector ───────────────────────────────────────────────────────────

def detect_heading(line: str) -> str | None:
    """
    Return a normalised heading string if `line` looks like a section heading,
    otherwise None.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None

    words = stripped.split()
    if len(words) == 0 or len(words) > 12:
        return None

    # Caption lines ("Figure 1 ...", "Table 2 ...") are not headings
    if words[0].lower().rstrip(".") in _CAPTION_PREFIXES:
        return None

    # Lines with math symbols are not headings
    if any(ch in _MATH_SYMBOLS for ch in stripped):
        return None

    normalized = re.sub(r"[:.]+$", "", stripped).strip()
    lowered = normalized.lower()

    # Known section name (exact)
    if lowered in _KNOWN_SECTION_NAMES:
        return normalized.title()

    # Numbered heading: "1 Introduction", "2.3 Methods", "IV Results"
    if _HEADING_NUMBER_RE.match(normalized):
        return normalized

    # ALL-CAPS heading (3–80 chars, ≤ 6 words)
    non_space = [ch for ch in stripped if not ch.isspace()]
    if non_space:
        digit_ratio = sum(ch.isdigit() for ch in non_space) / len(non_space)
        if digit_ratio > 0.35:
            return None
    if stripped[:-1] and re.search(r"[.?!]", stripped[:-1]):
        return None
    if normalized.isupper() and 3 <= len(normalized) <= 80 and len(words) <= 6:
        return normalized.title()

    return None