"""
backend/src/multimodal/ingestion/section.py

Section span detection — unified coordinate system.

DESIGN: Why one coordinate space, not two
─────────────────────────────────────────
There are two callers of section resolution:

  resolve_section()         → text.py / equation.py
      input: (page_number, line_index: int, spans)

  resolve_section_spatial() → table.py / image.py / equation.py
      input: (page_number, bbox: (x0,y0,x1,y1), spans)
      bbox.y0 may be a real PDF pixel coord OR a pseudo-pixel value.

ALL spans store line_start as a 0-based *line index*.
resolve_section_spatial() converts the caller's bbox.y0 to an estimated
line index using _APPROX_LINE_HEIGHT before comparing — keeping both
paths in the same unit.

_APPROX_LINE_HEIGHT is imported by equation.py and image.py so the
pseudo-bbox construction in those files uses the identical constant.
Accuracy in absolute terms is NOT required; only consistency matters.

BUGS FIXED IN THIS FILE
───────────────────────
Bug 1 (original): block-based spans stored line_start = int(y0) in raw
  pixel coordinates (e.g. 150), while text-fallback spans stored
  line_start = line_index (e.g. 5). resolve_section() comparing
  span.line_start > line_index would break immediately for all pixel-based
  spans, leaving every chunk with section = "Document".

Bug 2 (introduced in first fix attempt): resolve_section_spatial() was
  changed to compare span.bbox[1] > bbox_y0. But text-fallback spans
  stored bbox.y0 = line_index * 12 (scaled), while table.py / image.py
  pass real PDF pixel coords (e.g. 350). Different scales → wrong
  section assignment for tables and images.

Final fix: ALL spans use line_start as a line index.
  resolve_section_spatial() converts its input y0 via
  int(y0 / _APPROX_LINE_HEIGHT) before comparing, making both callers
  consistent.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..types import PageBlocks, SectionSpan

# Single constant shared with equation.py and image.py.
# Must be imported from here — do not redefine in those modules.
_APPROX_LINE_HEIGHT: float = 12.0

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

    Primary:  reads block["type"] == "section_header" / "header" from page.blocks.
    Fallback: scans page.text line by line with detect_heading().

    ALL spans store line_start as a 0-based line index.
    Text-fallback spans also store bbox.y0 = line_index * _APPROX_LINE_HEIGHT
    so resolve_section_spatial() (which divides by _APPROX_LINE_HEIGHT) yields
    the original line index, keeping the comparison consistent.
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
        page_lines = page.text.splitlines() if page.text else []

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

            # Convert pixel y0 → line index.
            line_idx = _y0_to_line_index(y0, page_lines, section_name)
            column = 1 if page.width > 0 and x0 > page.width / 2 else 0

            spans.append(SectionSpan(
                page_number=page.page_number,
                # Store bbox.y0 as line_idx * _APPROX_LINE_HEIGHT so that
                # resolve_section_spatial()'s int(y0/_APPROX_LINE_HEIGHT)
                # round-trips correctly back to line_idx.
                bbox=(x0,
                      float(line_idx) * _APPROX_LINE_HEIGHT,
                      x1,
                      float(line_idx + 1) * _APPROX_LINE_HEIGHT),
                line_start=line_idx,
                section_name=section_name,
                column=column,
            ))
            block_spans_found = True

        # ── Fallback: scan page.text for heading lines ────────────────────────
        if not block_spans_found and page.text:
            for line_index, raw_line in enumerate(page_lines):
                heading = detect_heading(raw_line.strip())
                if heading:
                    spans.append(SectionSpan(
                        page_number=page.page_number,
                        bbox=(0.0,
                              float(line_index) * _APPROX_LINE_HEIGHT,
                              page.width,
                              float(line_index + 1) * _APPROX_LINE_HEIGHT),
                        line_start=line_index,
                        section_name=heading,
                        column=0,
                    ))

    return spans


def _y0_to_line_index(y0: float, page_lines: list[str], section_name: str) -> int:
    """
    Convert a PDF pixel y0 coordinate to a 0-based line index.

    Preferred: scan page_lines for exact or partial heading match (precise).
    Fallback:  int(y0 / _APPROX_LINE_HEIGHT) (approximate).
    """
    if page_lines:
        needle = section_name.lower().strip()
        for idx, raw_line in enumerate(page_lines):
            if raw_line.strip().lower() == needle:
                return idx
        for idx, raw_line in enumerate(page_lines):
            if needle in raw_line.strip().lower():
                return idx
    return max(0, int(y0 / _APPROX_LINE_HEIGHT))


# ── Resolution helpers ─────────────────────────────────────────────────────────

def resolve_section(
    page_number: int,
    line_index: int,
    spans: Sequence[SectionSpan],
) -> str | None:
    """
    Return the active section name at (page_number, line_index).
    Called by text.py and equation.py (line-index domain).
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
    bbox = (x0, y0, x1, y1).

    y0 may come from three sources:
      • Real PDF pixel coord from pdfplumber   (table.py, image.py)
      • Pseudo-pixel = line_index * _APPROX_LINE_HEIGHT  (equation.py)

    In all cases y0 is divided by _APPROX_LINE_HEIGHT to estimate the
    line index, then compared against span.line_start (also a line index).
    This makes the comparison consistent regardless of y0's origin.

    Note: when the caller passes a real PDF pixel y (e.g. 350.0), the
    estimated line index (350/12 ≈ 29) is an approximation. It is
    accurate enough to determine which section a table or image belongs to
    because section headings are typically tens of lines apart.
    """
    _x0, y0, _x1, _y1 = bbox
    caller_line = int(y0 / _APPROX_LINE_HEIGHT) if y0 > 0 else 0

    result: str | None = None
    for span in spans:
        if span.page_number > page_number:
            break
        if span.page_number == page_number and span.line_start > caller_line:
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

    if words[0].lower().rstrip(".") in _CAPTION_PREFIXES:
        return None

    if any(ch in _MATH_SYMBOLS for ch in stripped):
        return None

    normalized = re.sub(r"[:.]+$", "", stripped).strip()
    lowered = normalized.lower()

    if lowered in _KNOWN_SECTION_NAMES:
        return normalized.title()

    if _HEADING_NUMBER_RE.match(normalized):
        return normalized

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