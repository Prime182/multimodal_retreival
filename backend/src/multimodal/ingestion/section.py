from __future__ import annotations

import re
from typing import Sequence

from ..types import PageBlocks, SectionSpan


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


def build_section_spans_from_blocks(pages: list[PageBlocks]) -> list[SectionSpan]:
    spans = []
    # Start with a default Document span if there are pages
    if pages:
        spans.append(SectionSpan(
            page_number=pages[0].page_number,
            bbox=(0.0, 0.0, 0.0, 0.0),
            line_start=0,
            section_name="Document",
            column=0,
        ))

    for page in pages:
        for block in page.blocks:
            if not isinstance(block, dict) or "type" not in block:
                continue
            if block["type"] in ("section_header", "header"):
                if "bbox" not in block:
                    continue
                x0, y0, x1, y1 = block["bbox"]
                # Detect column: if x0 > page_width / 2 → right column
                column = 1 if x0 > page.width / 2 else 0
                section_name = block["text"].strip()
                if not section_name:
                    continue
                    
                spans.append(SectionSpan(
                    page_number=page.page_number,
                    bbox=tuple(block["bbox"]),
                    line_start=int(y0),  # use y0 as "line" for backward compatibility if needed
                    section_name=section_name,
                    column=column,
                ))
    return spans


def resolve_section_spatial(
    page_number: int,
    bbox: tuple[float, float, float, float],
    spans: Sequence[SectionSpan],
) -> str | None:
    """Matches sections by page and spatial proximity (Y position)."""
    x0, y0, x1, y1 = bbox
    result: str | None = None
    
    # Sections are generally ordered by page and then by Y
    for span in spans:
        if span.page_number > page_number:
            break
        
        # If on the same page, we need to consider reading order (column then Y)
        if span.page_number == page_number:
            # If the span is in a later column, it hasn't started for this column yet
            # (Assuming sections don't typically span across columns in a way that breaks this simple logic)
            span_x0 = span.bbox[0]
            # Simple column check for the span itself if it wasn't pre-calculated
            # but we have it in SectionSpan.column
            
            # If current block is in col 0 and span is in col 1, skip
            block_column = 1 if x0 > 300 else 0 # Rough midpoint if page width not available
            # Ideally we'd use page.width but here we just have bbox. 
            # Most PDFs are ~600pts wide.
            
            if span.column > 0 and x0 < 300: # span is in right col, but we are in left col
                continue
            
            if span.bbox[1] > y0: # span starts below current block
                if span.column == 0 and x0 > 300: # span in left col, we are in right col
                    # This span might still be the current one if it's the latest one from left col
                    pass
                else:
                    break
                    
        result = span.section_name
    return result


def resolve_section(
    page_number: int,
    line_index: int,
    spans: Sequence[SectionSpan],
) -> str | None:
    # Kept for backward compatibility during transition if needed
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
