"""
backend/src/multimodal/ingestion/table.py

BUGS FIXED IN THIS FILE
───────────────────────
Bug 1: TABLE_CAPTION_RE required a separator character after the table number.
  "Table 1" alone on a line (very common academic PDF pattern) did NOT match.
  Fix: add `|$` so end-of-line is also accepted.
  Old: r"^\s*Table\s+\d+[\s.:–-]"
  New: r"^\s*Table\s+\d+(?:[\s.:–-]|$)"

Bug 2: build_table_text_exclusion() used BOTH lines AND text strategies.
  The text strategy on a two-column PDF detects the entire page as one big
  "table", adding all prose to the exclusion set. This poisoned equation
  detection by vetoing real content lines.
  Fix: exclusion set uses ONLY the lines strategy.

Bug 3: _extract_first_table_below_y() did not account for two-column layout.
  For tables confined to one column, pdfplumber's text strategy merged both
  columns, producing garbled multi-column output.
  Fix: detect whether table content is left-column-only (max word x < midpoint)
  and crop to the appropriate column width before extraction. Fall back to
  extract_text() on the cropped region when extract_tables() produces too
  many columns (garbled output).
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Sequence

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore[assignment]

from ..types import PageBlocks, TableChunk
from .section import SectionSpan, resolve_section_spatial
from .utils import normalise_line


# FIX 1: `|$` allows "Table 1" at end-of-line to match.
TABLE_CAPTION_RE = re.compile(
    r"^\s*Table\s+\d+(?:[\s.:–-]|$)",
    re.IGNORECASE,
)

_SETTINGS_LINES = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
}

_SETTINGS_TEXT = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
    "text_tolerance": 3,
    "text_x_tolerance": 3,
    "text_y_tolerance": 3,
}

# Maximum number of columns expected in a real table.
# If extract_tables() returns more than this, the output is likely garbled
# (e.g. whole-page text fragmented by the text strategy) and we fall back
# to extract_text() on the cropped region.
_MAX_EXPECTED_COLUMNS = 8


def _find_caption_y(page, caption_text: str) -> float | None:
    m = re.match(r"^\s*(Table\s+\d+)", caption_text, re.IGNORECASE)
    if not m:
        return None
    anchor_norm = re.sub(r"\s+", "", m.group(1).lower())

    words = page.extract_words()
    for i, word in enumerate(words):
        w_norm = re.sub(r"\s+", "", word["text"].lower())
        if w_norm == anchor_norm:
            return float(word["bottom"])
        if w_norm == "table" and i + 1 < len(words):
            combined = w_norm + words[i + 1]["text"].lower()
            if combined == anchor_norm:
                return float(words[i + 1]["bottom"])
    return None


def _column_aware_x1(page, caption_y: float) -> float:
    """
    Return the right x-boundary to use for table cropping.

    For tables confined to one column of a two-column layout, crop to the
    column width to prevent the other column's text from polluting extraction.
    If the table spans more than half the page width, use the full width.
    """
    midpoint = page.width / 2
    # Sample words in the ~250pt region below the caption
    words = page.extract_words()
    table_words = [
        w for w in words
        if w["top"] > caption_y and w["top"] < caption_y + 250
    ]
    if not table_words:
        return page.width
    max_x = max(w["x1"] for w in table_words)
    return midpoint if max_x < midpoint else page.width


def _extract_first_table_below_y(page, y: float) -> list[list[str | None]] | None:
    """
    Crop the page below y, respecting column boundaries, and extract table.

    Strategy order:
    1. lines strategy (exact — works when table has ruling lines)
    2. text strategy with column-aware x1 (approximate — for lineless tables)
    3. extract_text() fallback on cropped region (for garbled text-strategy output)
    """
    x1 = _column_aware_x1(page, y)

    try:
        cropped = page.within_bbox((0, y, x1, page.height), relative=False)
    except Exception:
        return None

    # Strategy 1: lines (precise, works with ruled tables)
    try:
        tables = cropped.extract_tables(table_settings=_SETTINGS_LINES) or []
        valid = [t for t in tables if t and len(t) >= 2]
        if valid:
            return valid[0]
    except Exception:
        pass

    # Strategy 2: text (approximate, for lineless tables)
    try:
        tables = cropped.extract_tables(table_settings=_SETTINGS_TEXT) or []
        valid = [t for t in tables if t and len(t) >= 2]
        if valid:
            best = valid[0]
            # Sanity check: if way too many columns, the text strategy
            # fragmented words — fall through to strategy 3.
            if best and len(best[0]) <= _MAX_EXPECTED_COLUMNS:
                return best
    except Exception:
        pass

    # Strategy 3: plain text extraction from cropped region.
    # Returns a synthetic single-column table where each row is one line.
    # Column alignment is lost but all content is preserved for embedding.
    try:
        text = cropped.extract_text() or ""
        text = re.sub(r"\(cid:\d+\)", "−", text).strip()
        if not text:
            return None
        rows: list[list[str | None]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                rows.append([stripped])
        return rows if len(rows) >= 2 else None
    except Exception:
        return None


def build_table_text_exclusion(pdf_path: Path) -> tuple[set[str], set[str]]:
    """Build cell/row exclusion sets used by equation detection.

    FIX: Uses ONLY the lines strategy.
    The text strategy on a two-column PDF falsely detects the entire page
    as a table, which would add all prose to the exclusion set and veto
    every line in equation detection.
    """
    if pdfplumber is None:
        return set(), set()

    cell_ex: set[str] = set()
    row_ex: set[str] = set()

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            try:
                tables = page.extract_tables(table_settings=_SETTINGS_LINES) or []
            except Exception:
                tables = []
            for table in tables:
                for row in table:
                    cells = []
                    for cell in row:
                        if not cell:
                            continue
                        n = normalise_line(str(cell))
                        if n:
                            cell_ex.add(n)
                            cells.append(n)
                    if cells:
                        row_ex.add(" ".join(cells))

    return cell_ex, row_ex


def extract_tables(
    *,
    pdf_path: Path,
    pages: Sequence[PageBlocks],
    spans: Sequence[SectionSpan],
    journal_id: str,
    article_id: str,
    source_path: str,
) -> list[TableChunk]:
    """
    Caption-anchored table extraction with column-aware cropping.
    """
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required for table extraction.")

    page_by_num: dict[int, PageBlocks] = {page.page_number: page for page in pages}

    caption_map: dict[int, list[str]] = {}
    for page in pages:
        captions = [
            line.strip()
            for line in page.text.splitlines()
            if TABLE_CAPTION_RE.match(line.strip())
        ]
        if captions:
            caption_map[page.page_number] = captions

    chunks: list[TableChunk] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        n_pages = len(pdf.pages)

        for page_number, captions in caption_map.items():
            pdf_page = pdf.pages[page_number - 1]

            for caption in captions:
                caption_y = _find_caption_y(pdf_page, caption)
                raw_table: list[list[str | None]] | None = None

                if caption_y is not None:
                    raw_table = _extract_first_table_below_y(pdf_page, caption_y)

                if raw_table is None and page_number < n_pages:
                    next_pdf_page = pdf.pages[page_number]
                    raw_table = _extract_first_table_below_y(next_pdf_page, 0)

                if raw_table is None:
                    continue

                norm_rows = _normalize_table_rows(raw_table)
                if len(norm_rows) < 2:
                    continue

                header = norm_rows[0]
                data_rows = norm_rows[1:]

                page_obj = page_by_num.get(page_number)
                page_width = page_obj.width if page_obj else 600.0
                page_height = page_obj.height if page_obj else 800.0
                approx_y = caption_y if caption_y is not None else page_height * 0.3
                tbox = (0.0, approx_y, page_width, page_height)
                section = resolve_section_spatial(page_number, tbox, spans)

                if len(data_rows) <= 20:
                    chunks.append(
                        TableChunk(
                            chunk_id=f"{journal_id}_{article_id}_tbl_{len(chunks)+1:05d}",
                            journal_id=journal_id,
                            article_id=article_id,
                            source_path=source_path,
                            csv_data=_table_to_csv([header, *data_rows]),
                            header=",".join(header),
                            caption=caption,
                            row_index=None,
                            page_number=page_number,
                            section=section,
                        )
                    )
                else:
                    for ri, row in enumerate(data_rows, start=1):
                        chunks.append(
                            TableChunk(
                                chunk_id=f"{journal_id}_{article_id}_tbl_{len(chunks)+1:05d}",
                                journal_id=journal_id,
                                article_id=article_id,
                                source_path=source_path,
                                csv_data=_table_to_csv([header, row]),
                                header=",".join(header),
                                caption=caption,
                                row_index=ri,
                                page_number=page_number,
                                section=section,
                            )
                        )

    return chunks


def _normalize_table_rows(
    table: Sequence[Sequence[object | None]],
) -> list[list[str]]:
    rows = []
    for raw in table:
        row = [str(c).strip() if c is not None else "" for c in raw]
        if any(row):
            rows.append(row)
    return rows


def _table_to_csv(rows: Sequence[Sequence[str]]) -> str:
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return buf.getvalue().strip()