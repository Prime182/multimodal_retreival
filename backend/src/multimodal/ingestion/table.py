"""
backend/src/multimodal/ingestion/table.py

Caption-anchored table extraction.

BUG FIXED: pages[page_number - 1] indexed access assumes the `pages` list
is sorted in ascending page order AND that page numbers are exactly 1-indexed
with no gaps. If pymupdf4llm returns chunks out of order (possible with
complex multi-column PDFs), pages[0] is not necessarily page 1.

Fix: build a dict {page.page_number: page} for O(1) safe lookup by page
number.  This also makes the code resilient to missing pages (None-safe).
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


TABLE_CAPTION_RE = re.compile(
    r"^\s*Table\s+\d+[\s.:–-]",
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


def _extract_first_table_below_y(page, y: float) -> list[list[str | None]] | None:
    try:
        cropped = page.within_bbox((0, y, page.width, page.height), relative=False)
    except Exception:
        return None

    for settings in (_SETTINGS_LINES, _SETTINGS_TEXT):
        try:
            tables = cropped.extract_tables(table_settings=settings) or []
        except Exception:
            tables = []
        valid = [t for t in tables if t and len(t) >= 2]
        if valid:
            return valid[0]
    return None


def build_table_text_exclusion(pdf_path: Path) -> tuple[set[str], set[str]]:
    if pdfplumber is None:
        return set(), set()

    cell_ex: set[str] = set()
    row_ex: set[str] = set()

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for settings in (_SETTINGS_LINES, _SETTINGS_TEXT):
                try:
                    tables = page.extract_tables(table_settings=settings) or []
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
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required for table extraction.")

    # FIX: build a dict so lookup by page_number is safe regardless of list order.
    # Previously: pages[page_number - 1]  ← assumes sorted, 1-indexed, no gaps.
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
            pdf_page = pdf.pages[page_number - 1]  # pdfplumber is always 0-indexed

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

                # FIX: use page_by_num dict instead of pages[page_number - 1].
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