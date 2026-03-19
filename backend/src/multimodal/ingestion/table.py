from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Sequence

try:
    import pdfplumber
except ImportError:  # pragma: no cover - verified at runtime.
    pdfplumber = None  # type: ignore[assignment]

from ..types import PageBlocks, TableChunk
from .section import SectionSpan, resolve_section_spatial
from .utils import normalise_line


def build_table_text_exclusion(pdf_path: Path) -> tuple[set[str], set[str]]:
    if pdfplumber is None:
        return set(), set()

    cell_exclusion: set[str] = set()
    row_exclusion: set[str] = set()

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                for row in table:
                    row_cells: list[str] = []
                    for cell in row:
                        if not cell:
                            continue
                        normalised = normalise_line(str(cell))
                        if normalised:
                            cell_exclusion.add(normalised)
                            row_cells.append(normalised)
                    if row_cells:
                        row_exclusion.add(" ".join(row_cells))

    return cell_exclusion, row_exclusion


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

    page_lines = {
        page.page_number: page.text.splitlines()
        for page in pages
    }
    page_widths = {page.page_number: page.width for page in pages}
    page_heights = {page.page_number: page.height for page in pages}

    chunks: list[TableChunk] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            table_objects = page.find_tables() or []
            raw_tables = [table.extract() for table in table_objects]
            if not raw_tables:
                raw_tables = page.extract_tables() or []

            for table_index, raw_table in enumerate(raw_tables):
                normalized_rows = _normalize_table_rows(raw_table)
                if len(normalized_rows) < 2:
                    continue

                header = normalized_rows[0]
                data_rows = normalized_rows[1:]

                table_bbox = None
                if table_index < len(table_objects) and table_objects[table_index].bbox:
                    x0, y0, x1, y1 = table_objects[table_index].bbox
                    table_bbox = (float(x0), float(y0), float(x1), float(y1))
                    
                else:
                    # Fallback to estimating a bbox if pdfplumber didn't provide one readily
                    # This is a heuristic and might not be accurate.
                    page_width = page_widths.get(page_index, 600.0)
                    page_height = page_heights.get(page_index, 800.0)
                    # Estimate position based on header, and assign a rough bbox
                    estimated_line_pos = _estimate_line_position(header, page_lines.get(page_index, []))
                    estimated_y0 = estimated_line_pos * 10.0 # Assuming 10 units per line
                    table_bbox = (50.0, estimated_y0, page_width - 50.0, estimated_y0 + 50.0) # A rough block

                section = resolve_section_spatial(page_index, table_bbox, spans) if table_bbox else None
                caption = _find_table_caption(
                    table_index + 1,
                    int(table_bbox[1] / 10) if table_bbox else 0, # Pass estimated line pos to old caption finder
                    page_lines.get(page_index, []),
                )

                if len(data_rows) <= 20:
                    csv_data = _table_to_csv([header, *data_rows])
                    chunks.append(
                        TableChunk(
                            chunk_id=f"{journal_id}_{article_id}_tbl_{len(chunks) + 1:05d}",
                            journal_id=journal_id,
                            article_id=article_id,
                            source_path=source_path,
                            csv_data=csv_data,
                            header=",".join(header),
                            caption=caption,
                            row_index=None,
                            page_number=page_index,
                            section=section,
                        )
                    )
                    continue

                for row_index, row in enumerate(data_rows, start=1):
                    csv_data = _table_to_csv([header, row])
                    chunks.append(
                        TableChunk(
                            chunk_id=f"{journal_id}_{article_id}_tbl_{len(chunks) + 1:05d}",
                            journal_id=journal_id,
                            article_id=article_id,
                            source_path=source_path,
                            csv_data=csv_data,
                            header=",".join(header),
                            caption=caption,
                            row_index=row_index,
                            page_number=page_index,
                            section=section,
                        )
                    )

    return chunks


def _normalize_table_rows(table: Sequence[Sequence[object | None]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_row in table:
        row = [str(cell).strip() if cell is not None else "" for cell in raw_row]
        if any(cell for cell in row):
            rows.append(row)
    return rows


def _table_to_csv(rows: Sequence[Sequence[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)
    return buffer.getvalue().strip()


def _estimate_line_position(header_row: Sequence[str], page_lines: Sequence[str]) -> int:
    header_cells = [normalise_line(cell) for cell in header_row if normalise_line(cell)]
    if not header_cells:
        return 0

    minimum_hits = min(2, len(header_cells))
    for index, line in enumerate(page_lines):
        normalised = normalise_line(line)
        cell_hits = sum(1 for cell in header_cells if len(cell) > 1 and cell in normalised)
        if cell_hits >= minimum_hits:
            return index

    first_cell = header_cells[0]
    for index, line in enumerate(page_lines):
        if first_cell in normalise_line(line):
            return index

    return 0


def _find_table_caption(
    table_idx: int,
    line_position: int,
    page_lines: Sequence[str],
) -> str | None:
    search_start = max(0, line_position - 5)
    search_window = page_lines[search_start:line_position + 2]
    indexed_pattern = re.compile(rf"\bTable\s+{table_idx}\b", re.IGNORECASE)
    any_pattern = re.compile(r"\bTable\s+\d+\b", re.IGNORECASE)

    for line in search_window:
        stripped = line.strip()
        if indexed_pattern.search(stripped):
            return stripped
    for line in search_window:
        stripped = line.strip()
        if any_pattern.search(stripped):
            return stripped
    return None
