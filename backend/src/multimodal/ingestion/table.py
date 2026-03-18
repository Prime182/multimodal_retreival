from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Sequence

try:
    import pdfplumber
except ImportError:  # pragma: no cover - verified at runtime.
    pdfplumber = None  # type: ignore[assignment]

from ..types import TableChunk
from .section import SectionSpan, resolve_section
from .utils import PageText, normalise_line


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
    pages: Sequence[PageText],
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

                if table_index < len(table_objects):
                    top_y = table_objects[table_index].bbox[1]
                    line_pos = max(0, int(top_y / 12))
                else:
                    line_pos = _estimate_line_position(header, page_lines.get(page_index, []))

                section = resolve_section(page_index, line_pos, spans)

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
