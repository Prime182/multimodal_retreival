from __future__ import annotations

from pathlib import Path

from ..types import IngestedDocument, build_document_id, parse_pdf_filename
from .equation import extract_equations
from .image import extract_images
from .section import build_section_spans_from_blocks
from .table import build_table_text_exclusion, extract_tables
from .text import build_text_chunks
from .utils import ensure_file_exists, extract_page_blocks


class PDFIngestionAgent:
    """Convert a PDF into text, equation, table, and image records."""

    def __init__(self, *, chunk_token_limit: int = 1200) -> None:
        self.chunk_token_limit = max(1, int(chunk_token_limit))

    def process_pdf(
        self,
        pdf_path: str | Path,
        assets_dir: str | Path,
    ) -> IngestedDocument:
        source_pdf = Path(pdf_path).expanduser().resolve()
        assets_root = Path(assets_dir).expanduser().resolve()
        ensure_file_exists(source_pdf)
        assets_root.mkdir(parents=True, exist_ok=True)

        journal_id, article_id = parse_pdf_filename(source_pdf)
        document_id = build_document_id(source_pdf)
        image_dir = assets_root / document_id / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        pages = extract_page_blocks(source_pdf)
        spans = build_section_spans_from_blocks(pages)
        cell_exclusion, row_exclusion = build_table_text_exclusion(source_pdf)

        return IngestedDocument(
            document_id=document_id,
            journal_id=journal_id,
            article_id=article_id,
            source_path=str(source_pdf),
            text_chunks=build_text_chunks(
                pages=pages,
                spans=spans,
                journal_id=journal_id,
                article_id=article_id,
                source_path=str(source_pdf),
                token_limit=self.chunk_token_limit,
            ),
            equation_chunks=extract_equations(
                pages=pages,
                spans=spans,
                journal_id=journal_id,
                article_id=article_id,
                source_path=str(source_pdf),
                cell_exclusion=cell_exclusion,
                row_exclusion=row_exclusion,
            ),
            table_chunks=extract_tables(
                pdf_path=source_pdf,
                pages=pages,
                spans=spans,
                journal_id=journal_id,
                article_id=article_id,
                source_path=str(source_pdf),
            ),
            images=extract_images(
                pdf_path=source_pdf,
                image_dir=image_dir,
                pages=pages,
                spans=spans,
                journal_id=journal_id,
                article_id=article_id,
                source_path=str(source_pdf),
            ),
        )


def ingest_pdf(
    pdf_path: str | Path,
    assets_dir: str | Path,
    *,
    chunk_token_limit: int = 1200,
) -> IngestedDocument:
    return PDFIngestionAgent(chunk_token_limit=chunk_token_limit).process_pdf(
        pdf_path,
        assets_dir,
    )
