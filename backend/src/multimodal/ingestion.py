"""PDF ingestion helpers for multimodal article processing."""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import pdfplumber
except ImportError:  # pragma: no cover - verified at runtime.
    pdfplumber = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow is optional for metadata only.
    Image = None

from .types import (
    EquationChunk,
    ExtractedImage,
    IngestedDocument,
    TableChunk,
    TextChunk,
    build_document_id,
    parse_pdf_filename,
)

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_HEADING_NUMBER_RE = re.compile(r"^\s*(\d+(\.\d+)*|[IVXLC]+)\s+[A-Z]")
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
_MATH_SYMBOLS = set("=+-*/^_<>%()[]{}|\\∑∫√≈≠≤≥±∞∂∆∇λμσπθβα")
_LATEX_MARKERS = (
    "\\frac",
    "\\sum",
    "\\int",
    "\\alpha",
    "\\beta",
    "\\theta",
    "\\lambda",
    "\\mu",
    "\\sigma",
    "\\pi",
    "\\sqrt",
)
_IMAGE_NAME_RE = re.compile(r"(\d+)")
_PDFIMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(slots=True)
class _PageText:
    page_number: int
    text: str
    token_count: int


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
        _ensure_file_exists(source_pdf)
        assets_root.mkdir(parents=True, exist_ok=True)

        journal_id, article_id = parse_pdf_filename(source_pdf)
        document_id = build_document_id(source_pdf)
        image_dir = assets_root / document_id / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        pages = _extract_page_text(source_pdf)
        section_map = _build_section_map(pages)

        return IngestedDocument(
            document_id=document_id,
            journal_id=journal_id,
            article_id=article_id,
            source_path=str(source_pdf),
            text_chunks=_build_text_chunks(
                pages=pages,
                section_map=section_map,
                journal_id=journal_id,
                article_id=article_id,
                source_path=str(source_pdf),
                token_limit=self.chunk_token_limit,
            ),
            equation_chunks=_extract_equations(
                pages=pages,
                section_map=section_map,
                journal_id=journal_id,
                article_id=article_id,
                source_path=str(source_pdf),
            ),
            table_chunks=_extract_tables(
                pdf_path=source_pdf,
                section_map=section_map,
                journal_id=journal_id,
                article_id=article_id,
                source_path=str(source_pdf),
            ),
            images=_extract_images(
                pdf_path=source_pdf,
                image_dir=image_dir,
                section_map=section_map,
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


def _extract_page_text(pdf_path: Path) -> list[_PageText]:
    _require_command("pdftotext")

    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".txt", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        command = [
            "pdftotext",
            "-layout",
            "-enc",
            "UTF-8",
            "-eol",
            "unix",
            str(pdf_path),
            str(output_path),
        ]
        _run_command(command, "extracting text from PDF")
        raw_text = output_path.read_text(encoding="utf-8", errors="replace")
    finally:
        output_path.unlink(missing_ok=True)

    return [
        _PageText(
            page_number=page_number,
            text=page_text.strip(),
            token_count=_estimate_tokens(page_text.strip()),
        )
        for page_number, page_text in enumerate(raw_text.split("\f"), start=1)
    ]


def _build_section_map(pages: Sequence[_PageText]) -> dict[int, str]:
    section_map: dict[int, str] = {}
    current_section = "Document"

    for page in pages:
        detected = _detect_page_section(page.text)
        if detected:
            current_section = detected
        section_map[page.page_number] = current_section

    return section_map


def _detect_page_section(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:12]:
        normalized = re.sub(r"[:.]+$", "", line).strip()
        lowered = normalized.lower()
        if lowered in _KNOWN_SECTION_NAMES:
            return normalized.title()
        if _HEADING_NUMBER_RE.match(normalized):
            return normalized
        if normalized.isupper() and 3 <= len(normalized) <= 80 and len(normalized.split()) <= 8:
            return normalized.title()
    return None


def _build_text_chunks(
    *,
    pages: Sequence[_PageText],
    section_map: dict[int, str],
    journal_id: str,
    article_id: str,
    source_path: str,
    token_limit: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    buffer_parts: list[str] = []
    buffer_pages: list[int] = []
    buffer_tokens = 0
    current_section: str | None = None

    def flush() -> None:
        nonlocal buffer_parts, buffer_pages, buffer_tokens
        if not buffer_parts or not buffer_pages:
            buffer_parts = []
            buffer_pages = []
            buffer_tokens = 0
            return

        text = "\n\n".join(part for part in buffer_parts if part).strip()
        if text:
            chunks.append(
                TextChunk(
                    chunk_id=f"{journal_id}_{article_id}_text_{len(chunks) + 1:05d}",
                    journal_id=journal_id,
                    article_id=article_id,
                    source_path=source_path,
                    text=text,
                    token_count=_estimate_tokens(text),
                    page_start=buffer_pages[0],
                    page_end=buffer_pages[-1],
                    section=current_section,
                    caption=_build_text_caption(current_section, buffer_pages[0], buffer_pages[-1]),
                )
            )
        buffer_parts = []
        buffer_pages = []
        buffer_tokens = 0

    for page in pages:
        if not page.text:
            continue

        page_section = section_map.get(page.page_number)
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page.text) if part.strip()]

        if current_section is not None and page_section != current_section and buffer_parts:
            flush()

        current_section = page_section
        for paragraph in paragraphs:
            paragraph_tokens = _estimate_tokens(paragraph)

            if paragraph_tokens > token_limit:
                flush()
                for fragment in _split_long_text(paragraph, token_limit):
                    if not fragment:
                        continue
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{journal_id}_{article_id}_text_{len(chunks) + 1:05d}",
                            journal_id=journal_id,
                            article_id=article_id,
                            source_path=source_path,
                            text=fragment,
                            token_count=_estimate_tokens(fragment),
                            page_start=page.page_number,
                            page_end=page.page_number,
                            section=current_section,
                            caption=_build_text_caption(
                                current_section,
                                page.page_number,
                                page.page_number,
                            ),
                        )
                    )
                continue

            if buffer_parts and buffer_tokens + paragraph_tokens > token_limit:
                flush()
                current_section = page_section

            buffer_parts.append(paragraph)
            buffer_pages.append(page.page_number)
            buffer_tokens += paragraph_tokens

    flush()
    return chunks


def _split_long_text(text: str, token_limit: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            chunks.append(" ".join(buffer).strip())
            buffer = []
            buffer_tokens = 0

    for sentence in sentences:
        if not sentence:
            continue
        sentence_tokens = _estimate_tokens(sentence)
        if sentence_tokens > token_limit:
            flush()
            chunks.extend(_token_slices(sentence, token_limit))
            continue
        if buffer and buffer_tokens + sentence_tokens > token_limit:
            flush()
        buffer.append(sentence)
        buffer_tokens += sentence_tokens

    flush()
    return chunks


def _extract_equations(
    *,
    pages: Sequence[_PageText],
    section_map: dict[int, str],
    journal_id: str,
    article_id: str,
    source_path: str,
) -> list[EquationChunk]:
    chunks: list[EquationChunk] = []

    for page in pages:
        lines = [line.rstrip() for line in page.text.splitlines()]
        block: list[str] = []
        block_start = 0

        def flush(end_index: int) -> None:
            nonlocal block, block_start
            if not block:
                return
            latex = "\n".join(line.strip() for line in block if line.strip()).strip()
            if latex:
                chunks.append(
                    EquationChunk(
                        chunk_id=f"{journal_id}_{article_id}_eq_{len(chunks) + 1:05d}",
                        journal_id=journal_id,
                        article_id=article_id,
                        source_path=source_path,
                        latex=latex,
                        context=_extract_equation_context(lines, block_start, end_index),
                        page_number=page.page_number,
                        section=section_map.get(page.page_number),
                    )
                )
            block = []
            block_start = 0

        for index, line in enumerate(lines):
            if _is_equation_line(line):
                if not block:
                    block_start = index
                block.append(line)
                continue
            flush(index)

        flush(len(lines))

    return chunks


def _is_equation_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    non_space_chars = [char for char in stripped if not char.isspace()]
    if not non_space_chars:
        return False

    math_hits = sum(char in _MATH_SYMBOLS for char in non_space_chars)
    math_density = math_hits / len(non_space_chars)

    if any(marker in stripped for marker in _LATEX_MARKERS):
        return True
    if math_density >= 0.15 and any(char in stripped for char in "=<>±∑∫√"):
        return True
    if re.search(r"\b([A-Za-z]\s*=\s*.+|.+\s*=\s*[A-Za-z0-9(])", stripped):
        return True
    if re.search(r"\([^)]+\)\s*[=<>]", stripped):
        return True
    return False


def _extract_equation_context(lines: Sequence[str], start: int, end: int) -> str:
    surrounding = list(lines[max(0, start - 2):start]) + list(lines[end:min(len(lines), end + 2)])
    text = " ".join(line.strip() for line in surrounding if line.strip())
    if not text:
        return "Equation extracted from surrounding document context."

    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        cleaned = sentence.strip()
        if cleaned:
            return cleaned
    return text.strip()


def _extract_tables(
    *,
    pdf_path: Path,
    section_map: dict[int, str],
    journal_id: str,
    article_id: str,
    source_path: str,
) -> list[TableChunk]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required for table extraction.")

    chunks: list[TableChunk] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            extracted_tables = page.extract_tables() or []
            for table in extracted_tables:
                normalized_rows = _normalize_table_rows(table)
                if len(normalized_rows) < 2:
                    continue

                header = normalized_rows[0]
                data_rows = normalized_rows[1:]
                section = section_map.get(page_index)

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


def _extract_images(
    *,
    pdf_path: Path,
    image_dir: Path,
    section_map: dict[int, str],
    journal_id: str,
    article_id: str,
    source_path: str,
) -> list[ExtractedImage]:
    _require_command("pdfimages")

    prefix = image_dir / f"{pdf_path.stem}_img"
    command = [
        "pdfimages",
        "-png",
        "-j",
        "-p",
        "-q",
        str(pdf_path),
        str(prefix),
    ]
    _run_command(command, "extracting images from PDF")

    extracted_files = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.name.startswith(prefix.name)
        ],
        key=_image_sort_key,
    )

    images: list[ExtractedImage] = []
    for path in extracted_files:
        suffix = path.suffix.lower()
        if suffix not in _PDFIMAGE_EXTENSIONS:
            continue

        page_number = _infer_page_number(path.name)
        width: int | None = None
        height: int | None = None
        if Image is not None:
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except Exception:
                width = height = None

        images.append(
            ExtractedImage(
                image_id=f"{journal_id}_{article_id}_img_{len(images) + 1:05d}",
                journal_id=journal_id,
                article_id=article_id,
                source_path=source_path,
                file_path=str(path),
                page_number=page_number,
                mime_type=_mime_type_for_suffix(suffix),
                width=width,
                height=height,
                caption=_build_image_caption(page_number, section_map.get(page_number)),
                section=section_map.get(page_number),
            )
        )

    return images


def _image_sort_key(path: Path) -> tuple[int, int, str]:
    numbers = [int(value) for value in _IMAGE_NAME_RE.findall(path.stem)]
    if len(numbers) >= 2:
        return (numbers[-2], numbers[-1], path.name)
    if len(numbers) == 1:
        return (numbers[0], 0, path.name)
    return (0, 0, path.name)


def _infer_page_number(file_name: str) -> int | None:
    numbers = [int(value) for value in _IMAGE_NAME_RE.findall(file_name)]
    if not numbers:
        return None
    if len(numbers) >= 2:
        return numbers[-2]
    return numbers[0]


def _build_text_caption(section: str | None, page_start: int, page_end: int) -> str:
    section_prefix = f"{section} " if section else ""
    if page_start == page_end:
        return f"{section_prefix}page {page_start}".strip()
    return f"{section_prefix}pages {page_start}-{page_end}".strip()


def _build_image_caption(page_number: int | None, section: str | None) -> str:
    if section and page_number is not None:
        return f"{section} image from page {page_number}"
    if page_number is not None:
        return f"Extracted image from page {page_number}"
    if section:
        return f"{section} image"
    return "Extracted image"


def _mime_type_for_suffix(suffix: str) -> str:
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/png"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


def _estimate_tokens(text: str) -> int:
    return len(_tokenize(text))


def _token_slices(text: str, token_limit: int) -> list[str]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    return [
        " ".join(tokens[index:index + token_limit]).strip()
        for index in range(0, len(tokens), token_limit)
        if tokens[index:index + token_limit]
    ]


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"PDF path is not a file: {path}")


def _require_command(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"Required command not found on PATH: {command_name}. "
            "This ingestion module expects Poppler utilities to be installed."
        )


def _run_command(command: Sequence[str], action: str) -> None:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            json.dumps(
                {
                    "action": action,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                },
                indent=2,
            )
        )
