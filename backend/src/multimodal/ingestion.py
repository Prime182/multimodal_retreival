"""PDF ingestion helpers for multimodal article processing.

This module keeps the dependency surface small by leaning on the Poppler
command-line tools already available in the environment:

* ``pdftotext`` for page text extraction
* ``pdfimages`` for extracting embedded JPEG/PNG-compatible images

The public entry point, :func:`ingest_pdf`, returns a plain dictionary that is
easy to pass to the next agent in the pipeline.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow is optional for metadata only.
    Image = None

from .types import (
    ExtractedImage as ImageRecord,
    IngestedDocument,
    TextChunk as TextChunkRecord,
    build_document_id,
)

__all__ = [
    "PageText",
    "TextChunk",
    "ExtractedImage",
    "IngestedPDF",
    "PDFIngestionAgent",
    "ingest_pdf",
]

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_IMAGE_NAME_RE = re.compile(r"(\d+)")
_PDFIMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(slots=True)
class PageText:
    """Text extracted from a single PDF page."""

    page_number: int
    text: str
    token_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TextChunk:
    """A chunk of text ready for embedding."""

    chunk_id: str
    page_start: int
    page_end: int
    text: str
    token_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExtractedImage:
    """An embedded image extracted from the PDF."""

    image_id: str
    page_number: int | None
    file_path: str
    mime_type: str
    width: int | None = None
    height: int | None = None
    caption: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IngestedPDF:
    """Complete ingestion output for downstream storage and embedding."""

    source_pdf: str
    assets_dir: str
    pages: list[PageText]
    text_chunks: list[TextChunk]
    images: list[ExtractedImage]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_pdf": self.source_pdf,
            "assets_dir": self.assets_dir,
            "pages": [page.as_dict() for page in self.pages],
            "text_chunks": [chunk.as_dict() for chunk in self.text_chunks],
            "images": [image.as_dict() for image in self.images],
        }


def ingest_pdf(
    pdf_path: str | Path,
    assets_dir: str | Path,
    *,
    chunk_token_limit: int = 8192,
) -> dict[str, Any]:
    """Ingest a PDF into page text, text chunks, and extracted image files.

    Args:
        pdf_path: PDF file to parse.
        assets_dir: Directory where extracted images should be stored.
        chunk_token_limit: Maximum approximate token count per text chunk.

    Returns:
        A JSON-friendly dictionary containing page text, chunked text, and
        image metadata.
    """

    source_pdf = Path(pdf_path).expanduser().resolve()
    assets_root = Path(assets_dir).expanduser().resolve()
    _ensure_file_exists(source_pdf)
    assets_root.mkdir(parents=True, exist_ok=True)

    token_limit = max(1, min(int(chunk_token_limit), 8192))
    run_id = uuid.uuid4().hex[:12]
    run_assets_dir = assets_root / source_pdf.stem / run_id
    image_dir = run_assets_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    pages = _extract_page_text(source_pdf)
    text_chunks = _chunk_page_text(pages, token_limit)
    images = _extract_images(source_pdf, image_dir)

    return IngestedPDF(
        source_pdf=str(source_pdf),
        assets_dir=str(run_assets_dir),
        pages=pages,
        text_chunks=text_chunks,
        images=images,
    ).as_dict()


class PDFIngestionAgent:
    """Convert a PDF into text chunks and extracted image records."""

    def __init__(self, *, chunk_token_limit: int = 8192) -> None:
        self.chunk_token_limit = chunk_token_limit

    def process_pdf(
        self,
        pdf_path: str | Path,
        assets_dir: str | Path,
    ) -> IngestedDocument:
        payload = ingest_pdf(
            pdf_path,
            assets_dir,
            chunk_token_limit=self.chunk_token_limit,
        )
        source_path = payload["source_pdf"]

        text_chunks = [
            TextChunkRecord(
                chunk_id=chunk["chunk_id"],
                source_path=source_path,
                text=chunk["text"],
                token_count=chunk["token_count"],
                page_start=chunk["page_start"],
                page_end=chunk["page_end"],
                caption=f"Pages {chunk['page_start']}-{chunk['page_end']}",
            )
            for chunk in payload["text_chunks"]
        ]

        images = [
            ImageRecord(
                image_id=image["image_id"],
                source_path=source_path,
                file_path=image["file_path"],
                page_number=image["page_number"] or 0,
                mime_type=image["mime_type"],
                width=image.get("width"),
                height=image.get("height"),
                caption=image.get("caption") or _build_image_caption(image),
            )
            for image in payload["images"]
        ]

        return IngestedDocument(
            document_id=build_document_id(source_path),
            source_path=source_path,
            text_chunks=text_chunks,
            images=images,
        )


def _extract_page_text(pdf_path: Path) -> list[PageText]:
    _require_command("pdftotext")

    with tempfile.NamedTemporaryFile(
        mode="w+b", suffix=".txt", delete=False
    ) as handle:
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

    pages: list[PageText] = []
    for page_number, page_text in enumerate(raw_text.split("\f"), start=1):
        normalized = page_text.strip()
        pages.append(
            PageText(
                page_number=page_number,
                text=normalized,
                token_count=_estimate_tokens(normalized),
            )
        )
    return pages


def _chunk_page_text(pages: Sequence[PageText], token_limit: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    buffer_text: list[str] = []
    buffer_pages: list[int] = []
    buffer_tokens = 0

    def flush_buffer() -> None:
        nonlocal buffer_text, buffer_pages, buffer_tokens
        if not buffer_text:
            return
        text = "\n\n".join(buffer_text).strip()
        if text:
            chunks.append(
                TextChunk(
                    chunk_id=f"chunk_{len(chunks) + 1:05d}",
                    page_start=buffer_pages[0],
                    page_end=buffer_pages[-1],
                    text=text,
                    token_count=buffer_tokens,
                )
            )
        buffer_text = []
        buffer_pages = []
        buffer_tokens = 0

    for page in pages:
        if not page.text:
            continue

        if page.token_count > token_limit:
            flush_buffer()
            for part in _split_long_text(page.text, token_limit):
                part_tokens = _estimate_tokens(part)
                if not part:
                    continue
                chunks.append(
                    TextChunk(
                        chunk_id=f"chunk_{len(chunks) + 1:05d}",
                        page_start=page.page_number,
                        page_end=page.page_number,
                        text=part,
                        token_count=part_tokens,
                    )
                )
            continue

        if buffer_text and buffer_tokens + page.token_count > token_limit:
            flush_buffer()

        buffer_text.append(page.text)
        buffer_pages.append(page.page_number)
        buffer_tokens += page.token_count

    flush_buffer()
    return chunks


def _split_long_text(text: str, token_limit: int) -> list[str]:
    """Split oversize text while keeping paragraph boundaries when possible."""

    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            chunks.append("\n\n".join(buffer).strip())
            buffer = []
            buffer_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = _estimate_tokens(paragraph)
        if paragraph_tokens > token_limit:
            flush()
            chunks.extend(_token_slices(paragraph, token_limit))
            continue

        if buffer and buffer_tokens + paragraph_tokens > token_limit:
            flush()

        buffer.append(paragraph)
        buffer_tokens += paragraph_tokens

    flush()
    return [chunk for chunk in chunks if chunk.strip()]


def _token_slices(text: str, token_limit: int) -> list[str]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    return [
        " ".join(tokens[index : index + token_limit]).strip()
        for index in range(0, len(tokens), token_limit)
        if tokens[index : index + token_limit]
    ]


def _extract_images(pdf_path: Path, image_dir: Path) -> list[ExtractedImage]:
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
    for index, path in enumerate(extracted_files, start=1):
        suffix = path.suffix.lower()
        if suffix not in _PDFIMAGE_EXTENSIONS:
            continue

        page_number = _infer_page_number(path.name)
        width: int | None = None
        height: int | None = None
        if Image is not None:
            try:
                with Image.open(path) as img:
                    width, height = img.size
            except Exception:
                width = height = None

        images.append(
            ExtractedImage(
                image_id=f"image_{index:05d}",
                page_number=page_number,
                file_path=str(path),
                mime_type=_mime_type_for_suffix(suffix),
                width=width,
                height=height,
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


def _build_image_caption(image: dict[str, Any]) -> str:
    page_number = image.get("page_number")
    if page_number:
        return f"Extracted image from page {page_number}"
    return f"Extracted image {image['image_id']}"


def _mime_type_for_suffix(suffix: str) -> str:
    if suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/png"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


def _estimate_tokens(text: str) -> int:
    return len(_tokenize(text))


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
