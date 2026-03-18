from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow is optional for metadata only.
    Image = None

from ..types import ExtractedImage
from .section import SectionSpan, resolve_section
from .utils import PageText, require_command, run_command


_IMAGE_NAME_RE = re.compile(r"(\d+)")
_PDFIMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def extract_images(
    *,
    pdf_path: Path,
    image_dir: Path,
    pages: Sequence[PageText],
    spans: Sequence[SectionSpan],
    journal_id: str,
    article_id: str,
    source_path: str,
) -> list[ExtractedImage]:
    require_command("pdfimages")

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
    run_command(command, "extracting images from PDF")

    extracted_files = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.name.startswith(prefix.name)
        ],
        key=_image_sort_key,
    )
    page_lines = {page.page_number: page.text.splitlines() for page in pages}

    images: list[ExtractedImage] = []
    for image_index, path in enumerate(extracted_files, start=1):
        suffix = path.suffix.lower()
        if suffix not in _PDFIMAGE_EXTENSIONS:
            continue

        page_number = _infer_page_number(path.name)
        section: str | None = None
        caption_line = 0
        if page_number is not None:
            caption_line = _find_caption_line(image_index, page_lines.get(page_number, []))
            section = resolve_section(page_number, caption_line, spans)

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
                caption=_build_image_caption(page_number, section),
                section=section,
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


def _find_caption_line(image_index: int, page_lines: Sequence[str]) -> int:
    numbered_pattern = re.compile(
        rf"\b(?:figure|fig\.?)\s*{image_index}\b",
        re.IGNORECASE,
    )
    generic_pattern = re.compile(r"^\s*(?:figure|fig\.?)\s*\d+", re.IGNORECASE)

    for index, line in enumerate(page_lines):
        if numbered_pattern.search(line):
            return index
    for index, line in enumerate(page_lines):
        if generic_pattern.search(line):
            return index
    return 0


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
