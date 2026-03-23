"""
backend/src/multimodal/ingestion/image.py

BUG FIXED: pseudo-bbox y0 for spatial section resolution used
`caption_line_index * 10.0` (hardcoded constant).  section.py
text-fallback spans store bbox.y0 = line_index * _APPROX_LINE_HEIGHT (12.0).
resolve_section_spatial() converts y0 back to a line index by dividing by
_APPROX_LINE_HEIGHT.  Using 10.0 here instead of 12.0 caused a ~20% offset,
producing wrong section assignments for images.

Fix: import _APPROX_LINE_HEIGHT from section.py and use it for the
pseudo-bbox construction.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow is optional for metadata only.
    Image = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from ..types import ExtractedImage, PageBlocks
from .section import SectionSpan, _APPROX_LINE_HEIGHT, resolve_section_spatial
from .utils import require_command, run_command


_IMAGE_NAME_RE = re.compile(r"(\d+)")
_PDFIMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def extract_images(
    *,
    pdf_path: Path,
    image_dir: Path,
    pages: Sequence[PageBlocks],
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

    # Build lookup by page_number for safe O(1) access (fixes table.py-style
    # pages[page_number - 1] unsorted-list assumption).
    page_by_num: dict[int, PageBlocks] = {page.page_number: page for page in pages}

    images: list[ExtractedImage] = []
    for image_index, path in enumerate(extracted_files, start=1):
        suffix = path.suffix.lower()
        if suffix not in _PDFIMAGE_EXTENSIONS:
            continue

        page_number = _infer_page_number(path.name)
        section: str | None = None
        caption: str | None = None
        context: str | None = None

        if page_number is not None:
            page_obj = page_by_num.get(page_number)
            lines_on_page = page_obj.text.splitlines() if page_obj and page_obj.text else []
            page_width = page_obj.width if page_obj else 600.0
            page_height = page_obj.height if page_obj else 800.0

            # Phase 6: Try pdfplumber-based extraction first (multi-sentence captions + context)
            caption, context = _extract_figure_caption_pdfplumber(
                str(pdf_path), page_number, image_index
            )

            # Fallback to line-scan if pdfplumber found nothing
            if caption is None:
                caption_line_index = _find_caption_line(image_index, lines_on_page)
                caption = _extract_figure_caption(caption_line_index, lines_on_page)
                context = None

            # FIX: use _APPROX_LINE_HEIGHT (imported from section.py) instead
            # of the previously hardcoded 10.0.
            # resolve_section_spatial() converts y0 via int(y0/_APPROX_LINE_HEIGHT)
            # to compare against span.line_start (a line index). Using the same
            # constant here ensures the round-trip is exact.
            caption_line_index = _find_caption_line(image_index, lines_on_page)
            pseudo_y0 = float(caption_line_index) * _APPROX_LINE_HEIGHT
            pseudo_y0 = max(0.0, min(pseudo_y0, page_height - _APPROX_LINE_HEIGHT))
            pseudo_bbox = (
                50.0,
                pseudo_y0,
                page_width - 50.0,
                pseudo_y0 + _APPROX_LINE_HEIGHT,
            )
            section = resolve_section_spatial(page_number, pseudo_bbox, spans)

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
                caption=caption or _build_image_caption(page_number, section),
                section=section,
                context=context,  # Phase 6: Pass context
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


# ──────────────────────────────────────────────────────────────────────────────
# Phase 6: Enhanced caption and context extraction using pdfplumber
# ──────────────────────────────────────────────────────────────────────────────

def _extract_figure_caption_pdfplumber(
    pdf_path: str,
    page_number: int,
    image_index: int,
) -> tuple[str | None, str | None]:
    """
    Extract (caption, context) using pdfplumber word stream.

    Returns:
        (caption, context) tuple where:
        - caption: full figure caption, multi-sentence, ends at natural boundary
        - context: up to 2 sentences of body text immediately before the figure
    """
    if pdfplumber is None:
        return None, None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not (0 < page_number <= len(pdf.pages)):
                return None, None
            page = pdf.pages[page_number - 1]
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
    except Exception:
        return None, None

    if not words:
        return None, None

    # Group words into lines by y-coordinate proximity
    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (round(w["top"] / 3) * 3, w["x0"])):
        if lines and abs(word["top"] - lines[-1][0]["top"]) <= 4:
            lines[-1].append(word)
        else:
            lines.append([word])

    line_texts = [" ".join(w["text"] for w in ln) for ln in lines]

    # Find caption start by matching "Figure N" or "Fig. N"
    fig_exact = re.compile(
        rf"\b(?:figure|fig\.?)\s*{image_index}\b", re.IGNORECASE
    )
    fig_any = re.compile(r"^\s*(?:figure|fig\.?)\s*\d+", re.IGNORECASE)

    caption_start = next(
        (i for i, t in enumerate(line_texts) if fig_exact.search(t)), -1
    )
    if caption_start == -1:
        caption_start = next(
            (i for i, t in enumerate(line_texts) if fig_any.search(t)), -1
        )
    if caption_start == -1:
        return None, None

    # Collect caption lines
    caption_parts = [line_texts[caption_start]]
    for i in range(caption_start + 1, min(len(line_texts), caption_start + 12)):
        text = line_texts[i].strip()
        if not text:
            break
        if re.match(r"^\s*(?:figure|fig\.?|table|scheme)\s*\d+", text, re.IGNORECASE):
            break
        if text.isupper() and len(text.split()) <= 5:  # section heading
            break
        if re.match(r"^\d+[\.\s]", text):  # numbered section
            break
        caption_parts.append(text)
        if text.endswith("."):
            break

    caption = " ".join(caption_parts).strip() or None

    # Collect context: 2 sentences before the caption
    raw_context = " ".join(
        ln.strip()
        for ln in line_texts[max(0, caption_start - 6) : caption_start]
        if ln.strip()
    )
    sentences = re.split(r"(?<=[.!?])\s+", raw_context)
    context = " ".join(sentences[-2:]).strip() if sentences else None

    return caption, context


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


def _extract_figure_caption(caption_line: int, page_lines: Sequence[str]) -> str | None:
    if caption_line < 0 or caption_line >= len(page_lines):
        return None

    caption_pattern = re.compile(r"^\s*(?:figure|fig\.?)\s*\d+\b", re.IGNORECASE)
    start_line = page_lines[caption_line].strip()
    if not caption_pattern.search(start_line):
        return None

    caption_parts = [start_line]
    for next_line in page_lines[caption_line + 1:caption_line + 5]:
        stripped = next_line.strip()
        if not stripped:
            break
        if caption_pattern.search(stripped):
            break
        if re.match(r"^\s*(?:table\s+\d+|references?)\b", stripped, re.IGNORECASE):
            break
        if len(stripped.split()) <= 3 and stripped.isupper():
            break
        caption_parts.append(stripped)
        if stripped.endswith("."):
            break

    caption = " ".join(caption_parts).strip()
    return caption or None


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