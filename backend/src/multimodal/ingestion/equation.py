from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency in some environments
    pdfplumber = None  # type: ignore[assignment]

try:
    from pdf2image import convert_from_path
except ImportError:  # pragma: no cover - optional dependency in some environments
    convert_from_path = None  # type: ignore[assignment]

from ..types import ExtractedImage, PageBlocks, SectionSpan
from .section import resolve_section_spatial
from .utils import require_command

DPI = 300
PADDING = 10
OVERLAP_TOL = 1.0
BLOCK_GAP = 10.0
MIN_BLOCK_WIDTH = 40.0
MIN_BLOCK_HEIGHT = 18.0
CENTRE_MIN = 0.20
CENTRE_MAX = 0.80
LINE_TOL = 4.0

EQUATION_FONTS = {
    "ATRMRS+ArnoPro-Regular",
    "ZDNDLO+ArnoPro-Regular",
    "IIVZSS+ArnoPro-Italic",
    "SPJFMW+ArnoPro-Regular",
    "WOCWKH+STIXGeneral-Regular",
    "FHKMDR+STIXGeneral-Regular",
    "KFDOGS+STIXGeneral-Regular",
}
MATH_OPERATOR_FONTS = {
    "JMRPAX+STIXGeneral-Regular",
    "KFDOGS+STIXGeneral-Regular",
    "FHKMDR+STIXGeneral-Regular",
    "WOCWKH+STIXGeneral-Regular",
}


def cluster_by_overlap(words: list[dict[str, Any]], tol: float = OVERLAP_TOL) -> list[list[Any]]:
    """Merge words whose vertical extents overlap into clusters."""
    if not words:
        return []

    words_sorted = sorted(words, key=lambda word: word["top"])
    clusters: list[list[Any]] = []
    for word in words_sorted:
        placed = False
        for cluster in clusters:
            if word["top"] < cluster[1] + tol:
                cluster[2].append(word)
                cluster[1] = max(cluster[1], word["bottom"])
                placed = True
                break
        if not placed:
            clusters.append([word["top"], word["bottom"], [word]])
    return clusters


def clusters_to_blocks(clusters: list[list[Any]], gap: float = BLOCK_GAP) -> list[list[list[Any]]]:
    """Merge consecutive clusters separated by <= gap into equation blocks."""
    if not clusters:
        return []

    blocks: list[list[list[Any]]] = []
    current = [clusters[0]]
    prev_bottom = clusters[0][1]
    for cluster in clusters[1:]:
        if cluster[0] - prev_bottom <= gap:
            current.append(cluster)
        else:
            blocks.append(current)
            current = [cluster]
        prev_bottom = max(prev_bottom, cluster[1])
    blocks.append(current)
    return blocks


def should_keep_equation_box(
    x0: float,
    top: float,
    x1: float,
    bottom: float,
    page_width: float,
) -> bool:
    block_width = x1 - x0
    block_height = bottom - top
    if block_width < MIN_BLOCK_WIDTH or block_height < MIN_BLOCK_HEIGHT:
        return False

    midpoint_fraction = ((x0 + x1) / 2.0) / page_width if page_width else 0.0
    return CENTRE_MIN <= midpoint_fraction <= CENTRE_MAX


def _line_bucket(word: dict[str, Any]) -> float:
    return round(word["top"] / LINE_TOL) * LINE_TOL


def _build_equation_boxes(pdf_path: str | Path) -> list[dict[str, float | int]]:
    boxes: list[dict[str, float | int]] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["fontname", "size"]) or []
            equation_words = [
                word for word in words if word.get("fontname", "") in EQUATION_FONTS
            ]
            if not equation_words:
                continue

            equation_lines = {_line_bucket(word) for word in equation_words}
            operator_words = [
                word
                for word in words
                if word.get("fontname", "") in MATH_OPERATOR_FONTS
                and _line_bucket(word) in equation_lines
            ]
            all_equation_words = equation_words + operator_words

            clusters = cluster_by_overlap(all_equation_words)
            blocks = clusters_to_blocks(clusters)

            for block in blocks:
                block_words = [word for cluster in block for word in cluster[2]]
                equation_only_words = [
                    word
                    for word in block_words
                    if word.get("fontname", "") in EQUATION_FONTS
                ]
                if not equation_only_words:
                    equation_only_words = block_words

                x0 = min(word["x0"] for word in equation_only_words)
                x1 = max(word["x1"] for word in equation_only_words)
                top = min(word["top"] for word in block_words)
                bottom = max(word["bottom"] for word in block_words)

                if not should_keep_equation_box(x0, top, x1, bottom, page.width):
                    continue

                boxes.append(
                    {
                        "page": page_number,
                        "x0": x0,
                        "top": top,
                        "x1": x1,
                        "bottom": bottom,
                        "page_width": page.width,
                        "page_height": page.height,
                    }
                )

    return sorted(boxes, key=lambda item: (item["page"], item["top"], item["x0"]))


def _crop_box_to_pixels(
    box: dict[str, float | int],
    *,
    image_width: int,
    image_height: int,
    padding: int = PADDING,
) -> tuple[int, int, int, int]:
    pdf_width = float(box["page_width"])
    pdf_height = float(box["page_height"])
    scale_x = image_width / pdf_width if pdf_width else 1.0
    scale_y = image_height / pdf_height if pdf_height else 1.0

    px_x0 = max(0, int(float(box["x0"]) * scale_x) - padding)
    px_top = max(0, int(float(box["top"]) * scale_y) - padding)
    px_x1 = min(image_width, int(float(box["x1"]) * scale_x) + padding)
    px_bottom = min(image_height, int(float(box["bottom"]) * scale_y) + padding)
    return px_x0, px_top, px_x1, px_bottom


def extract_equations(
    *,
    pdf_path: Path,
    equation_dir: Path,
    pages: Sequence[PageBlocks],
    spans: Sequence[SectionSpan],
    journal_id: str,
    article_id: str,
    source_path: str,
    cell_exclusion: set[str],
    row_exclusion: set[str],
) -> list[ExtractedImage]:
    del cell_exclusion
    del row_exclusion

    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required for equation extraction.")
    if convert_from_path is None:
        raise RuntimeError("pdf2image is required for equation crop extraction.")

    require_command("pdftoppm")
    require_command("pdfinfo")

    equation_dir.mkdir(parents=True, exist_ok=True)
    boxes = _build_equation_boxes(pdf_path)
    if not boxes:
        return []

    boxes_by_page: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for box in boxes:
        boxes_by_page[int(box["page"])].append(box)

    equation_images: list[ExtractedImage] = []
    for page_number in sorted(boxes_by_page):
        rendered_pages = convert_from_path(
            str(pdf_path),
            dpi=DPI,
            first_page=page_number,
            last_page=page_number,
        )
        if not rendered_pages:
            continue

        page_image = rendered_pages[0]
        image_width, image_height = page_image.size

        for equation_index, box in enumerate(boxes_by_page[page_number], start=1):
            crop_bounds = _crop_box_to_pixels(
                box,
                image_width=image_width,
                image_height=image_height,
            )
            crop = page_image.crop(crop_bounds)

            file_name = f"equation_p{page_number}_{equation_index}.png"
            file_path = equation_dir / file_name
            crop.save(file_path)

            section = resolve_section_spatial(
                page_number,
                (
                    float(box["x0"]),
                    float(box["top"]),
                    float(box["x1"]),
                    float(box["bottom"]),
                ),
                spans,
            )

            caption = (
                f"Equation on page {page_number}"
                if section is None
                else f"Equation in {section} on page {page_number}"
            )

            equation_images.append(
                ExtractedImage(
                    image_id=f"{journal_id}_{article_id}_eq_{len(equation_images) + 1:05d}",
                    journal_id=journal_id,
                    article_id=article_id,
                    source_path=source_path,
                    file_path=str(file_path),
                    page_number=page_number,
                    mime_type="image/png",
                    width=crop.width,
                    height=crop.height,
                    caption=caption,
                    section=section,
                    content_type="image",
                    asset_subtype="equation",
                    bbox_x0=float(box["x0"]),
                    bbox_top=float(box["top"]),
                    bbox_x1=float(box["x1"]),
                    bbox_bottom=float(box["bottom"]),
                )
            )

    return equation_images
