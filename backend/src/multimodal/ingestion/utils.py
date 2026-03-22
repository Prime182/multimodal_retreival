"""
backend/src/multimodal/ingestion/utils.py

ROOT CAUSE FIX: pymupdf4llm treats two-column academic PDF layout as images,
producing "**==> picture ... <==**" markers for text columns.
clean_page_text() then removes ALL of them, leaving page.text = '' for every page.
Result: text_chunks=0, equation_chunks=0.

FIX: After clean_page_text(), if page.text is empty (or suspiciously short),
fall back to pdfplumber.extract_text() for that page.  pdfplumber reliably
extracts text from typeset two-column PDFs regardless of layout complexity.

This makes extract_page_blocks() robust against pymupdf4llm's picture-marker
behavior while still using pymupdf4llm for the page dimensions and metadata
that are needed by the layout-detection path.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import pymupdf
import pymupdf4llm

try:
    import pdfplumber as _pdfplumber
except ImportError:
    _pdfplumber = None  # type: ignore[assignment]

from ..types import PageBlocks


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

_NOISE_RE = re.compile(
    r"\*\*==>.*?<==\*\*\n?"
    r"|!\[[^\]]*\]\([^)]*\)\n?",
    re.DOTALL,
)

# Minimum number of non-whitespace chars required to consider pymupdf4llm
# text usable.  If a page has fewer chars after noise stripping, pdfplumber
# is used as a fallback.  Typical academic pages have thousands of chars;
# 50 is a conservative threshold that catches empty/picture-only outputs.
_MIN_USABLE_TEXT_CHARS = 50


def clean_page_text(text: str) -> str:
    """Remove pymupdf4llm noise artefacts from a page's markdown text."""
    cleaned = _NOISE_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _pdfplumber_page_text(pdf_path: Path, page_idx: int) -> str:
    """
    Extract plain text from a single page using pdfplumber.

    Used as a fallback when pymupdf4llm produces empty/picture-only text.
    Returns '' if pdfplumber is not installed or extraction fails.
    """
    if _pdfplumber is None:
        return ""
    try:
        with _pdfplumber.open(str(pdf_path)) as pdf:
            if not (0 <= page_idx < len(pdf.pages)):
                return ""
            text = pdf.pages[page_idx].extract_text() or ""
            # Clean CID font artifacts (e.g. "(cid:0)" → "−")
            text = re.sub(r"\(cid:\d+\)", "−", text)
            return text.strip()
    except Exception:
        return ""


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


def estimate_tokens(text: str) -> int:
    return len(tokenize(text))


def normalise_line(line: str) -> str:
    text = line.replace("±", " +/- ")
    text = re.sub(r"\+\s*/\s*-", " +/- ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.replace("+/-", " plusminus ")


def extract_page_blocks(pdf_path: Path) -> list[PageBlocks]:
    """
    Returns layout-aware blocks with bounding boxes per page.

    Primary text source: pymupdf4llm (Markdown with noise stripped).
    Fallback text source: pdfplumber (plain text extraction).

    The fallback fires when pymupdf4llm produces empty or picture-only text —
    which happens with two-column academic PDFs whose layout pymupdf4llm
    represents as omitted-picture markers, leaving no usable text after
    clean_page_text() runs.
    """
    chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        show_progress=False,
    )

    doc = pymupdf.open(str(pdf_path))
    pages = []

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        if "page" in meta:
            page_idx = meta["page"]
            page_num = page_idx + 1
        elif "page_number" in meta:
            page_num = meta["page_number"]
            page_idx = page_num - 1
        else:
            page_idx = 0
            page_num = 1

        if not (0 <= page_idx < doc.page_count):
            continue

        raw_blocks = chunk.get("page_boxes", [])
        blocks = []
        for b in raw_blocks:
            if not isinstance(b, dict):
                continue
            if "type" not in b and "class" in b:
                b = {**b, "type": b["class"]}
            if "text" not in b:
                b = {**b, "text": ""}
            blocks.append(b)

        pdf_page = doc.load_page(page_idx)
        width = pdf_page.rect.width
        height = pdf_page.rect.height

        # ── Primary: pymupdf4llm text (noise-stripped) ─────────────────────
        primary_text = clean_page_text(chunk["text"])

        # ── Fallback: pdfplumber text ──────────────────────────────────────
        # Trigger when pymupdf4llm produces near-empty text, which happens
        # when it converts two-column text regions to picture-omitted markers.
        non_ws_chars = len(re.sub(r"\s", "", primary_text))
        if non_ws_chars < _MIN_USABLE_TEXT_CHARS:
            fallback_text = _pdfplumber_page_text(pdf_path, page_idx)
            if len(re.sub(r"\s", "", fallback_text)) > non_ws_chars:
                primary_text = fallback_text

        pages.append(PageBlocks(
            page_number=page_num,
            text=primary_text,
            blocks=blocks,
            width=width,
            height=height,
        ))

    doc.close()
    return pages


def ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"PDF path is not a file: {path}")


def require_command(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"Required command not found on PATH: {command_name}. "
            "Install Poppler utilities."
        )


def run_command(command: Sequence[str], action: str) -> None:
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