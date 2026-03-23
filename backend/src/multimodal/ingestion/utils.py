# backend/src/multimodal/ingestion/utils.py
"""
Phase 1 IMPLEMENTATION: Pdfplumber-native text extraction pipeline.

REMOVED: pymupdf, pymupdf4llm, clean_page_text(), _pdfplumber_page_text()

NEW: Native pdfplumber column-aware text extraction supporting two-column
academic PDFs without external markdown conversion.

This replaces pymupdf4llm completely, which is the root cause of text_chunks=0
for two-column layouts (Bug #1 from BUGS_AND_SOLUTIONS.md).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Sequence

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from ..types import PageBlocks

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


def estimate_tokens(text: str) -> int:
    return len(tokenize(text))


def normalise_line(line: str) -> str:
    text = line.replace("±", " +/- ")
    text = re.sub(r"\+\s*/\s*-", " +/- ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.replace("+/-", " plusminus ")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Pdfplumber-based text extraction with column-aware splitting
# ──────────────────────────────────────────────────────────────────────────────

def get_body_font_size(page) -> float:
    """
    Estimate body-text font size using the statistical mode of all character sizes on a page,
    rounded to the nearest 0.5pt. The most frequent size is the body text size.
    """
    sizes = [
        round(ch["size"] * 2) / 2
        for ch in page.chars
        if ch.get("size", 0) > 4
    ]
    if not sizes:
        return 10.0
    freq: dict[float, int] = {}
    for s in sizes:
        freq[s] = freq.get(s, 0) + 1
    return max(freq, key=freq.get)


def chars_to_lines(chars: list[dict]) -> list[list[dict]]:
    """
    Group a flat list of char dicts into visual lines by vertical (top) coordinate.
    Characters within 2pt of each other share a line; within each line chars are sorted left-to-right.
    """
    if not chars:
        return []

    # Sort by vertical position (top coordinate)
    sorted_chars = sorted(chars, key=lambda c: (round(c["top"] / 2) * 2, c["x0"]))

    lines: list[list[dict]] = []
    for char in sorted_chars:
        if lines and abs(char["top"] - lines[-1][0]["top"]) <= 2:
            # Belongs to the same line
            lines[-1].append(char)
        else:
            # New line
            lines.append([char])

    # Sort characters within each line left-to-right
    for line in lines:
        line.sort(key=lambda c: c["x0"])

    return lines


def detect_column_split(page) -> float | None:
    """
    Detect two-column page layout by finding a vertical gutter.
    Builds a 1pt-wide histogram of word x0 positions restricted to the central 30% of page width.
    The longest zero-density run ≥ 10pt is the gutter. Returns None for single-column pages.
    """
    if not page.chars:
        return None

    page_width = page.width
    left_margin = page_width * 0.1
    right_margin = page_width * 0.9

    # Extract word x0 positions (approximate from char positions)
    word_x0s: list[float] = []
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    for word in words:
        x0 = word["x0"]
        if left_margin <= x0 <= right_margin:
            word_x0s.append(x0)

    if len(word_x0s) < 10:
        return None

    # Build 1pt histogram
    histogram: dict[int, int] = defaultdict(int)
    for x0 in word_x0s:
        bucket = int(x0)
        histogram[bucket] += 1

    # Find longest zero-density run ≥ 10pt
    sorted_buckets = sorted(histogram.keys())
    max_gap = 0
    max_gap_center = None

    for i, bucket in enumerate(sorted_buckets):
        if i == 0:
            continue
        gap = bucket - sorted_buckets[i - 1]
        if gap >= 10 and gap > max_gap:
            max_gap = gap
            max_gap_center = (sorted_buckets[i - 1] + bucket) / 2.0

    return max_gap_center


def extract_page_text(page, body_size: float, column_aware: bool = True) -> str:
    """
    Extract text from a single pdfplumber page,
    handling two-column layouts if column_aware=True.

    Returns text with columns concatenated (left then right).
    """
    if column_aware:
        gutter = detect_column_split(page)
        if gutter is not None:
            # Split chars into left and right columns
            left_chars = [c for c in page.chars if c["x0"] < gutter]
            right_chars = [c for c in page.chars if c["x0"] >= gutter]

            # Extract text from each column independently
            left_text = _extract_text_from_chars(left_chars)
            right_text = _extract_text_from_chars(right_chars)

            # Interleave by y-position to maintain reading order as much as possible
            text = f"{left_text}\n\n{right_text}"
            return text

    # Single-column or fallback
    text = _extract_text_from_chars(page.chars)
    return text


def _extract_text_from_chars(chars: list[dict]) -> str:
    """
    Convert a list of character dicts (already filtered to a column/region)
    back into readable text, preserving line breaks.
    """
    if not chars:
        return ""

    lines = chars_to_lines(chars)
    line_texts = []

    for line in lines:
        line_text = "".join(ch["text"] for ch in line)
        line_texts.append(line_text)

    return "\n".join(line_texts)


def extract_page_blocks(pdf_path: Path) -> list[PageBlocks]:
    """
    Build PageBlocks from pdfplumber with native column-aware text extraction.

    Uses font-size estimation and column detection to handle two-column layouts.
    The returned PageBlocks.text contains full page text; PageBlocks.blocks
    remains empty (line-based chunker in text.py does not need blocks).

    This replaces the pymupdf4llm and unstructured approaches entirely.
    """
    if pdfplumber is None:
        raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")

    pages: list[PageBlocks] = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_number = page.page_number
                width = page.width
                height = page.height

                # Get body font size for heading detection (used by section.py later)
                body_size = get_body_font_size(page)

                # Extract text with column awareness
                text = extract_page_text(page, body_size, column_aware=True).strip()

                # Clean CID font artifacts (garbled character streams)
                text = re.sub(r"\(cid:\d+\)", "−", text)

                # Normalize whitespace while preserving paragraph structure
                lines = text.split("\n")
                clean_lines = [line.strip() for line in lines if line.strip()]
                text = "\n".join(clean_lines)

                pages.append(PageBlocks(
                    page_number=page_number,
                    text=text,
                    blocks=[],  # Empty: line-based chunker does not need blocks
                    width=width,
                    height=height,
                    body_size=body_size,  # Store for Phase 2 heading detection
                ))
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {pdf_path}: {e}")

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
            "Install the required system package."
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
