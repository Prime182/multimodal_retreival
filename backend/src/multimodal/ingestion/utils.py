"""
PATCH: backend/src/multimodal/ingestion/utils.py
Replace extract_page_blocks() and add clean_page_text().
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import pymupdf
import pymupdf4llm

from ..types import PageBlocks


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# ── NEW: strip noise injected by pymupdf4llm into markdown output ─────────────
# "**==> picture [249 x 503] intentionally omitted <==**"  ← pollutes text chunks
# "![img](path/to/img.png)"                                 ← stray markdown images
_NOISE_RE = re.compile(
    r"\*\*==>.*?<==\*\*\n?"          # picture-omitted markers
    r"|!\[[^\]]*\]\([^)]*\)\n?",     # stray markdown image refs
    re.DOTALL,
)


def clean_page_text(text: str) -> str:
    """Remove pymupdf4llm noise artefacts from a page's markdown text."""
    cleaned = _NOISE_RE.sub("", text)
    # collapse runs of 3+ blank lines down to two (keeps paragraph spacing)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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
    """Returns layout-aware blocks with bounding boxes per page."""
    chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        show_progress=False,
        # write_images=False is default — keeps images out of markdown
        # (we extract images separately with pdfimages)
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

        pages.append(PageBlocks(
            page_number=page_num,
            text=clean_page_text(chunk["text"]),   # ← noise stripped here
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