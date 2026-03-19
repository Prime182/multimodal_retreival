from __future__ import annotations

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
        page_chunks=True,  # one dict per page
        show_progress=False,
    )
    
    # We also need page dimensions for column detection
    doc = pymupdf.open(str(pdf_path))
    
    pages = []
    for chunk in chunks:
        # metadata["page"] is 0-indexed in pymupdf4llm, but we want 1-indexed
        # Fallback to 'page_number' if 'page' is not found, or use a default if neither is present.
        # NOTE: 'page_number' in chunk['metadata'] seems to be 1-indexed!
        if "page" in chunk["metadata"]:
            page_idx = chunk["metadata"]["page"]
            page_num = page_idx + 1
        elif "page_number" in chunk["metadata"]:
            page_num = chunk["metadata"]["page_number"]
            page_idx = page_num - 1
        else:
            page_idx = 0
            page_num = 1

        # Normalize blocks: pymupdf4llm uses "class" instead of "type" in some versions/configs
        raw_blocks = chunk.get("page_boxes", [])
        blocks = []
        for b in raw_blocks:
            if not isinstance(b, dict):
                continue
            # Ensure "type" exists and map from "class" if needed
            if "type" not in b and "class" in b:
                b["type"] = b["class"]
            # Ensure "text" exists and map from "text" or just empty string
            if "text" not in b:
                # We might need to extract text from the markdown if it's not in the box
                # but usually it should be there. Let's look at the chunk text if needed.
                # For now just ensure it's a string.
                b["text"] = b.get("text", "")
            blocks.append(b)

        # Ensure page_idx is within valid range

        if not (0 <= page_idx < doc.page_count):
            print(f"[WARN] Invalid page_idx: {page_idx}, doc.page_count: {doc.page_count}. Skipping chunk.")
            continue
        
        # Get page dimensions
        pdf_page = doc.load_page(page_idx)
        width = pdf_page.rect.width
        height = pdf_page.rect.height
        
        pages.append(PageBlocks(
            page_number=page_num,
            text=chunk["text"],
            blocks=blocks,  # [{type, bbox, text}, ...]
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
            "This ingestion module expects Poppler utilities to be installed."
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
