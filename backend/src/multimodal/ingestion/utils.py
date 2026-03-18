from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(slots=True)
class PageText:
    page_number: int
    text: str
    token_count: int


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


def estimate_tokens(text: str) -> int:
    return len(tokenize(text))


def normalise_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()


def extract_page_text(pdf_path: Path) -> list[PageText]:
    require_command("pdftotext")

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
        run_command(command, "extracting text from PDF")
        raw_text = output_path.read_text(encoding="utf-8", errors="replace")
    finally:
        output_path.unlink(missing_ok=True)

    return [
        PageText(
            page_number=page_number,
            text=page_text.strip(),
            token_count=estimate_tokens(page_text.strip()),
        )
        for page_number, page_text in enumerate(raw_text.split("\f"), start=1)
    ]


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
