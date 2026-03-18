from __future__ import annotations

import re
from typing import Sequence

from ..types import EquationChunk
from .section import SectionSpan, resolve_section
from .utils import PageText, normalise_line


_MATH_SYMBOLS = frozenset("=<>±∑∫√≈≠≤≥∞∂∆∇λμσπθβα^")
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
_PROSE_VETO_WORDS = {
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "have",
    "which",
    "their",
    "were",
    "been",
    "than",
    "these",
    "those",
    "however",
    "therefore",
    "although",
    "results",
    "figure",
    "table",
    "study",
    "patients",
    "participants",
    "data",
    "analysis",
    "treatment",
}
_MATH_CONTEXT_WORDS = {
    "equation",
    "formula",
    "integral",
    "derivative",
    "matrix",
    "vector",
    "scalar",
    "coefficient",
    "eigenvalue",
    "function",
    "polynomial",
    "theorem",
    "proof",
    "lemma",
}
_OPERAND_PATTERN = r"(?:[A-Za-z][A-Za-z0-9_/]*(?:\^\d+)?|\d[\d.]*(?:[A-Za-z]{0,3})?)"


def extract_equations(
    *,
    pages: Sequence[PageText],
    spans: Sequence[SectionSpan],
    journal_id: str,
    article_id: str,
    source_path: str,
    cell_exclusion: set[str],
    row_exclusion: set[str],
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
                        context=_extract_context(lines, block_start, end_index),
                        page_number=page.page_number,
                        section=resolve_section(page.page_number, block_start, spans),
                    )
                )
            block = []
            block_start = 0

        for index, line in enumerate(lines):
            normalised = normalise_line(line)

            if normalised in row_exclusion:
                flush(index)
                continue

            cell_hits = sum(1 for cell in cell_exclusion if len(cell) > 3 and cell in normalised)
            if cell_hits >= 2:
                flush(index)
                continue

            if is_equation_line(line):
                if not block:
                    block_start = index
                block.append(line)
                continue

            flush(index)

        flush(len(lines))

    return chunks


def is_equation_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return False

    lower_tokens = {token.lower().strip(".,;:()[]") for token in stripped.split()}
    prose_hits = lower_tokens & _PROSE_VETO_WORDS
    if len(prose_hits) > 2:
        return False
    if len(stripped.split()) > 12 and prose_hits:
        return False

    if any(marker in stripped for marker in _LATEX_MARKERS):
        return True

    score = 0

    non_space = [char for char in stripped if not char.isspace()]
    if non_space and sum(char in _MATH_SYMBOLS for char in non_space) / len(non_space) >= 0.30:
        score += 1

    numeric_eq = re.search(
        r"(?<!\w)([A-Za-z]{1,4}|\d[\d.,]*)\s*[=<>]\s*(\d[\d.,]*[A-Za-z]{0,3}|[A-Za-z]{1,4})(?!\w)",
        stripped,
    )
    symbolic_eq = re.search(
        r"(?<!\w)[A-Za-z][A-Za-z0-9_/^]{0,12}\s*[=<>]\s*[A-Za-z0-9_^./]+\s*(?:[+*/^]|\s+-\s+)\s*[A-Za-z0-9_^./]+",
        stripped,
    )
    if numeric_eq or symbolic_eq:
        score += 1

    if re.search(
        rf"{_OPERAND_PATTERN}\s*[+*/^]\s*{_OPERAND_PATTERN}",
        stripped,
    ) or re.search(
        rf"{_OPERAND_PATTERN}\s+-\s+{_OPERAND_PATTERN}",
        stripped,
    ):
        score += 1

    if lower_tokens & _MATH_CONTEXT_WORDS:
        score += 1

    return score >= 2


def _extract_context(lines: Sequence[str], start: int, end: int) -> str:
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
