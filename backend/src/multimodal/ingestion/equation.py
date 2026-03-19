"""
backend/src/multimodal/ingestion/equation.py

BUG FIXED: pseudo-bbox y0 was `index * 10` (hardcoded).
section.py text-fallback spans store bbox.y0 = line_index * _APPROX_LINE_HEIGHT.
resolve_section_spatial() divides y0 by _APPROX_LINE_HEIGHT to get a line index.
Using a different constant (10 vs 12) caused a ~20% offset, meaning equations
near section boundaries were assigned to the wrong section.

Fix: import _APPROX_LINE_HEIGHT from section.py and use it for the pseudo-bbox.
The constant is defined once in section.py and shared across all three modules.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..types import EquationChunk, PageBlocks, SectionSpan
from .section import _APPROX_LINE_HEIGHT, detect_heading, resolve_section_spatial
from .utils import normalise_line


_LATEX_MARKERS = (
    "\\frac", "\\sum", "\\int", "\\alpha", "\\beta", "\\theta",
    "\\lambda", "\\mu", "\\sigma", "\\pi", "\\sqrt", "\\begin",
    "\\end", "\\left", "\\right",
)

_DISPLAY_MATH_RE = re.compile(
    r"""
    \([^)]{3,}\s*[−–\-]\s*[^)]{3,}\)\s*/\s*\(
    |
    ^\s*=\s*.{5,}[×÷*/]\s*\d
    |
    [=]\s*\([^)]+[−–\-][^)]+\)\s*/
    |
    \bOD\b.*[=\-−–/].*\bOD\b
    |
    \b\w[\w\s]{0,20}[(%]\s*\)\s*=\s*.+[/×÷]
    |
    ^\s*=\s*(?=.*[+\-−–×÷*/^])
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

_HARD_VETO_WORDS: frozenset[str] = frozenset(
    {
        "renal", "allograft", "hepatic", "viral", "serum", "plasma",
        "patients", "patient", "participants", "recipients",
        "therapy", "survival", "function", "toxicity", "efficacy",
        "transplant", "months", "years", "weeks", "days", "hours",
        "clinical", "medical", "surgical", "laboratory",
        "egfr", "hbv", "hbsag", "hbeag", "alt", "ast", "afp",
        "ml/min", "iu/ml", "u/l", "mmol/l", "umol/l",
        "dna", "rna", "pcr", "elisa",
        "compared", "showed", "demonstrated", "observed", "reported",
        "significant", "difference", "association", "correlation",
        "baseline", "respectively", "analysis",
        "result", "results", "mean", "median",
        "stable", "remained", "experienced", "naive",
        "the", "and", "that", "this", "vs", "with", "from",
        "have", "which", "their", "were", "been", "than",
        "these", "those", "however", "therefore", "although",
        "during", "after", "before", "between", "among", "within",
        "while", "because", "since", "also", "both", "further",
        "follow-up", "followup", "overall",
    }
)

_MATH_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "sin", "cos", "tan", "log", "exp", "det", "div", "inf",
        "max", "min", "mod", "arg", "sgn", "var", "cov", "std",
        "lim", "sup", "deg",
    }
)

_STATISTICAL_NOTATION_RE = re.compile(
    r"^\s*(?:p|n)\s*[=<>]\s*(?:\d+|0?\.\d+|ns)\s*$",
    re.IGNORECASE,
)


def extract_equations(
    *,
    pages: Sequence[PageBlocks],
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
        block_section: str | None = None

        def flush(end_index: int) -> None:
            nonlocal block, block_start, block_section
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
                        section=block_section,
                    )
                )
            block = []
            block_start = 0
            block_section = None

        for index, line in enumerate(lines):
            stripped = line.strip()

            if detect_heading(stripped):
                flush(index)
                continue

            norm = normalise_line(line)
            if norm in row_exclusion:
                flush(index)
                continue

            cell_hits = sum(1 for cell in cell_exclusion if len(cell) > 3 and cell in norm)
            if cell_hits >= 2:
                flush(index)
                continue

            if is_equation_line(line):
                if not block:
                    block_start = index
                    # FIX: use _APPROX_LINE_HEIGHT imported from section.py.
                    # resolve_section_spatial() divides y0 by _APPROX_LINE_HEIGHT
                    # to convert back to a line index; using the same constant
                    # here ensures the round-trip is exact.
                    # Previously: y0 = index * 10  (hardcoded, inconsistent)
                    pseudo_y0 = float(index) * _APPROX_LINE_HEIGHT
                    pseudo_bbox = (
                        100.0,
                        pseudo_y0,
                        500.0,
                        pseudo_y0 + _APPROX_LINE_HEIGHT,
                    )
                    block_section = resolve_section_spatial(
                        page.page_number, pseudo_bbox, spans
                    )
                block.append(line)
                continue

            flush(index)

        flush(len(lines))

    return chunks


def is_equation_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return False

    if any(marker in stripped for marker in _LATEX_MARKERS):
        return True

    if _DISPLAY_MATH_RE.search(stripped):
        return True

    if _STATISTICAL_NOTATION_RE.match(stripped):
        return False

    lower_tokens = {
        token.lower().strip(".,;:()[]{}\"'%-+*/^=<>")
        for token in stripped.split()
    }
    if lower_tokens & _HARD_VETO_WORDS:
        return False

    if not re.search(r"[=<>]", stripped):
        return False

    english_count, math_count = _classify_tokens(stripped.split())
    total = english_count + math_count
    if total == 0 or math_count < 2:
        return False

    return (math_count / total) >= 0.60


def _classify_tokens(tokens: list[str]) -> tuple[int, int]:
    english = 0
    math = 0
    for token in tokens:
        clean = token.strip(".,;:()[]{}\"'")
        if not clean:
            continue
        if re.fullmatch(r"[+\-*/=<>^±∑∫√≈≠≤≥∞∂∆∇λμσπθβα|\\]+", clean):
            math += 1; continue
        if re.fullmatch(r"\d[\d.,]*[A-Za-z]{0,4}", clean):
            math += 1; continue
        if re.fullmatch(r"[A-Za-z]{1,2}", clean):
            math += 1; continue
        if clean.lower() in _MATH_IDENTIFIERS:
            math += 1; continue
        if re.fullmatch(r"[A-Za-z]{1,3}_[A-Za-z0-9]{1,3}", clean):
            math += 1; continue
        if re.fullmatch(r"d[A-Za-z]+/d[A-Za-z]+", clean):
            math += 1; continue
        if re.fullmatch(r"[A-Za-z]{1,4}\^?\d+", clean):
            math += 1; continue
        if re.fullmatch(r"\d+[\^e]-?\d+", clean, re.IGNORECASE):
            math += 1; continue
        if re.fullmatch(r"[A-Za-z0-9]+/[A-Za-z0-9]+", clean):
            math += 1; continue
        if len(clean) >= 3 and re.search(r"[A-Za-z]{3}", clean):
            english += 1
    return english, math


def _extract_context(lines: Sequence[str], start: int, end: int) -> str:
    surrounding = list(lines[max(0, start - 2):start]) + list(
        lines[end:min(len(lines), end + 2)]
    )
    text = " ".join(line.strip() for line in surrounding if line.strip())
    if not text:
        return "Equation extracted from surrounding document context."
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if sentence.strip():
            return sentence.strip()
    return text.strip()