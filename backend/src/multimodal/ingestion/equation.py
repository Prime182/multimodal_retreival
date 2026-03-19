"""
PATCH: backend/src/multimodal/ingestion/equation.py

Root cause of the missed equations in your paper:
  _HARD_VETO_WORDS contains "control", "sample", "treatment" — which are
  used as VARIABLE NAMES in biomedical display formulas like:

    = (OD of sample − OD of −ve control)  × 100
      ────────────────────────────────────
      (OD of +ve control − OD of −ve control)

  and:

    Eradication of biofilm(%) =
        (OD in control − OD in treatment) / OD in control

Fix: Add a `_DISPLAY_MATH_RE` structural override that fires BEFORE the
veto-word check.  A line that matches this pattern is classified as an
equation regardless of the words it contains.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..types import EquationChunk, PageBlocks, SectionSpan
from .section import detect_heading, resolve_section_spatial
from .utils import normalise_line


_LATEX_MARKERS = (
    "\\frac", "\\sum", "\\int", "\\alpha", "\\beta", "\\theta",
    "\\lambda", "\\mu", "\\sigma", "\\pi", "\\sqrt", "\\begin",
    "\\end", "\\left", "\\right",
)

# ── NEW: structural patterns that identify display / block equations ──────────
# These fire BEFORE any veto-word check so that formulas whose variable names
# happen to be ordinary English words (OD, control, sample, treatment…) are
# still captured.
_DISPLAY_MATH_RE = re.compile(
    r"""
    # Fraction-like numerator/denominator separated by −, –, or -
    \([^)]{3,}\s*[−–\-]\s*[^)]{3,}\)\s*/\s*\(
    |
    # Line that IS just "= … × number" or "= … / …"  (formula continuation)
    ^\s*=\s*.{5,}[×÷*/]\s*\d
    |
    # Percentage-formula structure:  "= (A − B) / (C − D) × 100"
    [=]\s*\([^)]+[−–\-][^)]+\)\s*/
    |
    # OD-based formula (very common in microbiology/cell-biology papers)
    \bOD\b.*[=\-−–/].*\bOD\b
    |
    # Generic display equation: variable_phrase = fraction × scalar
    \b\w[\w\s]{0,20}[(%]\s*\)\s*=\s*.+[/×÷]
    |
    # Line starts with = and contains at least one operator  (continuation line)
    ^\s*=\s*(?=.*[+\-−–×÷*/^])
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

_HARD_VETO_WORDS: frozenset[str] = frozenset(
    {
        # ── Keep genuine prose-only vetoes ───────────────────────────────────
        # (removed: control, sample, treatment, od — these appear as
        #  variable names in biomedical formulas)
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
                    pseudo_bbox = (100.0, float(index * 10), 500.0, float(index * 10 + 10))
                    block_section = resolve_section_spatial(page.page_number, pseudo_bbox, spans)
                block.append(line)
                continue

            flush(index)

        flush(len(lines))

    return chunks


def is_equation_line(line: str) -> bool:
    """
    Return True when the line is predominantly mathematical notation.

    Order of checks (most specific → least specific):
    1. LaTeX markers          — unambiguous
    2. Display-math patterns  — structural override; bypasses veto words
    3. Statistical p/n=…      — never an equation
    4. Veto-word check        — blocks obvious prose
    5. Scoring                — requires ≥2 independent math signals
    """
    stripped = line.strip()
    if len(stripped) < 3:
        return False

    # ── 1. LaTeX — immediate pass ─────────────────────────────────────────────
    if any(marker in stripped for marker in _LATEX_MARKERS):
        return True

    # ── 2. Display / block math structural override ───────────────────────────
    # Catches biomedical formulas like:
    #   "= (OD of sample − OD of −ve control) / (OD of +ve control …) × 100"
    # even though those lines contain veto words.
    if _DISPLAY_MATH_RE.search(stripped):
        return True

    # ── 3. Statistical notation — immediate reject ────────────────────────────
    if _STATISTICAL_NOTATION_RE.match(stripped):
        return False

    # ── 4. Veto-word check ────────────────────────────────────────────────────
    lower_tokens = {
        token.lower().strip(".,;:()[]{}\"'%-+*/^=<>")
        for token in stripped.split()
    }
    if lower_tokens & _HARD_VETO_WORDS:
        return False

    # ── 5. Scoring: need ≥ 2 independent math signals ────────────────────────
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