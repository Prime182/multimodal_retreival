# backend/src/multimodal/ingestion/equation.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore[assignment]

from ..types import EquationChunk, PageBlocks, SectionSpan
from .section import _APPROX_LINE_HEIGHT, resolve_section_spatial

# ---------------------------------------------------------------------------
# Patterns — ported directly from formula_extractor.py
# ---------------------------------------------------------------------------

FORMULA_INTRO = re.compile(
    r"(?:using\s+the\s+following\s+formula"
    r"|calculated\s+using\s+the\s+following"
    r"|determine\s+by\s+using\s+the\s+following"
    r"|determined\s+by\s+the\s+following)",
    re.IGNORECASE,
)

MATH_SYM = re.compile(r"[×÷±√∑∫≤≥≠≈∞μαβγδλσφπΩ°]")

NAMED_KW = re.compile(
    r"(?:cell\s*viability|eradication\s*of\s*biofilm"
    r"|%\s*of\s*hemolysis|eradication\s*\(%\)"
    r"|hemolysis\s*at\s*\d+\s*nm)",
    re.IGNORECASE,
)

NOISE_RE = re.compile(
    r"^(?:figure\s*\d|table\s*\d|scheme\s*\d"
    r"|https?://|doi\s*[:\.]|\d{3,5}\s*$)",
    re.IGNORECASE,
)

SECTION_RE = re.compile(
    r"^(?:\d+[\.\d]*\.?\s+)[A-Z][^\n]{3,80}$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Column-aware text extraction — matches formula_extractor.page_columns()
# ---------------------------------------------------------------------------

def _page_columns(page) -> list[str]:
    """Return [left_col, right_col] for two-column pages, else [full_text]."""
    pw = page.width

    def crop(x0: float, x1: float) -> str:
        try:
            return (
                page.crop((x0, 0, x1, page.height)).extract_text(
                    x_tolerance=3, y_tolerance=3
                )
                or ""
            )
        except Exception:
            return ""

    left = crop(0, pw / 2)
    right = crop(pw / 2, pw)
    if len(left) > 150 and len(right) > 150:
        return [left, right]
    return [page.extract_text(x_tolerance=3, y_tolerance=3) or ""]


# ---------------------------------------------------------------------------
# Formula collection helpers
# ---------------------------------------------------------------------------

def _collect_display_formula(lines: list[str], intro_idx: int) -> str:
    """Collect lines immediately after a formula-introduction phrase."""
    parts: list[str] = []
    blank_count = 0
    for i in range(intro_idx + 1, min(len(lines), intro_idx + 12)):
        s = lines[i].strip()
        if not s:
            blank_count += 1
            if blank_count > 1:
                break
            continue
        if SECTION_RE.match(s) or NOISE_RE.match(s):
            break
        if parts and s.endswith(".") and len(s) > 60:
            break
        parts.append(s)
        blank_count = 0
    return " ".join(parts)


def _extract_context(lines: list[str], line_idx: int) -> str:
    """Return one sentence of surrounding context for an equation line."""
    before = " ".join(
        l.strip() for l in lines[max(0, line_idx - 2) : line_idx] if l.strip()
    )[-250:]
    if not before:
        return "Equation extracted from document."
    for sentence in re.split(r"(?<=[.!?])\s+", before):
        if sentence.strip():
            return sentence.strip()
    return before


def _is_real_equation(text: str) -> bool:
    """Require a meaningful LHS and RHS around the equals sign."""
    if "=" not in text:
        return False
    lhs, _, rhs = text.partition("=")
    lhs_ok = len(re.sub(r"\s", "", lhs)) > 3
    rhs_ok = len(re.sub(r"\s", "", rhs)) > 1
    return lhs_ok and rhs_ok


# ---------------------------------------------------------------------------
# Per-column scanner — matches formula_extractor.scan_column()
# ---------------------------------------------------------------------------

def _scan_column(text: str, page_num: int) -> list[dict]:
    if not text.strip():
        return []

    lines = text.split("\n")
    results: list[dict] = []
    seen: set[str] = set()

    def register(formula: str, line_idx: int) -> None:
        norm = re.sub(r"\s+", " ", formula).strip()
        if len(norm) < 6 or NOISE_RE.match(norm):
            return
        if not _is_real_equation(norm):
            return
        key = re.sub(r"[^\w=+\-*/×%]", "", norm.lower())[:65]
        if key in seen:
            return
        seen.add(key)
        results.append(
            {
                "raw_text": norm,
                "page_number": page_num,
                "context": _extract_context(lines, line_idx),
            }
        )

    # Pass A — formula-introduction phrases
    for i, line in enumerate(lines):
        if FORMULA_INTRO.search(line):
            formula = _collect_display_formula(lines, i)
            if formula and ("=" in formula or "%" in formula):
                register(formula, i)

    # Pass B — named biological/chemical calculation formulas
    for i, line in enumerate(lines):
        s = line.strip()
        if NAMED_KW.search(s) and ("=" in s or "%" in s):
            chunk = s
            for j in range(1, 6):
                nxt = lines[i + j].strip() if i + j < len(lines) else ""
                if not nxt or NOISE_RE.match(nxt) or SECTION_RE.match(nxt):
                    break
                chunk += " " + nxt
                if chunk.count("=") >= 1 and ("100" in chunk or ")" in chunk):
                    break
            register(chunk, i)

    # Pass C — inline math signals (math symbol + equals sign)
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or NOISE_RE.match(s):
            continue
        if (MATH_SYM.search(s) and "=" in s) or ("OD" in s and "=" in s):
            register(s, i)

    return results


# ---------------------------------------------------------------------------
# Public entry point — called by ingestion.py (signature unchanged)
# ---------------------------------------------------------------------------

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
    if pdfplumber is None:
        return []

    chunks: list[EquationChunk] = []

    with pdfplumber.open(source_path) as pdf:
        page_by_num = {p.page_number: p for p in pages}

        for plumber_page in pdf.pages:
            page_num = plumber_page.page_number
            page_meta = page_by_num.get(page_num)
            page_w = page_meta.width if page_meta else 600.0
            page_h = page_meta.height if page_meta else 800.0

            for col_text in _page_columns(plumber_page):
                for hit in _scan_column(col_text, page_num):
                    raw = hit["raw_text"]

                    # --- Table-region veto (Bug #2 fix from bugs_and_solutions.md) ---
                    norm = re.sub(r"\s+", " ", raw).strip().lower()
                    if norm in row_exclusion:
                        continue
                    cell_hits = sum(
                        1 for c in cell_exclusion if len(c) > 3 and c in norm
                    )
                    if cell_hits >= 2:
                        continue

                    # --- Section resolution ---
                    # Use mid-page y as approximation; good enough for section
                    # assignment because headings are tens of lines apart.
                    approx_y = page_h * 0.4
                    pseudo_bbox = (
                        50.0,
                        approx_y,
                        page_w - 50.0,
                        approx_y + _APPROX_LINE_HEIGHT,
                    )
                    section = resolve_section_spatial(page_num, pseudo_bbox, spans)

                    chunks.append(
                        EquationChunk(
                            chunk_id=f"{journal_id}_{article_id}_eq_{len(chunks)+1:05d}",
                            journal_id=journal_id,
                            article_id=article_id,
                            source_path=source_path,
                            latex=raw,
                            context=hit.get("context"),
                            page_number=page_num,
                            section=section,
                        )
                    )

    return chunks
