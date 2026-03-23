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
# Post-processing: clean_pdf_text  (ported from formula_extractor.py)
# ---------------------------------------------------------------------------

KNOWN_PAIRS = [
    ('treatedcells',          'treated cells'),
    ('untreatedcells',        'untreated cells'),
    ('cellviability',         'Cell viability'),
    ('eradicationofbiofilm',  'Eradication of biofilm'),
    ('odinsample',            'OD in sample'),
    ('odincontrol',           'OD in control'),
    ('odintreatment',         'OD in treatment'),
    ('odofsample',            'OD of sample'),
    ('odof',                  'OD of '),
    ('vecontrol',             've control'),
    ('ofhemolysisat',         'of hemolysis at '),
    ('%ofhemolysis',          '% of hemolysis'),
    ('standarddeviation',     'standard deviation'),
    ('shownhereasthe',        'shown here as the'),
    ('valuesare',             'values are'),
    ('areshownhere',          'are shown here'),
]

CAMEL_RE = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')

STATS_ONLY = re.compile(
    r'^(?:(?:values?\s*)?(?:are\s+)?shown\s+here\s+as\s+the\s+means?'
    r'|means?\s*[±\+]\s*standard\s+deviation'
    r'|\(n\s*=\s*\d+\))',
    re.IGNORECASE,
)


def clean_pdf_text(text: str) -> str:
    """Fix common PDF text extraction artefacts in equation strings."""
    s = text
    for run, spaced in KNOWN_PAIRS:
        s = re.sub(re.escape(run), spaced, s, flags=re.IGNORECASE)
    s = CAMEL_RE.sub(' ', s)
    s = re.sub(r'  +', ' ', s).strip()
    return s


def _is_real_equation_post(text: str) -> bool:
    """Post-processing quality check: reject stat annotations and trivial strings."""
    if STATS_ONLY.match(text.strip()):
        return False
    if text.strip() in ('= × 100', '= ×100', '='):
        return False
    if '=' not in text:
        return False
    lhs, _, rhs = text.partition('=')
    lhs_ok = len(re.sub(r'\s', '', lhs)) > 3
    rhs_ok = len(re.sub(r'\s', '', rhs)) > 1
    return lhs_ok and rhs_ok


# ---------------------------------------------------------------------------
# Post-processing: reconstruct_fractions  (ported from formula_extractor.py)
# ---------------------------------------------------------------------------

CELL_VIA_RE = re.compile(
    r'^(treated\s+cells\s+)?'
    r'(Cell\s+viability\s*\(%\))\s*=\s*[×\\times\s]+100\s*(untreated\s+cells)?',
    re.IGNORECASE,
)
HEMOLYSIS_RE = re.compile(
    r'%\s*of\s+hemolysis\s+at\s+\d+\s*nm'
    r'.*?\(OD\s+of\s+sample.*?OD\s+of.*?control\)\s*=\s*[×\\times\s]+100',
    re.IGNORECASE | re.DOTALL,
)
BIOERADICATION_RE = re.compile(
    r'(?:formula\s+)?Eradication\s+of\s+biofilm\s*\(%\)'
    r'\s+OD\s+in\s+control\s+OD\s+in\s+treatment\s*=\s*OD\s+in\s+control',
    re.IGNORECASE,
)


def reconstruct_fractions(raw_text: str) -> str:
    """
    Rewrite a single equation string if it matches a known garbled fraction pattern.
    Returns the cleaned string (unchanged if no pattern matches).
    """
    if CELL_VIA_RE.search(raw_text):
        return 'Cell viability (%) = (treated cells / untreated cells) × 100'
    if HEMOLYSIS_RE.search(raw_text):
        return (
            '% Hemolysis at 540 nm = '
            '(OD of sample − OD of −ve control) / '
            '(OD of +ve control − OD of −ve control) × 100'
        )
    if BIOERADICATION_RE.search(raw_text):
        return (
            'Eradication of biofilm (%) = '
            '(OD in control − OD in treatment) / OD in control'
        )
    return raw_text


# ---------------------------------------------------------------------------
# Post-processing: dedup  (ported from formula_extractor.py)
# ---------------------------------------------------------------------------

def _fp(text: str) -> str:
    """Fingerprint: alnum + operator chars, lowercased, first 65 chars."""
    return re.sub(r'[^\w=+\-*/×%]', '', text.lower())[:65]


def dedup(items: list[dict]) -> list[dict]:
    """
    Remove near-duplicates, keeping the LONGEST (most complete) version.
    Two items are considered duplicates if one's fingerprint is a prefix of the other.
    """
    items = sorted(items, key=lambda x: -len(x['raw_text']))
    seen: set[str] = set()
    out: list[dict] = []

    for eq in items:
        fp = _fp(eq['raw_text'])
        covered = any(
            fp.startswith(s[:40]) or s.startswith(fp[:40])
            for s in seen
        )
        if not covered:
            seen.add(fp)
            out.append(eq)

    return sorted(out, key=lambda x: x['page_number'])


# ---------------------------------------------------------------------------
# Column-aware text extraction
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
# Per-column scanner
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

    # ── Step 1: collect all raw hits across all pages ─────────────────────────
    all_hits: list[dict] = []

    with pdfplumber.open(source_path) as pdf:
        page_by_num = {p.page_number: p for p in pages}

        for plumber_page in pdf.pages:
            page_num = plumber_page.page_number
            for col_text in _page_columns(plumber_page):
                for hit in _scan_column(col_text, page_num):
                    all_hits.append(hit)

    # ── Step 2: apply text cleaning ───────────────────────────────────────────
    for hit in all_hits:
        hit["raw_text"] = clean_pdf_text(hit["raw_text"])

    # ── Step 3: reconstruct garbled fractions ─────────────────────────────────
    for hit in all_hits:
        hit["raw_text"] = reconstruct_fractions(hit["raw_text"])

    # ── Step 4: quality filter (post-cleaning) ────────────────────────────────
    all_hits = [h for h in all_hits if _is_real_equation_post(h["raw_text"])]

    # ── Step 5: global dedup, keep longest ────────────────────────────────────
    all_hits = dedup(all_hits)

    # ── Step 6: table-region veto + section resolution + chunk creation ───────
    chunks: list[EquationChunk] = []

    with pdfplumber.open(source_path) as pdf:
        page_by_num = {p.page_number: p for p in pages}

        for hit in all_hits:
            raw = hit["raw_text"]
            page_num = hit["page_number"]

            # Table-region veto
            norm = re.sub(r"\s+", " ", raw).strip().lower()
            if norm in row_exclusion:
                continue
            cell_hits = sum(1 for c in cell_exclusion if len(c) > 3 and c in norm)
            if cell_hits >= 2:
                continue

            # Section resolution
            page_meta = page_by_num.get(page_num)
            page_h = page_meta.height if page_meta else 800.0
            page_w = page_meta.width if page_meta else 600.0
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