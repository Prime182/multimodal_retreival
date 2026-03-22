# backend/src/multimodal/ingestion/table.py
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore[assignment]

from ..types import PageBlocks, TableChunk
from .section import SectionSpan, resolve_section_spatial
from .utils import normalise_line

# ---------------------------------------------------------------------------
# Patterns — ported from table_extractor.py (with Bug #1 fix: |$ suffix)
# ---------------------------------------------------------------------------

CAPTION_PATTERN = re.compile(
    r"^Table\s+\d+(?:[\s.:–-]|$)",
    re.IGNORECASE,
)

END_PATTERN = re.compile(
    r"^(Abbreviations?|Figure|Fig\.|Scheme|References?|Discussion|"
    r"Methods?|Conclusion|Note:|Supporting|Author|Funding|Abstract|"
    r"Introduction|\d+\.\d*\s+[A-Z]|\d+\.\s+[A-Z])",
    re.IGNORECASE,
)

_FOOTNOTE_MARKER = r"[a-dA-Dᵃᵇᶜᵈ]"
FOOTNOTE_PATTERN = re.compile(
    rf"^({_FOOTNOTE_MARKER}Reaction|{_FOOTNOTE_MARKER}Isolated"
    rf"|{_FOOTNOTE_MARKER}The|The values|Reaction condition|purificationby"
    rf"|[ᵃᵇᶜᵈ][A-Z])",
    re.IGNORECASE,
)

DATA_HEADER = re.compile(
    r"(S\.?\s*No\.?|Entry|Groups?|PDB|Ligand|Treatment|Compound)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Unicode super/subscript helpers
# ---------------------------------------------------------------------------

SUPER_MAP = str.maketrans(
    "0123456789abcdefghijklmnoprstuvwxyz+-=()",
    "⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ⁺⁻⁼⁽⁾",
)
SUB_MAP = str.maketrans(
    "0123456789aehijklmnoprstuvx+-=()",
    "₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ₊₋₌₍₎",
)


def _to_sup(s: str) -> str:
    return s.translate(SUPER_MAP)


def _to_sub(s: str) -> str:
    return s.translate(SUB_MAP)


# ---------------------------------------------------------------------------
# Word rebuilder — preserves super/subscripts from PDF character stream
# ---------------------------------------------------------------------------

def _rebuild_word(word: dict, all_chars: list[dict]) -> str:
    wx0, wx1 = word["x0"] - 1, word["x1"] + 1
    ytop, ybot = word["top"] - 3, word["bottom"]
    wchars = [
        c for c in all_chars
        if wx0 <= c["x0"] <= wx1 and ytop <= c["top"] <= ybot
        and c.get("text", "").strip()
    ]
    if not wchars:
        return word["text"]
    top_c = Counter(round(c["top"], 1) for c in wchars)
    base_y = top_c.most_common(1)[0][0]
    bl_sz = [c["size"] for c in wchars if abs(c["top"] - base_y) <= 1.5]
    if not bl_sz:
        return word["text"]
    dom = max(set(bl_sz), key=bl_sz.count)

    def kind(c: dict) -> str:
        dy = round(c["top"], 1) - base_y
        if dy < -1.0 and c["size"] < dom * 0.80:
            return "sup"
        if dy > +1.5:
            return "sub"
        return "base"

    if all(kind(c) == "base" for c in wchars):
        return word["text"]
    wchars.sort(key=lambda c: c["x0"])
    return "".join(
        _to_sup(c["text"]) if kind(c) == "sup"
        else _to_sub(c["text"]) if kind(c) == "sub"
        else c["text"]
        for c in wchars
    )


# ---------------------------------------------------------------------------
# Visual row builder
# ---------------------------------------------------------------------------

def _get_visual_rows(page, row_gap: int = 5) -> list[dict]:
    words = page.extract_words(x_tolerance=3, y_tolerance=2)
    all_chars = page.chars
    if not words:
        return []
    fixed = [{**w, "text": _rebuild_word(w, all_chars)} for w in words]

    def is_dangling(w: dict, others: list[dict]) -> bool:
        txt = w["text"].strip()
        if len(txt) > 3 or not txt.isalnum():
            return False
        return any(
            o is not w and o["x0"] <= w["x0"] and o["x1"] >= w["x1"]
            and (_to_sub(txt) in o["text"] or _to_sup(txt) in o["text"])
            for o in others
        )

    fixed.sort(key=lambda w: (w["top"], w["x0"]))
    clusters: list[list[dict]] = []
    for w in fixed:
        if not clusters or (w["top"] - clusters[-1][-1]["top"]) > row_gap:
            clusters.append([w])
        else:
            clusters[-1].append(w)

    rows = []
    for cl in clusters:
        cl.sort(key=lambda w: w["x0"])
        flt = [w for w in cl if not is_dangling(w, cl)]
        text = " ".join(w["text"] for w in flt).strip()
        if text:
            rows.append({
                "y": cl[0]["top"],
                "text": text,
                "x0": cl[0]["x0"],
                "x1": cl[-1]["x1"],
                "words": flt,
            })
    return rows


# ---------------------------------------------------------------------------
# Caption and body collectors
# ---------------------------------------------------------------------------

def _is_column_header_row(txt: str) -> bool:
    if len(txt) > 80:
        return False
    m = DATA_HEADER.search(txt)
    return bool(m and m.start() <= 25)


def _is_narrative(text: str) -> bool:
    words = text.split()
    has_digit = any(any(c.isdigit() for c in w) for w in words)
    return len(words) > 8 and not has_digit


def _collect_caption(rows: list[dict], idx: int) -> tuple[str, int, float]:
    parts = [rows[idx]["text"]]
    cap_x1 = rows[idx]["x1"]
    i = idx + 1
    while i < len(rows):
        gap = rows[i]["y"] - rows[i - 1]["y"]
        txt = rows[i]["text"]
        if gap > 18:
            break
        if _is_column_header_row(txt):
            break
        if CAPTION_PATTERN.match(txt):
            break
        parts.append(txt)
        cap_x1 = max(cap_x1, rows[i]["x1"])
        i += 1
    return " ".join(parts), i, cap_x1


def _collect_body(
    rows: list[dict],
    idx: int,
    pw: float,
    cap_x1: float,
    max_gap: int = 120,
) -> tuple[list[str], list[list[dict]], int]:
    is_narrow = cap_x1 < 0.50 * pw
    x_right = (cap_x1 + 20) if is_narrow else None
    table_left = rows[idx - 1]["x0"] if idx else 0

    lines: list[str] = []
    word_rows: list[list[dict]] = []
    i = idx
    prev_y = rows[idx - 1]["y"] if idx else 0

    while i < len(rows):
        row = rows[i]
        curr_y = row["y"]
        txt = row["text"]
        gap = curr_y - prev_y

        if END_PATTERN.match(txt):
            break
        if CAPTION_PATTERN.match(txt):
            break
        if FOOTNOTE_PATTERN.match(txt) and lines:
            break
        if gap > max_gap and lines:
            break
        if len(lines) >= 2 and _is_narrative(txt):
            break
        if (
            lines
            and row["x0"] < table_left - 10
            and len(txt.split()) > 6
            and not any(c.isdigit() for c in txt[:30])
        ):
            break

        fw = [w for w in row["words"] if not x_right or w["x0"] <= x_right]
        line = " ".join(w["text"] for w in fw).strip()
        if line:
            lines.append(line)
            word_rows.append(fw)
        prev_y = curr_y
        i += 1

    return lines, word_rows, i


# ---------------------------------------------------------------------------
# Column detection and grid assembly
# ---------------------------------------------------------------------------

def _detect_col_anchors(word_rows: list[list[dict]], min_gap: int = 15) -> list[float]:
    if not word_rows:
        return []
    n = len(word_rows)
    all_x = [round(w["x0"] / 4) * 4 for wr in word_rows for w in wr]
    counts = Counter(all_x)
    threshold = max(2, n * 0.40)
    anchors = sorted(x for x, c in counts.items() if c >= threshold)
    if not anchors:
        return []

    merged = [anchors[0]]
    for x in anchors[1:]:
        if x - merged[-1] >= min_gap:
            merged.append(x)

    if len(merged) < 2:
        return merged

    gaps = [(merged[i + 1] - merged[i], i) for i in range(len(merged) - 1)]
    max_gap_size, max_gap_idx = max(gaps)
    sorted_gaps = sorted(g for g, _ in gaps)
    second_largest = sorted_gaps[-2] if len(sorted_gaps) >= 2 else 0

    if max_gap_size >= max(30, second_largest * 1.5):
        return merged[max_gap_idx + 1 :]
    return merged


def _assign_cols(word_row: list[dict], anchors: list[float], tol: int = 14) -> list[str]:
    if not anchors:
        return [" ".join(w["text"] for w in word_row)]
    label_words = [w for w in word_row if w["x0"] < anchors[0] - tol]
    data_words = [w for w in word_row if w["x0"] >= anchors[0] - tol]
    cells = [""] * (len(anchors) + 1)
    cells[0] = " ".join(w["text"] for w in label_words).strip()
    for w in data_words:
        idx = min(range(len(anchors)), key=lambda i: abs(anchors[i] - w["x0"]))
        cells[idx + 1] = (cells[idx + 1] + " " + w["text"]).strip()
    while cells and not cells[-1]:
        cells.pop()
    return cells


def _build_grid(table: dict) -> list[list[str]]:
    wr = table.get("word_rows", [])
    if not wr:
        return [[line] for line in table["lines"]]
    anchors = _detect_col_anchors(wr)
    if not anchors:
        return [[line] for line in table["lines"]]
    grid = [_assign_cols(row, anchors) for row in wr]
    ncols = max(len(r) for r in grid)
    return [r + [""] * (ncols - len(r)) for r in grid]


def _merge_wrapped_rows(grid: list[list[str]]) -> list[list[str]]:
    if not grid:
        return grid
    merged = [list(grid[0])]
    for row in grid[1:]:
        non_empty = [c for c in row if c.strip()]
        is_continuation = (
            len(non_empty) == 1
            and row[0].strip()
            and (row[0][0].islower() or row[0][0] in "(")
        )
        if is_continuation and merged:
            merged[-1][0] = (merged[-1][0] + " " + row[0]).strip()
        else:
            merged.append(list(row))
    return merged


# ---------------------------------------------------------------------------
# Exclusion set for equation detection (Bug #2 fix — lines strategy only)
# ---------------------------------------------------------------------------

_SETTINGS_LINES = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
}


def build_table_text_exclusion(pdf_path: Path) -> tuple[set[str], set[str]]:
    """
    Build cell/row text exclusion sets consumed by equation detection.
    Uses ONLY the lines strategy — the text strategy on two-column PDFs
    falsely detects the whole page as a table and poisons equation detection.
    """
    if pdfplumber is None:
        return set(), set()

    cell_ex: set[str] = set()
    row_ex: set[str] = set()

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            try:
                tables = page.extract_tables(table_settings=_SETTINGS_LINES) or []
            except Exception:
                tables = []
            for table in tables:
                for row in table:
                    cells: list[str] = []
                    for cell in row:
                        if not cell:
                            continue
                        n = normalise_line(str(cell))
                        if n:
                            cell_ex.add(n)
                            cells.append(n)
                    if cells:
                        row_ex.add(" ".join(cells))

    return cell_ex, row_ex


# ---------------------------------------------------------------------------
# Public entry point — called by ingestion.py (signature unchanged)
# ---------------------------------------------------------------------------

def extract_tables(
    *,
    pdf_path: Path,
    pages: Sequence[PageBlocks],
    spans: Sequence[SectionSpan],
    journal_id: str,
    article_id: str,
    source_path: str,
) -> list[TableChunk]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required for table extraction.")

    page_by_num = {p.page_number: p for p in pages}
    all_raw: list[dict] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, plumber_page in enumerate(pdf.pages, start=1):
            pw = plumber_page.width
            rows = _get_visual_rows(plumber_page)
            if not rows:
                continue
            i = 0
            while i < len(rows):
                if not CAPTION_PATTERN.match(rows[i]["text"]):
                    i += 1
                    continue
                caption, body_start, cap_x1 = _collect_caption(rows, i)
                if body_start >= len(rows):
                    i = body_start
                    continue
                lines, word_rows, i = _collect_body(rows, body_start, pw, cap_x1)
                if lines:
                    all_raw.append({
                        "page": page_num,
                        "caption": caption,
                        "lines": lines,
                        "word_rows": word_rows,
                    })

    chunks: list[TableChunk] = []

    for tbl in all_raw:
        page_num: int = tbl["page"]
        caption: str = tbl["caption"]
        grid = _merge_wrapped_rows(_build_grid(tbl))
        if len(grid) < 2:
            continue

        buf = io.StringIO()
        csv.writer(buf).writerows(grid)
        csv_data = buf.getvalue().strip()
        header = ",".join(str(c) for c in grid[0])
        data_rows = grid[1:]

        page_meta = page_by_num.get(page_num)
        page_h = page_meta.height if page_meta else 800.0
        page_w = page_meta.width if page_meta else 600.0
        approx_y = page_h * 0.4
        section = resolve_section_spatial(
            page_num, (0.0, approx_y, page_w, page_h), spans
        )

        if len(data_rows) <= 20:
            chunks.append(
                TableChunk(
                    chunk_id=f"{journal_id}_{article_id}_tbl_{len(chunks)+1:05d}",
                    journal_id=journal_id,
                    article_id=article_id,
                    source_path=source_path,
                    csv_data=csv_data,
                    header=header,
                    caption=caption,
                    row_index=None,
                    page_number=page_num,
                    section=section,
                )
            )
        else:
            for ri, row in enumerate(data_rows, start=1):
                row_buf = io.StringIO()
                csv.writer(row_buf).writerows([grid[0], row])
                chunks.append(
                    TableChunk(
                        chunk_id=f"{journal_id}_{article_id}_tbl_{len(chunks)+1:05d}",
                        journal_id=journal_id,
                        article_id=article_id,
                        source_path=source_path,
                        csv_data=row_buf.getvalue().strip(),
                        header=header,
                        caption=caption,
                        row_index=ri,
                        page_number=page_num,
                        section=section,
                    )
                )

    return chunks
