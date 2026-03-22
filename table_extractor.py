import pdfplumber, re, csv, os
from collections import Counter

CAPTION_PATTERN  = re.compile(r"^Table\s+\d+[\.\:]?", re.IGNORECASE)

# FIX 1 — END_PATTERN: add "Abbreviations" sub-pattern that fires mid-table too,
#          and add section-number headings like "3.3." and "2. RESULTS"
END_PATTERN      = re.compile(
    r"^(Abbreviations?|Figure|Fig\.|Scheme|References?|Discussion|"
    r"Methods?|Conclusion|Note:|Supporting|Author|Funding|Abstract|"
    r"Introduction|\d+\.\d*\s+[A-Z]|\d+\.\s+[A-Z])",
    re.IGNORECASE
)

# Unicode superscript letters that rebuild_word produces for footnote markers.
# e.g. ASCII 'a' → 'ᵃ' (U+1D43), 'b' → 'ᵇ', 'c' → 'ᶜ', 'd' → 'ᵈ'
_FOOTNOTE_MARKER = r"[a-dA-Dᵃᵇᶜᵈ]"

FOOTNOTE_PATTERN = re.compile(
    rf"^({_FOOTNOTE_MARKER}Reaction|{_FOOTNOTE_MARKER}Isolated"
    rf"|{_FOOTNOTE_MARKER}The|The values"
    rf"|Reaction condition|purificationby"
    # Also catch any line that starts with a Unicode superscript marker
    # immediately followed by a capital letter (e.g. ᵃNote, ᵇValues …)
    rf"|[ᵃᵇᶜᵈ][A-Z])",
    re.IGNORECASE
)

DATA_HEADER = re.compile(
    r"(S\.?\s*No\.?|Entry|Groups?|PDB|Ligand|Treatment|Compound)",
    re.IGNORECASE
)

# Stricter version used only inside collect_caption.
# Requires the keyword to appear near the START of the line (within first 25 chars)
# AND the line to be short (≤ 80 chars) — ruling out mid-sentence prose matches.
def is_column_header_row(txt):
    """Return True only when the line looks like an actual table column header."""
    if len(txt) > 80:
        return False
    # keyword must start within the first 25 characters
    m = DATA_HEADER.search(txt)
    if m and m.start() <= 25:
        return True
    return False

# FIX 3 — SECTION_HEADER_PATTERN: two-column papers interleave body text with
#          table rows at the same y-level; detect section headers that signal
#          we've left the table zone even if gap is small.
SECTION_HEADER_PATTERN = re.compile(
    r"^(\d+\.\d*\s|\d+\.\s|[A-Z][A-Z ]{4,}$)",
)

def is_narrative(text):
    words = text.split()
    has_digit = any(any(c.isdigit() for c in w) for w in words)
    return len(words) > 8 and not has_digit


# ── Unicode super/subscript maps ──────────────────────────────────────────────
SUPER_MAP = str.maketrans(
    "0123456789abcdefghijklmnoprstuvwxyz+-=()",
    "⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ⁺⁻⁼⁽⁾"
)
SUB_MAP = str.maketrans(
    "0123456789aehijklmnoprstuvx+-=()",
    "₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ₊₋₌₍₎"
)
def to_sup(s): return s.translate(SUPER_MAP)
def to_sub(s): return s.translate(SUB_MAP)


# ── Word rebuilder ────────────────────────────────────────────────────────────
def rebuild_word(word, all_chars):
    wx0, wx1 = word["x0"] - 1, word["x1"] + 1
    ytop, ybot = word["top"] - 3, word["bottom"]
    wchars = [c for c in all_chars
               if wx0<=c["x0"]<=wx1 and ytop<=c["top"]<=ybot
               and c.get("text","").strip()]
    if not wchars:
        return word["text"]
    top_c  = Counter(round(c["top"],1) for c in wchars)
    base_y = top_c.most_common(1)[0][0]
    bl_sz  = [c["size"] for c in wchars if abs(c["top"]-base_y)<=1.5]
    if not bl_sz: return word["text"]
    dom    = max(set(bl_sz), key=bl_sz.count)
    def kind(c):
        dy = round(c["top"],1) - base_y
        if dy < -1.0 and c["size"] < dom*0.80: return "sup"
        if dy > +1.5:                            return "sub"
        return "base"
    if all(kind(c)=="base" for c in wchars): return word["text"]
    wchars.sort(key=lambda c: c["x0"])
    out = []
    for c in wchars:
        k = kind(c)
        out.append(to_sup(c["text"]) if k=="sup" else
                   to_sub(c["text"]) if k=="sub" else c["text"])
    return "".join(out)


# ── Visual row builder ────────────────────────────────────────────────────────
def get_visual_rows(page, row_gap=5):
    words     = page.extract_words(x_tolerance=3, y_tolerance=2)
    all_chars = page.chars
    if not words: return []
    fixed = [{**w, "text": rebuild_word(w, all_chars)} for w in words]

    def is_dangling(w, others):
        txt = w["text"].strip()
        if len(txt) > 3 or not txt.isalnum(): return False
        for o in others:
            if o is w: continue
            if o["x0"] <= w["x0"] and o["x1"] >= w["x1"]:
                if to_sub(txt) in o["text"] or to_sup(txt) in o["text"]:
                    return True
        return False

    fixed.sort(key=lambda w: (w["top"], w["x0"]))
    clusters = []
    for w in fixed:
        if not clusters or (w["top"]-clusters[-1][-1]["top"]) > row_gap:
            clusters.append([w])
        else:
            clusters[-1].append(w)

    rows = []
    for cl in clusters:
        cl.sort(key=lambda w: w["x0"])
        flt  = [w for w in cl if not is_dangling(w, cl)]
        text = " ".join(w["text"] for w in flt).strip()
        if text:
            rows.append({"y":cl[0]["top"],"text":text,
                         "x0":cl[0]["x0"],"x1":cl[-1]["x1"],"words":flt})
    return rows


# ── Caption / body collectors ─────────────────────────────────────────────────
def collect_caption(rows, idx):
    """
    Collect all lines belonging to the table caption.

    Stop conditions (in order):
      1. Vertical gap > 18pt  → blank line separates caption from table body
      2. is_column_header_row → short line whose first token looks like a column
         header (S.No., Entry, Groups, Treatment …). Uses the stricter check to
         avoid false-positive stops on mid-sentence words like "treatment."
      3. Another Table caption starts
    """
    parts, cap_x1 = [rows[idx]["text"]], rows[idx]["x1"]
    i = idx + 1
    while i < len(rows):
        gap = rows[i]["y"] - rows[i-1]["y"]
        txt = rows[i]["text"]
        if gap > 18:
            break
        if is_column_header_row(txt):
            break
        if CAPTION_PATTERN.match(txt):
            break
        parts.append(txt)
        cap_x1 = max(cap_x1, rows[i]["x1"])
        i += 1
    return " ".join(parts), i, cap_x1


def collect_body(rows, idx, pw, cap_x1, max_gap=120):
    """
    FIX 4 — Tighter body collection with three extra stop conditions:
      a) FOOTNOTE_PATTERN fires on the current line  → stop immediately
      b) A pure body-text paragraph starts (left-flush, long, no digits) → stop
      c) x0 suddenly jumps far left to the page margin AND the text is long →
         likely we've crossed into the two-column prose area
    """
    is_narrow = cap_x1 < 0.50*pw
    x_right   = (cap_x1+20) if is_narrow else None

    # Determine the left margin of the table (caption's x0)
    table_left = rows[idx-1]["x0"] if idx else 0

    lines, word_rows, i, prev_y = [], [], idx, rows[idx-1]["y"] if idx else 0
    while i < len(rows):
        row, curr_y, txt = rows[i], rows[i]["y"], rows[i]["text"]
        gap = curr_y - prev_y

        # ── stop conditions ──────────────────────────────────────────────────
        if END_PATTERN.match(txt):
            break
        if CAPTION_PATTERN.match(txt):
            break
        # FIX 4a — footnote line stops immediately (don't include it)
        if FOOTNOTE_PATTERN.match(txt) and lines:
            break
        if gap > max_gap and lines:
            break
        # FIX 4b — long narrative prose with no data = body text, not table row
        if len(lines) >= 2 and is_narrative(txt):
            break
        # FIX 4c — two-column bleed: row starts well to the left of the table,
        #           text is long prose → we've left the table zone
        if (lines and row["x0"] < table_left - 10
                and len(txt.split()) > 6
                and not any(c.isdigit() for c in txt[:30])):
            break

        fw   = [w for w in row["words"] if not x_right or w["x0"]<=x_right]
        line = " ".join(w["text"] for w in fw).strip()
        if line:
            lines.append(line)
            word_rows.append(fw)
        prev_y=curr_y; i+=1
    return lines, word_rows, i


def extract_all_tables(pdf_path):
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            pw, rows = page.width, get_visual_rows(page)
            if not rows: continue
            i = 0
            while i < len(rows):
                if not CAPTION_PATTERN.match(rows[i]["text"]): i+=1; continue
                caption, body_start, cap_x1 = collect_caption(rows, i)
                if body_start >= len(rows): i=body_start; continue
                lines, word_rows, i = collect_body(rows, body_start, pw, cap_x1)
                if lines:
                    results.append({"page":page_num,"caption":caption,
                                    "lines":lines,"word_rows":word_rows})
    return results


# ── Column detection ──────────────────────────────────────────────────────────
def detect_col_anchors(word_rows, min_gap=15):
    if not word_rows: return []
    n = len(word_rows)
    all_x = [round(w["x0"]/4)*4 for wr in word_rows for w in wr]
    counts = Counter(all_x)
    threshold = max(2, n*0.40)
    anchors = sorted(x for x, c in counts.items() if c >= threshold)
    if not anchors: return []

    merged = [anchors[0]]
    for x in anchors[1:]:
        if x - merged[-1] >= min_gap:
            merged.append(x)

    if len(merged) < 2:
        return merged

    gaps = [(merged[i+1]-merged[i], i) for i in range(len(merged)-1)]
    max_gap_size, max_gap_idx = max(gaps)
    sorted_gaps = sorted(g for g, _ in gaps)
    second_largest = sorted_gaps[-2] if len(sorted_gaps) >= 2 else 0

    if max_gap_size >= max(30, second_largest * 1.5):
        return merged[max_gap_idx+1:]
    else:
        return merged


def assign_cols(word_row, anchors, tol=14):
    if not anchors:
        return [" ".join(w["text"] for w in word_row)]

    label_words = [w for w in word_row if w["x0"] < anchors[0] - tol]
    data_words  = [w for w in word_row if w["x0"] >= anchors[0] - tol]

    cells = [""] * (len(anchors)+1)
    cells[0] = " ".join(w["text"] for w in label_words).strip()

    for w in data_words:
        idx = min(range(len(anchors)), key=lambda i: abs(anchors[i]-w["x0"]))
        cells[idx+1] = (cells[idx+1]+" "+w["text"]).strip()

    while cells and not cells[-1]: cells.pop()
    return cells


def build_grid(table):
    wr = table.get("word_rows", [])
    if not wr: return [[l] for l in table["lines"]]
    anchors = detect_col_anchors(wr)
    if not anchors: return [[l] for l in table["lines"]]
    grid  = [assign_cols(row, anchors) for row in wr]
    ncols = max(len(r) for r in grid)
    return [r+[""]*(ncols-len(r)) for r in grid]


# ── FIX 5 — Merge continuation rows into their parent ────────────────────────
def merge_wrapped_rows(grid):
    """
    When a label wraps onto the next line (indented, no data values),
    append it to the previous row's label cell instead of creating a new row.
    Heuristic: a row is a "continuation" if:
      - it has only 1 non-empty cell (the label cell, cell[0])
      - cell[0] starts with a lowercase letter or an open-paren
        (continuation of a phrase, not a new entry)
    """
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


# ── CSV export ────────────────────────────────────────────────────────────────
def export_csv(tables, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    for tbl in tables:
        safe  = re.sub(r"[^\w\s-]","",tbl["caption"][:50]).strip()
        safe  = re.sub(r"\s+","_",safe)
        fname = os.path.join(output_dir, f"p{tbl['page']}_{safe}.csv")
        grid  = merge_wrapped_rows(build_grid(tbl))
        with open(fname,"w",newline="",encoding="utf-8-sig") as f:
            w = csv.writer(f)
            for row in grid: w.writerow(row)
        print(f"  ✓  {os.path.basename(fname)}")


# ── Print ─────────────────────────────────────────────────────────────────────
def print_tables(tables):
    if not tables: print("No tables found."); return
    print(f"\nFound {len(tables)} table(s)\n")
    for idx, tbl in enumerate(tables, 1):
        print("="*70)
        print(f"  TABLE {idx}  (page {tbl['page']})")
        print("="*70)
        print(f"  Caption: {tbl['caption'][:110]}")
        print("-"*70)
        grid = merge_wrapped_rows(build_grid(tbl))
        for row in grid:
            print("  |  ".join(str(c) for c in row))
        print()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    paths = ["/content/AO_5c05577.pdf",
             "/content/BJ_100833.pdf"]
    for path in paths:
        name = os.path.basename(path)
        print(f"\n{'#'*70}\n  {name}\n{'#'*70}")
        tables = extract_all_tables(path)
        print_tables(tables)
        stem = name.replace(".pdf","")
        print(f"\nExporting CSVs → /content/csv_out/{stem}/")
        export_csv(tables, f"/content/csv_out{stem}")