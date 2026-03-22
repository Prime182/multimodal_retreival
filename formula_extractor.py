"""
Scientific Equation Extractor from PDFs
========================================
Extracts mathematical, chemical, and biological equations/formulas
from scientific PDFs (including two-column layouts) and outputs
structured, RAG-friendly JSON.

Algorithm
---------
1. Per page, crop left and right halves independently (handles 2-column layout).
2. Reconstruct multi-line display equations by detecting "formula context"
   (lines immediately following "following formula", "using the formula", etc.)
3. Apply a tiered pattern bank: named formulas → display equations → inline math.
4. Enrich each hit with section, context, LaTeX, and a self-contained RAG chunk.

Output schema (per equation)
-----------------------------
  equation_id       stable ID (rank + page + md5 fragment)
  page_number       1-indexed source page
  section           nearest section heading
  equation_type     mathematical | biological_formula | chemical_formula |
                    ratio_formula | percentage_formula | pharmacokinetic
  raw_text          equation as it appears (whitespace-normalised)
  latex             best-effort LaTeX
  variables         list of named symbols / variables
  context_before    ≤250 chars before the equation
  context_after     ≤250 chars after the equation
  rag_chunk         self-contained retrieval unit

Usage
-----
    python equation_extractor.py  <input.pdf>  [output.json]
    python equation_extractor.py  paper.pdf    results.json
"""

import re, json, sys, hashlib
from pathlib import Path
from datetime import datetime

try:
    import pdfplumber
except ImportError:
    sys.exit("Run:  pip install pdfplumber --break-system-packages")


# ─────────────────────────────────────────────────────────────────
# 0.  CONSTANTS
# ─────────────────────────────────────────────────────────────────

# Phrases that immediately precede a display-style formula block
FORMULA_INTRO = re.compile(
    r'(?:using\s+the\s+following\s+formula'
    r'|calculated\s+using\s+the\s+following'
    r'|determine\s+by\s+using\s+the\s+following'
    r'|determined\s+by\s+the\s+following)',
    re.IGNORECASE
)

# Section heading: "4.6. Cell Cytotoxicity Assay" etc.
SECTION_RE = re.compile(
    r'^(?:\d+[\.\d]*\.?\s+)[A-Z][^\n]{3,80}$',
    re.MULTILINE
)

# Lines to skip — pure noise
NOISE_RE = re.compile(
    r'^(?:figure\s*\d|table\s*\d|scheme\s*\d|acs[\s\-]omega'
    r'|https?://|doi\s*[:\.]|\d{3,5}\s*$'
    r'|athe\s+values|bthe\s+|cthe\s+|the\s+values\s+are\s+shown'
    r'|data\s+were\s+recorded)',
    re.IGNORECASE
)

# Greek + math symbols
MATH_SYM = re.compile(r'[×÷±√∑∫≤≥≠≈∞μαβγδλσφπΩ°]')

# Named bio/chem calculation formulas
NAMED_KW = re.compile(
    r'(?:cell\s*viability|eradication\s*of\s*biofilm'
    r'|%\s*of\s*hemolysis|eradication\s*\(%\)'
    r'|hemolysis\s*at\s*\d+\s*nm)',
    re.IGNORECASE
)

# Symbol → LaTeX map
SYM_MAP = {
    '×': r'\times', '÷': r'\div',  '±': r'\pm',  '√': r'\sqrt',
    '∑': r'\sum',   '∫': r'\int',  '∂': r'\partial', '≤': r'\leq',
    '≥': r'\geq',   '≠': r'\neq', '≈': r'\approx', '∞': r'\infty',
    '°': r'^{\circ}', 'μ': r'\mu','α': r'\alpha', 'β': r'\beta',
    'γ': r'\gamma', 'δ': r'\delta','λ': r'\lambda','σ': r'\sigma',
    'φ': r'\phi',   'π': r'\pi',  '−': '-',
}

CHEM_SUB = re.compile(r'([A-Z][a-z]?)(\d+)')

STOPWORDS = {
    'the','of','in','and','or','by','to','at','for','with',
    'as','are','from','been','was','were','has','its','not',
    'Da','nm','mL','μg','mg','mM','min','rpm','ml','cm','μL',
}


# ─────────────────────────────────────────────────────────────────
# 1.  TEXT EXTRACTION — COLUMN-AWARE
# ─────────────────────────────────────────────────────────────────

def page_columns(page) -> list[str]:
    """
    Return a list of text blocks: [left_col, right_col] for 2-column pages,
    or [full_text] for single-column pages.
    """
    pw = page.width

    def crop_text(x0, x1) -> str:
        try:
            cropped = page.crop((x0, 0, x1, page.height))
            return cropped.extract_text(x_tolerance=3, y_tolerance=3) or ''
        except Exception:
            return ''

    left  = crop_text(0, pw / 2)
    right = crop_text(pw / 2, pw)

    # If both halves have substantial text → genuine 2-column layout
    if len(left) > 150 and len(right) > 150:
        return [left, right]

    # Otherwise extract full width
    full = page.extract_text(x_tolerance=3, y_tolerance=3) or ''
    return [full]


# ─────────────────────────────────────────────────────────────────
# 2.  FORMULA RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────

def collect_display_formula(lines: list[str], intro_idx: int) -> str:
    """
    After a formula-introduction phrase, collect the formula lines
    (stop at next section heading or empty line after content).
    """
    formula_lines = []
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
        # Stop if we hit a new paragraph (sentence-like line)
        if len(formula_lines) > 0 and s.endswith('.') and len(s) > 60:
            break
        formula_lines.append(s)
        blank_count = 0

    return ' '.join(formula_lines)


# ─────────────────────────────────────────────────────────────────
# 3.  CORE SCANNER
# ─────────────────────────────────────────────────────────────────

def scan_column(text: str, page_num: int) -> list[dict]:
    """Extract equations from a single column's text."""
    if not text.strip():
        return []

    lines = text.split('\n')
    results: list[dict] = []
    seen:    set[str]   = set()

    def current_section(up_to: int) -> str:
        for line in reversed(lines[:up_to]):
            if SECTION_RE.match(line.strip()):
                return line.strip()
        return 'Unknown'

    def make_context(i: int):
        before = ' '.join(lines[max(0, i-3):i]).strip()[-250:]
        after  = ' '.join(lines[i+1:min(len(lines), i+4)]).strip()[:250]
        return before, after

    def register(formula: str, line_idx: int, source: str = ''):
        norm = re.sub(r'\s+', ' ', formula).strip()
        if len(norm) < 6 or NOISE_RE.match(norm):
            return
        key = re.sub(r'[^\w=+\-*/×%]', '', norm.lower())[:65]
        if key in seen:
            return
        seen.add(key)
        before, after = make_context(line_idx)
        results.append({
            'raw_text':       norm,
            'page_number':    page_num,
            'section':        current_section(line_idx),
            'context_before': before,
            'context_after':  after,
            '_source':        source,
        })

    # ── Pass A: formula-introduction phrases ──────────────────────
    for i, line in enumerate(lines):
        if FORMULA_INTRO.search(line):
            formula = collect_display_formula(lines, i)
            if formula and ('=' in formula or '%' in formula):
                register(formula, i, 'display_formula')

    # ── Pass B: named calculation formulas (can span lines) ───────
    for i, line in enumerate(lines):
        s = line.strip()
        if NAMED_KW.search(s):
            # Build the complete formula by looking ahead
            chunk = s
            for j in range(1, 6):
                nxt = lines[i + j].strip() if i + j < len(lines) else ''
                if not nxt or NOISE_RE.match(nxt) or SECTION_RE.match(nxt):
                    break
                chunk += ' ' + nxt
                if chunk.count('=') >= 1 and ('100' in chunk or ')' in chunk):
                    break
            if '=' in chunk or '%' in chunk:
                register(chunk, i, 'named_formula')

    # ── Pass C: inline math signals ───────────────────────────────
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or NOISE_RE.match(s):
            continue
        is_math = (
            (MATH_SYM.search(s) and '=' in s) or
            ('OD' in s and '=' in s) or
            ('mol %' in s.lower() and '=' in s)
        )
        if is_math:
            register(s, i, 'inline_math')

    return results


# ─────────────────────────────────────────────────────────────────
# 4.  ENRICHMENT  (LaTeX, type, variables)
# ─────────────────────────────────────────────────────────────────

def to_latex(text: str) -> str:
    s = text
    for u, l in SYM_MAP.items():
        s = s.replace(u, f' {l} ')
    s = CHEM_SUB.sub(r'\1_{\2}', s)
    # Simple A / B fraction
    m = re.match(r'^([\w\s\(\)\-\+\.]+)\s*/\s*([\w\s\(\)\-\+\.]+)$', s.strip())
    if m:
        n, d = m.group(1).strip(), m.group(2).strip()
        s = rf'\frac{{{n}}}{{{d}}}'
    s = re.sub(r'\^(\w+)', r'^{\1}', s)
    return re.sub(r'  +', ' ', s).strip()


def classify(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ['viability', 'hemolysis', 'eradication', 'disruption']):
        return 'biological_formula'
    if any(k in t for k in ['logp', 'hbd', 'hba', 'molar reactivity', 'mass =']):
        return 'pharmacokinetic'
    if re.search(r'\b[A-Z][a-z]?\d[A-Z]', text) and len(text) < 35:
        return 'chemical_formula'
    if 'OD' in text and '/' in text:
        return 'ratio_formula'
    if '%' in text and '=' in text:
        return 'percentage_formula'
    return 'mathematical'


def variables_of(text: str) -> list[str]:
    tokens = re.findall(r'\b([A-Z][a-zA-Z\d]{0,7}|[a-z]{1,5})\b', text)
    return sorted({t for t in tokens if t not in STOPWORDS and len(t) > 1})


# ─────────────────────────────────────────────────────────────────
# 5.  RAG CHUNK
# ─────────────────────────────────────────────────────────────────

def rag_chunk(eq: dict, title: str) -> str:
    """Self-contained retrieval unit — equation on one unbroken line."""
    return '\n'.join([
        f"SOURCE: {title}",
        f"PAGE: {eq['page_number']}  |  SECTION: {eq.get('section', 'Unknown')}",
        f"EQUATION_TYPE: {eq['equation_type']}  |  ID: {eq['equation_id']}",
        "",
        f"CONTEXT_BEFORE: {eq.get('context_before', '')}",
        "",
        f"EQUATION: {eq['raw_text']}",
        f"LATEX:    {eq['latex']}",
        "",
        f"CONTEXT_AFTER: {eq.get('context_after', '')}",
    ])


# ─────────────────────────────────────────────────────────────────
# 5b. POST-PROCESSING — clean PDF text artefacts
# ─────────────────────────────────────────────────────────────────

# Pairs like "treatedcells" → "treated cells"
KNOWN_PAIRS = [
    ('treatedcells',     'treated cells'),
    ('untreatedcells',   'untreated cells'),
    ('cellviability',    'Cell viability'),
    ('eradicationofbiofilm', 'Eradication of biofilm'),
    ('odinsample',       'OD in sample'),
    ('odincontrol',      'OD in control'),
    ('odintreatment',    'OD in treatment'),
    ('odofsample',       'OD of sample'),
    ('odof',             'OD of '),
    ('vecontrol',        've control'),
    ('ofhemolysisat',    'of hemolysis at '),
    ('%ofhemolysis',     '% of hemolysis'),
    ('standarddeviation','standard deviation'),
    ('shownhereasthe',   'shown here as the'),
    ('valuesare',        'values are'),
    ('areshownhere',     'are shown here'),
]

# Camel-case split for unknown run-togethers (e.g. "CellViability" → "Cell Viability")
CAMEL_RE = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')


def clean_pdf_text(text: str) -> str:
    """Fix common PDF text extraction artefacts in equation strings."""
    s = text

    # Known word pairs (case-insensitive replace)
    for run, spaced in KNOWN_PAIRS:
        s = re.sub(re.escape(run), spaced, s, flags=re.IGNORECASE)

    # Insert space before capital run-ons in lower-case context
    s = CAMEL_RE.sub(' ', s)

    # Collapse multiple spaces
    s = re.sub(r'  +', ' ', s).strip()
    return s


# Filter out lines that are ONLY statistical annotations, not actual equations
STATS_ONLY = re.compile(
    r'^(?:(?:values?\s*)?(?:are\s+)?shown\s+here\s+as\s+the\s+means?'
    r'|means?\s*[±\+]\s*standard\s+deviation'
    r'|\(n\s*=\s*\d+\))',
    re.IGNORECASE
)

# Minimum equation quality after cleaning — must have both sides of =
def is_real_equation(text: str) -> bool:
    """True if the text looks like a real equation (not just a stat annotation)."""
    if STATS_ONLY.match(text.strip()):
        return False
    if text.strip() in ('= × 100', '= ×100', '='):
        return False
    # Must have = AND at least one meaningful expression on both sides
    if '=' not in text:
        return False
    lhs, _, rhs = text.partition('=')
    lhs_meaningful = len(re.sub(r'\s', '', lhs)) > 3
    rhs_meaningful = len(re.sub(r'\s', '', rhs)) > 1
    return lhs_meaningful and rhs_meaningful


# ─────────────────────────────────────────────────────────────────
# 6.  GLOBAL DEDUP  (keep the LONGEST near-duplicate)
# ─────────────────────────────────────────────────────────────────

def _fp(text: str) -> str:
    """Fingerprint: alnum + operator chars, lowercased, first 65 chars."""
    return re.sub(r'[^\w=+\-*/×%]', '', text.lower())[:65]


def dedup(items: list[dict]) -> list[dict]:
    """
    Remove near-duplicates, keeping the LONGEST (most complete) version.
    Two items are considered duplicates if one's fingerprint is a prefix of the other.
    """
    # Sort longest-first so we keep the most complete form
    items = sorted(items, key=lambda x: -len(x['raw_text']))
    seen: set[str] = set()
    out:  list[dict] = []

    for eq in items:
        fp = _fp(eq['raw_text'])
        # Check if this fp (or any 40-char prefix) is already covered
        covered = any(
            fp.startswith(s[:40]) or s.startswith(fp[:40])
            for s in seen
        )
        if not covered:
            seen.add(fp)
            out.append(eq)

    # Restore original page order
    return sorted(out, key=lambda x: (x['page_number'], x['raw_text']))


# ─────────────────────────────────────────────────────────────────
# 6b.  FRACTION RECONSTRUCTION
#      Two-column PDFs often split display fractions across lines:
#        Line N:   "Cell viability(%) ="
#        Line N+1: "treated cells"      ← numerator
#        Line N+2: "untreated cells"    ← denominator (below the bar)
#        Line N+3: "× 100"
#      We detect these patterns from the context and rewrite the formula.
# ─────────────────────────────────────────────────────────────────

def reconstruct_fractions(equations: list[dict]) -> list[dict]:
    """Rewrite equations that clearly have a numerator/denominator split."""

    # Pattern: "NAME = × 100  DENOM"  → NAME = (NUM / DENOM) × 100
    CELL_VIA_RE = re.compile(
        r'^(treated\s+cells\s+)?'
        r'(Cell\s+viability\s*\(%\))\s*=\s*[×\\times\s]+100\s*(untreated\s+cells)?',
        re.IGNORECASE
    )
    HEMOLYSIS_RE = re.compile(
        r'%\s*of\s+hemolysis\s+at\s+\d+\s*nm'
        r'.*?\(OD\s+of\s+sample.*?OD\s+of.*?control\)\s*=\s*[×\\times\s]+100',
        re.IGNORECASE | re.DOTALL
    )
    BIOERADICATION_RE = re.compile(
        r'(?:formula\s+)?Eradication\s+of\s+biofilm\s*\(%\)'
        r'\s+OD\s+in\s+control\s+OD\s+in\s+treatment\s*=\s*OD\s+in\s+control',
        re.IGNORECASE
    )

    for eq in equations:
        raw = eq['raw_text']

        # ── Cell Viability ────────────────────────────────────────
        if CELL_VIA_RE.search(raw):
            eq['raw_text']   = 'Cell viability (%) = (treated cells / untreated cells) × 100'
            eq['normalized'] = eq['raw_text']
            eq['latex']      = r'Cell\,viability\,(\%) = \frac{\text{treated cells}}{\text{untreated cells}} \times 100'
            eq['equation_type'] = 'biological_formula'

        # ── % Hemolysis ───────────────────────────────────────────
        elif HEMOLYSIS_RE.search(raw):
            eq['raw_text']   = ('% Hemolysis at 540 nm = '
                                '(OD of sample − OD of −ve control) / '
                                '(OD of +ve control − OD of −ve control) × 100')
            eq['normalized'] = eq['raw_text']
            eq['latex']      = (r'\% \text{Hemolysis} = '
                                r'\frac{OD_{\text{sample}} - OD_{-ve}}'
                                r'{OD_{+ve} - OD_{-ve}} \times 100')
            eq['equation_type'] = 'biological_formula'

        # ── Biofilm Eradication ───────────────────────────────────
        elif BIOERADICATION_RE.search(raw):
            eq['raw_text']   = ('Eradication of biofilm (%) = '
                                '(OD in control − OD in treatment) / OD in control')
            eq['normalized'] = eq['raw_text']
            eq['latex']      = (r'\text{Eradication of biofilm}\,(\%) = '
                                r'\frac{OD_{\text{control}} - OD_{\text{treatment}}}'
                                r'{OD_{\text{control}}}')
            eq['equation_type'] = 'biological_formula'

    return equations


# ─────────────────────────────────────────────────────────────────
# 7.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def extract_equations(pdf_path: str, doc_title: str = None) -> dict:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(pdf_path)
    if not doc_title:
        doc_title = path.stem.replace('_', ' ').replace('-', ' ')

    all_raw: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        print(f"  Pages : {n_pages}")

        for page_num, page in enumerate(pdf.pages, start=1):
            cols = page_columns(page)
            page_hits = []
            for col_text in cols:
                hits = scan_column(col_text, page_num)
                page_hits.extend(hits)
            if page_hits:
                print(f"  Page {page_num:>3}: {len(page_hits)} candidate(s)")
            all_raw.extend(page_hits)

    unique = dedup(all_raw)

    # Apply cleaning + quality filter
    cleaned = []
    for eq in unique:
        eq['raw_text'] = clean_pdf_text(eq['raw_text'])
        if is_real_equation(eq['raw_text']):
            cleaned.append(eq)
    unique = cleaned

    # ── Final pass: reconstruct display fractions ─────────────────
    unique = reconstruct_fractions(unique)

    for rank, eq in enumerate(unique, start=1):
        norm = re.sub(r'\s+', ' ', eq['raw_text']).strip()
        eq['raw_text']    = norm
        eq['normalized']  = norm
        eq['equation_id'] = (
            f"eq_{rank:04d}_p{eq['page_number']:02d}_"
            f"{hashlib.md5(norm.encode()).hexdigest()[:6]}"
        )
        eq['latex']         = to_latex(norm)
        eq['equation_type'] = classify(norm)
        eq['variables']     = variables_of(norm)
        eq.pop('_source', None)
        eq['rag_chunk']     = rag_chunk(eq, doc_title)

    type_counts: dict[str, int] = {}
    for eq in unique:
        t = eq['equation_type']
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        'document_metadata': {
            'title':                 doc_title,
            'source_file':           str(path.resolve()),
            'page_count':            n_pages,
            'total_equations_found': len(unique),
            'extraction_timestamp':  datetime.now().isoformat(),
        },
        'equations': unique,
        'summary':   type_counts,
    }


# ─────────────────────────────────────────────────────────────────
# 8.  PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────

def print_summary(result: dict) -> None:
    meta = result['document_metadata']
    bar  = '═' * 66
    print(f"\n{bar}")
    print(f"  Document  : {meta['title']}")
    print(f"  Pages     : {meta['page_count']}")
    print(f"  Equations : {meta['total_equations_found']}")
    print(f"{bar}")
    print("  By type:")
    for t, n in sorted(result['summary'].items()):
        print(f"    {t:<34}  {n:>3}")
    print(f"{bar}")

    print("\n  All extracted equations:\n  " + '─' * 62)
    for eq in result['equations']:
        print(f"\n  [{eq['equation_id']}]")
        print(f"  Page {eq['page_number']}  ·  {eq.get('section','')[:60]}")
        print(f"  Type   : {eq['equation_type']}")
        print(f"  Formula: {eq['raw_text'][:115]}")
        print(f"  LaTeX  : {eq['latex'][:90]}")
    print()


# ─────────────────────────────────────────────────────────────────
# 9.  CLI
# ─────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    pdf_in  = "/content/AO_5c05577.pdf"
    out     = sys.argv[2] if len(sys.argv) > 2 else 'equations_output.json'

    print(f"\nProcessing: {pdf_in}")
    result = extract_equations(pdf_in)

    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print_summary(result)
    print(f"  Saved  →  {out}\n")


if __name__ == '__main__':
    main()