"""
backend/src/multimodal/ingestion/text.py

Text chunking with graceful fallback.

BUG FIXED: flush() in _chunk_from_text was missing `nonlocal` declarations
for buf_lines, buf_pages, buf_tokens, cur_section.

Python sees the assignment `buf_lines = []` at the END of flush() and
therefore treats `buf_lines` as a LOCAL variable throughout the entire
function body — including the guard `if not buf_lines` at the TOP.
This causes an UnboundLocalError on the first real flush call, meaning
zero text chunks are ever produced.

Fix: add `nonlocal buf_lines, buf_pages, buf_tokens, cur_section` at the
start of flush() so Python correctly mutates the enclosing scope's variables.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..types import PageBlocks, SectionSpan, TextChunk
from .section import detect_heading, resolve_section
from .utils import estimate_tokens, tokenize


def build_text_chunks(
    *,
    pages: Sequence[PageBlocks],
    spans: Sequence[SectionSpan],
    journal_id: str,
    article_id: str,
    source_path: str,
    token_limit: int,
) -> list[TextChunk]:
    """
    Build text chunks from pages.

    Tries layout-aware (block-based) chunking first.
    If page.blocks is empty (pymupdf4llm default), falls back to
    line-by-line processing of page.text.
    """
    has_blocks = any(
        isinstance(b, dict) and b.get("type") == "text"
        for page in pages
        for b in page.blocks
    )

    if has_blocks:
        return _chunk_from_blocks(
            pages=pages, spans=spans,
            journal_id=journal_id, article_id=article_id,
            source_path=source_path, token_limit=token_limit,
        )
    else:
        return _chunk_from_text(
            pages=pages, spans=spans,
            journal_id=journal_id, article_id=article_id,
            source_path=source_path, token_limit=token_limit,
        )


# ── Path A: block-based ────────────────────────────────────────────────────────

def _chunk_from_blocks(
    *,
    pages: Sequence[PageBlocks],
    spans: Sequence[SectionSpan],
    journal_id: str,
    article_id: str,
    source_path: str,
    token_limit: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []

    for page in pages:
        midpoint = page.width / 2 if page.width > 0 else 300.0

        left_blocks = sorted(
            [b for b in page.blocks
             if isinstance(b, dict) and b.get("type") == "text"
             and "bbox" in b and b["bbox"][0] < midpoint],
            key=lambda b: b["bbox"][1],
        )
        right_blocks = sorted(
            [b for b in page.blocks
             if isinstance(b, dict) and b.get("type") == "text"
             and "bbox" in b and b["bbox"][0] >= midpoint],
            key=lambda b: b["bbox"][1],
        )

        for col_idx, col_blocks in enumerate([left_blocks, right_blocks]):
            if not col_blocks:
                continue

            buf_text: list[str] = []
            buf_tokens = 0
            cur_section: str | None = None
            buf_bbox: tuple | None = None

            # _chunk_from_blocks flush takes all state as parameters → no nonlocal needed
            def flush(sect, col, btext, btokens, bbbox):
                if not btext:
                    return
                text = " ".join(btext).strip()
                if not text:
                    return
                chunks.append(TextChunk(
                    chunk_id=f"{journal_id}_{article_id}_text_{len(chunks)+1:05d}",
                    journal_id=journal_id, article_id=article_id,
                    source_path=source_path, text=text,
                    token_count=estimate_tokens(text),
                    page_start=page.page_number, page_end=page.page_number,
                    section=sect,
                    caption=_caption(sect, page.page_number, page.page_number),
                    bbox=bbbox, column=col, block_type="text",
                ))

            for block in col_blocks:
                block_text = block["text"].strip()
                if not block_text:
                    continue
                block_bbox = tuple(block["bbox"])
                line_proxy = int(block_bbox[1])
                section = resolve_section(page.page_number, line_proxy, spans)
                block_tokens = estimate_tokens(block_text)

                if (cur_section is not None and section != cur_section) or \
                        (buf_tokens + block_tokens > token_limit):
                    flush(cur_section, col_idx, buf_text, buf_tokens, buf_bbox)
                    buf_text, buf_tokens, buf_bbox = [], 0, None

                cur_section = section
                buf_text.append(block_text)
                buf_tokens += block_tokens
                if buf_bbox is None:
                    buf_bbox = block_bbox
                else:
                    buf_bbox = (
                        min(buf_bbox[0], block_bbox[0]),
                        min(buf_bbox[1], block_bbox[1]),
                        max(buf_bbox[2], block_bbox[2]),
                        max(buf_bbox[3], block_bbox[3]),
                    )

            flush(cur_section, col_idx, buf_text, buf_tokens, buf_bbox)

    return chunks


# ── Path B: line-based fallback ────────────────────────────────────────────────

def _chunk_from_text(
    *,
    pages: Sequence[PageBlocks],
    spans: Sequence[SectionSpan],
    journal_id: str,
    article_id: str,
    source_path: str,
    token_limit: int,
) -> list[TextChunk]:
    """
    Process page.text line by line.

    FIX: flush() declares `nonlocal buf_lines, buf_pages, buf_tokens,
    cur_section`. Without this, Python treats those names as local variables
    throughout flush() (because of the assignments at the end), raising
    UnboundLocalError even on the first `if not buf_lines` guard.
    """
    chunks: list[TextChunk] = []

    buf_lines: list[str] = []
    buf_pages: list[int] = []
    buf_tokens = 0
    cur_section: str | None = None

    def flush() -> None:
        nonlocal buf_lines, buf_pages, buf_tokens, cur_section  # ← THE FIX
        if not buf_lines or not buf_pages:
            buf_lines, buf_pages, buf_tokens = [], [], 0
            return
        text = "\n".join(buf_lines).strip()
        if text:
            chunks.append(TextChunk(
                chunk_id=f"{journal_id}_{article_id}_text_{len(chunks)+1:05d}",
                journal_id=journal_id, article_id=article_id,
                source_path=source_path, text=text,
                token_count=estimate_tokens(text),
                page_start=buf_pages[0], page_end=buf_pages[-1],
                section=cur_section,
                caption=_caption(cur_section, buf_pages[0], buf_pages[-1]),
            ))
        buf_lines, buf_pages, buf_tokens = [], [], 0

    for page in pages:
        if not page.text:
            continue

        for line_index, raw_line in enumerate(page.text.splitlines()):
            stripped = raw_line.strip()
            section = resolve_section(page.page_number, line_index, spans)

            if detect_heading(stripped):
                flush()
                cur_section = section
                continue

            if not stripped:
                if buf_lines and buf_lines[-1] != "":
                    buf_lines.append("")
                continue

            line_tokens = estimate_tokens(stripped)

            if line_tokens > token_limit:
                flush()
                cur_section = section
                for fragment in _token_slices(stripped, token_limit):
                    if not fragment:
                        continue
                    chunks.append(TextChunk(
                        chunk_id=f"{journal_id}_{article_id}_text_{len(chunks)+1:05d}",
                        journal_id=journal_id, article_id=article_id,
                        source_path=source_path, text=fragment,
                        token_count=estimate_tokens(fragment),
                        page_start=page.page_number, page_end=page.page_number,
                        section=cur_section,
                        caption=_caption(cur_section, page.page_number, page.page_number),
                    ))
                continue

            if buf_lines and (section != cur_section or
                              buf_tokens + line_tokens > token_limit):
                flush()

            cur_section = section
            buf_lines.append(stripped)
            buf_pages.append(page.page_number)
            buf_tokens += line_tokens

    flush()
    return chunks


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _caption(section: str | None, page_start: int, page_end: int) -> str:
    prefix = f"{section} " if section else ""
    if page_start == page_end:
        return f"{prefix}page {page_start}".strip()
    return f"{prefix}pages {page_start}-{page_end}".strip()


def _token_slices(text: str, token_limit: int) -> list[str]:
    tokens = tokenize(text)
    if not tokens:
        return []
    return [
        " ".join(tokens[i: i + token_limit]).strip()
        for i in range(0, len(tokens), token_limit)
        if tokens[i: i + token_limit]
    ]