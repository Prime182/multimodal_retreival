from __future__ import annotations

from typing import Sequence

from ..types import PageBlocks, SectionSpan, TextChunk
from .section import detect_heading, resolve_section_spatial
from .utils import estimate_tokens, tokenize


def build_text_chunks_layout_aware(
    *,
    pages: list[PageBlocks],
    spans: Sequence[SectionSpan],
    journal_id: str,
    article_id: str,
    source_path: str,
    token_limit: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []

    for page in pages:
        # Group blocks by column (left vs right)
        # Using a fixed midpoint or page.width / 2
        midpoint = page.width / 2 if page.width > 0 else 300
        
        left_blocks = [
            b for b in page.blocks 
            if isinstance(b, dict) and b.get("type") == "text" and "bbox" in b and b["bbox"][0] < midpoint
        ]
        right_blocks = [
            b for b in page.blocks 
            if isinstance(b, dict) and b.get("type") == "text" and "bbox" in b and b["bbox"][0] >= midpoint
        ]

        for column_idx, column_blocks in enumerate([left_blocks, right_blocks]):
            if not column_blocks:
                continue
                
            # Sort by Y position (top-to-bottom reading order)
            sorted_blocks = sorted(column_blocks, key=lambda b: b["bbox"][1])
            
            buffer_text = []
            buffer_tokens = 0
            current_section = None
            buffer_bbox = None

            def flush_buffer(sect, col, b_text, b_tokens, b_bbox):
                if not b_text:
                    return
                text = " ".join(b_text).strip()
                if not text:
                    return
                
                chunks.append(
                    TextChunk(
                        chunk_id=f"{journal_id}_{article_id}_text_{len(chunks) + 1:05d}",
                        journal_id=journal_id,
                        article_id=article_id,
                        source_path=source_path,
                        text=text,
                        token_count=estimate_tokens(text),
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section=sect,
                        caption=_build_text_caption(sect, page.page_number, page.page_number),
                        bbox=b_bbox,
                        column=col,
                        block_type="text"
                    )
                )

            for block in sorted_blocks:
                block_text = block["text"].strip()
                if not block_text:
                    continue
                
                block_bbox = tuple(block["bbox"])
                section = resolve_section_spatial(page.page_number, block_bbox, spans)
                block_tokens = estimate_tokens(block_text)

                # If section changed or token limit exceeded, flush
                if (current_section is not None and section != current_section) or \
                   (buffer_tokens + block_tokens > token_limit):
                    flush_buffer(current_section, column_idx, buffer_text, buffer_tokens, buffer_bbox)
                    buffer_text = []
                    buffer_tokens = 0
                    buffer_bbox = None

                current_section = section
                buffer_text.append(block_text)
                buffer_tokens += block_tokens
                if buffer_bbox is None:
                    buffer_bbox = block_bbox
                else:
                    # Update bounding box to encompass the new block
                    buffer_bbox = (
                        min(buffer_bbox[0], block_bbox[0]),
                        min(buffer_bbox[1], block_bbox[1]),
                        max(buffer_bbox[2], block_bbox[2]),
                        max(buffer_bbox[3], block_bbox[3])
                    )

            flush_buffer(current_section, column_idx, buffer_text, buffer_tokens, buffer_bbox)

    return chunks


def build_text_chunks(
    *,
    pages: Sequence[any],  # Generic to avoid type conflict during transition
    spans: Sequence[SectionSpan],
    journal_id: str,
    article_id: str,
    source_path: str,
    token_limit: int,
) -> list[TextChunk]:
    # This is the original function, redirected to layout-aware if we have PageBlocks
    if pages and isinstance(pages[0], PageBlocks):
        return build_text_chunks_layout_aware(
            pages=list(pages),
            spans=spans,
            journal_id=journal_id,
            article_id=article_id,
            source_path=source_path,
            token_limit=token_limit
        )
    
    # This part was accidentally modified with comments instead of keeping the fallback logic.
    # We've already implemented build_text_chunks_layout_aware, which is the preferred way.
    # We will ensure this function calls it correctly if PageBlocks are provided.

    chunks: list[TextChunk] = []

    buffer_lines: list[str] = []
    buffer_pages: list[int] = []
    buffer_tokens = 0
    current_section: str | None = None

    def flush() -> None:
        nonlocal buffer_lines, buffer_pages, buffer_tokens, current_section
        if not buffer_lines or not buffer_pages:
            buffer_lines = []
            buffer_pages = []
            buffer_tokens = 0
            return

        text = "\n".join(buffer_lines).strip()
        if text:
            chunks.append(
                TextChunk(
                    chunk_id=f"{journal_id}_{article_id}_text_{len(chunks) + 1:05d}",
                    journal_id=journal_id,
                    article_id=article_id,
                    source_path=source_path,
                    text=text,
                    token_count=estimate_tokens(text),
                    page_start=buffer_pages[0],
                    page_end=buffer_pages[-1],
                    section=current_section,
                    caption=_build_text_caption(current_section, buffer_pages[0], buffer_pages[-1]),
                )
            )
        buffer_lines = []
        buffer_pages = []
        buffer_tokens = 0

    for page in pages:
        if not page.text:
            continue

        for line_index, raw_line in enumerate(page.text.splitlines()):
            stripped = raw_line.strip()
            section = resolve_section_spatial(page.page_number, line_index, spans)

            if detect_heading(stripped):
                flush()
                current_section = section
                continue

            if not stripped:
                if buffer_lines and buffer_lines[-1] != "":
                    buffer_lines.append("")
                continue

            line_tokens = estimate_tokens(stripped)
            if line_tokens > token_limit:
                flush()
                current_section = section
                for fragment in _token_slices(stripped, token_limit):
                    if not fragment:
                        continue
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{journal_id}_{article_id}_text_{len(chunks) + 1:05d}",
                            journal_id=journal_id,
                            article_id=article_id,
                            source_path=source_path,
                            text=fragment,
                            token_count=estimate_tokens(fragment),
                            page_start=page.page_number,
                            page_end=page.page_number,
                            section=current_section,
                            caption=_build_text_caption(
                                current_section,
                                page.page_number,
                                page.page_number,
                            ),
                        )
                    )
                continue

            if buffer_lines and (section != current_section or buffer_tokens + line_tokens > token_limit):
                flush()

            current_section = section
            buffer_lines.append(stripped)
            buffer_pages.append(page.page_number)
            buffer_tokens += line_tokens

    flush()
    return chunks


def _build_text_caption(section: str | None, page_start: int, page_end: int) -> str:
    section_prefix = f"{section} " if section else ""
    if page_start == page_end:
        return f"{section_prefix}page {page_start}".strip()
    return f"{section_prefix}pages {page_start}-{page_end}".strip()


def _token_slices(text: str, token_limit: int) -> list[str]:
    tokens = tokenize(text)
    if not tokens:
        return []
    return [
        " ".join(tokens[index:index + token_limit]).strip()
        for index in range(0, len(tokens), token_limit)
        if tokens[index:index + token_limit]
    ]
