from __future__ import annotations

from typing import Sequence

from ..types import TextChunk
from .section import SectionSpan, detect_heading, resolve_section
from .utils import PageText, estimate_tokens, tokenize


def build_text_chunks(
    *,
    pages: Sequence[PageText],
    spans: Sequence[SectionSpan],
    journal_id: str,
    article_id: str,
    source_path: str,
    token_limit: int,
) -> list[TextChunk]:
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
            section = resolve_section(page.page_number, line_index, spans)

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
