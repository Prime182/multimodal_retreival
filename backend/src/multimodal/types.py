from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import re


ContentType = Literal["text", "equation", "table", "image"]
_PDF_STEM_RE = re.compile(r"^(?P<jid>[A-Za-z]{2})(?:_(?P<aid1>[A-Za-z0-9]+)|(?P<aid2>[A-Za-z0-9]+))$")


def parse_pdf_filename(path: str | Path) -> tuple[str, str]:
    stem = Path(path).stem
    match = _PDF_STEM_RE.fullmatch(stem)
    if not match:
        raise ValueError(
            f"PDF filename '{stem}' must match '<JJ><ARTICLE>' or '<JJ>_<ARTICLE>' with a two-letter journal prefix."
        )
    article_id = match.group("aid1") or match.group("aid2")
    if not article_id:
        raise ValueError(f"Could not derive article_id from '{stem}'.")
    return match.group("jid").upper(), article_id


def build_document_id(path: str | Path) -> str:
    return Path(path).stem


@dataclass(slots=True)
class TextChunk:
    chunk_id: str
    journal_id: str
    article_id: str
    source_path: str
    text: str
    token_count: int
    page_start: int
    page_end: int
    section: str | None = None
    caption: str | None = None
    content_type: ContentType = "text"

    @property
    def embed_text(self) -> str:
        parts = [self.section, self.text]
        return "\n\n".join(part for part in parts if part)

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.content_type
        return payload


@dataclass(slots=True)
class EquationChunk:
    chunk_id: str
    journal_id: str
    article_id: str
    source_path: str
    latex: str
    context: str | None = None
    page_number: int | None = None
    section: str | None = None
    content_type: ContentType = "equation"

    @property
    def embed_text(self) -> str:
        parts = [self.section, self.context, self.latex]
        return "\n\n".join(part for part in parts if part)

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.content_type
        return payload


@dataclass(slots=True)
class TableChunk:
    chunk_id: str
    journal_id: str
    article_id: str
    source_path: str
    csv_data: str
    header: str
    row_index: int | None = None
    page_number: int | None = None
    section: str | None = None
    content_type: ContentType = "table"

    @property
    def embed_text(self) -> str:
        parts = [self.section, self.header, self.csv_data]
        return "\n\n".join(part for part in parts if part)

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.content_type
        return payload


@dataclass(slots=True)
class ExtractedImage:
    image_id: str
    journal_id: str
    article_id: str
    source_path: str
    file_path: str
    page_number: int | None
    mime_type: str
    width: int | None = None
    height: int | None = None
    caption: str | None = None
    image_url: str | None = None
    section: str | None = None
    content_type: ContentType = "image"

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.content_type
        payload["imageUrl"] = self.image_url
        payload["mimeType"] = self.mime_type
        return payload


@dataclass(slots=True)
class IngestedDocument:
    document_id: str
    journal_id: str
    article_id: str
    source_path: str
    text_chunks: list[TextChunk] = field(default_factory=list)
    equation_chunks: list[EquationChunk] = field(default_factory=list)
    table_chunks: list[TableChunk] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    item_id: str
    distance: float
    metadata: dict[str, Any]
    score: float | None = None
