"""Local EPUB/Markdown extraction with paragraph-level source provenance."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub

from brooks_trader.models import BookParagraph

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_MARKDOWN_DECORATION = re.compile(r"(?:\*\*|__|`|~~)")


def parse_book(path: str | Path, *, book_title: str | None = None) -> list[BookParagraph]:
    """Parse one supported local source without sending licensed text externally."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"book source does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".epub":
        return parse_epub(source, book_title=book_title)
    if suffix in {".md", ".markdown"}:
        return parse_markdown(source, book_title=book_title)
    raise ValueError(f"unsupported book format {suffix!r}; expected EPUB or Markdown")


def parse_epub(path: str | Path, *, book_title: str | None = None) -> list[BookParagraph]:
    """Extract ordered text documents from an EPUB using ebooklib and BeautifulSoup."""
    source = Path(path).expanduser()
    book = epub.read_epub(str(source), options={"ignore_ncx": True})
    title = book_title or _metadata_title(book) or source.stem
    paragraphs: list[BookParagraph] = []
    chapter_counter = 0
    for item in _ordered_document_items(book):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        heading = soup.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        fallback = Path(item.get_name()).stem.replace("_", " ").replace("-", " ").strip()
        chapter = _normalize_text(heading.get_text(" ", strip=True)) if heading else fallback
        if not chapter:
            chapter_counter += 1
            chapter = f"Document {chapter_counter}"
        current_section: str | None = None
        paragraph_number = 0
        for element in soup.find_all(["h2", "h3", "h4", "h5", "h6", "p", "li"]):
            text = _normalize_text(element.get_text(" ", strip=True))
            if not text:
                continue
            if element.name in {"h2", "h3", "h4", "h5", "h6"}:
                current_section = text
                continue
            paragraph_number += 1
            paragraphs.append(
                BookParagraph(
                    book=title,
                    chapter=chapter,
                    section=current_section,
                    paragraph=paragraph_number,
                    text=text,
                    source_file=source.name,
                    source_document=item.get_name(),
                )
            )
    if not paragraphs:
        raise ValueError(f"EPUB contains no extractable paragraphs: {source}")
    return paragraphs


def parse_markdown(
    path: str | Path,
    *,
    book_title: str | None = None,
) -> list[BookParagraph]:
    """Extract headings and paragraph blocks from Markdown without an HTML renderer."""
    source = Path(path).expanduser()
    title = book_title or source.stem
    chapter = "Front Matter"
    section: str | None = None
    paragraph_number = 0
    paragraphs: list[BookParagraph] = []
    block: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal paragraph_number
        text = _normalize_markdown(" ".join(block))
        block.clear()
        if not text:
            return
        paragraph_number += 1
        paragraphs.append(
            BookParagraph(
                book=title,
                chapter=chapter,
                section=section,
                paragraph=paragraph_number,
                text=text,
                source_file=source.name,
                source_document=source.name,
            )
        )

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("```") or line.startswith("~~~"):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = _MARKDOWN_HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            heading_text = _normalize_markdown(heading.group(2))
            if level <= 2:
                chapter = heading_text
                section = None
                paragraph_number = 0
            else:
                section = heading_text
            continue
        if not line:
            flush()
        else:
            block.append(line)
    flush()
    if not paragraphs:
        raise ValueError(f"Markdown contains no extractable paragraphs: {source}")
    return paragraphs


def write_paragraphs_jsonl(
    paragraphs: Sequence[BookParagraph],
    path: str | Path,
) -> Path:
    """Atomically write the canonical processed-books JSONL artifact."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for paragraph in paragraphs:
            stream.write(json.dumps(paragraph.model_dump(mode="json"), ensure_ascii=False))
            stream.write("\n")
    temporary.replace(destination)
    return destination


def read_paragraphs_jsonl(path: str | Path) -> list[BookParagraph]:
    """Load and validate a processed-books JSONL artifact."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"processed books file does not exist: {source}")
    paragraphs: list[BookParagraph] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                paragraphs.append(BookParagraph.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid paragraph at JSONL line {line_number}") from error
    if not paragraphs:
        raise ValueError(f"processed books file is empty: {source}")
    return paragraphs


def _metadata_title(book: Any) -> str | None:
    values = book.get_metadata("DC", "title")
    if not values:
        return None
    value = values[0][0]
    return _normalize_text(str(value)) or None


def _ordered_document_items(book: Any) -> list[Any]:
    items: list[Any] = []
    seen: set[str] = set()
    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ITEM_DOCUMENT or isinstance(item, epub.EpubNav):
            continue
        items.append(item)
        seen.add(item.get_id())
    if items:
        return items
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        if item.get_id() not in seen and not isinstance(item, epub.EpubNav):
            items.append(item)
    return items


def _normalize_markdown(text: str) -> str:
    value = re.sub(r"^[-*+]\s+", "", text)
    value = re.sub(r"^\d+[.)]\s+", "", value)
    value = re.sub(r"!\[([^]]*)]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", value)
    value = _MARKDOWN_DECORATION.sub("", value)
    return _normalize_text(value)


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())
