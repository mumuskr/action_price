"""Parse local EPUB/Markdown books and build the Phase 8 FAISS knowledge index."""

from __future__ import annotations

import argparse
from pathlib import Path

from brooks_trader.knowledge import (
    FaissKnowledgeBase,
    chunk_paragraphs,
    load_knowledge_config,
    parse_book,
    write_paragraphs_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="EPUB/Markdown sources; defaults to supported files in books/raw",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("books/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("books/processed"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/knowledge.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = args.sources or sorted(
        path
        for path in args.raw_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".epub", ".md", ".markdown"}
    )
    if not sources:
        raise SystemExit(
            "No EPUB or Markdown books found. Place licensed files in books/raw or pass paths."
        )
    config = load_knowledge_config(args.config)
    paragraphs = []
    for source in sources:
        extracted = parse_book(source)
        paragraphs.extend(extracted)
        print(f"Parsed {len(extracted):,} paragraphs from {source}")
    processed_path = write_paragraphs_jsonl(
        paragraphs,
        args.processed_root / "processed_books.jsonl",
    )
    chunks = chunk_paragraphs(paragraphs, config=config.chunk)
    index_path = FaissKnowledgeBase.build(chunks, config=config).save(args.processed_root / "faiss")
    print(f"Paragraphs: {len(paragraphs):,}")
    print(f"Chunks: {len(chunks):,}")
    print(f"Processed books: {processed_path}")
    print(f"FAISS index: {index_path}")


if __name__ == "__main__":
    main()
