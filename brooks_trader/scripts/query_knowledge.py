"""Query the local Phase 8 Brooks knowledge index with source citations."""

from __future__ import annotations

import argparse
from pathlib import Path

from brooks_trader.knowledge import FaissKnowledgeBase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Retrieval query")
    parser.add_argument("--index", type=Path, default=Path("books/processed/faiss"))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--book", help="Optional exact book-title filter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    knowledge_base = FaissKnowledgeBase.load(args.index)
    results = knowledge_base.search(args.query, top_k=args.top_k, book=args.book)
    if not results:
        print("No matching knowledge chunks found.")
        return
    for rank, result in enumerate(results, start=1):
        print(f"[{rank}] score={result.score:.4f} {result.chunk.source_reference}")
        print(result.chunk.text)
        print()


if __name__ == "__main__":
    main()
