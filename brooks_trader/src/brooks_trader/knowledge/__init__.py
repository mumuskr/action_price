"""Public Brooks book, rule-library, and local retrieval API."""

from brooks_trader.knowledge.epub_parser import (
    parse_book,
    parse_epub,
    parse_markdown,
    read_paragraphs_jsonl,
    write_paragraphs_jsonl,
)
from brooks_trader.knowledge.rag import (
    ChunkConfig,
    EmbeddingConfig,
    FaissKnowledgeBase,
    HashingEmbedder,
    KnowledgeConfig,
    RetrievalConfig,
    RetrievalResult,
    chunk_paragraphs,
    load_knowledge_config,
)
from brooks_trader.knowledge.rule_extractor import (
    approved_rules,
    candidate_rule,
    load_rule,
    load_rule_library,
    write_rule,
)

__all__ = [
    "ChunkConfig",
    "EmbeddingConfig",
    "FaissKnowledgeBase",
    "HashingEmbedder",
    "KnowledgeConfig",
    "RetrievalConfig",
    "RetrievalResult",
    "approved_rules",
    "candidate_rule",
    "chunk_paragraphs",
    "load_knowledge_config",
    "load_rule",
    "load_rule_library",
    "parse_book",
    "parse_epub",
    "parse_markdown",
    "read_paragraphs_jsonl",
    "write_paragraphs_jsonl",
    "write_rule",
]
