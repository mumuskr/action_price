from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from ebooklib import epub

from brooks_trader.knowledge import (
    ChunkConfig,
    FaissKnowledgeBase,
    approved_rules,
    candidate_rule,
    chunk_paragraphs,
    load_knowledge_config,
    load_rule_library,
    parse_epub,
    parse_markdown,
    read_paragraphs_jsonl,
    write_paragraphs_jsonl,
)
from brooks_trader.models import BookParagraph, BrooksRule, RuleStatus

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_epub(path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("phase8-test")
    book.set_title("Price Action Test Book")
    book.set_language("en")
    second = epub.EpubHtml(title="Second", file_name="second.xhtml", lang="en")
    second.content = "<h1>Second Chapter</h1><p>Trading range breakout failure.</p>"
    first = epub.EpubHtml(title="First", file_name="first.xhtml", lang="en")
    first.content = (
        "<h1>First Chapter</h1><h2>Bull Flags</h2>"
        "<p>An H2 is a second upward attempt.</p><p>Context remains important.</p>"
    )
    book.add_item(second)
    book.add_item(first)
    book.toc = (first, second)
    book.spine = ["nav", first, second]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)


def test_epub_parser_uses_spine_order_and_records_provenance(tmp_path: Path) -> None:
    source = tmp_path / "test.epub"
    make_epub(source)

    paragraphs = parse_epub(source)

    assert [paragraph.chapter for paragraph in paragraphs] == [
        "First Chapter",
        "First Chapter",
        "Second Chapter",
    ]
    assert paragraphs[0].section == "Bull Flags"
    assert paragraphs[0].paragraph == 1
    assert paragraphs[0].source_file == "test.epub"
    assert paragraphs[0].source_document == "first.xhtml"


def test_markdown_parser_and_jsonl_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "sample.md"
    source.write_text(
        "# Test Book\n\n## Pullbacks\n\nFirst paragraph.\n\n"
        "### Second Entries\n\nAn **H2** requires a renewed attempt.\n",
        encoding="utf-8",
    )

    paragraphs = parse_markdown(source, book_title="Explicit Title")
    destination = write_paragraphs_jsonl(paragraphs, tmp_path / "processed.jsonl")

    assert paragraphs[0].book == "Explicit Title"
    assert paragraphs[0].chapter == "Pullbacks"
    assert paragraphs[1].section == "Second Entries"
    assert paragraphs[1].text == "An H2 requires a renewed attempt."
    assert read_paragraphs_jsonl(destination) == paragraphs


def test_chunking_stays_within_source_boundaries_and_size_limit() -> None:
    paragraphs = [
        BookParagraph(
            book="Book",
            chapter="One",
            section="H2",
            paragraph=index,
            text=("second entry bull flag " * 8).strip(),
            source_file="book.md",
            source_document="book.md",
        )
        for index in range(1, 4)
    ] + [
        BookParagraph(
            book="Book",
            chapter="Two",
            section=None,
            paragraph=1,
            text="trading range",
            source_file="book.md",
            source_document="book.md",
        )
    ]
    config = ChunkConfig(target_characters=100, maximum_characters=140, overlap_paragraphs=1)

    chunks = chunk_paragraphs(paragraphs, config=config)

    assert chunks
    assert all(len(chunk.text) <= config.maximum_characters for chunk in chunks)
    assert all("One" not in chunk.text or chunk.chapter == "One" for chunk in chunks)
    assert chunks[-1].chapter == "Two"
    assert chunks[0].source_reference.startswith("Book | One | H2")


def test_faiss_index_round_trip_returns_source_backed_h2_match(tmp_path: Path) -> None:
    paragraphs = [
        BookParagraph(
            book="Trends",
            chapter="Second Entries",
            paragraph=1,
            text="H2 second entry bull flag strong bull trend stop above the signal bar.",
            source_file="trends.md",
            source_document="trends.md",
        ),
        BookParagraph(
            book="Ranges",
            chapter="Breakouts",
            paragraph=1,
            text="A failed breakout returns into a trading range after breaking support.",
            source_file="ranges.md",
            source_document="ranges.md",
        ),
    ]
    config = load_knowledge_config(PROJECT_ROOT / "config/knowledge.yaml")
    chunks = chunk_paragraphs(paragraphs, config=config.chunk)
    index_path = FaissKnowledgeBase.build(chunks, config=config).save(tmp_path / "faiss")

    restored = FaissKnowledgeBase.load(index_path)
    results = restored.search("H2 second entry bull flag", top_k=1)

    assert results[0].chunk.book == "Trends"
    assert results[0].chunk.source_reference.startswith("Trends | Second Entries")
    assert results[0].score > 0


def test_faiss_load_rejects_tampered_chunk_metadata(tmp_path: Path) -> None:
    config = load_knowledge_config(PROJECT_ROOT / "config/knowledge.yaml")
    paragraphs = [
        BookParagraph(
            book="Trends",
            chapter="H2",
            paragraph=1,
            text="Second entry bull flag.",
            source_file="trends.md",
            source_document="trends.md",
        )
    ]
    chunks = chunk_paragraphs(paragraphs, config=config.chunk)
    index_path = FaissKnowledgeBase.build(chunks, config=config).save(tmp_path / "faiss")
    chunks_path = index_path / "chunks.jsonl"
    chunks_path.write_text(chunks_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="chunk file failed integrity"):
        FaissKnowledgeBase.load(index_path)


def test_rule_library_defaults_to_candidates_and_approved_filter_is_strict() -> None:
    rules = load_rule_library(PROJECT_ROOT / "knowledge/patterns")

    assert len(rules) == 11
    assert all(rule.status == RuleStatus.CANDIDATE for rule in rules)
    assert approved_rules(rules) == []
    approved_values = rules[0].model_dump()
    approved_values.update(
        {
            "status": RuleStatus.APPROVED,
            "reviewed_by": "researcher",
            "reviewed_at": datetime(2026, 8, 10, tzinfo=UTC),
        }
    )
    approved_values["source"].update(
        {"chapter": "10", "source_reference": "Chapter 10, paragraph 12"}
    )
    approved = BrooksRule.model_validate(approved_values)
    assert approved_rules([*rules, approved]) == [approved]


def test_extracted_rule_is_forced_to_candidate_even_if_llm_claims_approved() -> None:
    values = {
        "name": "Extracted Rule",
        "category": "second_entry",
        "direction": "long",
        "source": {
            "book": "Trends",
            "chapter": "1",
            "source_reference": "paragraph 1",
        },
        "definition": "Candidate text.",
        "status": "approved",
    }

    rule = candidate_rule(values)

    assert rule.status == RuleStatus.CANDIDATE
    assert not rule.is_strategy_eligible


def test_invalid_rule_yaml_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "rules"
    directory.mkdir()
    (directory / "invalid.yaml").write_text(
        yaml.safe_dump({"name": "Incomplete", "status": "approved"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid Brooks rule"):
        load_rule_library(directory)


def test_approved_rule_requires_audited_review_and_verified_source() -> None:
    rule = load_rule_library(PROJECT_ROOT / "knowledge/patterns")[0]
    values = rule.model_dump()
    values["status"] = RuleStatus.APPROVED

    with pytest.raises(ValueError, match="reviewed_by and reviewed_at"):
        BrooksRule.model_validate(values)
