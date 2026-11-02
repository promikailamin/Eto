#!/usr/bin/env python3
"""Build a balanced, reproducible plain-text corpus for Eto training."""

import argparse
import re
from pathlib import Path


BOOKS = (
    "alice.txt",
    "drac.txt",
    "earnest.txt",
    "frank.txt",
    "ge.txt",
    "jane.txt",
    "moby.txt",
    "pp.txt",
    "sherlock.txt",
    "tom.txt",
    "wp.txt",
)
BOOK_BYTE_LIMIT = 256 * 1024
GRAMMAR_REPEATS = 16
CHAT_REPEATS = 4
CHAT_TOPIC = re.compile(
    r"\b(?:minecraft|creeper|crafting table|redstone|nether|bedrock edition|"
    r"ender dragon|villager|enderman)\b",
    re.IGNORECASE,
)
ULTRON = re.compile(r"\bultron\b", re.IGNORECASE)
MINECRAFT = re.compile(r"\bminecraft\b", re.IGNORECASE)
GUTENBERG_START = re.compile(
    r"^\*{3}\s*START OF (?:THE )?PROJECT GUTENBERG EBOOK[^\n]*\n?",
    re.IGNORECASE | re.MULTILINE,
)
GUTENBERG_END = re.compile(
    r"^(?:\*{3}\s*END OF (?:THE )?PROJECT GUTENBERG EBOOK|"
    r"End of (?:the )?Project Gutenberg)[^\n]*",
    re.IGNORECASE | re.MULTILINE,
)


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    start = GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = GUTENBERG_END.search(text)
    if end:
        text = text[:end.start()]
    return text


def paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def balanced_book(path: Path) -> str:
    parts = paragraphs(read_text(path))
    total_bytes = sum(len(part.encode("utf-8")) + 2 for part in parts)
    stride = max(1, (total_bytes + BOOK_BYTE_LIMIT - 1) // BOOK_BYTE_LIMIT)
    selected = parts[::stride]
    while len(selected) > 1 and sum(len(part.encode("utf-8")) + 2 for part in selected) > BOOK_BYTE_LIMIT:
        selected.pop()
    return "\n\n".join(selected)


def cleaned_chat(path: Path) -> str:
    text = read_text(path)
    starts = list(re.finditer(r"(?im)^user:\s*", text))
    unique_turns = []
    seen = set()
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        turn = text[match.start():end]
        turn = re.sub(r"(?m)^=+[^\n]*=+\s*$", "", turn)
        if CHAT_TOPIC.search(turn):
            continue
        turn = ULTRON.sub("Eto", turn).strip()
        key = " ".join(turn.split()).casefold()
        if key and key not in seen:
            seen.add(key)
            unique_turns.append(turn)
    return "\n\n".join(unique_turns)


def source_block(title: str, text: str) -> str:
    text = ULTRON.sub("Eto", text)
    text = MINECRAFT.sub("Mara", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return ""
    return f"Source: {title}\n\n{text}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pieces = []
    for filename in BOOKS:
        path = args.data_dir / filename
        if path.is_file():
            text = balanced_book(path)
            pieces.append(source_block(path.stem, text))

    for filename in ("grammar.txt", "english_grammar.txt"):
        path = args.data_dir / filename
        if path.is_file():
            block = source_block(path.stem, read_text(path))
            pieces.extend([block] * GRAMMAR_REPEATS)

    chat_path = args.data_dir / "chat.txt"
    if chat_path.is_file():
        block = source_block("English conversation", cleaned_chat(chat_path))
        pieces.extend([block] * CHAT_REPEATS)

    sample_path = args.data_dir / "sample.txt"
    if sample_path.is_file():
        pieces.append(source_block("English story", read_text(sample_path)))

    corpus = "\n\n---\n\n".join(piece for piece in pieces if piece) + "\n"
    if len(corpus.encode("utf-8")) < 16:
        raise SystemExit("prepared corpus is empty or too small")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus, encoding="utf-8", newline="\n")
    print(f"prepared {len(pieces)} weighted sections, {len(corpus.encode('utf-8')):,} bytes -> {args.output}")


if __name__ == "__main__":
    main()