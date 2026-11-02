#!/usr/bin/env python3
"""Convert downloaded public chat/story datasets into Eto's plain-text chat format.

Chat turns look like:      user: <text>\nEto: <text>\n\nuser: ...
Each converted dialogue chunk is kept short (<= MAX_CHUNK bytes) so that a whole
exchange fits inside the model's 256-byte context window.
"""
import json
import re
import sys
from pathlib import Path

MAX_CHUNK = 240
ALLOWED = re.compile(r"^[\x20-\x7e‘’“”]+$")


def clean(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def ok(text: str) -> bool:
    return bool(text) and ALLOWED.match(text) is not None


def chunk_dialogue(utterances):
    """Split an alternating dialogue into user/Eto chunks that start on a user turn."""
    chunks, current, size = [], [], 0
    for index, utterance in enumerate(utterances):
        speaker = "user" if index % 2 == 0 else "Eto"
        line = f"{speaker}: {utterance}"
        if size + len(line) + 1 > MAX_CHUNK and current:
            if len(current) >= 2:
                chunks.append(current)
            current, size = [], 0
            if speaker != "user":
                continue  # wait for the next user turn to restart
        if not current and speaker != "user":
            continue
        current.append(line)
        size += len(line) + 1
    if len(current) >= 2:
        chunks.append(current)
    return ["\n".join(c) for c in chunks]


def alpaca(path: Path) -> str:
    out = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        instruction = clean(item.get("instruction", ""))
        extra = clean(item.get("input", ""))
        answer = clean(item.get("output", ""))
        prompt = f"{instruction} {extra}".strip()
        if not (ok(prompt) and ok(answer)):
            continue
        turn = f"user: {prompt}\nEto: {answer}"
        if len(turn) <= MAX_CHUNK:
            out.append(turn)
    return "\n\n".join(out)


def chatterbot(paths) -> str:
    out = []
    for path in paths:
        conversations, current = [], None
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.startswith("- - "):
                current = [raw[4:]]
                conversations.append(current)
            elif raw.startswith("  - ") and current is not None:
                current.append(raw[4:])
        for utterances in conversations:
            utterances = [clean(u.strip("'\"")) for u in utterances]
            if all(ok(u) for u in utterances):
                out.extend(chunk_dialogue(utterances))
    return "\n\n".join(out)


def dailydialog(path: Path) -> str:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split("__eou__") if p.strip()]
        fixed = []
        for p in parts:
            p = re.sub(r"\s+([,.!?;:%])", r"\1", p)
            p = re.sub(r"\s+'\s*(s|m|t|re|ve|ll|d)\b", r"'\1", p)
            p = re.sub(r"\b(\w+)\s+n't", r"\1n't", p)
            fixed.append(clean(p))
        if len(fixed) >= 2 and all(ok(p) for p in fixed):
            out.extend(chunk_dialogue(fixed))
    return "\n\n".join(out)


def tinystories(path: Path, limit_bytes: int) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    stories, total = [], 0
    for story in text.split("<|endoftext|>"):
        story = re.sub(r"\n{3,}", "\n\n", story.replace("\r", "")).strip()
        if len(story) < 200 or not ok(story.replace("\n", " ")):
            continue
        stories.append(story)
        total += len(story) + 2
        if total >= limit_bytes:
            break
    return "\n\n".join(stories)


def main() -> None:
    root = Path(__file__).resolve().parent / "external"
    raw = root / "raw"
    jobs = []
    if (raw / "alpaca_cleaned.json").is_file():
        jobs.append(("chat_alpaca.txt", lambda: alpaca(raw / "alpaca_cleaned.json")))
    cb = sorted(raw.glob("chatterbot_*.yml"))
    if cb:
        jobs.append(("chat_chatterbot.txt", lambda: chatterbot(cb)))
    if (raw / "dialogues_text.txt").is_file():
        jobs.append(("chat_dailydialog.txt", lambda: dailydialog(raw / "dialogues_text.txt")))
    if (raw / "TinyStories-valid.txt").is_file():
        jobs.append(("stories_tiny.txt", lambda: tinystories(raw / "TinyStories-valid.txt", 8 * 1024 * 1024)))
    for name, fn in jobs:
        text = fn()
        (root / name).write_text(text + "\n", encoding="utf-8", newline="\n")
        print(f"converted {name}: {len(text.encode()):,} bytes")


if __name__ == "__main__":
    sys.exit(main())
