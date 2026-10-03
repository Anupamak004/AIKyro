from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape


TIMESTAMP = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?"
)
TAG = re.compile(r"<[^>]+>")
BRACKETED = re.compile(r"\[(?:[^\]]{1,80})\]|\((?:[^)]{1,80})\)")


@dataclass(frozen=True)
class Cue:
    start: str
    end: str
    text: str
    index: int


def normalize_timestamp(value: str) -> str:
    return value.replace(",", ".")


def timestamp_seconds(value: str) -> float:
    hours, minutes, rest = normalize_timestamp(value).split(":")
    seconds, millis = rest.split(".")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def clean_dialogue(lines: list[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    text = unescape(TAG.sub("", text))
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\u200b\ufeff")


def parse_srt(content: str) -> list[Cue]:
    """Parse SRT cues while tolerating BOMs, CRLF, comma decimals, and cue settings."""
    content = content.lstrip("\ufeff")
    blocks = re.split(r"\r?\n\s*\r?\n", content.strip())
    cues: list[Cue] = []
    for fallback_index, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        timestamp_line = next((i for i, line in enumerate(lines) if TIMESTAMP.search(line)), None)
        if timestamp_line is None:
            continue
        match = TIMESTAMP.search(lines[timestamp_line])
        assert match is not None
        text = clean_dialogue(lines[timestamp_line + 1 :])
        if not text:
            continue
        cues.append(
            Cue(
                start=normalize_timestamp(match.group("start")),
                end=normalize_timestamp(match.group("end")),
                text=text,
                index=int(lines[0].strip()) if lines[0].strip().isdigit() else fallback_index,
            )
        )
    return cues


def group_cues(cues: list[Cue], max_chars: int = 850, max_seconds: float = 35) -> list[list[Cue]]:
    """Group adjacent cues into dialogue passages without crossing large scene gaps."""
    groups: list[list[Cue]] = []
    current: list[Cue] = []
    chars = 0
    for cue in cues:
        gap = timestamp_seconds(cue.start) - timestamp_seconds(current[-1].end) if current else 0
        span = timestamp_seconds(cue.end) - timestamp_seconds(current[0].start) if current else 0
        if current and (chars + len(cue.text) > max_chars or gap > max_seconds or span > max_seconds):
            groups.append(current)
            current, chars = [], 0
        current.append(cue)
        chars += len(cue.text)
    if current:
        groups.append(current)
    return groups
