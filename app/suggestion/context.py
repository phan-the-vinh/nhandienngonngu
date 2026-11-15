"""Suggestion Context Parser.

Analyzes text to extract the current active prefix and committed context
for offline dictionary search and contextual ranking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class ParsedContext:
    """Represents the parsed state of a typing session."""
    raw_text: str
    committed_text: str
    current_prefix: str
    last_committed_word: str
    recent_words: list[str]

    @property
    def has_prefix(self) -> bool:
        """Returns True if there is an active non-empty prefix."""
        return bool(self.current_prefix)

    @property
    def has_context(self) -> bool:
        """Returns True if there is at least one committed word preceding the prefix."""
        return bool(self.committed_text)


class SuggestionContext:
    """Static utility and parser for typing context and prefix segmentation."""

    # Sentinel words that must never be treated as valid text/prefix
    DISALLOWED_SENTINELS = frozenset({"UNCERTAIN", "NONE", "IDLE", "PAUSED"})

    @classmethod
    def parse(cls, text: str | None, max_recent_words: int = 5) -> ParsedContext:
        """Parse raw text into committed context and active prefix.

        Rules:
        1. If text is None or empty: prefix="", committed="", recent=[].
        2. If text ends with a space: all words are committed, prefix="".
        3. If text does not end with a space: the trailing segment is current_prefix,
           and any preceding text before the last space is committed_text.
        4. Current prefix is normalized to uppercase for dictionary lookup.
        5. Space is never part of the prefix.
        6. Disallowed sentinel values (e.g. UNCERTAIN) are excluded from prefix.
        7. Multiple consecutive spaces are treated as word separators.
        """
        if text is None:
            raw = ""
        else:
            raw = str(text)

        if not raw:
            return ParsedContext(
                raw_text="",
                committed_text="",
                current_prefix="",
                last_committed_word="",
                recent_words=[]
            )

        # Detect trailing space: prefix is empty, all tokens are committed
        if raw.endswith(" "):
            # All tokens before trailing spaces are committed
            clean_committed = raw.rstrip(" ")
            words = [w.upper() for w in clean_committed.split() if w]
            # Filter sentinels
            words = [w for w in words if w not in cls.DISALLOWED_SENTINELS]
            committed_str = " ".join(words)
            recent = words[-max_recent_words:] if words else []
            last_word = recent[-1] if recent else ""

            return ParsedContext(
                raw_text=raw,
                committed_text=committed_str,
                current_prefix="",
                last_committed_word=last_word,
                recent_words=recent
            )

        # Text does not end with space: split at last space
        last_space_idx = raw.rfind(" ")
        if last_space_idx == -1:
            # No space in text: entire text is current_prefix
            raw_prefix = raw.strip()
            clean_prefix = cls._clean_prefix(raw_prefix)
            return ParsedContext(
                raw_text=raw,
                committed_text="",
                current_prefix=clean_prefix,
                last_committed_word="",
                recent_words=[]
            )

        # Has at least one space
        committed_part = raw[:last_space_idx].strip()
        raw_prefix = raw[last_space_idx + 1:]
        clean_prefix = cls._clean_prefix(raw_prefix)

        words = [w.upper() for w in committed_part.split() if w]
        words = [w for w in words if w not in cls.DISALLOWED_SENTINELS]
        committed_str = " ".join(words)
        recent = words[-max_recent_words:] if words else []
        last_word = recent[-1] if recent else ""

        return ParsedContext(
            raw_text=raw,
            committed_text=committed_str,
            current_prefix=clean_prefix,
            last_committed_word=last_word,
            recent_words=recent
        )

    @classmethod
    def _clean_prefix(cls, raw_prefix: str) -> str:
        """Sanitize prefix: uppercase, letters only, no sentinels."""
        pfx = raw_prefix.strip().upper()
        if pfx in cls.DISALLOWED_SENTINELS:
            return ""
        # Keep only alphabetic characters in prefix for English dictionary search
        letters_only = re.sub(r"[^A-Z]", "", pfx)
        return letters_only
