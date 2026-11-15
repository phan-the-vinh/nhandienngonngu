"""Offline English dictionary with prefix search and frequency ranking.

Independent module: zero dependencies on camera, MediaPipe, or external packages.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


class _TrieNode:
    """Internal node for the prefix trie."""
    __slots__ = ("children", "is_word", "frequency", "word")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.is_word: bool = False
        self.frequency: int = 0
        self.word: str | None = None


class OfflineDictionary:
    """Prefix-based in-memory English dictionary with frequency prioritization.

    Guarantees:
    - Case-insensitive search.
    - Only returns words starting strictly with the normalized prefix.
    - Results sorted by frequency descending; ties broken alphabetically for stability.
    - Empty or whitespace prefix returns empty list [].
    - Zero external dependencies.
    """

    def __init__(self) -> None:
        self._root = _TrieNode()
        self._words: dict[str, int] = {}

    @classmethod
    def load(cls, path: str | Path) -> OfflineDictionary:
        """Factory method to construct and populate an OfflineDictionary from file."""
        instance = cls()
        instance.load_file(path)
        return instance

    def load_file(self, path: str | Path) -> int:
        """Load word entries from a file.

        Supported line formats:
        1. <WORD><tab><FREQUENCY>
        2. <WORD>,<FREQUENCY>
        3. <WORD> <FREQUENCY>
        4. <WORD> (defaults to frequency 1)

        Returns:
            Number of unique words successfully loaded.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Dictionary file not found: {file_path}")

        loaded_count = 0
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                word, freq = self._parse_line(line)
                if word:
                    self.add_word(word, freq)
                    loaded_count += 1

        return len(self._words)

    @staticmethod
    def _parse_line(line: str) -> tuple[str | None, int]:
        """Parse a dictionary line into (word, frequency)."""
        delimiters = ["\t", ",", " "]
        parts: list[str] = []
        for delim in delimiters:
            if delim in line:
                parts = line.split(delim, 1)
                break

        if not parts:
            parts = [line]

        raw_word = parts[0].strip()
        if not raw_word or not raw_word.isalpha():
            return None, 0

        frequency = 1
        if len(parts) > 1:
            raw_freq = parts[1].strip()
            if raw_freq:
                try:
                    frequency = int(raw_freq)
                    if frequency < 0:
                        frequency = 0
                except ValueError:
                    # Malformed frequency: fall back to default frequency 1
                    frequency = 1

        return raw_word.upper(), frequency

    def add_word(self, word: str, frequency: int = 1) -> None:
        """Insert or update a word with its frequency."""
        clean = word.strip().upper()
        if not clean or not clean.isalpha():
            return

        frequency = max(0, int(frequency))
        self._words[clean] = frequency

        curr = self._root
        for char in clean:
            if char not in curr.children:
                curr.children[char] = _TrieNode()
            curr = curr.children[char]

        curr.is_word = True
        curr.frequency = frequency
        curr.word = clean

    def search(self, prefix: str, limit: int = 5) -> list[str]:
        """Search for words matching the prefix exactly, ranked by frequency.

        Args:
            prefix: The prefix string to search for (case-insensitive).
            limit: Maximum number of candidate words to return (default: 5).

        Returns:
            List of unique words starting strictly with prefix, sorted by
            frequency (descending), then alphabetically for tie-breaking.
        """
        if limit <= 0:
            return []

        clean_prefix = prefix.strip().upper() if prefix else ""
        if not clean_prefix:
            return []

        # 1. Walk Trie to prefix terminal node
        curr = self._root
        for char in clean_prefix:
            if char not in curr.children:
                return []
            curr = curr.children[char]

        # 2. Gather all words in the subtree under curr
        candidates: list[tuple[int, str]] = []
        stack = [curr]
        while stack:
            node = stack.pop()
            if node.is_word and node.word is not None:
                candidates.append((node.frequency, node.word))
            for child in node.children.values():
                stack.append(child)

        # 3. Sort: primary key -frequency (descending), secondary key word (ascending)
        candidates.sort(key=lambda item: (-item[0], item[1]))

        # 4. Return top `limit` words
        return [word for _, word in candidates[:limit]]

    def contains(self, word: str) -> bool:
        """Check if a word exists in the dictionary (case-insensitive)."""
        clean = word.strip().upper() if word else ""
        if not clean:
            return False
        return clean in self._words

    def get_frequency(self, word: str) -> int:
        """Get the frequency of a word. Returns 0 if word not found."""
        clean = word.strip().upper() if word else ""
        if not clean:
            return 0
        return self._words.get(clean, 0)

    def clear(self) -> None:
        """Clear all entries from the dictionary."""
        self._root = _TrieNode()
        self._words.clear()

    def __len__(self) -> int:
        """Return the number of unique words in the dictionary."""
        return len(self._words)
