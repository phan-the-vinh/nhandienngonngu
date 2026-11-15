"""Offline Sentence Suggestion Validator.

Validates, cleans, and sanitizes Qwen sentence generation responses.
Guarantees:
- Strict JSON schema validation (requires key 'suggestions' as list).
- Rejection of markdown fences and non-JSON output.
- Rejection of empty sentences, control characters, and sentinels.
- Enforces word count (<= 15 words) and char length (<= 120 chars).
- Case-insensitive deduplication and whitespace normalization.
- Keeps at most 3 valid sentences.
- Completely independent of Qwen / network for offline unit testing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from app.suggestion.context import SuggestionContext


@dataclass(frozen=True)
class SentenceValidationResult:
    """Result of sentence validation."""
    is_valid: bool
    sentences: list[str]
    error: Optional[str] = None


class SentenceValidator:
    """Independent validator and sanitizer for sentence suggestions."""

    # Disallowed sentinels that must never appear in suggestions
    DISALLOWED_SENTINELS = SuggestionContext.DISALLOWED_SENTINELS

    # Regex for control characters (C0/C1, except standard space)
    CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x1f\x7f-\x9f]")

    @classmethod
    def validate(
        cls,
        raw_output: str | dict | Any,
        max_sentences: int = 3,
        max_words: int = 15,
        max_chars: int = 120
    ) -> SentenceValidationResult:
        """Validate raw generation output against strict safety and schema rules.

        Args:
            raw_output: Raw string from model, or already parsed dict.
            max_sentences: Maximum number of suggestions to return (default 3).
            max_words: Maximum word count per sentence (default 15).
            max_chars: Maximum character length per sentence (default 120).

        Returns:
            SentenceValidationResult with is_valid, list of sentences, and optional error.
        """
        if raw_output is None:
            return SentenceValidationResult(is_valid=False, sentences=[], error="EMPTY_OUTPUT")

        if isinstance(raw_output, str):
            trimmed = raw_output.strip()
            if not trimmed:
                return SentenceValidationResult(is_valid=False, sentences=[], error="EMPTY_OUTPUT")

            # Reject markdown fences (e.g. ```json ... ```)
            if "```" in trimmed:
                return SentenceValidationResult(is_valid=False, sentences=[], error="MARKDOWN_FENCE")

            try:
                data = json.loads(trimmed)
            except (json.JSONDecodeError, ValueError) as exc:
                return SentenceValidationResult(
                    is_valid=False,
                    sentences=[],
                    error=f"INVALID_JSON: {exc}"
                )
        elif isinstance(raw_output, dict):
            data = raw_output
        else:
            return SentenceValidationResult(is_valid=False, sentences=[], error="INVALID_SCHEMA")

        if not isinstance(data, dict):
            return SentenceValidationResult(is_valid=False, sentences=[], error="INVALID_SCHEMA")

        if "suggestions" not in data:
            return SentenceValidationResult(is_valid=False, sentences=[], error="MISSING_SUGGESTIONS")

        raw_list = data["suggestions"]
        if not isinstance(raw_list, list):
            return SentenceValidationResult(is_valid=False, sentences=[], error="SUGGESTIONS_NOT_LIST")

        valid_sentences: list[str] = []
        seen_normalized: set[str] = set()

        for item in raw_list:
            if not isinstance(item, str):
                continue

            # Check for control characters
            if cls.CONTROL_CHAR_REGEX.search(item):
                continue

            # Normalize whitespace
            clean = " ".join(item.strip().split())
            if not clean:
                continue

            # Length bounds
            if len(clean) > max_chars:
                continue

            words = clean.split()
            if len(words) > max_words:
                continue

            # Check sentinels (word-level and full sentence)
            words_upper = [w.upper().rstrip(".,!?;:") for w in words]
            if any(w in cls.DISALLOWED_SENTINELS for w in words_upper):
                continue

            # Normalize to uppercase continuation for ASL compatibility
            clean_upper = clean.upper()

            # Case-insensitive deduplication
            if clean_upper in seen_normalized:
                continue
            seen_normalized.add(clean_upper)

            valid_sentences.append(clean_upper)
            if len(valid_sentences) >= max_sentences:
                break

        if not valid_sentences:
            return SentenceValidationResult(is_valid=False, sentences=[], error="NO_VALID_SUGGESTIONS")

        return SentenceValidationResult(
            is_valid=True,
            sentences=valid_sentences[:max_sentences],
            error=None
        )

    @classmethod
    def clean_and_filter(cls, raw_output: str | dict | Any) -> list[str]:
        """Convenience method returning valid list of sentences, or empty list."""
        result = cls.validate(raw_output)
        return result.sentences if result.is_valid else []
