"""Offline word suggestion package for NhanDienCuChi.

Provides prefix-based offline dictionary search and contextual word ranking.
"""
from __future__ import annotations

from app.suggestion.context import ParsedContext, SuggestionContext
from app.suggestion.dictionary import OfflineDictionary
from app.suggestion.manager import SuggestionManager
from app.suggestion.qwen_ranker import QwenRanker, RankResult
from app.suggestion.sentence_generator import SentenceGenerator, SentenceResult
from app.suggestion.sentence_validator import SentenceValidationResult, SentenceValidator

__all__ = [
    "OfflineDictionary",
    "ParsedContext",
    "SuggestionContext",
    "QwenRanker",
    "RankResult",
    "SentenceGenerator",
    "SentenceResult",
    "SentenceValidationResult",
    "SentenceValidator",
    "SuggestionManager",
]
