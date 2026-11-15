"""Suggestion Manager.

Coordinates offline word and sentence suggestions using a shared worker architecture:
- Tier 1: Instant prefix search on OfflineDictionary (< 1ms).
- Tier 2: Background contextual re-ranking via QwenRanker (WORD_RANK).
- Tier 3: Background sentence continuation generation via SentenceGenerator (SENTENCE_GEN).

Guarantees:
- Strictly ONE background worker thread (shared by both word and sentence tasks).
- Exactly ONE LLM inference at any given time (zero concurrent Qwen requests).
- Shared generation token and latest-only scheduling.
- Word Suggestion takes 100% priority when user is spelling (non-empty prefix).
- Sentence Suggestion only triggers on word boundaries with committed context.
- Robust fallback to Tier 1 Trie ranking when Ollama is offline.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from app.suggestion.context import ParsedContext, SuggestionContext
from app.suggestion.dictionary import OfflineDictionary
from app.suggestion.qwen_ranker import QwenRanker, RankResult
from app.suggestion.sentence_generator import SentenceGenerator, SentenceResult


@dataclass
class _SuggestionTask:
    """Internal representation of a pending LLM task."""
    task_type: str  # "WORD_RANK" or "SENTENCE_GEN"
    gen_id: int
    context: str
    prefix: str
    candidates: list[str]
    created_at: float


class SuggestionManager:
    """Orchestrates dictionary prefix suggestions, Qwen word re-ranking, and sentence generation."""

    def __init__(
        self,
        dictionary: OfflineDictionary,
        ranker: Optional[QwenRanker] = None,
        sentence_generator: Optional[SentenceGenerator] = None,
        debounce_ms: float = 250.0,
        on_update_callback: Optional[Callable[[List[str]], None]] = None,
        on_sentence_callback: Optional[Callable[[List[str]], None]] = None,
    ) -> None:
        """Initialize the manager.

        Args:
            dictionary: Initialized OfflineDictionary instance.
            ranker: Optional QwenRanker instance for Tier 2 contextual word re-ranking.
            sentence_generator: Optional SentenceGenerator for Tier 3 sentence continuation.
            debounce_ms: Debounce delay in milliseconds before dispatching LLM request.
            on_update_callback: Optional callback invoked when word re-ranking completes.
            on_sentence_callback: Optional callback invoked when sentence generation completes.
        """
        self.dictionary = dictionary
        self.ranker = ranker
        self.sentence_generator = sentence_generator
        self.debounce_s = max(0.0, debounce_ms / 1000.0)
        self.on_update_callback = on_update_callback
        self.on_sentence_callback = on_sentence_callback

        self.current_generation: int = 0
        self.current_prefix: str = ""
        self.current_context: str = ""
        self.current_suggestions: list[str] = []
        self.current_sentence_suggestions: list[str] = []
        self.current_sentence_context: str = ""
        self.current_sentence_generation: int = 0
        self.mode: str = "ABC"

        # Concurrency & Scheduling controls
        self._lock = threading.Lock()
        self._running: bool = True
        self._wake_event = threading.Event()
        # Holds AT MOST ONE pending request (latest-only)
        self._pending_task: Optional[_SuggestionTask] = None

        # Metrics for benchmark and diagnostic reporting
        self.metric_requests_dispatched: int = 0
        self.metric_stale_dropped: int = 0
        self.metric_fallbacks: int = 0
        self.metric_last_latency_ms: float = 0.0

        self.metric_sentence_requests_dispatched: int = 0
        self.metric_sentence_fallbacks: int = 0
        self.metric_last_sentence_latency_ms: float = 0.0

        # Dedicated background worker thread (strictly 1 shared worker)
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="SuggestionManagerWorker",
            daemon=True
        )
        self._worker_thread.start()

    def set_mode(self, mode: str) -> None:
        """Update active mode (e.g. 'ABC', '123', 'JZ_PAUSED')."""
        with self._lock:
            self.mode = mode
            if mode != "ABC":
                self.current_suggestions = []
                self.current_sentence_suggestions = []
                self._pending_task = None
                self.current_generation += 1

    def update_text(self, text: str | None) -> list[str]:
        """Update suggestions based on full text string.

        Returns:
            Current active word suggestions (Tier 1 immediately).
        """
        parsed = SuggestionContext.parse(text)
        return self.update_parsed(parsed)

    def update_parsed(self, parsed: ParsedContext) -> list[str]:
        """Update suggestions given a pre-parsed context."""
        with self._lock:
            self.current_generation += 1
            gen_id = self.current_generation
            self.current_prefix = parsed.current_prefix
            self.current_context = parsed.committed_text

            # If mode is not ABC: clear all suggestions
            if self.mode != "ABC":
                self.current_suggestions = []
                self.current_sentence_suggestions = []
                self._pending_task = None
                return []

            # Priority 1: User is currently typing a word (has non-empty prefix)
            if parsed.has_prefix:
                # Word suggestion takes 100% priority; clear sentence suggestions
                self.current_sentence_suggestions = []

                candidates = self.dictionary.search(parsed.current_prefix, limit=15)
                if not candidates:
                    self.current_suggestions = []
                    self._pending_task = None
                    return []

                self.current_suggestions = candidates[:5]

                if self.ranker is not None:
                    # Latest-only: schedule background WORD_RANK task
                    self._pending_task = _SuggestionTask(
                        task_type="WORD_RANK",
                        gen_id=gen_id,
                        context=parsed.committed_text,
                        prefix=parsed.current_prefix,
                        candidates=candidates,
                        created_at=time.monotonic(),
                    )
                    self._wake_event.set()
                else:
                    self._pending_task = None

                return list(self.current_suggestions)

            # Priority 2: Prefix is empty (word boundary, e.g. after SPACE or commit_word)
            self.current_suggestions = []

            # Only trigger sentence generation when there is non-empty committed context
            if parsed.has_context and self.sentence_generator is not None:
                # Check if current sentence suggestions already match this exact context
                if (
                    self.current_sentence_context == parsed.committed_text
                    and self.current_sentence_suggestions
                ):
                    # Already valid and current
                    self._pending_task = None
                else:
                    # Schedule background SENTENCE_GEN task
                    self._pending_task = _SuggestionTask(
                        task_type="SENTENCE_GEN",
                        gen_id=gen_id,
                        context=parsed.committed_text,
                        prefix="",
                        candidates=[],
                        created_at=time.monotonic(),
                    )
                    self._wake_event.set()
            else:
                self.current_sentence_suggestions = []
                self._pending_task = None

            return []

    def get_suggestions(self) -> list[str]:
        """Retrieve current word suggestions thread-safely."""
        with self._lock:
            return list(self.current_suggestions)

    def get_sentence_suggestions(self) -> list[str]:
        """Retrieve current sentence suggestions thread-safely."""
        with self._lock:
            return list(self.current_sentence_suggestions)

    def select_suggestion(
        self,
        index_1_based: int,
        composer: Any = None
    ) -> Optional[str]:
        """Select a word suggestion by 1-based index (1 to 5).

        If a composer object with commit_word() is provided, the word is
        committed to the composer and suggestions are updated.

        Returns:
            The selected word, or None if index was invalid.
        """
        with self._lock:
            if not (1 <= index_1_based <= len(self.current_suggestions)):
                return None
            chosen_word = self.current_suggestions[index_1_based - 1]
            valid_list = list(self.current_suggestions)

        if composer is not None and hasattr(composer, "commit_word"):
            try:
                composer.commit_word(chosen_word, valid_suggestions=valid_list)
            except TypeError:
                composer.commit_word(chosen_word)
            if hasattr(composer, "text"):
                self.update_text(composer.text)

        return chosen_word

    def select_sentence_suggestion(
        self,
        index_1_based: int,
        composer: Any = None
    ) -> Optional[str]:
        """Select a sentence suggestion by 1-based index (1 to 3).

        If a composer object with commit_sentence() is provided, the sentence continuation
        is committed to the composer and suggestions are updated.

        Returns:
            The selected sentence continuation, or None if index was invalid.
        """
        with self._lock:
            if not (1 <= index_1_based <= len(self.current_sentence_suggestions)):
                return None
            chosen_sentence = self.current_sentence_suggestions[index_1_based - 1]
            valid_list = list(self.current_sentence_suggestions)
            expected_ctx = self.current_sentence_context

        if composer is not None and hasattr(composer, "commit_sentence"):
            try:
                ok = composer.commit_sentence(
                    chosen_sentence,
                    valid_suggestions=valid_list,
                    expected_context=expected_ctx
                )
            except TypeError:
                try:
                    ok = composer.commit_sentence(chosen_sentence, valid_suggestions=valid_list)
                except TypeError:
                    ok = composer.commit_sentence(chosen_sentence)
            if ok and hasattr(composer, "text"):
                self.update_text(composer.text)

        return chosen_sentence

    def _worker_loop(self) -> None:
        """Dedicated background loop with debounce and latest-only scheduling.

        Sequentially handles both WORD_RANK and SENTENCE_GEN tasks.
        Guarantees that at most one inference query is executed at any given time.
        """
        while self._running:
            task: Optional[_SuggestionTask] = None

            with self._lock:
                if self._pending_task is not None:
                    elapsed = time.monotonic() - self._pending_task.created_at
                    if elapsed >= self.debounce_s:
                        task = self._pending_task
                        self._pending_task = None
                        self._wake_event.clear()

            if task is None:
                self._wake_event.wait(timeout=0.05)
                continue

            # Pre-invocation check: is request already stale before network call?
            with self._lock:
                if not self._running or task.gen_id != self.current_generation:
                    self.metric_stale_dropped += 1
                    continue

            if task.task_type == "WORD_RANK":
                self.metric_requests_dispatched += 1
                rank_result = self.ranker.rank(task.context, task.prefix, task.candidates)
                self.metric_last_latency_ms = rank_result.latency_ms

                # Post-invocation check: is request stale after network call?
                with self._lock:
                    if not self._running or task.gen_id != self.current_generation:
                        self.metric_stale_dropped += 1
                        continue

                    if rank_result.success and rank_result.ranked_words:
                        self.current_suggestions = rank_result.ranked_words[:5]
                        updated_words = list(self.current_suggestions)
                    else:
                        self.metric_fallbacks += 1
                        updated_words = None

                if updated_words is not None and self.on_update_callback is not None:
                    try:
                        self.on_update_callback(updated_words)
                    except Exception:
                        pass

            elif task.task_type == "SENTENCE_GEN":
                self.metric_sentence_requests_dispatched += 1
                sentence_result = self.sentence_generator.generate(task.context)
                self.metric_last_sentence_latency_ms = sentence_result.latency_ms

                # Post-invocation check: is request stale after network call?
                with self._lock:
                    if not self._running or task.gen_id != self.current_generation:
                        self.metric_stale_dropped += 1
                        continue

                    if sentence_result.success and sentence_result.suggestions:
                        self.current_sentence_suggestions = sentence_result.suggestions[:3]
                        self.current_sentence_generation = task.gen_id
                        self.current_sentence_context = task.context
                        updated_sentences = list(self.current_sentence_suggestions)
                    else:
                        self.metric_sentence_fallbacks += 1
                        updated_sentences = None

                if updated_sentences is not None and self.on_sentence_callback is not None:
                    try:
                        self.on_sentence_callback(updated_sentences)
                    except Exception:
                        pass

    def close(self, timeout_s: float = 0.5) -> None:
        """Shutdown the manager and background worker without blocking camera."""
        with self._lock:
            self._running = False
            self._pending_task = None
            self._wake_event.set()

        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout_s)

    def shutdown(self, timeout_s: float = 0.5) -> None:
        """Alias for close()."""
        self.close(timeout_s=timeout_s)
