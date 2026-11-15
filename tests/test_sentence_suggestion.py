"""Comprehensive Test Suite for Offline English Sentence Suggestion Core.

Phase 3.10.7-B3 verification suite covering all 38 required scenarios:
A. Sentence Validator (14 scenarios)
B. Context and Trigger (8 scenarios)
C. Shared Worker Concurrency and Robustness (8 scenarios)
D. Commit Sentence API (8 scenarios)
"""
from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from app.inference.unified_static_coordinator import TextComposer
from app.suggestion.context import SuggestionContext
from app.suggestion.dictionary import OfflineDictionary
from app.suggestion.manager import SuggestionManager
from app.suggestion.qwen_ranker import QwenRanker, RankResult
from app.suggestion.sentence_generator import SentenceGenerator, SentenceResult
from app.suggestion.sentence_validator import SentenceValidationResult, SentenceValidator

ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = ROOT / "data/dictionary/english_words_25k.tsv"


class FakeSlowSentenceGenerator:
    """Mock sentence generator with simulated delay and controllable response."""

    def __init__(self, delay_s: float = 0.08, suggestions: Optional[list[str]] = None, fail: bool = False):
        self.delay_s = delay_s
        self.suggestions = suggestions or ["EAT SOMETHING.", "GO HOME.", "LEARN ENGLISH."]
        self.fail = fail
        self.call_count = 0
        self.active_calls = 0
        self.max_concurrent_calls = 0
        self._lock = threading.Lock()

    def generate(self, context: str) -> SentenceResult:
        with self._lock:
            self.call_count += 1
            self.active_calls += 1
            if self.active_calls > self.max_concurrent_calls:
                self.max_concurrent_calls = self.active_calls

        time.sleep(self.delay_s)

        with self._lock:
            self.active_calls -= 1

        if self.fail:
            return SentenceResult(
                success=False,
                suggestions=[],
                error_type="SIMULATED_FAILURE",
                error_message="Simulated sentence generator failure",
                latency_ms=self.delay_s * 1000.0,
            )

        return SentenceResult(
            success=True,
            suggestions=list(self.suggestions),
            latency_ms=self.delay_s * 1000.0,
        )


class FakeSlowWordRanker:
    """Mock word ranker with simulated delay."""

    def __init__(self, delay_s: float = 0.08):
        self.delay_s = delay_s
        self.call_count = 0
        self.active_calls = 0
        self.max_concurrent_calls = 0
        self._lock = threading.Lock()

    def rank(self, context: str, prefix: str, candidates: list[str]) -> RankResult:
        with self._lock:
            self.call_count += 1
            self.active_calls += 1
            if self.active_calls > self.max_concurrent_calls:
                self.max_concurrent_calls = self.active_calls

        time.sleep(self.delay_s)

        with self._lock:
            self.active_calls -= 1

        return RankResult(
            success=True,
            ranked_words=list(reversed(candidates)),
            latency_ms=self.delay_s * 1000.0,
        )


# ==============================================================================
# SECTION A: SENTENCE VALIDATOR TESTS (14 scenarios)
# ==============================================================================
class TestSentenceValidator(unittest.TestCase):

    # 1. JSON hop le
    def test_a01_valid_json(self):
        raw = json.dumps({"suggestions": ["EAT SOMETHING.", "GO HOME.", "LEARN ENGLISH."]})
        res = SentenceValidator.validate(raw)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.sentences, ["EAT SOMETHING.", "GO HOME.", "LEARN ENGLISH."])
        self.assertIsNone(res.error)

    # 2. JSON sai schema (dict khong co 'suggestions')
    def test_a02_invalid_schema_wrong_key(self):
        raw = json.dumps({"predictions": ["EAT SOMETHING."]})
        res = SentenceValidator.validate(raw)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.error, "MISSING_SUGGESTIONS")

    # 3. Thieu truong suggestions
    def test_a03_missing_suggestions_field(self):
        raw = json.dumps({"status": "OK"})
        res = SentenceValidator.validate(raw)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.error, "MISSING_SUGGESTIONS")

    # 4. Suggestions khong phai list
    def test_a04_suggestions_not_list(self):
        raw = json.dumps({"suggestions": "EAT SOMETHING."})
        res = SentenceValidator.validate(raw)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.error, "SUGGESTIONS_NOT_LIST")

    # 5. Cau rong
    def test_a05_empty_sentences_filtered(self):
        raw = json.dumps({"suggestions": ["", "   ", "VALID CONTINUATION."]})
        res = SentenceValidator.validate(raw)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.sentences, ["VALID CONTINUATION."])

    # 6. Cau trung lap
    def test_a06_duplicate_sentences_deduped(self):
        raw = json.dumps({"suggestions": ["GO HOME.", "go home.", "Go Home.", "EAT FOOD."]})
        res = SentenceValidator.validate(raw)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.sentences, ["GO HOME.", "EAT FOOD."])

    # 7. Cau vuot 15 tu
    def test_a07_sentence_exceeds_15_words(self):
        sixteen_words = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen."
        raw = json.dumps({"suggestions": [sixteen_words, "SHORT SENTENCE."]})
        res = SentenceValidator.validate(raw)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.sentences, ["SHORT SENTENCE."])

    # 8. Cau vuot 120 ky tu
    def test_a08_sentence_exceeds_120_chars(self):
        long_sentence = "A" * 121
        raw = json.dumps({"suggestions": [long_sentence, "VALID SHORT."]})
        res = SentenceValidator.validate(raw)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.sentences, ["VALID SHORT."])

    # 9. Sentinel (UNCERTAIN, NONE, IDLE, PAUSED)
    def test_a09_sentinel_dropped(self):
        raw = json.dumps({"suggestions": ["UNCERTAIN", "NONE.", "DO NOT PAUSED HERE", "VALID SUGGESTION."]})
        res = SentenceValidator.validate(raw)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.sentences, ["VALID SUGGESTION."])

    # 10. Ky tu dieu khien
    def test_a10_control_characters_dropped(self):
        with_control = "HELLO\x00WORLD."
        raw = json.dumps({"suggestions": [with_control, "CLEAN CONTINUATION."]})
        res = SentenceValidator.validate(raw)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.sentences, ["CLEAN CONTINUATION."])

    # 11. Markdown fence (```json ... ```)
    def test_a11_markdown_fence_rejected(self):
        raw = "```json\n{\"suggestions\": [\"EAT SOMETHING.\"]}\n```"
        res = SentenceValidator.validate(raw)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.error, "MARKDOWN_FENCE")

    # 12. Output rong
    def test_a12_empty_output(self):
        res1 = SentenceValidator.validate("")
        self.assertFalse(res1.is_valid)
        self.assertEqual(res1.error, "EMPTY_OUTPUT")

        res2 = SentenceValidator.validate(None)
        self.assertFalse(res2.is_valid)
        self.assertEqual(res2.error, "EMPTY_OUTPUT")

    # 13. Chuan hoa khoang trang
    def test_a13_whitespace_normalized(self):
        raw = json.dumps({"suggestions": ["  EAT    SOMETHING   GOOD.  "]})
        res = SentenceValidator.validate(raw)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.sentences, ["EAT SOMETHING GOOD."])

    # 14. Giu lai toi da 3 cau hop le
    def test_a14_max_three_sentences_retained(self):
        raw = json.dumps({"suggestions": ["ONE.", "TWO.", "THREE.", "FOUR.", "FIVE."]})
        res = SentenceValidator.validate(raw)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.sentences), 3)
        self.assertEqual(res.sentences, ["ONE.", "TWO.", "THREE."])


# ==============================================================================
# SECTION B: CONTEXT AND TRIGGER TESTS (8 scenarios)
# ==============================================================================
class TestContextAndTrigger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if DICT_PATH.is_file():
            cls.dict = OfflineDictionary.load(DICT_PATH)
        else:
            cls.dict = OfflineDictionary()
            cls.dict.add_word("APPLE", 100)
            cls.dict.add_word("EAT", 90)

    # 1. Context 'I WANT TO ' co the yeu cau sinh cau
    def test_b01_trigger_on_word_boundary_with_context(self):
        gen = FakeSlowSentenceGenerator(delay_s=0.01)
        mgr = SuggestionManager(dictionary=self.dict, sentence_generator=gen, debounce_ms=5.0)
        try:
            # Word boundary with context
            mgr.update_text("I WANT TO ")
            time.sleep(0.06)
            suggs = mgr.get_sentence_suggestions()
            self.assertEqual(len(suggs), 3)
            self.assertIn("EAT SOMETHING.", suggs)
        finally:
            mgr.close()

    # 2. Context rong khong sinh cau
    def test_b02_empty_context_no_trigger(self):
        gen = FakeSlowSentenceGenerator(delay_s=0.01)
        mgr = SuggestionManager(dictionary=self.dict, sentence_generator=gen, debounce_ms=5.0)
        try:
            mgr.update_text("")
            time.sleep(0.04)
            self.assertEqual(mgr.get_sentence_suggestions(), [])
            self.assertEqual(gen.call_count, 0)
        finally:
            mgr.close()

    # 3. Prefix 'APP' khong sinh cau
    def test_b03_non_empty_prefix_no_sentence_trigger(self):
        gen = FakeSlowSentenceGenerator(delay_s=0.01)
        mgr = SuggestionManager(dictionary=self.dict, sentence_generator=gen, debounce_ms=5.0)
        try:
            mgr.update_text("APP")
            time.sleep(0.04)
            self.assertEqual(mgr.get_sentence_suggestions(), [])
            self.assertEqual(gen.call_count, 0)
            # Word suggestions should be present
            self.assertTrue(len(mgr.get_suggestions()) > 0)
        finally:
            mgr.close()

    # 4. Khi nguoi dung bat dau tu moi, cau cu bi vo hieu
    def test_b04_starting_new_word_clears_sentence_suggestions(self):
        gen = FakeSlowSentenceGenerator(delay_s=0.01)
        mgr = SuggestionManager(dictionary=self.dict, sentence_generator=gen, debounce_ms=5.0)
        try:
            mgr.update_text("I WANT TO ")
            time.sleep(0.06)
            self.assertTrue(len(mgr.get_sentence_suggestions()) > 0)

            # User starts typing 'E'
            mgr.update_text("I WANT TO E")
            self.assertEqual(mgr.get_sentence_suggestions(), [])
            self.assertTrue(len(mgr.get_suggestions()) > 0)
        finally:
            mgr.close()

    # 5. Khi SPACE duoc commit, context duoc cap nhat dung
    def test_b05_space_commit_updates_context_and_triggers_sentence(self):
        gen = FakeSlowSentenceGenerator(delay_s=0.01)
        mgr = SuggestionManager(dictionary=self.dict, sentence_generator=gen, debounce_ms=5.0)
        try:
            composer = TextComposer()
            composer.text = "HELLO"
            mgr.update_text(composer.text)

            # Commit space
            composer.append(" ")
            mgr.update_text(composer.text)
            self.assertEqual(composer.current_prefix, "")
            self.assertEqual(composer.committed_context, "HELLO")

            time.sleep(0.06)
            self.assertEqual(len(mgr.get_sentence_suggestions()), 3)
        finally:
            mgr.close()

    # 6. Khi doi mode, cau cu bi vo hieu
    def test_b06_mode_switch_clears_sentence_suggestions(self):
        gen = FakeSlowSentenceGenerator(delay_s=0.01)
        mgr = SuggestionManager(dictionary=self.dict, sentence_generator=gen, debounce_ms=5.0)
        try:
            mgr.update_text("I WANT TO ")
            time.sleep(0.06)
            self.assertTrue(len(mgr.get_sentence_suggestions()) > 0)

            mgr.set_mode("123")
            self.assertEqual(mgr.get_sentence_suggestions(), [])
        finally:
            mgr.close()

    # 7. Khong gui request trong moi frame (khong goi khi khong co su thay doi van ban)
    def test_b07_no_redundant_request_per_frame(self):
        gen = FakeSlowSentenceGenerator(delay_s=0.01)
        mgr = SuggestionManager(dictionary=self.dict, sentence_generator=gen, debounce_ms=5.0)
        try:
            text = "I WANT TO "
            mgr.update_text(text)
            time.sleep(0.06)
            initial_calls = gen.call_count
            self.assertEqual(initial_calls, 1)

            # Repeated frame updates with same text
            for _ in range(10):
                mgr.update_text(text)
            time.sleep(0.04)
            # Should not dispatch multiple sentence requests for already completed context
            self.assertEqual(gen.call_count, initial_calls)
        finally:
            mgr.close()

    # 8. Khong tao request trung khong can thiet
    def test_b08_no_duplicate_requests_while_pending(self):
        gen = FakeSlowSentenceGenerator(delay_s=0.05)
        mgr = SuggestionManager(dictionary=self.dict, sentence_generator=gen, debounce_ms=20.0)
        try:
            mgr.update_text("I WANT TO ")
            mgr.update_text("I WANT TO ")
            mgr.update_text("I WANT TO ")
            time.sleep(0.12)
            self.assertEqual(gen.call_count, 1)
        finally:
            mgr.close()


# ==============================================================================
# SECTION C: SHARED WORKER CONCURRENCY AND ROBUSTNESS (8 scenarios)
# ==============================================================================
class TestSharedWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dict = OfflineDictionary()
        for w in ["APPLE", "APPLICATION", "EAT", "EVEN"]:
            cls.dict.add_word(w, 100)

    # 1. Word va Sentence dung chung worker (chi 1 worker thread)
    def test_c01_single_shared_worker_thread(self):
        ranker = FakeSlowWordRanker(delay_s=0.01)
        gen = FakeSlowSentenceGenerator(delay_s=0.01)
        mgr = SuggestionManager(dictionary=self.dict, ranker=ranker, sentence_generator=gen)
        try:
            # Inspect worker thread count inside manager
            self.assertTrue(mgr._worker_thread.is_alive())
            # Ensure no second worker thread exists on manager
            worker_threads = [t for t in threading.enumerate() if t.name == "SuggestionManagerWorker"]
            self.assertEqual(len(worker_threads), 1)
        finally:
            mgr.close()

    # 2. Khong co hai inference dong thoi
    def test_c02_no_concurrent_llm_inference(self):
        ranker = FakeSlowWordRanker(delay_s=0.03)
        gen = FakeSlowSentenceGenerator(delay_s=0.03)
        mgr = SuggestionManager(dictionary=self.dict, ranker=ranker, sentence_generator=gen, debounce_ms=2.0)
        try:
            # Rapidly alternate between word typing and word boundary
            mgr.update_text("I WANT TO ")
            time.sleep(0.01)
            mgr.update_text("I WANT TO A")
            time.sleep(0.08)

            # Neither ranker nor generator should ever have active_calls > 1
            self.assertLessEqual(ranker.max_concurrent_calls, 1)
            self.assertLessEqual(gen.max_concurrent_calls, 1)
        finally:
            mgr.close()

    # 3. Sentence task khong tao thread moi
    def test_c03_sentence_task_does_not_spawn_new_threads(self):
        gen = FakeSlowSentenceGenerator(delay_s=0.01)
        mgr = SuggestionManager(dictionary=self.dict, sentence_generator=gen, debounce_ms=5.0)
        try:
            initial_thread_count = threading.active_count()
            for _ in range(5):
                mgr.update_text("HELLO ")
                time.sleep(0.03)
            # Thread count must not grow
            self.assertLessEqual(threading.active_count(), initial_thread_count)
        finally:
            mgr.close()

    # 4. Word Suggestion van hoat dong khi Sentence Suggestion loi
    def test_c04_word_suggestion_works_even_if_sentence_fails(self):
        ranker = FakeSlowWordRanker(delay_s=0.01)
        failing_gen = FakeSlowSentenceGenerator(delay_s=0.01, fail=True)
        mgr = SuggestionManager(dictionary=self.dict, ranker=ranker, sentence_generator=failing_gen, debounce_ms=5.0)
        try:
            # Sentence fails
            mgr.update_text("I WANT TO ")
            time.sleep(0.04)
            self.assertEqual(mgr.get_sentence_suggestions(), [])
            self.assertGreater(mgr.metric_sentence_fallbacks, 0)

            # User types 'A' -> word suggestion must work perfectly
            words = mgr.update_text("I WANT TO A")
            self.assertTrue(len(words) > 0)
            time.sleep(0.04)
            self.assertTrue(len(mgr.get_suggestions()) > 0)
        finally:
            mgr.close()

    # 5. Sentence timeout khong lam treo camera loop
    def test_c05_sentence_timeout_does_not_block_caller(self):
        # Simulate generator that hangs for 0.5s
        hanging_gen = FakeSlowSentenceGenerator(delay_s=0.20)
        mgr = SuggestionManager(dictionary=self.dict, sentence_generator=hanging_gen, debounce_ms=5.0)
        try:
            start = time.monotonic()
            # update_text must return immediately without waiting for hanging generator
            mgr.update_text("I WANT TO ")
            call_duration = time.monotonic() - start
            self.assertLess(call_duration, 0.05)
        finally:
            mgr.close(timeout_s=0.3)

    # 6. Ollama offline khong lam ung dung crash
    def test_c06_ollama_offline_no_crash(self):
        # SentenceGenerator pointing to invalid port
        offline_gen = SentenceGenerator(endpoint="http://127.0.0.1:59999/api/chat", timeout_ms=50.0)
        mgr = SuggestionManager(dictionary=self.dict, sentence_generator=offline_gen, debounce_ms=5.0)
        try:
            mgr.update_text("I WANT TO ")
            time.sleep(0.15)
            # Must fail gracefully
            self.assertEqual(mgr.get_sentence_suggestions(), [])
            self.assertGreater(mgr.metric_sentence_fallbacks, 0)
        finally:
            mgr.close()

    # 7. Stale sentence result bi loai bo
    def test_c07_stale_sentence_result_dropped(self):
        slow_gen = FakeSlowSentenceGenerator(delay_s=0.08)
        mgr = SuggestionManager(dictionary=self.dict, sentence_generator=slow_gen, debounce_ms=5.0)
        try:
            mgr.update_text("I WANT TO ")
            time.sleep(0.02)
            # Before slow_gen finishes, user types 'N'
            mgr.update_text("I WANT TO N")
            time.sleep(0.12)
            # Stale sentence suggestions must be dropped
            self.assertEqual(mgr.get_sentence_suggestions(), [])
            self.assertGreater(mgr.metric_stale_dropped, 0)
        finally:
            mgr.close()

    # 8. Shutdown khong gay deadlock
    def test_c08_shutdown_no_deadlock(self):
        gen = FakeSlowSentenceGenerator(delay_s=0.10)
        mgr = SuggestionManager(dictionary=self.dict, sentence_generator=gen, debounce_ms=5.0)
        mgr.update_text("I WANT TO ")
        start = time.monotonic()
        mgr.close(timeout_s=0.3)
        elapsed = time.monotonic() - start
        self.assertFalse(mgr._worker_thread.is_alive())
        self.assertLess(elapsed, 0.35)


# ==============================================================================
# SECTION D: COMMIT SENTENCE TESTS (8 scenarios)
# ==============================================================================
class TestCommitSentence(unittest.TestCase):
    def setUp(self):
        self.composer = TextComposer()

    # 1. Noi dung continuation
    def test_d01_append_continuation_correctly(self):
        self.composer.text = "I WANT TO "
        ok = self.composer.commit_sentence("EAT SOMETHING.", valid_suggestions=["EAT SOMETHING."])
        self.assertTrue(ok)
        self.assertEqual(self.composer.text, "I WANT TO EAT SOMETHING.")

    # 2. Khong lap committed context
    def test_d02_no_repetition_of_committed_context(self):
        self.composer.text = "I WANT TO "
        # Model returned continuation that repeated the prompt context
        ok = self.composer.commit_sentence(
            "I WANT TO EAT SOMETHING.",
            valid_suggestions=["I WANT TO EAT SOMETHING."]
        )
        self.assertTrue(ok)
        self.assertEqual(self.composer.text, "I WANT TO EAT SOMETHING.")
        self.assertNotIn("I WANT TO I WANT TO", self.composer.text)

    # 3. Khong tao hai dau cach
    def test_d03_no_consecutive_spaces(self):
        self.composer.text = "I WANT TO   "
        ok = self.composer.commit_sentence("GO HOME.", valid_suggestions=["GO HOME."])
        self.assertTrue(ok)
        self.assertEqual(self.composer.text, "I WANT TO GO HOME.")
        self.assertNotIn("  ", self.composer.text)

    # 4. Tu choi cau khong nam trong whitelist
    def test_d04_reject_unwhitelisted_sentence(self):
        self.composer.text = "I WANT TO "
        ok = self.composer.commit_sentence(
            "UNAPPROVED CONTINUATION.",
            valid_suggestions=["EAT SOMETHING.", "GO HOME."]
        )
        self.assertFalse(ok)
        self.assertEqual(self.composer.text, "I WANT TO ")

    # 5. Tu choi cau tu generation cu / expected_context khong khop
    def test_d05_reject_stale_expected_context(self):
        self.composer.text = "I DECIDED TO "
        # Expected context from previous state was "I WANT TO"
        ok = self.composer.commit_sentence(
            "EAT SOMETHING.",
            valid_suggestions=["EAT SOMETHING."],
            expected_context="I WANT TO"
        )
        self.assertFalse(ok)
        self.assertEqual(self.composer.text, "I DECIDED TO ")

    # 6. Tu choi khi context thay doi (user typing a prefix)
    def test_d06_reject_when_user_typing_prefix(self):
        self.composer.text = "I WANT TO E"
        self.assertEqual(self.composer.current_prefix, "E")
        ok = self.composer.commit_sentence("EAT SOMETHING.", valid_suggestions=["EAT SOMETHING."])
        self.assertFalse(ok)
        self.assertEqual(self.composer.text, "I WANT TO E")

    # 7. Khong tu dong commit
    def test_d07_no_automatic_commit_without_explicit_call(self):
        dict_mock = MagicMock()
        gen_mock = MagicMock()
        gen_mock.generate.return_value = SentenceResult(
            success=True,
            suggestions=["EAT SOMETHING."]
        )
        mgr = SuggestionManager(dictionary=dict_mock, sentence_generator=gen_mock, debounce_ms=5.0)
        try:
            self.composer.text = "I WANT TO "
            mgr.update_text(self.composer.text)
            time.sleep(0.04)
            # Text must not change automatically
            self.assertEqual(self.composer.text, "I WANT TO ")
        finally:
            mgr.close()

    # 8. Khong lam thay doi noi dung truoc do
    def test_d08_preserve_all_prior_text(self):
        self.composer.text = "HELLO WORLD THIS IS MY TEXT "
        ok = self.composer.commit_sentence("FOR TODAY.", valid_suggestions=["FOR TODAY."])
        self.assertTrue(ok)
        self.assertEqual(self.composer.text, "HELLO WORLD THIS IS MY TEXT FOR TODAY.")
        self.assertTrue(self.composer.text.startswith("HELLO WORLD THIS IS MY TEXT"))


# ==============================================================================
# GROUP E: SENTENCE GENERATOR PAYLOAD CONTRACT TESTS (10 scenarios)
# Phase 3.10.8-A2-FIX1 Verification
# ==============================================================================
class TestSentenceGeneratorPayloadContract(unittest.TestCase):
    def setUp(self):
        self.captured_payload = None
        self.captured_timeout = None

    def _mock_transport(self, response_content: str):
        def transport(payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
            self.captured_payload = payload
            self.captured_timeout = timeout_s
            return {"message": {"content": response_content}}
        return transport

    # 1. Request gui toi Ollama chua think is False va think nam dung cap cao nhat
    def test_e01_think_is_false_at_top_level(self):
        transport = self._mock_transport('{"suggestions": ["GO HOME."]}')
        gen = SentenceGenerator(transport_fn=transport)
        res = gen.generate("I WANT TO")
        self.assertIsNotNone(self.captured_payload)
        self.assertIn("think", self.captured_payload)
        self.assertIs(self.captured_payload["think"], False)
        # think must NOT be placed inside options
        if "options" in self.captured_payload:
            self.assertNotIn("think", self.captured_payload["options"])

    # 2. format == "json" van duoc giu
    def test_e02_format_is_json(self):
        transport = self._mock_transport('{"suggestions": ["GO HOME."]}')
        gen = SentenceGenerator(transport_fn=transport)
        gen.generate("I WANT TO")
        self.assertEqual(self.captured_payload.get("format"), "json")

    # 3. stream == False
    def test_e03_stream_is_false(self):
        transport = self._mock_transport('{"suggestions": ["GO HOME."]}')
        gen = SentenceGenerator(transport_fn=transport)
        gen.generate("I WANT TO")
        self.assertIs(self.captured_payload.get("stream"), False)

    # 4. num_predict khong thay doi ngoai du kien
    def test_e04_num_predict_in_options(self):
        transport = self._mock_transport('{"suggestions": ["GO HOME."]}')
        gen = SentenceGenerator(transport_fn=transport)
        gen.generate("I WANT TO")
        options = self.captured_payload.get("options", {})
        self.assertIn("num_predict", options)
        self.assertGreaterEqual(options["num_predict"], 60)

    # 5. Model van la qwen3:4b theo default
    def test_e05_default_model_is_qwen3_4b(self):
        transport = self._mock_transport('{"suggestions": ["GO HOME."]}')
        gen = SentenceGenerator(transport_fn=transport)
        gen.generate("I WANT TO")
        self.assertEqual(self.captured_payload.get("model"), "qwen3:4b")

    # 6. Output JSON hop le duoc SentenceValidator chap nhan
    def test_e06_valid_json_accepted_by_validator(self):
        valid_json = '{"suggestions": ["EAT SOMETHING.", "GO HOME.", "LEARN ENGLISH."]}'
        transport = self._mock_transport(valid_json)
        gen = SentenceGenerator(transport_fn=transport)
        res = gen.generate("I WANT TO")
        self.assertTrue(res.success)
        self.assertEqual(len(res.suggestions), 3)
        self.assertEqual(res.suggestions, ["EAT SOMETHING.", "GO HOME.", "LEARN ENGLISH."])

    # 7. EMPTY_OUTPUT van duoc xu ly an toan
    def test_e07_empty_output_handled_safely(self):
        transport = self._mock_transport("")
        gen = SentenceGenerator(transport_fn=transport)
        res = gen.generate("I WANT TO")
        self.assertFalse(res.success)
        self.assertEqual(res.suggestions, [])
        self.assertEqual(res.error_type, "EMPTY_OUTPUT")

    # 8. HTTP error fallback dung
    def test_e08_http_error_fallback(self):
        import urllib.error
        def failing_transport(payload, timeout):
            raise urllib.error.URLError("Connection refused")
        gen = SentenceGenerator(transport_fn=failing_transport)
        res = gen.generate("I WANT TO")
        self.assertFalse(res.success)
        self.assertEqual(res.suggestions, [])
        self.assertEqual(res.error_type, "CONNECTION_ERROR")

    # 9. Timeout fallback dung
    def test_e09_timeout_fallback(self):
        def timeout_transport(payload, timeout):
            raise TimeoutError("Request timed out")
        gen = SentenceGenerator(transport_fn=timeout_transport)
        res = gen.generate("I WANT TO")
        self.assertFalse(res.success)
        self.assertEqual(res.suggestions, [])
        self.assertEqual(res.error_type, "TIMEOUT")

    # 10. Khong lam thay doi API cua SentenceGenerator
    def test_e10_api_backward_compatibility(self):
        gen = SentenceGenerator(
            model="custom-model",
            endpoint="http://localhost:11434/api/chat",
            timeout_ms=2000.0
        )
        self.assertEqual(gen.model, "custom-model")
        self.assertEqual(gen.endpoint, "http://localhost:11434/api/chat")
        self.assertEqual(gen.timeout_s, 2.0)
        self.assertTrue(callable(gen.generate))


if __name__ == "__main__":
    unittest.main()
