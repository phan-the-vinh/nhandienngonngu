"""Unit tests for QwenRanker with mock transports.
"""
from __future__ import annotations

import json
import unittest
from typing import Any, Dict

from app.suggestion.qwen_ranker import QwenRanker, RankResult


class TestQwenRanker(unittest.TestCase):
    """Test suite for QwenRanker client and guardrails."""

    def test_successful_ranking(self):
        def fake_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            # Verify required parameters
            self.assertEqual(payload.get("think"), False)
            self.assertEqual(payload.get("format"), "json")
            return {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"ranked_words": ["APPLE", "APPLICATION", "APPOINTMENT"]})
                }
            }

        ranker = QwenRanker(transport_fn=fake_transport)
        candidates = ["APPOINTMENT", "APPLICATION", "APPLE", "APPROACH"]
        res = ranker.rank(context="I WANT TO EAT AN", prefix="APP", candidates=candidates)

        self.assertTrue(res.success)
        self.assertEqual(res.ranked_words, ["APPLE", "APPLICATION", "APPOINTMENT"])
        self.assertIsNone(res.error_type)

    def test_whitelist_sanitization(self):
        def fake_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            return {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "ranked_words": ["INVENTED_WORD", "APPLE", "BANANA", "APPOINTMENT"]
                    })
                }
            }

        ranker = QwenRanker(transport_fn=fake_transport)
        candidates = ["APPLE", "APPLICATION", "APPOINTMENT"]
        res = ranker.rank(context="I LIKE", prefix="APP", candidates=candidates)

        self.assertTrue(res.success)
        # "INVENTED_WORD" is not in candidates -> dropped
        # "BANANA" does not start with "APP" and not in candidates -> dropped
        self.assertEqual(res.ranked_words, ["APPLE", "APPOINTMENT"])

    def test_deduplication_preserving_order(self):
        def fake_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            return {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "ranked_words": ["BALL", "BACK", "BALL", "BAG", "BANK"]
                    })
                }
            }

        ranker = QwenRanker(transport_fn=fake_transport)
        candidates = ["BACK", "BAD", "BAG", "BALL", "BANK"]
        res = ranker.rank(context="HE KICKED THE", prefix="B", candidates=candidates)

        self.assertTrue(res.success)
        self.assertEqual(res.ranked_words, ["BALL", "BACK", "BAG", "BANK"])
        self.assertEqual(len(res.ranked_words), len(set(res.ranked_words)))

    def test_clamped_to_max_5(self):
        def fake_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            return {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "ranked_words": ["ABOUT", "AFTER", "AGAIN", "ALL", "ALONE", "ALWAYS", "AND"]
                    })
                }
            }

        ranker = QwenRanker(transport_fn=fake_transport)
        candidates = ["ABOUT", "AFTER", "AGAIN", "ALL", "ALONE", "ALWAYS", "AND"]
        res = ranker.rank(context="", prefix="A", candidates=candidates)

        self.assertTrue(res.success)
        self.assertEqual(len(res.ranked_words), 5)
        self.assertEqual(res.ranked_words, ["ABOUT", "AFTER", "AGAIN", "ALL", "ALONE"])

    def test_malformed_json_fallback(self):
        def fake_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            return {"message": {"content": "This is plain text not JSON"}}

        ranker = QwenRanker(transport_fn=fake_transport)
        res = ranker.rank(context="TEST", prefix="AP", candidates=["APPLE"])

        self.assertFalse(res.success)
        self.assertEqual(res.error_type, "INVALID_JSON")
        self.assertEqual(res.ranked_words, [])

    def test_missing_ranked_words_key(self):
        def fake_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            return {"message": {"content": json.dumps({"unrelated_key": [1, 2, 3]})}}

        ranker = QwenRanker(transport_fn=fake_transport)
        res = ranker.rank(context="TEST", prefix="AP", candidates=["APPLE"])

        self.assertFalse(res.success)
        self.assertEqual(res.error_type, "INVALID_FORMAT")
        self.assertEqual(res.ranked_words, [])

    def test_timeout_handling(self):
        def fake_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            raise TimeoutError("HTTP request timed out after 1.5s")

        ranker = QwenRanker(transport_fn=fake_transport)
        res = ranker.rank(context="TEST", prefix="AP", candidates=["APPLE"])

        self.assertFalse(res.success)
        self.assertEqual(res.error_type, "TIMEOUT")
        self.assertEqual(res.ranked_words, [])

    def test_connection_error_handling(self):
        def fake_transport(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
            raise ConnectionError("Connection refused by Ollama daemon")

        ranker = QwenRanker(transport_fn=fake_transport)
        res = ranker.rank(context="TEST", prefix="AP", candidates=["APPLE"])

        self.assertFalse(res.success)
        self.assertEqual(res.error_type, "CONNECTION_ERROR")
        self.assertEqual(res.ranked_words, [])

    def test_empty_candidates_or_prefix(self):
        ranker = QwenRanker()
        res1 = ranker.rank(context="TEST", prefix="", candidates=["APPLE"])
        self.assertTrue(res1.success)
        self.assertEqual(res1.ranked_words, [])

        res2 = ranker.rank(context="TEST", prefix="AP", candidates=[])
        self.assertTrue(res2.success)
        self.assertEqual(res2.ranked_words, [])


if __name__ == "__main__":
    unittest.main()
