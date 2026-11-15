"""Qwen Offline Language Ranker.

Interacts with local Ollama service to perform contextual re-ranking
of prefix candidate words for ASL typing assistance.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class RankResult:
    """Outcome of a contextual re-ranking attempt."""
    success: bool
    ranked_words: list[str]
    error_type: Optional[str] = None  # None, "TIMEOUT", "CONNECTION_ERROR", "INVALID_JSON", "EMPTY_RESPONSE", "EXCEPTION"
    error_message: Optional[str] = None
    latency_ms: float = 0.0


SYSTEM_PROMPT = (
    "You are an English word completion assistant for sign language typing. "
    "Given sentence context, a word prefix, and candidate words from a dictionary: "
    "select and rank up to 5 most likely candidate words to complete the prefix. "
    "Rules:\n"
    "1. Rank ONLY words from the provided candidate list. Never invent or add words.\n"
    "2. Never change the prefix or spelling of candidate words.\n"
    "3. Return ONLY a valid JSON object with the exact key 'ranked_words' containing a list of strings.\n"
    "4. Return at most 5 words.\n"
    "Example: {\"ranked_words\": [\"APPLE\", \"APPLICATION\"]}"
)


class QwenRanker:
    """Client for local Ollama-based candidate re-ranking."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        timeout_ms: float = 1500.0,
        transport_fn: Optional[Callable[[Dict[str, Any], float], Dict[str, Any]]] = None
    ) -> None:
        """Initialize the ranker.

        Args:
            model: Ollama model name (default: 'qwen3:4b').
            endpoint: Local Ollama chat API URL.
            timeout_ms: Request timeout in milliseconds (default: 1500.0 ms).
            transport_fn: Optional custom transport callable (for unit testing without real Ollama).
        """
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_ms / 1000.0
        self._transport_fn = transport_fn or self._default_http_transport

    def rank(
        self,
        context: str,
        prefix: str,
        candidates: Sequence[str]
    ) -> RankResult:
        """Rank candidate words given preceding context and active prefix.

        Guarantees:
        - Whitelist candidate filtering: No words outside candidates are returned.
        - Deduplication: No duplicate words in returned list.
        - Prefix matching: All words must start strictly with the prefix.
        - Max 5 results.
        - If any error occurs (timeout, offline, malformed JSON), returns a failed
          RankResult so caller can fall back to dictionary ranking.
        """
        clean_prefix = prefix.strip().upper() if prefix else ""
        clean_candidates = [c.strip().upper() for c in candidates if c and c.strip()]

        # Edge cases: no prefix or empty candidates
        if not clean_prefix or not clean_candidates:
            return RankResult(
                success=True,
                ranked_words=[],
                latency_ms=0.0
            )

        # Prepare minimal prompt payload (only context + prefix + candidates)
        clean_context = context.strip().upper() if context else ""
        user_msg = (
            f"Context: {clean_context}\n"
            f"Prefix: {clean_prefix}\n"
            f"Candidates: {json.dumps(clean_candidates)}\n"
            "Output JSON:"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            "think": False,  # Mandatory: suppress reasoning mode to prevent 12s latency
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_predict": 128
            }
        }

        t0 = time.perf_counter()
        try:
            response_body = self._transport_fn(payload, self.timeout_s)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except TimeoutError as ex:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return RankResult(
                success=False,
                ranked_words=[],
                error_type="TIMEOUT",
                error_message=str(ex),
                latency_ms=latency_ms
            )
        except ConnectionError as ex:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return RankResult(
                success=False,
                ranked_words=[],
                error_type="CONNECTION_ERROR",
                error_message=str(ex),
                latency_ms=latency_ms
            )
        except Exception as ex:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return RankResult(
                success=False,
                ranked_words=[],
                error_type="EXCEPTION",
                error_message=str(ex),
                latency_ms=latency_ms
            )

        # Parse and validate output
        raw_content = response_body.get("message", {}).get("content", "")
        if not raw_content:
            return RankResult(
                success=False,
                ranked_words=[],
                error_type="EMPTY_RESPONSE",
                error_message="Ollama returned empty message content",
                latency_ms=latency_ms
            )

        try:
            parsed = json.loads(raw_content)
        except Exception as ex:
            return RankResult(
                success=False,
                ranked_words=[],
                error_type="INVALID_JSON",
                error_message=f"JSON decode failed: {ex}",
                latency_ms=latency_ms
            )

        if not isinstance(parsed, dict) or "ranked_words" not in parsed:
            return RankResult(
                success=False,
                ranked_words=[],
                error_type="INVALID_FORMAT",
                error_message="JSON missing 'ranked_words' list key",
                latency_ms=latency_ms
            )

        raw_words = parsed.get("ranked_words", [])
        if not isinstance(raw_words, list):
            return RankResult(
                success=False,
                ranked_words=[],
                error_type="INVALID_FORMAT",
                error_message="'ranked_words' is not a list",
                latency_ms=latency_ms
            )

        # 1. Whitelist filtering & prefix validation
        candidate_set = set(clean_candidates)
        whitelisted = []
        for item in raw_words:
            w = str(item).strip().upper()
            if w in candidate_set and w.startswith(clean_prefix):
                whitelisted.append(w)

        # 2. Deduplication preserving order
        deduped = list(dict.fromkeys(whitelisted))

        # 3. Limit to top 5
        final_ranked = deduped[:5]

        return RankResult(
            success=True,
            ranked_words=final_ranked,
            latency_ms=latency_ms
        )

    def _default_http_transport(self, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
        """Default HTTP transport using standard urllib."""
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=req_data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as ex:
            # Check for timeout vs connection refused
            msg = str(ex).lower()
            if "timed out" in msg:
                raise TimeoutError(f"HTTP request timed out after {timeout_s:.1f}s") from ex
            if "connection refused" in msg or "10061" in msg or "winerror 10061" in msg:
                raise ConnectionError(f"Cannot connect to Ollama at {self.endpoint}") from ex
            raise
        except TimeoutError:
            raise
        except Exception as ex:
            raise RuntimeError(f"Ollama request error: {ex}") from ex
