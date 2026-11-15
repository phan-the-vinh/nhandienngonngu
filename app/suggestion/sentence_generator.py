"""Offline Qwen Sentence Generator.

Queries local Ollama service to generate 3 short, natural English sentence
continuations based on committed context.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app.suggestion.sentence_validator import SentenceValidationResult, SentenceValidator


@dataclass(frozen=True)
class SentenceResult:
    """Outcome of a sentence generation attempt."""
    success: bool
    suggestions: list[str]
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: float = 0.0


SENTENCE_SYSTEM_PROMPT = (
    "You are an English sentence completion assistant for sign language typing. "
    "Given sentence context that the user has already typed, generate exactly 3 short, "
    "natural English sentence continuations to complete their thought. "
    "Rules:\n"
    "1. Do NOT repeat the words already typed in the context. Only provide the continuation.\n"
    "2. Each continuation must be a short, sensible continuation (at most 10 words).\n"
    "3. Return ONLY a valid JSON object with the exact key 'suggestions' containing a list of strings.\n"
    "4. Return at most 3 continuations.\n"
    "Example: {\"suggestions\": [\"EAT SOMETHING.\", \"GO HOME.\", \"LEARN ENGLISH.\"]}"
)


class SentenceGenerator:
    """Client for local Ollama-based sentence continuation generation."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        timeout_ms: float = 1500.0,
        transport_fn: Optional[Callable[[Dict[str, Any], float], Dict[str, Any]]] = None
    ) -> None:
        """Initialize the sentence generator.

        Args:
            model: Ollama model tag (default: 'qwen3:4b').
            endpoint: Ollama chat completion API URL.
            timeout_ms: Socket timeout in milliseconds (default: 1500.0).
            transport_fn: Optional transport callable for unit testing / mocking.
        """
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_ms / 1000.0
        self.transport_fn = transport_fn or self._http_transport

    def _http_transport(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """Perform HTTP POST request to Ollama service."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)

    def generate(self, context: str) -> SentenceResult:
        """Generate 3 sentence continuations given the committed context.

        Args:
            context: Committed words string, e.g. 'I WANT TO'.

        Returns:
            SentenceResult with success, validated suggestions, and latency.
        """
        clean_context = context.strip()
        if not clean_context:
            return SentenceResult(
                success=False,
                suggestions=[],
                error_type="EMPTY_CONTEXT",
                error_message="Cannot generate sentence for empty context",
                latency_ms=0.0
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SENTENCE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: \"{clean_context}\"\nGenerate 3 short continuations:"}
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_predict": 128,
            }
        }

        start_time = time.monotonic()
        try:
            resp = self.transport_fn(payload, self.timeout_s)
            latency_ms = (time.monotonic() - start_time) * 1000.0
        except urllib.error.URLError as exc:
            latency_ms = (time.monotonic() - start_time) * 1000.0
            reason_str = str(exc.reason) if hasattr(exc, "reason") else str(exc)
            err_type = "TIMEOUT" if "timed out" in reason_str.lower() else "CONNECTION_ERROR"
            return SentenceResult(
                success=False,
                suggestions=[],
                error_type=err_type,
                error_message=f"Network error: {reason_str}",
                latency_ms=latency_ms
            )
        except TimeoutError:
            latency_ms = (time.monotonic() - start_time) * 1000.0
            return SentenceResult(
                success=False,
                suggestions=[],
                error_type="TIMEOUT",
                error_message="Socket read timed out",
                latency_ms=latency_ms
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start_time) * 1000.0
            return SentenceResult(
                success=False,
                suggestions=[],
                error_type="EXCEPTION",
                error_message=f"Unexpected error: {exc}",
                latency_ms=latency_ms
            )

        # Parse Ollama response format
        raw_content = ""
        if isinstance(resp, dict):
            raw_content = resp.get("message", {}).get("content", "")

        val_result = SentenceValidator.validate(raw_content, max_sentences=3)
        if not val_result.is_valid:
            return SentenceResult(
                success=False,
                suggestions=[],
                error_type=val_result.error or "VALIDATION_FAILED",
                error_message=f"Validator rejected output: {val_result.error}",
                latency_ms=latency_ms
            )

        return SentenceResult(
            success=True,
            suggestions=val_result.sentences,
            error_type=None,
            error_message=None,
            latency_ms=latency_ms
        )
