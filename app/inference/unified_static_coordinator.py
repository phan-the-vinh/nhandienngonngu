"""Experimental ABC/123 recognition coordination; no camera side effects."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Protocol, Sequence

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from ai.preprocessing.landmark_utils import normalize_landmarks
from app.inference.letter_inference import Prediction, TemporalSmoother
from app.inference.push_confirmation_camera_test import observe_hand
from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings
from app.inference.space_predictor import SpacePredictor
from app.suggestion.context import SuggestionContext

ROOT = Path(__file__).resolve().parents[2]
DIGIT_CONFIG = ROOT / "configs/digits.json"
PREPROCESSING_DIR = ROOT / "ai/preprocessing"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_digit_model() -> Pipeline:
    """Load the active digits_final Pipeline without changing its preprocessing."""
    config = json.loads(DIGIT_CONFIG.read_text(encoding="utf-8"))
    if (config.get("status") != "ACTIVE" or config.get("feature_count") != 63
            or config.get("mirror_before_mediapipe") is not False
            or config.get("classes") != [str(i) for i in range(10)]
            or config.get("hand_landmarker_path") != "models/mediapipe/hand_landmarker.task"):
        raise ValueError("Unexpected active digit configuration")
    model_path = ROOT / config["model_path"]
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Digit model file not found at '{model_path}'. "
            "This public repository is distributed source-only without unverified pre-trained model weights. "
            "Please see docs/MODEL_DISTRIBUTION.md for details."
        )
    manifest_path = ROOT / config["deployment_manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "ACTIVE"
            or manifest.get("model", {}).get("path") != config["model_path"]
            or manifest.get("feature_count") != 63
            or manifest.get("runtime", {}).get("mirror_before_mediapipe") is not False
            or manifest.get("runtime", {}).get("mediapipe_path") != config["hand_landmarker_path"]
            or _sha256(model_path) != manifest.get("model", {}).get("sha256")):
        raise ValueError("Digit model/manifest integrity check failed")
    if str(PREPROCESSING_DIR) not in sys.path:
        sys.path.insert(0, str(PREPROCESSING_DIR))  # joblib's recorded module path
    model = joblib.load(model_path)
    if (not isinstance(model, Pipeline) or model.n_features_in_ != 63
            or list(model.named_steps) != ["orientation_2d", "scaler", "mlp"]
            or [str(value) for value in model.classes_] != config["classes"]):
        raise ValueError("Unexpected digit production Pipeline")
    return model


class DigitPredictor:
    """Apply canonical 63 features; the saved Pipeline owns 2D alignment/scaling."""

    def __init__(self, model: Pipeline):
        self.model = model
        self.labels = tuple(str(value) for value in model.classes_)
        self.smoother = TemporalSmoother(class_count=len(self.labels))

    def reset(self) -> None:
        self.smoother.reset()

    def predict_landmarks(self, hand: Sequence[object]) -> Prediction:
        features = np.asarray(normalize_landmarks(hand), dtype=np.float64)
        if features.shape != (63,) or not np.isfinite(features).all():
            self.reset()
            raise ValueError("Digit inference requires 63 finite canonical features")
        probabilities = np.asarray(self.model.predict_proba(features.reshape(1, 63))[0],
                                   dtype=np.float64)
        if (probabilities.shape != (10,) or not np.isfinite(probabilities).all()
                or not np.isclose(probabilities.sum(), 1.0)):
            self.reset()
            raise ValueError("Invalid digit probabilities")
        instant = int(np.argmax(probabilities))
        stable, stable_confidence = self.smoother.update(instant, probabilities)
        return Prediction(
            instant_label=self.labels[instant],
            instant_confidence=float(probabilities[instant]),
            stable_label=None if stable is None else self.labels[stable],
            stable_confidence=stable_confidence,
            features=tuple(map(float, features)),
            probabilities=tuple(map(float, probabilities)),
        )


class Predictor(Protocol):
    def predict_landmarks(self, hand: Sequence[object]) -> Prediction: ...
    def reset(self) -> None: ...


class TextComposer:
    def __init__(self):
        self.text = ""
        self.on_change = None

    def _notify(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change(self.text)
            except Exception:
                pass

    @property
    def current_prefix(self) -> str:
        """Uppercase prefix of the word currently being typed."""
        return SuggestionContext.parse(self.text).current_prefix

    @property
    def committed_context(self) -> str:
        """Committed words preceding the current prefix."""
        return SuggestionContext.parse(self.text).committed_text

    def append(self, character: str) -> None:
        if character in SuggestionContext.DISALLOWED_SENTINELS:
            return
        if len(character) != 1 or not (character.isalnum() or character == " "):
            raise ValueError("Expected one letter, digit, or space")
        self.text += character
        self._notify()

    def commit_word(self, word: str, valid_suggestions: Sequence[str] | None = None) -> bool:
        """Commit a suggested word, replacing the active prefix with exactly one trailing space.

        Args:
            word: The candidate word to commit.
            valid_suggestions: Optional sequence of currently valid suggestions.
                               If provided, word must be present in this sequence.

        Returns:
            True if word was successfully committed, False otherwise.
        """
        if not word or not isinstance(word, str):
            return False

        word_clean = word.strip().upper()
        if not word_clean or not word_clean.isalpha():
            return False
        if word_clean in SuggestionContext.DISALLOWED_SENTINELS:
            return False

        current_pfx = self.current_prefix
        if current_pfx and not word_clean.startswith(current_pfx):
            return False

        if valid_suggestions is not None:
            valid_set = {s.strip().upper() for s in valid_suggestions if isinstance(s, str)}
            if word_clean not in valid_set:
                return False

        ctx = self.committed_context
        if ctx:
            self.text = f"{ctx} {word_clean} "
        else:
            self.text = f"{word_clean} "
        self._notify()
        return True

    def commit_sentence(
        self,
        sentence: str,
        valid_suggestions: Sequence[str] | None = None,
        expected_context: str | None = None,
    ) -> bool:
        """Commit a suggested sentence continuation.

        Args:
            sentence: The sentence continuation to commit.
            valid_suggestions: Optional sequence of currently valid sentence suggestions.
            expected_context: Optional context string expected to match current committed context.

        Returns:
            True if committed successfully, False otherwise.
        """
        if not sentence or not isinstance(sentence, str):
            return False

        # Reject if user is currently spelling a word
        if self.current_prefix:
            return False

        sentence_clean = " ".join(sentence.strip().split())
        if not sentence_clean:
            return False

        # Reject sentinels
        words = [w.upper().rstrip(".,!?;:") for w in sentence_clean.split()]
        if any(w in SuggestionContext.DISALLOWED_SENTINELS for w in words):
            return False

        # Check whitelist if provided
        if valid_suggestions is not None:
            valid_set = {" ".join(s.strip().split()).upper() for s in valid_suggestions if isinstance(s, str)}
            if sentence_clean.upper() not in valid_set:
                return False

        current_ctx = self.committed_context
        if expected_context is not None:
            clean_expected = " ".join(expected_context.strip().split()).upper()
            if current_ctx != clean_expected:
                return False

        # Check if sentence continuation repeats current_ctx prefix
        clean_sentence_upper = sentence_clean.upper()
        if current_ctx and clean_sentence_upper.startswith(current_ctx):
            continuation = sentence_clean[len(current_ctx):].strip()
        else:
            continuation = sentence_clean

        if not continuation:
            return False

        # Append continuation without double space
        if current_ctx:
            self.text = f"{current_ctx} {continuation}"
        else:
            self.text = continuation

        self._notify()
        return True

    def clear(self) -> None:
        self.text = ""
        self._notify()

    def backspace(self) -> None:
        if self.text:
            self.text = self.text[:-1]
            self._notify()


class RecognitionCoordinator:
    """Route one detected hand to one model and one unchanged HOLD instance."""

    def __init__(self, letter_predictor: Predictor, digit_predictor: Predictor,
                 hold_settings: HoldSettings, space_predictor: SpacePredictor | None = None,
                 enable_space: bool = False):
        self.predictors = {"ABC": letter_predictor, "123": digit_predictor}
        self.mode = "ABC"
        self.hold_settings = hold_settings
        self.hold = HoldConfirmation(hold_settings)
        self.composer = TextComposer()
        self.space_predictor = space_predictor
        self.enable_space = enable_space

    def switch_mode(self, mode: str) -> None:
        if mode not in (*self.predictors, "JZ_PAUSED"):
            raise ValueError(f"Invalid recognition mode: {mode}")
        if mode == self.mode:
            return
        for predictor in self.predictors.values():
            predictor.reset()
        if self.space_predictor is not None:
            self.space_predictor.reset()
        self.mode = mode
        self.hold = HoldConfirmation(self.hold_settings)
        self.hold.text = self.composer.text  # Existing HOLD UI displays this text.

    def missing_hand(self, now: float) -> None:
        if self.mode in self.predictors:
            self.predictors[self.mode].reset()
        if self.space_predictor is not None:
            self.space_predictor.reset()
        self.hold.update(None, timestamp=now)

    def process_hand(self, hand: Sequence[object], frame_shape: tuple[int, ...],
                     now: float) -> tuple[Prediction, str | None]:
        if self.mode == "JZ_PAUSED":
            raise ValueError("J/Z live recognition is disabled; static confirmation is paused")

        # Prioritized SPACE evaluation: evaluate before static letter/digit classification
        if self.enable_space and self.space_predictor is not None:
            try:
                prob_space, is_space = self.space_predictor.predict_hand(hand, mode=self.mode)
            except Exception:
                prob_space, is_space = 0.0, False

            if is_space:
                features = tuple(normalize_landmarks(hand))
                prediction = Prediction(
                    instant_label="SPACE",
                    instant_confidence=prob_space,
                    stable_label="SPACE",
                    stable_confidence=prob_space,
                    features=features,
                    probabilities=(1.0 - prob_space, prob_space)
                )
                observation = observe_hand(hand, prediction, frame_shape)
                committed = self.hold.update(observation, timestamp=now)
                if committed is not None:
                    character = " " if committed == "SPACE" else committed
                    self.composer.append(character)
                    self.hold.text = self.composer.text
                    self.predictors[self.mode].reset()
                    self.space_predictor.reset()
                elif self.hold.rearmed and self.hold.state == "IDLE":
                    self.predictors[self.mode].reset()
                    self.space_predictor.reset()
                return prediction, committed

        # Baseline letter/digit predictor evaluation
        predictor = self.predictors[self.mode]
        prediction = predictor.predict_landmarks(hand)
        observation = observe_hand(hand, prediction, frame_shape)
        committed = self.hold.update(observation, timestamp=now)
        if committed is not None:
            character = " " if committed == "SPACE" else committed
            self.composer.append(character)
            self.hold.text = self.composer.text
            predictor.reset()  # Do not carry confirmed votes into the next attempt.
            if self.space_predictor is not None:
                self.space_predictor.reset()
        elif self.hold.rearmed and self.hold.state == "IDLE":
            predictor.reset()
            if self.space_predictor is not None:
                self.space_predictor.reset()
        return prediction, committed
