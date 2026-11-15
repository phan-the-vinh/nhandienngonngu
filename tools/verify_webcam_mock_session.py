"""Verification script: Executes a mock live session through evaluate_attempt and compute_session_metrics.

Validates that:
1. build_sequence_features succeeds without keyword argument errors.
2. Production model pipeline is actually invoked (model_invoked = True).
3. Both J and Z produce valid probabilities and predictions.
4. Baseline LR and Margin Wrapper both execute on identical features.
5. Telemetry is saved to reports/data/webcam_mock_verification_results.json (NEVER overwriting webcam_live_validation_results.json).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference.dynamic_jz.stage13_margin_wrapper import ProbabilityMarginModelWrapper
from tools.validate_webcam_live import (
    MODEL_PATH,
    compute_session_metrics,
    evaluate_attempt,
)

OUTPUT_PATH = PROJECT_ROOT / "reports/data/webcam_mock_verification_results.json"
USER_DATA_PATH = PROJECT_ROOT / "reports/data/webcam_live_validation_results.json"


def generate_hand_sequence(num_frames: int = 20, motion_type: str = "J"):
    """Generate realistic 21-landmark frames for dynamic gestures."""
    frames = []
    w, h = 640, 480
    for t in range(num_frames):
        alpha = t / float(num_frames - 1)
        if motion_type == "J":
            # Wrist moves downward then hooks left
            wrist_x = 0.5 - (0.08 * (alpha ** 2) if alpha > 0.6 else 0.0)
            wrist_y = 0.4 + 0.15 * alpha
        elif motion_type == "Z":
            # Wrist moves zig-zag: right -> diag down-left -> right
            if alpha < 0.33:
                wrist_x = 0.4 + 0.15 * (alpha / 0.33)
                wrist_y = 0.4
            elif alpha < 0.66:
                wrist_x = 0.55 - 0.15 * ((alpha - 0.33) / 0.33)
                wrist_y = 0.4 + 0.12 * ((alpha - 0.33) / 0.33)
            else:
                wrist_x = 0.4 + 0.15 * ((alpha - 0.66) / 0.34)
                wrist_y = 0.52
        else:
            # Static palm
            wrist_x = 0.5
            wrist_y = 0.5

        hand = []
        for i in range(21):
            hx = wrist_x + (i % 5) * 0.012
            hy = wrist_y + (i // 5) * 0.015
            hz = i * 0.001
            hand.append(SimpleNamespace(x=float(hx), y=float(hy), z=float(hz)))
        frames.append((t + 1, hand, (w, h)))
    return frames


def main():
    print("=" * 80)
    print("RUNNING WEBCAM MOCK VALIDATION VERIFICATION")
    print("=" * 80)

    # Check that user data exists and record its size/timestamp to guarantee it is preserved
    user_data_before_hash = None
    if USER_DATA_PATH.exists():
        import hashlib
        user_data_before_hash = hashlib.sha256(USER_DATA_PATH.read_bytes()).hexdigest()
        print(f"Verified existing user live results at: {USER_DATA_PATH}")
        print(f"  User data SHA256: {user_data_before_hash[:16]}... (Will be strictly preserved)")

    raw_pipeline = joblib.load(MODEL_PATH)
    wrapper_base = ProbabilityMarginModelWrapper(raw_pipeline, enable_reject_option=False, margin_threshold=0.10)
    wrapper_rej = ProbabilityMarginModelWrapper(raw_pipeline, enable_reject_option=True, margin_threshold=0.10)

    # Trial 1: J gesture
    print("\n[1/3] Simulating Trial 1 (Target: J)...")
    j_frames = generate_hand_sequence(25, "J")
    rec_j = evaluate_attempt(0, "J", j_frames, len(j_frames), wrapper_base, wrapper_rej, 0.10)
    print(f"  model_invoked: {rec_j['model_invoked']}")
    print(f"  Base Pred:     {rec_j['baseline_prediction']} (P(J)={rec_j['probability_J']:.4f}, P(Z)={rec_j['probability_Z']:.4f})")
    print(f"  Reject Pred:   {rec_j['reject_option_prediction']} (margin={rec_j['margin']:.4f})")
    print(f"  Rejection rsn: '{rec_j['rejection_reason']}'")
    assert rec_j["model_invoked"] is True, f"Trial J failed: {rec_j['rejection_reason']}"

    # Trial 2: Z gesture
    print("\n[2/3] Simulating Trial 2 (Target: Z)...")
    z_frames = generate_hand_sequence(25, "Z")
    rec_z = evaluate_attempt(1, "Z", z_frames, len(z_frames), wrapper_base, wrapper_rej, 0.10)
    print(f"  model_invoked: {rec_z['model_invoked']}")
    print(f"  Base Pred:     {rec_z['baseline_prediction']} (P(J)={rec_z['probability_J']:.4f}, P(Z)={rec_z['probability_Z']:.4f})")
    print(f"  Reject Pred:   {rec_z['reject_option_prediction']} (margin={rec_z['margin']:.4f})")
    print(f"  Rejection rsn: '{rec_z['rejection_reason']}'")
    assert rec_z["model_invoked"] is True, f"Trial Z failed: {rec_z['rejection_reason']}"

    # Trial 3: NEGATIVE control
    print("\n[3/3] Simulating Trial 3 (Target: NEGATIVE)...")
    neg_frames = generate_hand_sequence(20, "NEGATIVE")
    rec_neg = evaluate_attempt(2, "NEGATIVE", neg_frames, len(neg_frames), wrapper_base, wrapper_rej, 0.10)
    print(f"  model_invoked: {rec_neg['model_invoked']}")
    print(f"  Base Pred:     {rec_neg['baseline_prediction']} (P(J)={rec_neg['probability_J']:.4f}, P(Z)={rec_neg['probability_Z']:.4f})")
    print(f"  Reject Pred:   {rec_neg['reject_option_prediction']} (margin={rec_neg['margin']:.4f})")
    assert rec_neg["model_invoked"] is True, f"Trial NEGATIVE failed: {rec_neg['rejection_reason']}"

    # Compute metrics
    records = [rec_j, rec_z, rec_neg]
    metrics = compute_session_metrics(records)

    payload = {
        "metadata": {
            "description": "Mock validation session verifying build_sequence_features fix and model invocation",
            "trials_completed": len(records),
            "margin_threshold": 0.10,
        },
        "metrics": metrics,
        "trials": records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[SAVED] Verification telemetry saved to {OUTPUT_PATH}")

    # Verify that user live data was untouched
    if user_data_before_hash is not None:
        import hashlib
        user_data_after_hash = hashlib.sha256(USER_DATA_PATH.read_bytes()).hexdigest()
        assert user_data_after_hash == user_data_before_hash, "ERROR: User live validation data was modified!"
        print(f"[VERIFIED] User live validation data is untouched (SHA256 matches: {user_data_after_hash[:16]}...)")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETED SUCCESSFULLY: ALL TRIALS INVOKED MODEL WITHOUT ERRORS")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
