"""Replay opt-in J/Z diagnostic attempts; read-only, no camera/MediaPipe calls.

Event-only logs (including historical attempts 5/7) lack exact input arrays and
cannot be turned into replay data. No inferred landmarks or features are used.
"""
import argparse
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from app.inference.jz_attempt_capture import load_attempt,geometry_replay,model_replay


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir",type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only",action="store_true",help="Validate NPZ/hash/schema only; no model inference")
    mode.add_argument("--geometry-replay",action="store_true",help="Rebuild canonical features from saved landmarks and explain geometry")
    mode.add_argument("--model-replay",action="store_true",help="Predict using the EXACT saved pre-scaler matrix")
    mode.add_argument("--replay",action="store_true",help="Both geometry and model replay")
    parser.add_argument("--z-start-pose-shadow",action="store_true",
                        help="Also measure phase-A Z pose hypotheses using saved raw observations; no gate")
    args=parser.parse_args(argv)
    try:
        metadata,arrays=load_attempt(args.attempt_dir)
        report={"session_id":metadata["session_id"],"session_generation":metadata["session_generation"],
                "attempt_id":metadata["attempt_id"],"camera_opened":False,"mediapipe_executed":False,
                "files_written":0,"human_label":"UNREVIEWED (consult separate human review manifest)"}
        matched=True
        if args.geometry_replay or args.replay:
            report["geometry"]=geometry_replay(metadata,arrays)
            matched &= report["geometry"]["exact_feature_match"] and report["geometry"]["geometry_matches_saved"]
        if args.model_replay or args.replay:
            from app.inference.dynamic_jz_camera_test import load_experiment_model
            model,model_hash=load_experiment_model()
            report["model"]=model_replay(metadata,arrays,model,model_hash)
            matched &= report["model"]["prediction_match"] and report["model"]["probabilities_exact_match"]
        if args.z_start_pose_shadow and not args.check_only:
            from app.inference.z_start_pose_experimental import replay_start_pose_shadow
            report["z_start_pose_shadow"]=replay_start_pose_shadow(metadata,arrays)
        report["status"]="PASS" if matched else "MISMATCH_REVIEW_REQUIRED"
        print(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2))
        return 0 if matched else 1
    except (OSError,ValueError,KeyError,TypeError,EOFError,BadZipFile) as exc:
        print(json.dumps({"status":"REPLAY_UNAVAILABLE","reason":str(exc),"files_written":0}),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
