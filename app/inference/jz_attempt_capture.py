"""Opt-in, bounded landmark-only attempt capture. Not a training-data writer.

No filesystem writes at import. Callers explicitly create a session only after
--check-only has returned. All callbacks are observational and fail open for
recognition, with errors retained for the user (never silently accepted replay).
"""
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from ai.feature_extraction.dynamic_sequence import build_sequence_features, sample_valid_frame_indices
from ai.feature_extraction.hand_landmarks import landmarks_to_array
from app.inference.jz_shadow_diagnostics import geometry_predicates, trajectory_diagnostics

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "data/experiments/dynamic_jz_v1"
HUMAN_LABELS = ("UNREVIEWED","J_TRUE","Z_TRUE","STATIC_TRANSITION","HAND_ENTRY_EXIT","OTHER_MOTION","UNCERTAIN")
REVIEW_COLUMNS = ("session_id","session_generation","attempt_id","human_label","note",
                  "model_prediction","geometry_candidates","candidate_status","committed_character","attempt_path")
CODE_FILES = ("app/inference/automatic_jz_experiment.py","app/inference/first16_jz_segmenter.py",
              "app/inference/dynamic_jz_auto_camera_experiment.py","app/inference/integrated_jz_validation.py",
              "app/inference/unified_abc123_jz_auto_coordinator.py","app/inference/jz_shadow_diagnostics.py",
              "app/inference/jz_attempt_capture.py","app/inference/unified_abc123_jz_auto_experiment.py",
              "app/inference/dynamic_jz_camera_test.py","ai/feature_extraction/dynamic_sequence.py",
              "app/inference/z_start_pose_experimental.py",
              "ai/preprocessing/landmark_utils.py")


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def checked_output(path, root=EXPERIMENT_ROOT):
    path=Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Diagnostic output must remain under the J/Z experiment")
    return path


class DiagnosticModelProxy:
    """Copy the exact pre-scaler model input, without modifying or re-running it."""
    def __init__(self, model, recorder):
        self.model,self.recorder=model,recorder

    def __getattr__(self,name):
        return getattr(self.model,name)

    def predict_proba(self, features):
        saved=self.recorder.safe(lambda:np.asarray(features).copy())
        result=self.model.predict_proba(features)
        if saved is not None:
            self.recorder.safe(self.recorder.model_input,saved,np.asarray(result).copy())
        return result


class AttemptCapture:
    MAX_OBSERVATIONS=4096
    MAX_EVENTS=512

    def __init__(self, output, *, model_sha256, config, root=EXPERIMENT_ROOT, session_id=None):
        output=checked_output(output,root)
        self.session_id=session_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_")+uuid4().hex[:8])
        if not self.session_id or any(c not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-" for c in self.session_id):
            raise ValueError("Unsafe diagnostic session ID")
        self.path=output/self.session_id
        self.path.mkdir(parents=True,exist_ok=False)
        self.config=config
        self.model_sha256=model_sha256
        self.code_hashes={name:sha256(ROOT/name) for name in CODE_FILES}
        self.environment={"python":platform.python_version(),
                          **{name:version(name) for name in ("numpy","scikit-learn","mediapipe","joblib")}}
        self.history=deque(maxlen=512)
        self.current=None
        self.errors=[]
        self.completed_keys=set()
        self.stream=(self.path/"human_review_manifest.csv").open("x",encoding="utf-8",newline="")
        self.writer=csv.DictWriter(self.stream,fieldnames=REVIEW_COLUMNS)
        self.writer.writeheader()
        self.stream.flush()

    def safe(self, fn, *args, **kwargs):
        try:
            return fn(*args,**kwargs)
        except Exception as exc:  # Diagnostics must not alter recognition on disk/analysis failure.
            self.errors.append(type(exc).__name__+": "+str(exc))
            self.errors=self.errors[-8:]
            if self.current is not None:
                self.current["diagnostic_error"]=self.errors[-1]
            return None

    def observe(self,index,hand,size,now,*,hand_count,error,mode):
        if mode != "ABC":
            return
        raw=None
        if hand is not None and not error and hand_count in (None,1):
            try:
                raw=landmarks_to_array(hand).copy()
            except (ValueError,TypeError,AttributeError):
                pass
        row=(int(index),float(now),tuple(size),raw)
        self.history.append(row)
        if self.current is not None:
            if len(self.current["observations"]) >= self.MAX_OBSERVATIONS:
                self.current["diagnostic_error"]="OBSERVATION_LIMIT_EXCEEDED; replay incomplete"
            else:
                self.current["observations"].append(row)

    def event(self,event,generation):
        name=event["event"]
        aid=event.get("attempt_id")
        generation=event.get("session_generation",generation)
        key=(generation,aid)
        if name=="STATE_TRANSITION" and event.get("new_state")=="CAPTURING" and aid is not None:
            if self.current is None or self.current["key"]!=key:
                self.finish("NEXT_ATTEMPT")
                if key in self.completed_keys:
                    raise ValueError("Diagnostic attempt ID reused in same generation")
                first=event.get("segment",{}).get("first_frame",event["frame_id"])
                self.current={"key":key,"onset_frame":event["frame_id"],"onset_timestamp":event["timestamp_monotonic"],
                              "observations":[r for r in self.history if r[0]>=first],"events":[],
                              "candidate_status":"NO_CANDIDATE","committed_character":"","model_result":None}
        c=self.current
        if c is None or key!=c["key"]:
            return
        if (event.get("z_start_pose_shadow") or {}).get("attempt"):
            c["z_start_pose_shadow"]=event["z_start_pose_shadow"]
        if name not in ("MOTION_SAMPLE","RUNTIME_SAMPLE","OBSERVATION_CHANGE","ENDPOINT_EVIDENCE_CHANGE"):
            c["events"].append({k:event.get(k) for k in ("event","frame_id","timestamp_monotonic","previous_state",
                  "new_state","reason","cancellation_reason","character","hold_state","dynamic_hold_state")})
            if len(c["events"])>self.MAX_EVENTS:
                c["events"]=c["events"][-self.MAX_EVENTS:]
                c["event_trace_truncated"]=True
        if name=="ATTEMPT_FINISHED":
            c["result_status"]=event["result"]["status"]
            c["model_result"]=event["result"].get("model_result")
            c["finish_reason"]=event["result"].get("reason")
        if name=="JZ_CANDIDATE_CREATED":
            c["candidate_status"]="CREATED"
            c["endpoint_frame"]=event["frame_id"]
        elif name=="JZ_CANDIDATE_CANCELLED":
            c["candidate_status"]="CANCELLED"
            c["resolution"]=event.get("cancellation_reason","")
        elif name=="JZ_CONFIRMED_EXPERIMENTAL":
            c["candidate_status"]="COMMITTED"
            c["committed_character"]=event["character"]

    def bout(self,entries,rows,scale,metrics,validation):
        if self.current is None:
            raise ValueError("No diagnostic onset for completed bout")
        c=self.current
        by_id={r[0]:r for r in c["observations"]}
        if any(e[0] not in by_id for e in entries):
            raise ValueError("Missing captured timestamps; cannot reconstruct exact bout")
        c["capture"]=[(e[0],by_id[e[0]][1],e[2],landmarks_to_array(e[1]).copy()) for e in entries]
        c["detector_scale_px"]=scale
        c["motion_metrics"]=metrics
        c["technical_validation"]=validation

    def model_input(self,flat,probabilities):
        if self.current is None:
            raise ValueError("No diagnostic attempt for model input")
        if flat.shape!=(1,1104) or not np.isfinite(flat).all():
            raise ValueError("Unexpected actual J/Z model input")
        self.current["features"]=flat.reshape(16,69,order="C").copy()
        self.current["model_probabilities"]=probabilities.tolist()

    def after_update(self):
        c=self.current
        if c and (c["candidate_status"] in ("CANCELLED","COMMITTED")
                  or c.get("result_status") in ("REJECTED","UNCERTAIN")):
            self.finish("ATTEMPT_RESOLVED")

    def finish(self,reason):
        c=self.current
        if c is None:
            return
        # Clear before writing so an I/O error cannot attach the next attempt
        # to stale data. Partial output is retained, never deleted/overwritten.
        self.current=None
        self.completed_keys.add(c["key"])
        generation,aid=c["key"]
        folder=self.path/f"generation_{generation:03d}_attempt_{aid:05d}"
        folder.mkdir(exist_ok=False)
        observed=c.pop("observations")
        capture=c.pop("capture",None)
        if capture is None:
            capture=[r for r in observed if r[3] is not None]
            c["capture_provenance"]="OBSERVED_TO_RESOLUTION_NO_MODEL_BOUT"
        else:
            c["capture_provenance"]="EXACT_ENTRIES_PASSED_TO_INFERENCE"
        features=c.pop("features",np.empty((0,69),dtype=float))
        def arrays(rows):
            valid=[r for r in rows if r[3] is not None]
            return (np.asarray([r[0] for r in valid],dtype=np.int64),
                    np.asarray([r[1] for r in valid],dtype=np.float64),
                    np.asarray([r[2] for r in valid],dtype=np.int64).reshape(-1,2),
                    np.asarray([r[3] for r in valid],dtype=np.float64).reshape(-1,21,3))
        ids,times,sizes,hands=arrays(capture)
        oids,otimes,osizes,ohands=arrays(observed)
        selected=(c.get("model_result") or {}).get("selected_original_frame_indices",[])
        npz=folder/"replay_data.npz"
        with npz.open("xb") as stream:
            np.savez(stream,frame_indices=ids,timestamps=times,frame_sizes=sizes,landmarks=hands,
                     selected_frame_indices=np.asarray(selected,dtype=np.int64),features=features,
                     observed_frame_indices=oids,observed_timestamps=otimes,observed_sizes=osizes,observed_landmarks=ohands,
                     timeline_indices=np.asarray([r[0] for r in observed],dtype=np.int64),
                     timeline_timestamps=np.asarray([r[1] for r in observed]),
                     timeline_valid=np.asarray([r[3] is not None for r in observed],dtype=bool))
        c.update(schema_version=1,artifact_kind="DIAGNOSTIC_NOT_TRAINING_DATA",session_id=self.session_id,
                 session_generation=generation,attempt_id=aid,saved_at_utc=datetime.now(timezone.utc).isoformat(),
                 finalization_reason=reason,model_sha256=self.model_sha256,config=self.config,code_sha256=self.code_hashes,
                 environment=self.environment,
                 config_sha256=hashlib.sha256(json.dumps(self.config,sort_keys=True,allow_nan=False).encode()).hexdigest(),
                 inference_mode="ABC/IMAGE",mirror_convention="RAW_NO_FLIP; DISPLAY_MIRRORED_SEPARATELY",
                 array_sha256=sha256(npz),feature_shape=list(features.shape),valid_frame_count=len(ids),
                 capture_start_frame=int(ids[0]) if len(ids) else None,capture_end_frame=int(ids[-1]) if len(ids) else None,
                 capture_start_timestamp=float(times[0]) if len(times) else None,
                 capture_end_timestamp=float(times[-1]) if len(times) else None,
                 capture_duration_seconds=float(times[-1]-times[0]) if len(times) else None,
                 capture_seconds_after_onset=float(times[-1]-c["onset_timestamp"]) if len(times) else None,
                 pre_roll_frame_indices=[int(i) for i in ids if i<c["onset_frame"]],
                 tracking_gap_frame_indices=[r[0] for r in observed if r[3] is None],
                 replay_available=bool(len(ids)>=16 and features.shape==(16,69) and not c.get("diagnostic_error")),
                 human_label="UNREVIEWED",human_labels_allowed=list(HUMAN_LABELS))
        if len(ids):
            c["capture_trajectory"]=trajectory_diagnostics(hands,sizes,ids,times)
        if len(oids):
            c["observed_trajectory_through_resolution"]=trajectory_diagnostics(ohands,osizes,oids,otimes)
            endpoint=c.get("endpoint_frame")
            if endpoint in oids:
                pos=int(np.flatnonzero(oids==endpoint)[0])
                xy=ohands[:,:,:2]*osizes[:,None,:]
                center=xy[:,[0,5,9,17]].mean(axis=1)
                scale=c["observed_trajectory_through_resolution"]["palm_scale_px"][pos]
                c["post_endpoint_stability"]={"start_frame":endpoint,"through_frame":int(oids[-1]),
                    "observation_count":len(oids)-pos,
                    "max_palm_center_drift_px":float(np.max(np.linalg.norm(center[pos:]-center[pos],axis=1))),
                    "reference_scale_px":scale,"note":"Raw observations, no new acceptance threshold; gaps remain unverified"}
        if features.shape==(16,69):
            c["geometry_predicates"]=geometry_predicates(features,include_series=True)
        with (folder/"attempt.json").open("x",encoding="utf-8") as stream:
            json.dump(c,stream,ensure_ascii=False,allow_nan=False,indent=2)
        self.writer.writerow(dict(session_id=self.session_id,session_generation=generation,attempt_id=aid,
            human_label="UNREVIEWED",note="",model_prediction=(c.get("model_result") or {}).get("prediction",""),
            geometry_candidates=json.dumps(c.get("geometry_predicates",{}).get("geometry_candidates",[])),
            candidate_status=c["candidate_status"],committed_character=c["committed_character"],
            attempt_path=folder.relative_to(self.path).as_posix()))
        self.stream.flush()

    def close(self):
        try:
            self.safe(self.finish,"SESSION_END")
        finally:
            self.stream.close()


def load_attempt(folder):
    """Strict, pickle-free reader. Old event-only logs are not replay artefacts."""
    folder=Path(folder)
    metadata=json.loads((folder/"attempt.json").read_text(encoding="utf-8"))
    missing=[name for name in ("schema_version","session_id","session_generation","attempt_id","replay_available") if name not in metadata]
    if missing:
        raise ValueError("Missing replay metadata: "+", ".join(missing))
    if metadata.get("schema_version")!=1:
        raise ValueError("Unsupported/missing schema_version")
    if not metadata.get("replay_available"):
        raise ValueError("Exact replay unavailable: "+metadata.get("diagnostic_error","missing capture/actual model features"))
    if sha256(folder/"replay_data.npz")!=metadata.get("array_sha256"):
        raise ValueError("Replay NPZ SHA256 mismatch")
    required=("frame_indices","timestamps","frame_sizes","landmarks","selected_frame_indices","features")
    with np.load(folder/"replay_data.npz",allow_pickle=False) as archive:
        missing=[k for k in required if k not in archive]
        if missing:
            raise ValueError("Missing replay fields: "+", ".join(missing))
        optional=("observed_frame_indices","observed_timestamps","observed_sizes","observed_landmarks",
                  "timeline_indices","timeline_timestamps","timeline_valid")
        arrays={k:archive[k].copy() for k in (*required,*optional) if k in archive}
    ids,times=arrays["frame_indices"],arrays["timestamps"]
    if (ids.ndim!=1 or ids.dtype.kind not in "iu" or len(ids)<16 or np.any(ids<0) or np.any(np.diff(ids)<=0)
            or times.shape!=ids.shape or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or arrays["landmarks"].shape!=(len(ids),21,3) or not np.isfinite(arrays["landmarks"]).all()
            or arrays["frame_sizes"].shape!=(len(ids),2) or arrays["frame_sizes"].dtype.kind not in "iu"
            or np.any(arrays["frame_sizes"]<=0) or arrays["features"].shape!=(16,69)
            or not np.isfinite(arrays["features"]).all()):
        raise ValueError("Invalid replay shapes, finite values or chronological order")
    selected=sample_valid_frame_indices(ids.tolist())
    if selected!=arrays["selected_frame_indices"].tolist():
        raise ValueError("Saved selected frame indices do not match canonical valid-frame sampler")
    if arrays["selected_frame_indices"].dtype.kind not in "iu":
        raise ValueError("Selected frame indices must be integers")
    if selected!=(metadata.get("model_result") or {}).get("selected_original_frame_indices"):
        raise ValueError("Metadata/model selected indices mismatch")
    return metadata,arrays


def geometry_replay(metadata,arrays):
    mapping={int(i):n for n,i in enumerate(arrays["frame_indices"])}
    positions=[mapping[int(i)] for i in arrays["selected_frame_indices"]]
    hands=[[SimpleNamespace(x=x,y=y,z=z) for x,y,z in hand] for hand in arrays["landmarks"][positions]]
    rebuilt,scale=build_sequence_features(hands,[tuple(map(int,s)) for s in arrays["frame_sizes"][positions]])
    equal=np.array_equal(rebuilt,arrays["features"])
    predicates=geometry_predicates(rebuilt,include_series=True)
    return {"feature_shape":list(rebuilt.shape),"flatten_order":"C","flattened_features":1104,
            "exact_feature_match":bool(equal),"max_feature_difference":float(np.max(np.abs(rebuilt-arrays["features"]))),
            "builder_scale_px":scale,"geometry_predicates":predicates,
            "geometry_matches_saved":predicates==metadata.get("geometry_predicates"),
            "code_hash_matches":{p:sha256(ROOT/p)==h for p,h in metadata.get("code_sha256",{}).items() if p in CODE_FILES}}


def model_replay(metadata,arrays,model,model_hash):
    if model_hash!=metadata["model_sha256"]:
        raise ValueError("Model hash differs from captured inference model")
    if list(model.classes_)!=["J","Z"]:
        raise ValueError("Model classes/order must be J,Z")
    probabilities=np.asarray(model.predict_proba(arrays["features"].reshape(1,1104,order="C")))
    if probabilities.shape!=(1,2) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid replay model probabilities")
    previous=np.asarray(metadata.get("model_probabilities"))
    if previous.shape!=(1,2):
        raise ValueError("Missing exact recorded model_probabilities")
    label=str(model.classes_[int(np.argmax(probabilities[0]))])
    return {"prediction":label,"probabilities":probabilities.tolist(),
            "prediction_match":label==metadata["model_result"]["prediction"],
            "probabilities_exact_match":bool(np.array_equal(previous,probabilities)),
            "max_probability_difference":float(np.max(np.abs(previous-probabilities)))}
