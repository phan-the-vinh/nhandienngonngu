"""User-run, four-file SHADOW timeline audit. No training or dataset output.

--check-only hashes exactly the named J/Z counterparts, without video decoding,
MediaPipe/model execution or output creation. --audit requires confirmation.
The actual integrated VIDEO/IMAGE detection, classifiers and coordinator are
reused; reported onset is DETECTOR onset, not annotated linguistic onset.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from app.inference.jz_attempt_capture import sha256,checked_output
from app.inference.z_start_pose_experimental import TemporalReadyEvidence
from app.inference.unified_abc123_jz_auto_experiment import (
    create_hand_landmarker,detect_static,detect_frame,prepare_inference_frame,
    HAND_MODEL_PATH,DEFAULT_MODEL,load_official_artifacts,StaticAslInference,
    DigitPredictor,load_digit_model,load_experiment_model,UnifiedAutoCoordinator)

SOURCE_ROOT=ROOT/"data/raw/dynamic/external/signnteam_jz/SigNN Video Data"
REFERENCES={
    "3.avi":"87c89426eb9fac9deae9d36962693995cd8ae3c7a37fdc8b07957dcf83745976",
    "29.avi":"a3ff35a6a27bbaecdc083cce9d50a1f2f6cb99bcef0c01018928cb38d1dfa8bc",
    "68.avi":"61fc722f17bf7a30a803e314682efa4ab6bc5ed7b888cbee4e7ce9aee84bf0af",
    "380.avi":"28db72cebaa458a0b1f305e3385ff443c299f69e2019ec602e64228a98dfd20c",
}


def preflight(source_root=SOURCE_ROOT):
    root=Path(source_root).resolve()
    matched=[]
    counterparts=[]
    for filename,expected in REFERENCES.items():
        hits=[]
        for label in ("J","Z"):
            path=(root/label/filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Source path escapes dataset root")
            if path.is_file():
                digest=sha256(path)
                row={"filename":filename,"source_path":str(path),"source_folder_label":label,
                     "sha256":digest,"matches_user_reference":digest==expected}
                counterparts.append(row)
                if digest==expected:
                    hits.append(row)
        if len(hits)!=1:
            raise ValueError(f"{filename}: expected exactly one hash match, found {len(hits)}")
        matched.append({**hits[0],"provenance_status":"USER_REFERENCE_SHA256_MATCHED",
                        "gesture_ground_truth":"SOURCE_LABEL_ONLY_NOT_INDEPENDENT_ASL_VALIDATION"})
    if not HAND_MODEL_PATH.is_file():
        raise FileNotFoundError(HAND_MODEL_PATH)
    return matched,counterparts


def summarize_timeline(rows,events,fps):
    """Summarize actual contiguous frame observations, not selected 16 frames."""
    counts={status:sum(r["pose_status"]==status for r in rows) for status in ("VALID","INVALID","UNCERTAIN")}
    ranges=[]
    ready=[]
    for r in rows:
        if r["pose_status"]=="VALID":
            if ranges and r["frame_id"]==ranges[-1]["end_frame"]+1:
                ranges[-1]["end_frame"]=r["frame_id"]
            else:
                ranges.append({"start_frame":r["frame_id"],"end_frame":r["frame_id"]})
        item=r.get("ready")
        if item and item not in ready:
            ready.append(item)
    for item in ranges:
        item.update(frame_count=item["end_frame"]-item["start_frame"]+1,
                    duration_sample_span_seconds=(item["end_frame"]-item["start_frame"])/fps)
    onsets=[]
    for event in events:
        if event["event"]=="STATE_TRANSITION" and event.get("new_state")=="CAPTURING" and event.get("previous_state")!="CAPTURING":
            frame=event["frame_id"]
            # A READY seen sometime earlier may have expired. The frozen
            # per-attempt association, not min(ready timestamps), is decisive.
            row=next((r for r in rows if r["frame_id"]==frame),{})
            attempt=row.get("shadow_attempt") or {}
            onsets.append({"attempt_id":event.get("attempt_id"),"frame_id":frame,"seconds":frame/fps,
                           "ready_at_onset":attempt.get("ready"),
                           "movement_since":attempt.get("movement_since"),
                           "start_pose_linked":attempt.get("start_pose_linked"),
                           "temporal_evidence":(row.get("temporal_evidence") or {}).get("attempt"),
                           "reason":attempt.get("reason","NO_FROZEN_ASSOCIATION_AVAILABLE")})
    first=next((r["frame_id"] for r in rows if r["pose_status"]=="VALID"),None)
    return {"decoded_frame_count":len(rows),"pose_counts":counts,"first_valid_pose_frame":first,
            "first_valid_pose_seconds":first/fps if first is not None else None,
            "contiguous_valid_pose_ranges":ranges,"ready_events":ready,"detector_onsets":onsets,
            "z_candidate_events":[{"frame_id":e["frame_id"],"attempt_id":e.get("attempt_id")}
                                  for e in events if e["event"]=="JZ_CANDIDATE_CREATED" and e.get("candidate")=="Z"],
            "true_motion_onset":"UNANNOTATED_REQUIRES_HUMAN_REVIEW",
            "missed_gesture_start":"UNDETERMINED_WITHOUT_HUMAN_ONSET",
            "semantic_acceptance":"UNCHANGED_SHADOW_ONLY","recall":"NOT_ESTIMABLE_FROM_DETECTED_ATTEMPTS"}


def audit_video(reference,folder,model,model_hash):
    source=Path(reference["source_path"])
    if sha256(source)!=reference["sha256"]:
        raise ValueError("Source changed after preflight")
    cap=cv2.VideoCapture(str(source))  # File path ONLY; never a camera index.
    try:
        if not cap.isOpened():
            raise OSError("Cannot open "+str(source))
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        reported_count=float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(reported_count) or reported_count<0:
            reported_count=None
        if not np.isfinite(fps) or fps<=0:
            raise ValueError("Invalid source FPS; no invented clock")
        letters=StaticAslInference(load_official_artifacts(DEFAULT_MODEL))
        digits=DigitPredictor(load_digit_model())
        events=[]
        coordinator=UnifiedAutoCoordinator(letters,digits,model,z_start_pose_shadow=True,event_sink=events.append)
        rows=[]
        with (create_hand_landmarker(HAND_MODEL_PATH,running_mode="VIDEO",num_hands=1,
                                    min_tracking_confidence=.5) as video_task,
              create_hand_landmarker(HAND_MODEL_PATH,running_mode="IMAGE",num_hands=1) as image_task,
              (folder/"frame_timeline.jsonl").open("x",encoding="utf-8") as stream):
            index=0
            last_ms=-1
            while True:
                ok,frame=cap.read()
                if not ok:
                    break
                if frame is None or frame.size==0:
                    raise ValueError(f"Empty decoded frame at {index}")
                now=index/fps
                ms=max(last_ms+1,int(now*1000));last_ms=ms
                video_error=""
                try:
                    static_hand=detect_static(video_task,prepare_inference_frame(frame,"ABC"),ms)
                except (ValueError,RuntimeError,TypeError,AttributeError) as exc:
                    static_hand=None;video_error=str(exc)
                raw_hand,count,error=detect_frame(image_task,frame)
                coordinator.update(index,static_hand,raw_hand,frame.shape,now,
                                   dynamic_error=error,raw_hand_count=count,static_error=video_error)
                report=coordinator.z_start_pose_report or {}
                pose=report.get("pose") or {}
                row={"frame_id":index,"seconds":now,"image_hand_count":count,
                     "video_hand_detected":static_hand is not None,"image_error":error,"video_error":video_error,
                     "pose_status":pose.get("pose_status","UNCERTAIN"),"pose_score":pose.get("pose_score"),
                     "predicates":pose.get("predicates",{}),"pose_reason":pose.get("reason"),
                     "ready":report.get("ready"),"shadow_attempt":report.get("attempt"),
                     "temporal_evidence":report.get("temporal_evidence"),
                     "shadow_state":report.get("state"),"detector_state":coordinator.session.last_detector_state,
                     "raw_prediction":coordinator.raw_dynamic_prediction,"candidate":coordinator.display_dynamic_result}
                rows.append(row)
                stream.write(json.dumps(row,allow_nan=False)+"\n")
                index+=1
        if not rows:
            raise ValueError("No frames decoded")
        summary=summarize_timeline(rows,events,fps)
        summary.update(reference=reference,fps=fps,reported_frame_count_not_decode_truth=reported_count,
                       decode_count_matches_header=(len(rows)==reported_count) if reported_count is not None else None,
                       clock="original frame index / source reported FPS; not wall clock",
                       model_sha256=model_hash,task_sha256=sha256(HAND_MODEL_PATH),
                       pose_settings=vars(coordinator.z_start_pose_shadow.settings),
                       camera_opened=False,training_performed=False,dataset_created=False,
                       prefix_history="clip starts a fresh session; pre-clip readiness is UNKNOWN",
                       hand_identity="UNVERIFIED; num_hands=1 does not prove only one physical hand",
                       note="Read failure marks decode end; header mismatch needs review, never padded")
        if sha256(source)!=reference["sha256"]:
            raise ValueError("Source bytes changed during audit; do not use these diagnostics")
        with (folder/"summary.json").open("x",encoding="utf-8") as stream:
            json.dump(summary,stream,allow_nan=False,indent=2)
        with (folder/"events.jsonl").open("x",encoding="utf-8") as stream:
            for event in events: stream.write(json.dumps(event,allow_nan=False)+"\n")
        return summary
    finally:
        cap.release()


def replay_reports(run):
    """Replay ONLY saved pose/READY reports, not landmarks or recognition.

    Old timelines lack full per-frame joint geometry. A new association is a
    report-level hypothesis, conditional on continuity guards not rechecked.
    Never silently present this as a new MediaPipe/recognition audit.
    """
    run=checked_output(run)
    results=[]
    for filename,expected in REFERENCES.items():
        folder=(run/Path(filename).stem).resolve()
        if not folder.is_relative_to(run) or any(not (folder/name).resolve().is_relative_to(run)
                for name in ("summary.json","frame_timeline.jsonl","events.jsonl")):
            raise ValueError("Saved report path escapes run")
        summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
        reference=summary["reference"]
        if reference["filename"]!=filename or reference["sha256"]!=expected or not reference["matches_user_reference"]:
            raise ValueError("Saved reference identity mismatch: "+filename)
        rows=[json.loads(line) for line in (folder/"frame_timeline.jsonl").read_text(encoding="utf-8").splitlines()]
        events=[json.loads(line) for line in (folder/"events.jsonl").read_text(encoding="utf-8").splitlines()]
        fps=summary["fps"]
        if (not np.isfinite(fps) or fps<=0 or len(rows)!=summary["decoded_frame_count"]
                or [r["frame_id"] for r in rows]!=list(range(len(rows)))
                or any(abs(r["seconds"]-r["frame_id"]/fps)>1e-9 for r in rows)
                or any(r["pose_status"] not in ("VALID","INVALID","UNCERTAIN") for r in rows)):
            raise ValueError("Invalid saved timeline: "+filename)
        if summary["pose_counts"] != {s:sum(r['pose_status']==s for r in rows) for s in ('VALID','INVALID','UNCERTAIN')}:
            raise ValueError("Saved pose counts mismatch: "+filename)
        if any(not isinstance(e.get('frame_id'),int) or not 0<=e['frame_id']<len(rows) for e in events):
            raise ValueError("Saved event frame outside timeline")
        temporal=TemporalReadyEvidence()
        associations=[]
        by_frame={}
        for event in events:
            by_frame.setdefault(event["frame_id"],[]).append(event)
        generation=track_generation=0
        for row in rows:
            index,now=row["frame_id"],row["seconds"]
            old=row.get("shadow_attempt") or {}
            ready=row.get("ready")
            track_generation=(ready or old).get("track_generation",track_generation)
            frame_events=by_frame.get(index,[])
            generation=next((e["session_generation"] for e in frame_events if "session_generation" in e),generation)
            discontinuity=next((e.get("reason") for e in frame_events
                if e.get("reason") in ("PALM_POSITION_DISCONTINUITY","TEMPORAL_DISCONTINUITY")),"")
            pose={"pose_status":row["pose_status"],"predicates":row.get("predicates",{}),
                  "reason":row.get("pose_reason")}
            temporal.observe(index,now,pose,ready,track_generation=track_generation,
                session_generation=generation,tracking_valid=row["image_hand_count"]==1 and not row["image_error"]
                    and row.get('pose_reason', 'POSE_PREDICATES_NOT_MET') in ('POSE_PREDICATES_NOT_MET','POSE_HYPOTHESIS_MET'),
                discontinuity=discontinuity)
            for event in frame_events:
                if (event["event"]=="STATE_TRANSITION" and event.get("new_state")=="CAPTURING"
                        and event.get("previous_state")!="CAPTURING"):
                    temporal.begin_attempt(event["attempt_id"],index,now,old.get("movement_since"),old.get("start_pose_linked"))
                    associations.append(temporal.snapshot()["attempt"])
            for event in frame_events:
                if event["event"]=="ATTEMPT_FINISHED":
                    temporal.end_attempt("SAVED_ATTEMPT_FINISHED")
            temporal.finish_frame(index,now)
        results.append({"video": "Z/"+filename,"pose_counts":summary["pose_counts"],
            "associations":associations,"true_motion_onset":summary["true_motion_onset"],
            "continuity_guard_recheck":"UNAVAILABLE_FULL_PER_FRAME_JOINT_GEOMETRY_NOT_SAVED",
            "association_scope":"REPORT_LEVEL_HYPOTHESIS_NOT_NEW_RECOGNITION_RESULT"})
    return {"status":"SHADOW_REPORT_REPLAY_ONLY","acceptance_effect":False,
            "files_written":0,"mediapipe_executed":False,"video_decoded":False,"videos":results}


ANNOTATION_MARKS=("preparation_start","preparation_stable","true_z_motion_start","true_z_motion_end")


def annotation_record(reference,frame_count,fps,reviewer,marks,uncertain_ranges):
    """Validate human marks without enforcing READY-before-motion as truth."""
    if not reviewer.strip() or frame_count<=0 or not np.isfinite(fps) or fps<=0:
        raise ValueError("Reviewer and actual decoded frame count/FPS are required")
    if set(marks)!=set(ANNOTATION_MARKS):
        raise ValueError("Unexpected annotation fields")
    def valid_frame(value):
        return isinstance(value,int) and not isinstance(value,bool) and 0<=value<frame_count
    if any(value is not None and not valid_frame(value) for value in marks.values()):
        raise ValueError("Human annotation frame outside decoded clip")
    for first,last in (("preparation_start","preparation_stable"),("true_z_motion_start","true_z_motion_end")):
        if marks[first] is not None and marks[last] is not None and marks[first]>marks[last]:
            raise ValueError("Human annotation range reversed")
    if any(len(r)!=2 or not all(valid_frame(v) for v in r) or r[0]>r[1] for r in uncertain_ranges):
        raise ValueError("Invalid uncertain/occluded frame range")
    return {"schema_version":1,"reference":reference,"reviewer":reviewer,
        "annotation_source":"HUMAN_FRAME_REVIEW_NOT_DETECTOR_ONSET","frame_index_base":0,
        "frame_ranges_inclusive":True,"decoded_frame_count":frame_count,"fps":fps,
        "marks":dict(marks),"uncertain_or_occluded_ranges":list(uncertain_ranges),
        "annotation_status":"COMPLETE" if all(v is not None for v in marks.values()) else "PARTIAL",
        "asl_label_independently_verified":False,"acceptance_decision_changed":False}


def annotate_reference(reference,output,reviewer):
    """User-only frame viewer. Original pixels/orientation, no MediaPipe."""
    source=Path(reference["source_path"])
    if sha256(source)!=reference["sha256"]:
        raise ValueError("Reference changed before human annotation")
    cap=cv2.VideoCapture(str(source))
    frames=[]
    try:
        if not cap.isOpened(): raise OSError("Cannot open reference video")
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        while True:
            ok,frame=cap.read()
            if not ok: break
            if frame is None or not frame.size: raise ValueError("Empty decoded frame")
            frames.append(frame)
    finally:
        cap.release()
    if not frames or not np.isfinite(fps) or fps<=0: raise ValueError("Invalid video frames/FPS")
    marks={name:None for name in ANNOTATION_MARKS}
    uncertain=[]
    uncertainty_start=None
    position=0
    window="Human Z temporal annotation / original orientation"
    message=""
    try:
        while True:
            # Footer is outside source pixels. Frame stepping, no interpolation.
            image=cv2.copyMakeBorder(frames[position],0,145,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
            lines=[f"{reference['filename']}  original frame {position} / {len(frames)-1}  {position/fps:.3f}s",
                   "A/D previous/next | 1 prep start | 2 prep stable | 3 Z start | 4 Z end",
                   "5 uncertainty start | 6 uncertainty end | S save new JSON | Q cancel",
                   " ".join(f"{n}={marks[k]}" for n,k in enumerate(ANNOTATION_MARKS,1)),message]
            for n,line in enumerate(lines):
                cv2.putText(image,line,(8,frames[position].shape[0]+22+26*n),cv2.FONT_HERSHEY_SIMPLEX,.44,(0,0,0),1,cv2.LINE_AA)
            cv2.imshow(window,image)
            key=cv2.waitKey(50)&0xff
            if key in (27,ord('q')): return None
            if key==ord('a'): position=max(0,position-1)
            elif key==ord('d'): position=min(len(frames)-1,position+1)
            elif ord('1')<=key<=ord('4'): marks[ANNOTATION_MARKS[key-ord('1')]]=position
            elif key==ord('5'): uncertainty_start=position
            elif key==ord('6'):
                if uncertainty_start is not None and position>=uncertainty_start:
                    uncertain.append([uncertainty_start,position]);uncertainty_start=None
                else: message="Select an uncertainty start <= end"
            elif key==ord('s'):
                if uncertainty_start is not None:
                    message="Close the uncertainty range with 6 before saving";continue
                try: record=annotation_record(reference,len(frames),fps,reviewer,marks,uncertain)
                except ValueError as exc: message=str(exc);continue
                if sha256(source)!=reference['sha256']: raise ValueError("Source changed during annotation")
                output.mkdir(parents=True,exist_ok=True)
                path=output/(Path(reference['filename']).stem+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ_')+uuid4().hex[:8]+'.json')
                record['saved_at_utc']=datetime.now(timezone.utc).isoformat()
                with path.open('x',encoding='utf-8') as stream: json.dump(record,stream,indent=2,allow_nan=False)
                return path
    finally:
        cv2.destroyAllWindows()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only",action="store_true")
    mode.add_argument("--audit",action="store_true")
    mode.add_argument("--replay-reports",type=Path,metavar="SAVED_RUN",help="Read saved JSONL only; no video/model/MediaPipe or writes")
    mode.add_argument("--annotate",action="store_true",help="User-run frame stepping; save separate human annotation JSON")
    parser.add_argument("--reference",choices=list(REFERENCES))
    parser.add_argument("--reviewer",default="")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"data/experiments/dynamic_jz_v1/z_start_pose_reference_audit")
    args=parser.parse_args(argv)
    try:
        output=checked_output(args.output_dir)
        if args.replay_reports:
            print(json.dumps(replay_reports(args.replay_reports),indent=2,allow_nan=False))
            return 0
        references,counterparts=preflight()
        if args.check_only:
            print(json.dumps({"status":"PASS","matches":references,"counterparts":counterparts,
                              "camera_opened":False,"mediapipe_executed":False,"files_written":0},indent=2))
            return 0
        if args.annotate:
            if args.reference is None or not args.reviewer.strip():
                raise ValueError("--annotate requires --reference and --reviewer")
            if input("Open one reference video for human frame annotation? [y/N] ").strip().lower()!='y':
                print("CANCELLED; no output created");return 0
            reference=next(r for r in references if r['filename']==args.reference)
            path=annotate_reference(reference,checked_output(output/'human_annotations'),args.reviewer)
            print("ANNOTATION="+str(path) if path else "CANCELLED; no output created")
            return 0
        if input("Run four-video SHADOW timeline audit? [y/N] ").strip().lower()!="y":
            print("CANCELLED; no output created")
            return 0
        model,model_hash=load_experiment_model()
        run=output/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_")+uuid4().hex[:8])
        run.mkdir(parents=True,exist_ok=False)
        for reference in references:
            folder=run/Path(reference["filename"]).stem
            folder.mkdir(exist_ok=False)
            audit_video(reference,folder,model,model_hash)
        print("AUDIT_OUTPUT="+str(run))
        return 0
    except (OSError,ValueError,RuntimeError,EOFError) as exc:
        print("AUDIT_ERROR="+str(exc),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
