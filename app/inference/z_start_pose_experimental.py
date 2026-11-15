"""Phase-A Z start-pose hypotheses. SHADOW ONLY, never an acceptance gate.

Angles/ratios are invariant to translation, scale, rigid pseudo-3D rotation and
reflection. MediaPipe image Z is relative/pseudo depth, not metric 3D; viewpoint
and occlusion errors remain unverified. No static classifier label is consumed.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy

import numpy as np


@dataclass(frozen=True)
class ZStartPoseSettings:
    index_pip_min_degrees: float = 155.
    index_dip_min_degrees: float = 150.
    index_reach_min: float = .80
    folded_pip_max_degrees: float = 140.
    folded_reach_max: float = .70
    uncertain_angle_margin_degrees: float = 8.
    uncertain_reach_margin: float = .05
    stable_frames: int = 3
    stable_seconds: float = .10
    max_joint_angle_step_degrees: float = 18.
    max_missing_frames: int = 6
    max_gap_seconds: float = .4
    max_palm_step_ratio: float = .65
    min_presence_score: float = .5


def _angle(a,b,c):
    u,v=a-b,c-b
    denominator=np.linalg.norm(u)*np.linalg.norm(v)
    if denominator <= 1e-12:
        raise ValueError("DEGENERATE_FINGER_BONES")
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/denominator,-1.,1.))))


def _geometry(hand,size):
    if isinstance(hand,np.ndarray):
        raw=np.asarray(hand,dtype=np.float64)
    else:
        raw=np.asarray([(p.x,p.y,p.z) for p in hand],dtype=np.float64)
    if raw.shape!=(21,3) or not np.isfinite(raw).all():
        raise ValueError("EXPECTED_21_FINITE_LANDMARKS")
    width,height=size
    if not np.isfinite([width,height]).all() or min(width,height)<=0:
        raise ValueError("INVALID_FRAME_SIZE")
    # X/Y account for aspect ratio. MP Z uses approximately image-X units;
    # this is a documented pseudo-3D diagnostic, not camera depth recovery.
    points=raw*np.asarray([width,height,width])
    palm=points[[0,5,9,17]].mean(axis=0)
    scale=float(np.median(np.linalg.norm(points[[5,9,17]]-points[0],axis=1)))
    if scale <= 1e-6:
        raise ValueError("DEGENERATE_PALM")
    fingers={}
    for name,base in (("index",5),("middle",9),("ring",13),("pinky",17)):
        p=points[base:base+4]
        chain=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        if chain <= 1e-6:
            raise ValueError("DEGENERATE_FINGER_CHAIN")
        fingers[name]={"pip_angle_degrees":_angle(*p[:3]),"dip_angle_degrees":_angle(*p[1:]),
                       "reach_ratio":float(np.linalg.norm(p[3]-p[0])/chain)}
    # Thumb is measured, not constrained: do not silently exclude variants.
    thumb={"ip_angle_degrees":_angle(points[2],points[3],points[4]),
           "tip_to_index_mcp_palm_ratio":float(np.linalg.norm(points[4]-points[5])/scale),
           "acceptance_constraint":"NONE_UNCALIBRATED_VARIANTS"}
    return fingers,thumb,palm,scale


class ZStartPoseValidator:
    """Per-frame hypothesis: pose_score is NOT a calibrated probability."""
    def __init__(self,settings=ZStartPoseSettings()):
        values=asdict(settings)
        if any(not np.isfinite(v) or v<=0 for v in values.values()):
            raise ValueError("Z pose hypotheses must be finite and positive")
        if not isinstance(settings.stable_frames,int) or not isinstance(settings.max_missing_frames,int):
            raise ValueError("Frame counts must be integers")
        if (settings.index_reach_min>1 or settings.folded_reach_max>1 or settings.min_presence_score>1
                or settings.folded_pip_max_degrees>=settings.index_pip_min_degrees
                or settings.index_pip_min_degrees>180 or settings.index_dip_min_degrees>180):
            raise ValueError("Invalid angle/ratio hypotheses")
        self.settings=settings

    def evaluate(self,hand,size,*,hand_count=1,presence_score=None,handedness=None):
        out={"pose_status":"UNCERTAIN","pose_score":0.,"reason":"","predicates":{},
             "policy":"SHADOW_ONLY_UNCALIBRATED","landmark_confidence":"NOT_PROVIDED",
             "handedness":handedness,"identity_status":"NOT_VERIFIED_BY_HANDEDNESS"}
        if hand is None or hand_count not in (None,1):
            out["reason"]="MISSING_OR_MULTIPLE_HANDS"
            return out
        try:
            fingers,thumb,_,_=_geometry(hand,size)
        except (ValueError,TypeError,AttributeError,IndexError) as exc:
            out["reason"]=str(exc)
            return out
        if presence_score is not None:
            out["landmark_confidence"]="CALLER_SUPPLIED_PRESENCE_NOT_HANDEDNESS_CONFIDENCE"
            if not np.isfinite(presence_score) or not 0<=presence_score<=1 or presence_score<self.settings.min_presence_score:
                out["reason"]="LOW_OR_INVALID_PRESENCE_SCORE"
                return out
        s=self.settings
        measures=[]
        def check(name,value,threshold,minimum,margin,units):
            delta=(value-threshold) if minimum else (threshold-value)
            status="PASS" if delta>=0 else "UNCERTAIN" if delta>=-margin else "FAIL"
            out["predicates"][name]={"value":value,"threshold":threshold,"operator":">=" if minimum else "<=",
                                      "status":status,"units":units}
            measures.append(1. if status=="PASS" else .5 if status=="UNCERTAIN" else 0.)
        index=fingers["index"]
        check("index_pip_extended",index["pip_angle_degrees"],s.index_pip_min_degrees,True,s.uncertain_angle_margin_degrees,"degrees")
        check("index_dip_extended",index["dip_angle_degrees"],s.index_dip_min_degrees,True,s.uncertain_angle_margin_degrees,"degrees")
        check("index_reach",index["reach_ratio"],s.index_reach_min,True,s.uncertain_reach_margin,"tip-MCP distance / chain length")
        for name in ("middle","ring","pinky"):
            f=fingers[name]
            # Either actual joint flexion OR a shortened chain reach supports
            # folding. Near thresholds remains UNCERTAIN, not quietly VALID.
            pip=f["pip_angle_degrees"]
            reach=f["reach_ratio"]
            passed=pip<=s.folded_pip_max_degrees or reach<=s.folded_reach_max
            uncertain=pip<=s.folded_pip_max_degrees+s.uncertain_angle_margin_degrees or reach<=s.folded_reach_max+s.uncertain_reach_margin
            status="PASS" if passed else "UNCERTAIN" if uncertain else "FAIL"
            out["predicates"][name+"_folded"]={"value":{"pip_degrees":pip,"reach_ratio":reach},
                "threshold":{"pip_max_degrees":s.folded_pip_max_degrees,"reach_max":s.folded_reach_max},
                "operator":"PIP <= threshold OR reach <= threshold","status":status,"units":"degrees; ratio"}
            measures.append(1. if passed else .5 if uncertain else 0.)
        statuses=[v["status"] for v in out["predicates"].values()]
        out.update(pose_status="INVALID" if "FAIL" in statuses else "UNCERTAIN" if "UNCERTAIN" in statuses else "VALID",
                   pose_score=float(np.mean(measures)),finger_geometry=fingers,thumb=thumb,
                   reason="POSE_HYPOTHESIS_MET" if all(v=="PASS" for v in statuses) else "POSE_PREDICATES_NOT_MET",
                   quality_note="Finite geometry only; per-landmark visibility/confidence and true hand identity are unavailable")
        return out


class TemporalReadyEvidence:
    """SHADOW comparison, not a semantic or acceptance decision.

    No uncalibrated TTL: eligibility requires a continuously refreshed VALID
    observation, or the immediately preceding VALID observation in the SAME
    onset-frame transaction. An UNCERTAIN boundary is never carried forward.
    Historical evidence is retained only as diagnostics, never as reusable READY.
    """
    def __init__(self):
        self.reset(0)

    def reset(self, generation):
        self.generation = generation
        self.evidence = None
        self.retired = None
        self.attempt = None
        self.last_association = None
        self.last_observation = None
        self.reason = "SESSION_RESET"
        self.state = "EMPTY"
        self.generation_started_at = None

    def invalidate(self, reason, index=None, now=None):
        if self.evidence is not None:
            self.retired = {**deepcopy(self.evidence), "invalidation_reason": reason,
                            "invalidated_frame": index, "invalidated_at": now}
        self.evidence = None
        self.state = "EMPTY"
        self.reason = reason

    def observe(self, index, now, pose, legacy_ready, *, track_generation,
                session_generation=0, tracking_valid=True, discontinuity=""):
        if session_generation != self.generation:
            self.invalidate("SESSION_GENERATION_CHANGED", index, now)
            self.attempt = None
            self.generation = session_generation
            self.last_observation = None
            self.generation_started_at = now
        if self.generation_started_at is None:
            self.generation_started_at = now
        previous = self.last_observation
        status = pose.get("pose_status", "UNCERTAIN")
        self.last_observation = {"frame_id": index, "timestamp": now,
                                 "pose_status": status, "track_generation": track_generation,
                                 "tracking_status": "OBSERVED_ONLY_IDENTITY_UNVERIFIED" if tracking_valid
                                     else "TRACKING_UNVERIFIED"}
        if previous and (index != previous["frame_id"] + 1 or now <= previous["timestamp"]):
            discontinuity = discontinuity or "FRAME_OR_TIME_DISCONTINUITY"
        if self.evidence and track_generation != self.evidence["track_generation"]:
            discontinuity = discontinuity or "TRACK_GENERATION_CHANGED"
        if not tracking_valid or discontinuity:
            self.invalidate(discontinuity or "TRACKING_LOST_OR_AMBIGUOUS", index, now)
            if self.attempt:
                self.attempt["continuity_status"] = "UNVERIFIED_GAP_OR_DISCONTINUITY"
            return
        # A frozen pre-onset association is history: pose during capture does
        # not retroactively erase it, nor create another preparation token.
        if self.attempt is not None:
            self.reason = "ATTEMPT_OPEN_NO_READY_REUSE"
            return
        if status == "INVALID":
            self.invalidate("POSE_INVALID_REQUIRES_FRESH_PREPARATION", index, now)
            return
        if status == "VALID":
            if legacy_ready is None or legacy_ready["timestamp"] < self.generation_started_at:
                self.invalidate("LEGACY_STABILITY_NOT_ESTABLISHED", index, now)
                return
            if self.evidence is None:
                self.evidence = {**dict(legacy_ready), "ready_created_at": legacy_ready["timestamp"],
                    "session_generation": session_generation,
                    "tracking_status": "OBSERVED_ONLY_IDENTITY_UNVERIFIED",
                    "pose_at_ready": "VALID", "linked_attempt_id": None}
            self.evidence.update(last_valid_frame=index, last_valid_timestamp=now,
                                 last_valid_pose=deepcopy(pose), boundary_frame=None)
            self.state, self.reason = "READY", "CONTINUOUS_VALID_PREPARATION"
            return
        if (self.evidence and previous and previous["pose_status"] == "VALID"
                and self.evidence["last_valid_frame"] == index - 1):
            self.evidence["boundary_frame"] = index
            self.state, self.reason = "ONSET_BOUNDARY_PENDING", "UNCERTAIN_CURRENT_FRAME_PREVIOUS_POSE_VALID"
        else:
            self.invalidate("UNCERTAIN_WITHOUT_ADJACENT_VALID_EVIDENCE", index, now)

    def begin_attempt(self, attempt_id, onset_frame, onset_time, movement_since, legacy_linked):
        evidence = deepcopy(self.evidence)
        current = self.last_observation or {}
        reason = "NO_ELIGIBLE_TEMPORAL_READY"
        if evidence:
            if evidence["session_generation"] != self.generation:
                reason = "SESSION_GENERATION_CHANGED"
            elif current.get("frame_id") != onset_frame or current.get("timestamp") != onset_time:
                reason = "ONSET_OBSERVATION_NOT_CURRENT"
            elif evidence["timestamp"] >= onset_time:
                reason = "READY_NOT_BEFORE_ONSET"
            elif movement_since is None or movement_since < evidence["timestamp"]:
                reason = "MOTION_ALREADY_STARTED_OR_UNAVAILABLE"
            elif (evidence["last_valid_frame"] != onset_frame
                  and evidence.get("boundary_frame") != onset_frame):
                reason = "PREPARATION_NOT_CONTINUOUS_TO_ONSET"
            else:
                reason = "PRE_ONSET_READY_LINKED_SHADOW_ONLY"
        linked = reason == "PRE_ONSET_READY_LINKED_SHADOW_ONLY"
        self.attempt = {"attempt_id": attempt_id, "session_generation": self.generation,
            "onset_frame": onset_frame, "onset_timestamp": onset_time, "movement_since": movement_since,
            "legacy_start_pose_linked": bool(legacy_linked), "temporal_start_pose_linked": linked,
            "ready_evidence": evidence, "reason": reason,
            "ready_to_onset_seconds": onset_time-evidence["timestamp"] if evidence else None,
            "pose_at_onset": current.get("pose_status"),
            "continuity_status": "OBSERVED_ONLY_IDENTITY_UNVERIFIED",
            "semantic_motion_status": "UNVERIFIED", "would_allow_z_candidate": False}
        if evidence:
            self.attempt["ready_evidence"]["linked_attempt_id"] = attempt_id if linked else None
        self.invalidate("CONSUMED_BY_ATTEMPT" if linked else "ATTEMPT_STARTED_NOT_LINKED", onset_frame, onset_time)
        self.state = "ATTEMPT_OPEN"
        self.last_association = deepcopy(self.attempt)

    def finish_frame(self, index, now):
        # This is a transaction boundary, not a guessed time-to-live threshold.
        if self.state == "ONSET_BOUNDARY_PENDING":
            self.invalidate("UNCERTAIN_FRAME_WITHOUT_ONSET", index, now)

    def end_attempt(self, reason="ATTEMPT_FINISHED_OR_REARM"):
        if self.attempt is not None:
            self.last_association = {**deepcopy(self.attempt), "completion_reason": reason}
        self.attempt = None
        self.invalidate(reason)

    def snapshot(self):
        return deepcopy({"policy": "SHADOW_ONLY_NO_ACCEPTANCE_EFFECT", "state": self.state,
            "reason": self.reason, "session_generation": self.generation,
            "evidence": self.evidence, "retired_evidence": self.retired,
            "attempt": self.attempt, "last_association": self.last_association,
            "last_observation": self.last_observation,
            "ttl_seconds": None, "ttl_policy": "NOT_CALIBRATED_NO_STALE_TOKEN_CARRY",
            "identity_status": "UNVERIFIED_NOT_HANDEDNESS_BASED"})


class ZStartPoseShadow:
    """Observational READY/onset association; never resets the real detector.

    Missing observations invalidate READY (no inferred identity across a gap).
    An open shadow attempt retains counts across short gaps but marks evidence
    unresolved. Generic onset before READY cannot become valid retrospectively.
    Motion-specific validation is intentionally UNVERIFIED in phase A.
    """
    def __init__(self,settings=ZStartPoseSettings()):
        self.validator=ZStartPoseValidator(settings)
        self.settings=settings
        self.reset()

    def reset(self, session_generation=0):
        self.state="IDLE"
        self.track_generation=0
        self.previous=None
        self.streak=[]
        self.ready=None
        self.last_pose=None
        self.attempt=None
        self.missing=0
        self.temporal = TemporalReadyEvidence()
        self.temporal.reset(session_generation)

    def observe(self,index,hand,size,now,*,hand_count=1,error="",session_generation=0):
        if session_generation != self.temporal.generation:
            self.reset(session_generation=session_generation)
        previous = self.previous
        old_generation = self.track_generation
        self._observe_legacy(index,hand,size,now,hand_count=hand_count,error=error)
        pose = self.last_pose
        tracking_valid = not error and "finger_geometry" in pose
        discontinuity = ""
        if old_generation != self.track_generation:
            discontinuity = "TRACK_OR_TEMPORAL_DISCONTINUITY"
        elif previous and tracking_valid:
            angles = np.asarray([f[k] for f in pose["finger_geometry"].values()
                                 for k in ("pip_angle_degrees","dip_angle_degrees")])
            if np.max(np.abs(angles-previous[4])) > self.settings.max_joint_angle_step_degrees:
                discontinuity = "JOINT_GEOMETRY_DISCONTINUITY"
        self.temporal.observe(index,now,pose,self.ready,track_generation=self.track_generation,
                              session_generation=session_generation,tracking_valid=tracking_valid,
                              discontinuity=discontinuity)
        return self.snapshot()

    def _observe_legacy(self,index,hand,size,now,*,hand_count=1,error=""):
        if (not isinstance(index,int) or isinstance(index,bool) or index<0
                or not np.isfinite(now)):
            raise ValueError("INVALID_SHADOW_FRAME_OR_TIMESTAMP")
        pose=self.validator.evaluate(hand if not error else None,size,hand_count=hand_count)
        self.last_pose=pose
        discontinuity=""
        if pose["reason"] in ("MISSING_OR_MULTIPLE_HANDS",) or error or "finger_geometry" not in pose:
            self.missing+=1
            self.streak=[]
            self.ready=None
            if self.attempt is not None:
                self.attempt["missing_frames"].append(index)
                self.attempt["continuity_status"]="UNVERIFIED_GAP"
            if (self.missing>self.settings.max_missing_frames or hand_count not in (None,0,1)
                    or (self.previous and now-self.previous[1]>self.settings.max_gap_seconds)):
                self.track_generation+=1
                self.previous=None
                if self.attempt is not None:
                    self.attempt["continuity_status"]="DISCONTINUITY_REVIEW_REQUIRED"
            self.state="Z_CAPTURING" if self.attempt and self.attempt["start_pose_linked"] else "IDLE"
            return self.snapshot()
        _,_,center,scale=_geometry(hand,size)
        angles=np.asarray([f[k] for f in pose["finger_geometry"].values() for k in ("pip_angle_degrees","dip_angle_degrees")])
        if self.previous:
            old_index,old_time,old_center,old_scale,old_angles=self.previous
            if index<=old_index or now<=old_time or now-old_time>self.settings.max_gap_seconds:
                discontinuity="TEMPORAL_DISCONTINUITY"
            elif np.linalg.norm(center-old_center)/old_scale>self.settings.max_palm_step_ratio:
                discontinuity="PALM_POSITION_DISCONTINUITY"
            if discontinuity:
                self.track_generation+=1
                self.streak=[]
                self.ready=None
                if self.attempt is not None:
                    self.attempt["continuity_status"]="DISCONTINUITY_REVIEW_REQUIRED"
            if np.max(np.abs(angles-old_angles))>self.settings.max_joint_angle_step_degrees or index!=old_index+1:
                self.streak=[]
                self.ready=None
        self.previous=(index,now,center,scale,angles)
        self.missing=0
        if pose["pose_status"]=="VALID":
            self.streak.append((index,now))
            if self.ready is None and len(self.streak)>=self.settings.stable_frames and now-self.streak[0][1]>=self.settings.stable_seconds:
                self.ready={"frame_id":index,"timestamp":now,"track_generation":self.track_generation}
            # Only the start of the stable window is needed; memory stays bounded.
            if len(self.streak)>self.settings.stable_frames:
                self.streak=[self.streak[0],*self.streak[-self.settings.stable_frames:]]
        else:
            self.streak=[]
            self.ready=None
        if self.attempt is not None:
            self.attempt["observed_frames"]+=1
            self.attempt["valid_pose_frames"]+=pose["pose_status"]=="VALID"
            self.attempt["last_frame"]=index
            self.attempt["last_timestamp"]=now
        self.state=("Z_CAPTURING" if self.attempt["start_pose_linked"] else "IDLE") if self.attempt else "Z_POSE_READY" if self.ready else "IDLE"
        return self.snapshot()

    def begin_attempt(self,attempt_id,onset_frame,onset_time,movement_since):
        ready=dict(self.ready) if self.ready else None
        linked=bool(ready and ready["timestamp"]<onset_time and ready["track_generation"]==self.track_generation
                    and movement_since is not None and movement_since>=ready["timestamp"])
        self.attempt={"attempt_id":attempt_id,"onset_frame":onset_frame,"onset_timestamp":onset_time,
                      "movement_since":movement_since,"ready":ready,"start_pose_linked":linked,
                      "reason":"READY_BEFORE_NEW_MOTION" if linked else "NO_READY_BEFORE_NEW_MOTION",
                      "track_generation":self.track_generation,"continuity_status":"OBSERVED_ONLY_IDENTITY_UNVERIFIED",
                      "observed_frames":1,"valid_pose_frames":int((self.last_pose or {}).get("pose_status")=="VALID"),"missing_frames":[],
                      "motion_status":"UNVERIFIED_JZ_SPECIFIC_MOTION_NOT_CALIBRATED",
                      "would_allow_z_candidate":False}
        self.state="Z_CAPTURING" if linked else "IDLE"
        self.temporal.begin_attempt(attempt_id,onset_frame,onset_time,movement_since,linked)

    def end_attempt(self):
        self.attempt=None
        self.streak=[]
        self.ready=None
        self.state="IDLE"
        self.temporal.end_attempt()

    def finish_frame(self,index,now):
        self.temporal.finish_frame(index,now)

    def snapshot(self):
        return {"policy":"SHADOW_ONLY_NO_ACCEPTANCE_EFFECT","phase":"A","state":self.state,
                "settings":asdict(self.settings),"pose":self.last_pose,"ready":dict(self.ready) if self.ready else None,
                "attempt":({**self.attempt,"missing_frames":list(self.attempt["missing_frames"]),
                             "pose_valid_fraction":self.attempt["valid_pose_frames"]/max(1,self.attempt["observed_frames"])}
                           if self.attempt else None), "temporal_evidence":self.temporal.snapshot()}


def replay_start_pose_shadow(metadata,arrays):
    """Pose-only replay from opt-in raw observations, not from model features.

    Older attempts lack the original movement_since/READY history. Explicitly
    report that limitation, never infer a start pose from classifier output.
    """
    required=("observed_frame_indices","observed_timestamps","observed_sizes","observed_landmarks",
              "timeline_indices","timeline_timestamps","timeline_valid")
    missing=[name for name in required if name not in arrays]
    if missing:
        return {"status":"UNAVAILABLE","missing_fields":missing,"acceptance_effect":False}
    indices=arrays["observed_frame_indices"]
    timeline=arrays["timeline_indices"]
    timestamps=arrays["timeline_timestamps"]
    valid=arrays["timeline_valid"]
    landmarks=arrays["observed_landmarks"]
    sizes=arrays["observed_sizes"]
    if (indices.ndim!=1 or timeline.ndim!=1 or timestamps.shape!=timeline.shape or valid.shape!=timeline.shape
            or valid.dtype.kind!="b" or landmarks.shape!=(len(indices),21,3) or sizes.shape!=(len(indices),2)
            or not np.isfinite(landmarks).all() or not np.isfinite(timestamps).all()
            or np.any(np.diff(indices)<=0) or np.any(np.diff(timeline)<=0) or np.any(np.diff(timestamps)<=0)
            or not np.array_equal(timeline[valid],indices)
            or not np.array_equal(timestamps[valid],arrays["observed_timestamps"])):
        raise ValueError("INVALID_Z_POSE_REPLAY_TIMELINE")
    old=metadata.get("z_start_pose_shadow") or (metadata.get("technical_validation") or {}).get("z_start_pose_shadow") or {}
    old_attempt=old.get("attempt") or {}
    settings=ZStartPoseSettings(**old["settings"]) if old.get("settings") else ZStartPoseSettings()
    shadow=ZStartPoseShadow(settings)
    mapping={int(index):i for i,index in enumerate(indices)}
    counts={"VALID":0,"INVALID":0,"UNCERTAIN":0}
    ready_events=[]
    onset_report=None
    for index,now,is_valid in zip(timeline,timestamps,valid):
        pos=mapping.get(int(index))
        report=shadow.observe(int(index),landmarks[pos] if is_valid else None,
                              tuple(sizes[pos]) if is_valid else (1,1),float(now),hand_count=1 if is_valid else 0)
        counts[report["pose"]["pose_status"]]+=1
        if report.get("ready") and report["ready"] not in ready_events:
            ready_events.append(report["ready"])
        if int(index)==metadata["onset_frame"]:
            shadow.begin_attempt(metadata["attempt_id"],int(index),float(now),old_attempt.get("movement_since"))
            onset_report=shadow.snapshot()
        shadow.finish_frame(int(index),float(now))
    return {"status":"SHADOW_ONLY","acceptance_effect":False,"pose_frame_counts":counts,
            "ready_events":ready_events,"onset_report":onset_report,
            "final_report":shadow.snapshot(),
            "movement_since_available":old_attempt.get("movement_since") is not None,
            "history_limitation":"Only saved pre-roll through resolution; earlier camera READY history may be unavailable",
            "semantic_motion_status":"UNVERIFIED_NOT_AN_ACCEPTANCE_GATE"}
