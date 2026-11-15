"""Read-only explanations of historical geometry; NEVER recognition rules.

All predicates reproduce geometry_candidates(), including its limitations.
Detector metrics, raw pixel paths and model-feature paths have different units.
"""
import numpy as np

from ai.preprocessing.landmark_utils import normalize_landmarks
from types import SimpleNamespace


def geometry_predicates(features, *, include_series=False):
    values = np.asarray(features)
    if values.shape != (16, 69) or not np.isfinite(values).all():
        raise ValueError("Geometry diagnostics require finite (16,69) features")
    shape = values[:, :63].reshape(16, 21, 3)
    wrist = values[:, 63:65]
    result = {"policy": "SHADOW_ONLY_NOT_ASL_STANDARD", "labels": {}}
    for label, tip, pip, col, folded in (("J",20,18,67,(8,12,16)), ("Z",8,6,65,(12,16,20))):
        predicates = {}

        def record(name, measured, threshold, passed, units, rule):
            predicates[name] = {"value": measured, "threshold": threshold, "pass": bool(passed),
                                "units": units, "source_condition": rule,
                                "failure_reason": "" if passed else name+"_FAILED"}

        tip_norm = np.linalg.norm(shape[:,tip],axis=1)
        pip_norm = np.linalg.norm(shape[:,pip],axis=1)
        extended = tip_norm > 1.1*pip_norm
        folded_ok = np.ones(16,dtype=bool)
        finger_values = {}
        for other in folded:
            a = np.linalg.norm(shape[:,other],axis=1)
            b = np.linalg.norm(shape[:,other-2],axis=1)
            condition = a < 1.3*b
            folded_ok &= condition
            finger_values[str(other)] = {"fraction":float(condition.mean()), "multiplier":1.3}
            if include_series:
                finger_values[str(other)].update(tip_norm=a.tolist(),pip_norm=b.tolist())
        fraction = float(np.mean(extended & folded_ok))
        record("extended_and_folded_fraction",fraction,.75,fraction >= .75,"fraction of 16 frames",
               "mean(extended & folded_ok) >= .75; extended: tip_norm > 1.1*pip_norm; folded: tip_norm < 1.3*pip_norm")
        path = values[:,col:col+2]
        relative = path-wrist
        relative_excursion = float(np.max(np.linalg.norm(relative-relative[0],axis=1)))
        excursion = float(np.max(np.linalg.norm(path-path[0],axis=1)))
        record("relative_excursion",relative_excursion,.08,relative_excursion >= .08,"builder hand scales",
               "max(norm(relative-relative[0])) >= .08")
        record("trajectory_excursion",excursion,.30,excursion >= .30,"builder hand scales",
               "max(norm(path-path[0])) >= .30")
        details = {"extended_multiplier":1.1,"extended_fraction":float(extended.mean()),
                   "folded_fingers":finger_values,"full_series_included":include_series}
        if include_series:
            details.update(extended_tip_norm=tip_norm.tolist(),extended_pip_norm=pip_norm.tolist(),
                           path_xy=path.tolist(),relative_xy=relative.tolist())
        if label == "J":
            chord = path[-1]-path[0]
            length = float(np.linalg.norm(chord))
            record("nonzero_chord",length,1e-9,length > 1e-9,"builder hand scales","norm(chord) > 1e-9")
            delta = path-path[0]
            distance = np.abs(delta[:,0]*chord[1]-delta[:,1]*chord[0])
            turn = int(np.argmax(distance[1:-1]))+1
            first, hook = path[turn]-path[0],path[-1]-path[turn]
            angle_limit = float(.6*np.linalg.norm(first)*np.linalg.norm(hook))
            record("first_downward",float(first[1]),.15,first[1] > .15,"builder hand scales; image Y down","first[1] > .15")
            record("hook_horizontal",float(abs(hook[0])),.12,abs(hook[0]) > .12,"builder hand scales","abs(hook[0]) > .12")
            record("first_hook_dot",float(np.dot(first,hook)),angle_limit,np.dot(first,hook) < angle_limit,
                   "squared builder hand scales","dot(first,hook) < .6*norm(first)*norm(hook)")
            details.update(turn_sample_position=turn,first_leg=first.tolist(),hook_leg=hook.tolist())
        else:
            legs = np.diff(path[[0,5,10,15]],axis=0)
            dx,dy = legs[:,0],legs[:,1]
            record("horizontal_legs",np.abs(dx).tolist(),.12,np.all(np.abs(dx) > .12),
                   "builder hand scales","all(abs(dx) > .12)")
            record("first_direction_reversal",float(dx[0]*dx[1]),0.,dx[0]*dx[1] < 0,
                   "squared builder hand scales","dx[0]*dx[1] < 0")
            record("second_direction_reversal",float(dx[1]*dx[2]),0.,dx[1]*dx[2] < 0,
                   "squared builder hand scales","dx[1]*dx[2] < 0")
            record("middle_downward",float(dy[1]),.10,dy[1] > .10,"builder hand scales; image Y down","dy[1] > .10")
            for index,name in ((0,"first_horizontal"),(2,"last_horizontal")):
                limit = float(abs(dx[index])*.6)
                record(name,float(abs(dy[index])),limit,abs(dy[index]) < limit,
                       "builder hand scales","abs(dy[leg]) < .6*abs(dx[leg])")
            details.update(leg_sample_positions=[0,5,10,15],legs_xy=legs.tolist())
        failed = [name for name,p in predicates.items() if not p["pass"]]
        result["labels"][label] = {"pass":not failed,"failed_predicates":failed,"predicates":predicates,
                                   "details":details,
                                   "evaluation_note":"All predicates measured, including those skipped by source short-circuit"}
    result["geometry_candidates"] = [label for label,item in result["labels"].items() if item["pass"]]
    return result


def trajectory_diagnostics(landmarks, sizes, indices, timestamps):
    """Unsmoothed raw-image geometry, not detector EMA or model features.

    Steps/directions never span gaps. Returns exact XY/time plus step headings,
    signed turning angles, axis reversals, shape RMS and scale changes.
    No movement threshold or semantic acceptance is introduced.
    """
    a = np.asarray(landmarks,dtype=float)
    sizes = np.asarray(sizes)
    ids,times = np.asarray(indices),np.asarray(timestamps)
    if (a.shape != (len(ids),21,3) or sizes.shape != (len(ids),2)
            or times.shape != ids.shape or not len(ids) or not np.isfinite(a).all()
            or not np.isfinite(times).all() or np.any(np.diff(ids)<=0) or np.any(np.diff(times)<=0)):
        raise ValueError("Invalid diagnostic trajectory timeline")
    xy = a[:,:,:2]*sizes[:,None,:]
    scales = np.median(np.linalg.norm(xy[:,[5,9,17]]-xy[:,0:1],axis=2),axis=1)
    normalized = np.asarray([normalize_landmarks([SimpleNamespace(x=x,y=y,z=z) for x,y,z in hand]) for hand in a])
    edges = np.diff(ids)==1
    def series(points):
        steps=np.diff(points,axis=0)
        heading=np.arctan2(steps[:,1],steps[:,0])
        output={"xy_px":points.tolist(),"edge_end_frame_indices":ids[1:].tolist(),
                "turn_frame_indices":ids[1:-1].tolist(),"headings_radians":[],"speed_px_per_second":[],
                "turns_radians":[],"x_reversal_frames":[],"y_reversal_frames":[]}
        for i,step in enumerate(steps):
            nonzero=np.linalg.norm(step)>1e-12
            output["headings_radians"].append(float(heading[i]) if edges[i] and nonzero else None)
            output["speed_px_per_second"].append(float(np.linalg.norm(step)/(times[i+1]-times[i])) if edges[i] else None)
            if i:
                valid=edges[i] and edges[i-1] and nonzero and np.linalg.norm(steps[i-1])>1e-12
                angle=np.arctan2(np.sin(heading[i]-heading[i-1]),np.cos(heading[i]-heading[i-1]))
                output["turns_radians"].append(float(angle) if valid else None)
                for axis,name in ((0,"x"),(1,"y")):
                    if valid and steps[i,axis]*steps[i-1,axis]<0:
                        output[name+"_reversal_frames"].append(int(ids[i]))
        return output
    shape_steps=np.sqrt(np.mean(np.diff(normalized,axis=0)**2,axis=1))
    return {"definition":"RAW_UNSMOOTHED_PIXEL_PATHS_NOT_DETECTOR_EMA_NOT_MODEL_FEATURES",
            "frame_indices":ids.tolist(),"timestamps_monotonic_seconds":times.tolist(),
            "contiguous_edges":edges.tolist(),"palm_scale_px":scales.tolist(),
            "scale_ratio_to_previous":[float(scales[i+1]/scales[i]) if edges[i] and scales[i]>1e-6 else None for i in range(len(edges))],
            "canonical_shape_step_rms":[float(v) if edge else None for v,edge in zip(shape_steps,edges)],
            "whole_hand_palm_center":series(xy[:,[0,5,9,17]].mean(axis=1)),
            "landmarks":{str(k):series(xy[:,k]) for k in (0,8,20)},
            "tip_relative_to_wrist":{str(k):series(xy[:,k]-xy[:,0]) for k in (8,20)}}
