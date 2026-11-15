"""Audit script to verify Stage 2 Validation evaluation reproducibility and metric calculations."""

import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

base = Path(__file__).resolve().parents[1]
split_file = base / "data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json"
data_file = base / "data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl"
baseline_model_file = base / "data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib"
template_file = base / "models/dynamic_jz/dtw_templates/templates_config_c_valid_only_k2.json"

import sys
sys.path.insert(0, str(base))
from app.inference.dynamic_jz.finger_tracker import FingerTracker
from app.inference.dynamic_jz.template_store import TemplateStore
from app.inference.dynamic_jz.dtw_matcher import DTWMatcher

# 1. Load splits
with open(split_file, "r", encoding="utf-8") as f:
    splits = json.load(f)

train_list = splits["splits"]["train"]
val_list = splits["splits"]["validation"]
test_list = splits["splits"]["test"]

print(f"Split counts: Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")

# Set intersection check
train_paths = {x["source_path"] for x in train_list}
val_paths = {x["source_path"] for x in val_list}
test_paths = {x["source_path"] for x in test_list}

assert len(train_paths & val_paths) == 0, "Train and Val overlap!"
assert len(train_paths & test_paths) == 0, "Train and Test overlap!"
assert len(val_paths & test_paths) == 0, "Val and Test overlap!"
print("Zero split overlap confirmed.")

# 2. Load dataset entries
samples_by_path = {}
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        samples_by_path[item["source_path"]] = item

val_j_count = sum(1 for m in val_list if m["label"] == "J")
val_z_count = sum(1 for m in val_list if m["label"] == "Z")
print(f"Validation label distribution: J={val_j_count}, Z={val_z_count}, Total={len(val_list)}")

# 3. Evaluate Baseline LogisticRegression on Validation
baseline_model = joblib.load(baseline_model_file)
X_val = []
y_val = []
for m in val_list:
    raw = samples_by_path[m["source_path"]]
    feat = np.array(raw["features"], dtype=np.float64).flatten()
    X_val.append(feat)
    y_val.append(m["label"])

y_pred_base = baseline_model.predict(np.array(X_val))
acc_base = accuracy_score(y_val, y_pred_base)
prec_base, rec_base, f1_base, _ = precision_recall_fscore_support(y_val, y_pred_base, labels=["J", "Z"])
cm_base = confusion_matrix(y_val, y_pred_base, labels=["J", "Z"])
print("\n--- Baseline LogisticRegression Evaluation ---")
print(f"Accuracy: {acc_base:.4f}")
print(f"J - Prec: {prec_base[0]:.4f}, Rec: {rec_base[0]:.4f}, F1: {f1_base[0]:.4f}")
print(f"Z - Prec: {prec_base[1]:.4f}, Rec: {rec_base[1]:.4f}, F1: {f1_base[1]:.4f}")
print(f"Confusion Matrix (labels=['J', 'Z']):\n{cm_base}")

# 4. Evaluate DTW CONFIG_C K=2 R=4 on Validation
store = TemplateStore.load_json(template_file)
matcher_r4 = DTWMatcher(store, config_name="CONFIG_C", radius=4, threshold_j=1e9, threshold_z=1e9, min_margin=0.0)
matcher_r3 = DTWMatcher(store, config_name="CONFIG_C", radius=3, threshold_j=1e9, threshold_z=1e9, min_margin=0.0)

tracker = FingerTracker()

for r_name, matcher in [("R=4", matcher_r4), ("R=3", matcher_r3)]:
    y_pred_dtw = []
    margins = []
    winning_dists = []
    losing_dists = []

    for m in val_list:
        raw = samples_by_path[m["source_path"]]
        seq = np.array(raw["features"], dtype=np.float64)
        indices = raw.get("selected_original_frame_indices")
        res_j = tracker.extract(seq, letter_target="J", selected_original_frame_indices=indices)
        res_z = tracker.extract(seq, letter_target="Z", selected_original_frame_indices=indices)
        match_res = matcher.match_dual(res_j, res_z)
        y_pred_dtw.append(match_res.predicted_label)
        margins.append(match_res.distance_margin)
        if match_res.predicted_label == "J":
            winning_dists.append(match_res.distance_j)
            losing_dists.append(match_res.distance_z)
        else:
            winning_dists.append(match_res.distance_z)
            losing_dists.append(match_res.distance_j)

    acc_dtw = accuracy_score(y_val, y_pred_dtw)
    prec_dtw, rec_dtw, f1_dtw, _ = precision_recall_fscore_support(y_val, y_pred_dtw, labels=["J", "Z"])
    cm_dtw = confusion_matrix(y_val, y_pred_dtw, labels=["J", "Z"])
    print(f"\n--- DTW CONFIG_C VALID_ONLY K=2 {r_name} Evaluation ---")
    print(f"Accuracy: {acc_dtw:.4f} ({sum(1 for a, b in zip(y_val, y_pred_dtw) if a == b)}/{len(y_val)})")
    print(f"J - Prec: {prec_dtw[0]:.4f}, Rec: {rec_dtw[0]:.4f}, F1: {f1_dtw[0]:.4f}")
    print(f"Z - Prec: {prec_dtw[1]:.4f}, Rec: {rec_dtw[1]:.4f}, F1: {f1_dtw[1]:.4f}")
    print(f"Confusion Matrix (labels=['J', 'Z']):\n{cm_dtw}")
    print(f"Mean Margin: {np.mean(margins):.4f}, Median Margin: {np.median(margins):.4f}")
    print(f"Mean Winning Dist: {np.mean(winning_dists):.4f}, Mean Losing Dist: {np.mean(losing_dists):.4f}")
