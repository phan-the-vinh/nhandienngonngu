"""Audit script for Stage 2 Templates: Verifies provenance, leakage, and metadata."""

import json
import glob
from pathlib import Path

base = Path(__file__).resolve().parents[1]
template_dir = base / "models/dynamic_jz/dtw_templates"
split_file = base / "data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json"

with open(split_file, "r", encoding="utf-8") as f:
    splits_data = json.load(f)

train_sources = {x["source_path"]: x["label"] for x in splits_data["splits"]["train"]}
val_sources = {x["source_path"]: x["label"] for x in splits_data["splits"]["validation"]}
test_sources = {x["source_path"]: x["label"] for x in splits_data["splits"]["test"]}

template_files = sorted(glob.glob(str(template_dir / "*.json")))
print(f"Total template files: {len(template_files)}")

table_rows = []
leakage_found = []

for tf in template_files:
    fname = Path(tf).name
    with open(tf, "r", encoding="utf-8") as f:
        t_data = json.load(f)

    templates = t_data.get("templates", [])
    j_count = sum(1 for t in templates if t["label"] == "J")
    z_count = sum(1 for t in templates if t["label"] == "Z")
    valid_count = sum(1 for t in templates if t["tracking_quality"] == "VALID")
    suspicious_count = sum(1 for t in templates if t["tracking_quality"] == "SUSPICIOUS")

    # Check each template provenance
    for t in templates:
        src = t["source_sample_id"]
        split = t["source_split"]
        if split != "train":
            leakage_found.append((fname, t["template_id"], "source_split != train", split))
        if src in val_sources:
            leakage_found.append((fname, t["template_id"], "IN VALIDATION SPLIT", src))
        if src in test_sources:
            leakage_found.append((fname, t["template_id"], "IN TEST SPLIT", src))
        if src not in train_sources:
            leakage_found.append((fname, t["template_id"], "NOT IN TRAIN SPLIT", src))

    parts = fname.replace("templates_", "").replace(".json", "").split("_")
    cfg = parts[0].upper() + "_" + parts[1].upper()
    if "suspicious" in parts:
        policy = "VALID_AND_SUSPICIOUS"
        k_val = parts[-1].upper()
    else:
        policy = "VALID_ONLY"
        k_val = parts[-1].upper()

    table_rows.append({
        "filename": fname,
        "config": cfg,
        "policy": policy,
        "k": k_val,
        "j_count": j_count,
        "z_count": z_count,
        "valid": valid_count,
        "suspicious": suspicious_count,
        "total": len(templates),
    })

print(f"Leakages found: {len(leakage_found)}")
for leak in leakage_found:
    print("  LEAK:", leak)

print("\nDetailed Table of 32 template sets:")
print(f"{'Config':<10} {'Policy':<22} {'K':<4} {'J':<3} {'Z':<3} {'VALID':<6} {'SUSPICIOUS':<10} {'Total':<5}")
print("-" * 75)
for r in table_rows:
    print(f"{r['config']:<10} {r['policy']:<22} {r['k']:<4} {r['j_count']:<3} {r['z_count']:<3} {r['valid']:<6} {r['suspicious']:<10} {r['total']:<5}")
