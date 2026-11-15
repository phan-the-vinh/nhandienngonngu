"""Audit and benchmark latency for CONFIG_C vs CONFIG_A across 500 iterations."""

import json
import time
from pathlib import Path
import numpy as np

base = Path(__file__).resolve().parents[1]
split_file = base / "data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json"
data_file = base / "data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl"
template_dir = base / "models/dynamic_jz/dtw_templates"

import sys
sys.path.insert(0, str(base))
from app.inference.dynamic_jz.finger_tracker import FingerTracker
from app.inference.dynamic_jz.template_store import TemplateStore
from app.inference.dynamic_jz.dtw_matcher import DTWMatcher

# Load sample sequence
with open(split_file, "r", encoding="utf-8") as f:
    splits = json.load(f)
val_path = splits["splits"]["validation"][0]["source_path"]

sample_seq = None
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item["source_path"] == val_path:
            sample_seq = np.array(item["features"], dtype=np.float64)
            break

tracker = FingerTracker()

print("=== LATENCY AUDIT: CONFIG_C (9D) vs CONFIG_A (6D) ===")
print("500 iterations after 10 warm-up runs\n")

for cfg in ["CONFIG_A", "CONFIG_C"]:
    print(f"--- Configuration: {cfg} ---")
    print(f"{'K':<4} {'Templates':<10} {'Extract Mean':<14} {'Match Mean':<12} {'Match Median':<14} {'Match P95':<12} {'Match P99':<12} {'Total Mean':<12} {'Total P99':<12}")
    for k in [1, 2, 4, 8]:
        tf = template_dir / f"templates_{cfg.lower()}_valid_only_k{k}.json"
        store = TemplateStore.load_json(tf)
        matcher = DTWMatcher(store, config_name=cfg, radius=4 if cfg == "CONFIG_C" else 3)

        # Warm-up (10 runs)
        for _ in range(10):
            rj = tracker.extract(sample_seq, letter_target="J")
            rz = tracker.extract(sample_seq, letter_target="Z")
            _ = matcher.match_dual(rj, rz)

        ext_times = []
        match_times = []
        tot_times = []

        for _ in range(500):
            t0 = time.perf_counter()
            rj = tracker.extract(sample_seq, letter_target="J")
            rz = tracker.extract(sample_seq, letter_target="Z")
            t1 = time.perf_counter()
            _ = matcher.match_dual(rj, rz)
            t2 = time.perf_counter()

            ext_times.append((t1 - t0) * 1000.0)
            match_times.append((t2 - t1) * 1000.0)
            tot_times.append((t2 - t0) * 1000.0)

        print(
            f"K={k:<2} {2*k:<10} "
            f"{np.mean(ext_times):>10.3f} ms   "
            f"{np.mean(match_times):>8.3f} ms   "
            f"{np.median(match_times):>10.3f} ms   "
            f"{np.percentile(match_times, 95):>8.3f} ms   "
            f"{np.percentile(match_times, 99):>8.3f} ms   "
            f"{np.mean(tot_times):>8.3f} ms   "
            f"{np.percentile(tot_times, 99):>8.3f} ms"
        )
    print()
