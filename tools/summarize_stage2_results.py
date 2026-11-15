import json
from pathlib import Path

res_path = Path(__file__).resolve().parents[1] / "reports/data/stage2_experiment_results.json"
with open(res_path, "r", encoding="utf-8") as f:
    data = json.load(f)

cfg_summary = {}
for item in data["all_evaluations"]:
    cfg = item["config"]
    if cfg not in cfg_summary or item["accuracy"] > cfg_summary[cfg]["accuracy"] or (
        item["accuracy"] == cfg_summary[cfg]["accuracy"] and item["mean_margin"] > cfg_summary[cfg]["mean_margin"]
    ):
        cfg_summary[cfg] = item

print("=== BEST PER CONFIGURATION ===")
for cfg, item in sorted(cfg_summary.items()):
    print(
        f"{cfg}: Acc={item['accuracy']*100:.2f}%, J-F1={item['j_f1']:.4f}, Z-F1={item['z_f1']:.4f}, "
        f"Margin={item['mean_margin']:.4f}, Policy={item['policy']}, K={item['k']}, R={item['radius']}"
    )

print("\n=== CALIBRATION SWEEP (Optimal thresholds on best config) ===")
# Find sweep entries where acc_on_decided == 1.0 and unknown_rate is lowest
valid_sweeps = [s for s in data["calibration_sweep"] if s["acc_on_decided"] == 1.0]
valid_sweeps.sort(key=lambda s: (s["unknown_rate"], -s["margin"]))
print("Top 5 Threshold/Margin settings (Zero errors on decided samples):")
for s in valid_sweeps[:5]:
    print(
        f"Threshold={s['threshold']:.2f}, Margin={s['margin']:.2f} -> "
        f"Unknown={s['unknown']}/64 ({s['unknown_rate']*100:.1f}%), Error={s['error']}, Decided Acc={s['acc_on_decided']*100:.2f}%"
    )

print("\n=== LATENCY BENCHMARK ===")
for k_str, bench in data["latency_benchmark"].items():
    print(
        f"{k_str} ({bench['templates_count']} templates): "
        f"Match Mean={bench['matching_mean_ms']:.3f}ms, P95={bench['matching_p95_ms']:.3f}ms, P99={bench['matching_p99_ms']:.3f}ms | "
        f"Total Mean={bench['total_mean_ms']:.3f}ms, P99={bench['total_p99_ms']:.3f}ms"
    )
