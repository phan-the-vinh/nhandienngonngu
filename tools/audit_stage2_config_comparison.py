"""Analyze and compare all configurations from stage2_experiment_results.json."""

import json
from pathlib import Path

base = Path(__file__).resolve().parents[1]
res_file = base / "reports/data/stage2_experiment_results.json"

with open(res_file, "r", encoding="utf-8") as f:
    data = json.load(f)

evals = data["all_evaluations"]
print(f"Total evaluation combinations: {len(evals)}")

# 1. Compare configs across same settings (VALID_ONLY, K=2, R=4 and R=3)
print("\n--- CONFIG COMPARISON (VALID_ONLY, K=2) ---")
print(f"{'Config':<10} {'R':<3} {'Accuracy':<10} {'Mean Margin':<12} {'Median Margin':<14} {'Win Dist':<10} {'Lose Dist':<10}")
for e in evals:
    if e["policy"] == "VALID_ONLY" and e["k"] == 2 and e["radius"] in (3, 4):
        print(f"{e['config']:<10} {e['radius']:<3} {e['accuracy']*100:>6.2f}%    {e['mean_margin']:>10.4f}   {e['median_margin']:>12.4f}   {e['mean_winning_dist']:>8.4f}   {e['mean_losing_dist']:>8.4f}")

# 2. Compare R=3 vs R=4 for CONFIG_C across all K and policies
print("\n--- R=3 vs R=4 COMPARISON FOR CONFIG_C ---")
print(f"{'Policy':<22} {'K':<3} {'R=3 Margin':<12} {'R=4 Margin':<12} {'Diff (R4 - R3)':<14} {'R=3 WinDist':<12} {'R=4 WinDist':<12}")
for pol in ["VALID_ONLY", "VALID_AND_SUSPICIOUS"]:
    for k in [1, 2, 4, 8]:
        r3_entry = next(e for e in evals if e["config"] == "CONFIG_C" and e["policy"] == pol and e["k"] == k and e["radius"] == 3)
        r4_entry = next(e for e in evals if e["config"] == "CONFIG_C" and e["policy"] == pol and e["k"] == k and e["radius"] == 4)
        diff = r4_entry["mean_margin"] - r3_entry["mean_margin"]
        print(f"{pol:<22} {k:<3} {r3_entry['mean_margin']:>10.4f}   {r4_entry['mean_margin']:>10.4f}   {diff:>12.4f}   {r3_entry['mean_winning_dist']:>10.4f}   {r4_entry['mean_winning_dist']:>10.4f}")

# 3. Compare VALID_ONLY vs VALID_AND_SUSPICIOUS for CONFIG_C
print("\n--- VALID_ONLY vs VALID_AND_SUSPICIOUS FOR CONFIG_C (R=4) ---")
print(f"{'K':<3} {'VALID_ONLY Margin':<18} {'VALID+SUSP Margin':<18} {'Diff (VO - V+S)':<16} {'VO WinDist':<12} {'V+S WinDist':<12}")
for k in [1, 2, 4, 8]:
    vo_entry = next(e for e in evals if e["config"] == "CONFIG_C" and e["policy"] == "VALID_ONLY" and e["k"] == k and e["radius"] == 4)
    vs_entry = next(e for e in evals if e["config"] == "CONFIG_C" and e["policy"] == "VALID_AND_SUSPICIOUS" and e["k"] == k and e["radius"] == 4)
    diff = vo_entry["mean_margin"] - vs_entry["mean_margin"]
    print(f"{k:<3} {vo_entry['mean_margin']:>16.4f}   {vs_entry['mean_margin']:>16.4f}   {diff:>14.4f}   {vo_entry['mean_winning_dist']:>10.4f}   {vs_entry['mean_winning_dist']:>10.4f}")

# 4. K-Medoids stability: Margin and Distances across K for CONFIG_C VALID_ONLY R=4
print("\n--- K STABILITY FOR CONFIG_C VALID_ONLY R=4 ---")
print(f"{'K':<3} {'Accuracy':<10} {'Mean Margin':<12} {'Median Margin':<14} {'Mean Win Dist':<14} {'Mean Lose Dist':<14}")
for k in [1, 2, 4, 8]:
    e = next(e for e in evals if e["config"] == "CONFIG_C" and e["policy"] == "VALID_ONLY" and e["k"] == k and e["radius"] == 4)
    print(f"{k:<3} {e['accuracy']*100:>6.2f}%    {e['mean_margin']:>10.4f}   {e['median_margin']:>12.4f}   {e['mean_winning_dist']:>12.4f}   {e['mean_losing_dist']:>12.4f}")
