"""Script to independently recompute and verify all Stage 13 metrics from attempt-level records."""

import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "reports/data/stage13_results.json", encoding="utf-8") as f:
    d13 = json.load(f)

per_attempt = d13["per_attempt_webcam"]
print(f"Total per_attempt records in stage13_results.json: {len(per_attempt)}")

models = ["Production LR (Baseline)", "Production LR + Margin 0.10"]

# Audit Trial 10 and Trial 30 explicitly
print("\n--- AUDIT OF TRIAL 10 AND TRIAL 30 ---")
for a in per_attempt:
    if a["attempt_id"] in (10, 30, 35, 40):
        print(f"Attempt {a['attempt_id']:2d}: GT={a['ground_truth']} | Invoked={a['model_invoked']}")
        for m in models:
            pred = a["predictions"][m]
            margin = a["margins"].get(m)
            prob_j = (a["probabilities"].get(m) or {}).get("probability_J")
            prob_z = (a["probabilities"].get(m) or {}).get("probability_Z")
            print(f"   [{m}] Pred={pred} | Margin={margin} | P(J)={prob_j} | P(Z)={prob_z}")

for m in models:
    print(f"\n=================== {m} ===================")
    invoked_pos = []
    invoked_neg = []
    non_invoked_pos = []
    non_invoked_neg = []

    cm = Counter()

    for a in per_attempt:
        gt = a["ground_truth"]
        inv = a["model_invoked"]
        pred = a["predictions"][m]

        if inv:
            if gt in ("J", "Z"):
                invoked_pos.append(a)
                cm[(gt, pred)] += 1
            else:
                invoked_neg.append(a)
        else:
            if gt in ("J", "Z"):
                non_invoked_pos.append(a)
            else:
                non_invoked_neg.append(a)

    pos_j = sum(1 for a in invoked_pos if a["ground_truth"] == "J")
    pos_z = sum(1 for a in invoked_pos if a["ground_truth"] == "Z")
    print(f"Invoked Positives: {len(invoked_pos)} (J: {pos_j}, Z: {pos_z})")
    print(f"Invoked Negatives: {len(invoked_neg)}")
    non_j = sum(1 for a in non_invoked_pos if a["ground_truth"] == "J")
    non_z = sum(1 for a in non_invoked_pos if a["ground_truth"] == "Z")
    print(f"Non-invoked Positives: {len(non_invoked_pos)} (J: {non_j}, Z: {non_z})")
    print(f"Non-invoked Negatives: {len(non_invoked_neg)}")
    print(f"Total Attempts: {len(invoked_pos) + len(invoked_neg) + len(non_invoked_pos) + len(non_invoked_neg)}")

    print("\nConfusion Matrix on Invoked Positives (40 attempts):")
    for g in ["J", "Z"]:
        for p in ["J", "Z", "UNKNOWN"]:
            print(f"  GT={g} -> PRED={p}: {cm[(g, p)]}")

    tp_j = cm[("J", "J")]
    fp_j = cm[("Z", "J")]
    fn_j = cm[("J", "Z")] + cm[("J", "UNKNOWN")]
    prec_j = tp_j / (tp_j + fp_j) if (tp_j + fp_j) > 0 else 0
    rec_j = tp_j / (tp_j + fn_j) if (tp_j + fn_j) > 0 else 0
    f1_j = 2 * prec_j * rec_j / (prec_j + rec_j) if (prec_j + rec_j) > 0 else 0

    tp_z = cm[("Z", "Z")]
    fp_z = cm[("J", "Z")]
    fn_z = cm[("Z", "J")] + cm[("Z", "UNKNOWN")]
    prec_z = tp_z / (tp_z + fp_z) if (tp_z + fp_z) > 0 else 0
    rec_z = tp_z / (tp_z + fn_z) if (tp_z + fn_z) > 0 else 0
    f1_z = 2 * prec_z * rec_z / (prec_z + rec_z) if (prec_z + rec_z) > 0 else 0

    macro_f1 = (f1_j + f1_z) / 2
    acc = (tp_j + tp_z) / len(invoked_pos)
    cov = sum(1 for a in invoked_pos if a["predictions"][m] != "UNKNOWN") / len(invoked_pos)

    print(f"\nRecomputed Metrics (Invoked Positives - 40 attempts):")
    print(f"  Accuracy:    {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Coverage:    {cov:.4f} ({cov*100:.1f}%)")
    print(f"  J Precision: {prec_j:.4f} ({prec_j*100:.1f}%)")
    print(f"  J Recall:    {rec_j:.4f} ({rec_j*100:.1f}%)")
    print(f"  J F1:        {f1_j:.4f}")
    print(f"  Z Precision: {prec_z:.4f} ({prec_z*100:.1f}%)")
    print(f"  Z Recall:    {rec_z:.4f} ({rec_z*100:.1f}%)")
    print(f"  Z F1:        {f1_z:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f} ({macro_f1*100:.1f}%)")

    # Recomputed End-to-End Metrics (50 positive attempts)
    total_pos_j = pos_j + non_j  # 25
    total_pos_z = pos_z + non_z  # 25
    e2e_rec_j = tp_j / total_pos_j
    e2e_rec_z = tp_z / total_pos_z
    e2e_acc = (tp_j + tp_z) / (total_pos_j + total_pos_z)
    e2e_cov = sum(1 for a in per_attempt if a["ground_truth"] in ("J", "Z") and a["predictions"][m] in ("J", "Z")) / 50
    print(f"\nRecomputed End-to-End Metrics (All 50 Positives):")
    print(f"  End-to-End Accuracy: {e2e_acc:.4f} ({e2e_acc*100:.1f}%)")
    print(f"  End-to-End Coverage: {e2e_cov:.4f} ({e2e_cov*100:.1f}%)")
    print(f"  End-to-End J Recall: {e2e_rec_j:.4f} ({e2e_rec_j*100:.1f}%)")
    print(f"  End-to-End Z Recall: {e2e_rec_z:.4f} ({e2e_rec_z*100:.1f}%)")

    # Check Negatives
    neg_accepted = sum(1 for a in invoked_neg if a["predictions"][m] in ("J", "Z"))
    invoked_far = neg_accepted / len(invoked_neg)
    e2e_far = neg_accepted / (len(invoked_neg) + len(non_invoked_neg))
    print(f"\nRecomputed Negative FAR:")
    print(f"  Invoked Negatives Accepted: {neg_accepted}/{len(invoked_neg)} -> Invoked FAR = {invoked_far:.4f} ({invoked_far*100:.1f}%)")
    print(f"  Total Negatives Accepted:   {neg_accepted}/{len(invoked_neg) + len(non_invoked_neg)} -> End-to-End FAR = {e2e_far:.4f} ({e2e_far*100:.1f}%)")

# Inspect Latency measurements
print("\n--- LATENCY VERIFICATION ---")
lat = d13["latency_analysis"]
for m in models:
    print(f"{m}:")
    print(f"  Mean latency: {lat[m]['mean_latency_ms']:.4f} ms")
    print(f"  P50 latency:  {lat[m]['p50_latency_ms']:.4f} ms")
    print(f"  P95 latency:  {lat[m]['p95_latency_ms']:.4f} ms")
    print(f"  P99 latency:  {lat[m]['p99_latency_ms']:.4f} ms")
    print(f"  Throughput:   {lat[m]['throughput_fps']:.1f} FPS")
