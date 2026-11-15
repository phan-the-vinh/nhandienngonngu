# Preprocessing freeze review — standard ASL

Alphabet convention: Standard ASL manual alphabet A-Z; J and Z handled as dynamic gestures; M and N use standard ASL forms.

Static classes: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y. J/Z must not be proposed as alternative labels for these static samples. The former custom M/N convention is not used.

## Semantic review

The assistant inspected all 11 candidate original images and all three Q→A group images, with an independent assistant second pass. This is not human ASL certification. Model/kNN predictions are supporting triage signals, not semantic ground truth. A supported image label also does not certify its extracted landmark vector.

| Result | 11 candidates | Q→A group | Total |
| --- | ---: | ---: | ---: |
| SEMANTIC_ORIGINAL_LABEL_SUPPORTED | 2 | 0 | 2 |
| NEEDS_MANUAL_SEMANTIC_REVIEW | 9 | 3 | 12 |
| SEMANTIC_LABEL_MISMATCH_CANDIDATE | 0 | 0 | 0 |

The two supported originals are `dulieu/chucai/d/D4 (6).jpg` (D) and `dulieu/chucai/q/hand2_q_left_seg_2_cropped.jpeg` (Q). Their saved landmark/image correspondence still needs disposition; do not relabel them O/C merely because the landmark classifiers agree.

For `k/color_10_0460 (3).png`, extended upper digits do not support a closed T fist. The thumb/index/middle relationship is insufficiently resolved to confirm K, so its status remains manual review despite confidence 0.9998324928 and 7/7 T neighbors. No alternate semantic label is asserted.

For `DG-Q-c9d48eb2a9df1dc7`, two of three saved model predictions are A, while all seven saved neighbors are Q for each image. All three originals remain semantically unresolved due to blur, occlusion or foreshortening. No majority vote determines label correctness; do not propagate a judgment between group members.

The M images remain unresolved under standard ASL because thumb placement is not sufficiently visible. Ordinary ASL variations do not imply the project's former custom convention. Reference: [ASL University fingerspelling](https://www.lifeprint.com/asl101/fingerspelling/fingerspelling.htm), which documents variation in E/M/N/G forms; [manual alphabet overview](https://www.lifeprint.com/asl101/fingerspelling/).

Per-image observations and unchanged-label/sample actions are recorded in `preprocessing_semantic_observations.json`. No sample is deleted or relabeled. No alternative J/Z label is allowed. Static overlays audit geometry, not stability.

## Freeze status

Status remains **NEEDS_MINOR_FIXES**, not PREPROCESSING_READY_FOR_SPLIT. Convention is resolved; semantic review has been performed but 12 image judgments still need human resolution. Two missing-hand-blur cases remain unresolved: `a/color_0_0256.png` and `c/mrinnec2.png`, whose landmark-based ROIs are too small (7×33 and 43×9). Missing blur is not a pass. The observed landmark/image mismatches also need explicit disposition; no automated exclusion is authorized.

Canonical inputs remain `AI/dataset/phase2_quality_gate_v2/dataset_chucai_clean_reviewed_v2.csv` (89,428 samples) and `AI/dataset/sample_group_map.csv`. A pending manifest is not authorization to split or train. The finalizer records this exact alphabet convention in both pending and eventual final manifests and must refuse READY while semantic/landmark dispositions or integrity checks remain unresolved.

The reporting-only correction of `knn_nearest_alternative_class` means the first differently labeled neighbor in the saved distance-ordered top seven, not the modal alternative. The historical modal value belongs in `knn_modal_alternative_class`; an empty alternative means only NOT_OBSERVED_IN_TOP_K. Original reports and predictions must remain immutable; corrected reports use a new output directory.
