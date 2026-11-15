# Z start pose — phase A (SHADOW ONLY)

## Reference verification update (2026-10-14)

The user has now supplied SHA256 for all four references. Each matches the
corresponding file under `Z/` exactly; the same-named J/3, J/29, J/68 do NOT
match. Reference-byte identity is verified, not independent ASL correctness.
The earlier ambiguity described below is historical and now resolved by hash.

Continuous per-frame reference timelines are now available in two identical
four-video audit runs under `z_start_pose_reference_audit/`. They contain no
human-labelled true-motion boundaries. The older
three diagnostic sessions contain 121 UNREVIEWED attempts, no start-pose-shadow
snapshots and no source-video mapping. They cannot supply labelled recall or
negative rejection rates. Do not replace missing values with zero.

The user can run the dedicated, four-file diagnostic (no dataset/model output):

```powershell
python tools/audit_z_start_pose_references.py --check-only
python tools/audit_z_start_pose_references.py --audit
```

Check-only rechecks the hashes and creates no output. Audit asks confirmation,
then reuses the actual integrated VIDEO/IMAGE detection, classifier and SHADOW
coordinator with a source-FPS clock. It writes exclusive temporary per-frame
JSONL/events and summaries under
`data/experiments/dynamic_jz_v1/z_start_pose_reference_audit/<unique_run>/`.
It does not write features/training data, change thresholds, enable gates or
commit characters. A clip starts a fresh session; pre-clip history is unknown.
Detector onset is not manually annotated linguistic onset. Full end-to-end
recall additionally requires a trial log including gestures the detector missed.

This implements the first phase, not a fix that already blocks false Z
candidates. No recognition, candidate acceptance, static suppression, HOLD,
TextComposer, classifier, normalization, 16x69 builder, detector threshold or
production entry point is changed. The historical geometry gate remains shadow.
The optional monitor has no return value used by recognition decisions.

## Why no mandatory gate yet

`3.avi`, `29.avi`, `68.avi` exist in BOTH `J/` and `Z/` under
`data/raw/dynamic/external/signnteam_jz/SigNN Video Data/`. Names alone do not
identify which references the user intended or their true linguistic labels.
`380.avi` exists at `Z/380.avi`, but the user must confirm this is the supplied
reference. Folder labels are source annotations, not independently reviewed truth.

Read-only inspection found existing SUCCESS sequences for J/3, J/29, J/68,
Z/3, Z/68 and Z/380; Z/29 is absent from the original SUCCESS JSONL. No new
MediaPipe extraction, video decoding, model prediction or dataset changes were
performed for these references.

For the local Z/380 candidate only, a header-only read gives 640x480, reported
30 FPS / 90 frames (not a new decoded-frame audit). Existing selected shape
features preserve joint angles and reach ratios up to uniform scale. Measuring
them with aspect-ratio correction gives:

- VALID: original frames 0, 6, 12, 17, 89.
- UNCERTAIN: 23, 29.
- INVALID under the *provisional geometric hypotheses*: 35, 41, 46, 52, 58,
  64, 70, 75, 83.

These classifications describe estimated landmark geometry, NOT whether the
gesture is correct ASL. Sixteen sparse frames cannot establish stable pre-onset
readiness or continuous hand identity. The mixed results are a reason NOT to
enforce these rules throughout a real Z without reviewing the full motion and
viewpoint/occlusion effects. No positive/negative recall estimate is available.

## Per-frame validator

`ZStartPoseValidator.evaluate()` returns `pose_status` VALID/INVALID/UNCERTAIN,
`pose_score`, named predicate measurements/thresholds, and reasons. The score is
a mean of predicate support, not a calibrated probability or model confidence.

Coordinates are mapped to pseudo-3D `[x*width, y*height, z*width]` before angle
measurement. This handles image aspect ratio. MP Z is relative pseudo depth,
NOT an absolute camera distance. Translation, scale, reflection and rigid
rotation of this pseudo-3D geometry do not change the bone angles/ratios; real
viewpoint changes can still degrade MediaPipe estimates.

Provisional, uncalibrated hypotheses:

| Predicate | Landmarks | Hypothesis |
|---|---|---|
| Index PIP extended | 5,6,7 | Angle >=155 degrees. |
| Index DIP extended | 6,7,8 | Angle >=150 degrees. |
| Index reach | 5..8 | tip-to-MCP / summed bone lengths >=.80. |
| Middle folded | 9..12 | PIP angle <=140 degrees OR reach <=.70. |
| Ring folded | 13..16 | Same folded hypothesis. |
| Pinky folded | 17..20 | Same folded hypothesis. |
| Thumb | 2,3,4 and 5 | Measured only, no acceptance constraint. |

Near-threshold failures (8 degrees / .05 reach) are UNCERTAIN. Missing,
multiple, non-finite or degenerate landmarks are UNCERTAIN. Optional supplied
presence confidence is checked separately. Current camera inputs do not provide
per-landmark confidence; the report explicitly says NOT_PROVIDED. Handedness is
not used to determine pose or identity. No static letter label is an input.

## Temporal shadow

`ZStartPoseShadow` records an observational `IDLE -> Z_POSE_READY ->
Z_CAPTURING` progression. READY needs at least 3 consecutive valid observations
and .10 seconds of stable joint geometry; it does not add a 3-second pre-hold.
Joint-angle steps >18 degrees clear readiness. Numeric gap/palm-step hypotheses
are shadow diagnostics only, not changes to the real motion detector.

On generic capture onset it freezes attempt ID, READY timestamp/frame, observed
track generation, onset and the detector's `movement_since`. It links the start
pose only if READY predates onset AND `movement_since` is not before READY.
Readiness acquired later cannot retrospectively legitimize an earlier motion.
No pre-READY movement or pre-roll is credited as new Z evidence.

Missing observations clear readiness. An already open shadow attempt keeps gap
diagnostics but cannot claim verified identity; large gaps/palm jumps flag
discontinuity. Handedness alone never reconnects tracks. End/reset clears the
shadow attempt and permits fresh readiness without requiring hand withdrawal.
The actual detector, anchors, pre-roll, inference and HOLD are untouched.

Pose preservation during capture is measured as a fraction, not enforced.
Whole-hand Z is supported by pose measurements without requiring tip motion
relative to wrist. Generic motion and a valid index pose are still NOT Z proof.
`motion_status` remains UNVERIFIED and `would_allow_z_candidate` stays false.
Those are shadow fields; current recognition behavior remains unchanged.

## Commands (user runs camera)

```powershell
python app/inference/unified_abc123_jz_auto_experiment.py --check-only --z-start-pose-shadow
python app/inference/unified_abc123_jz_auto_experiment.py --debug --capture-jz-diagnostics --z-start-pose-shadow
```

The panel and debug events report pose/state. Capture stores the snapshot with
technical validation and keeps the original exact model input. Check-only does
not open camera, execute MediaPipe or write files. J/Z commit remains OFF by
default. To compare existing experimental HOLD/commit, the user may explicitly
add `--enable-experimental-jz`; this STILL does not enable a start-pose gate.

```powershell
python tools/replay_jz_attempt.py --attempt-dir "<attempt_dir>" --geometry-replay --z-start-pose-shadow
```

Replay measures saved raw observations, not guessed raw coordinates from model
predictions. It uses the captured pose settings when available. Older records
may lack movement_since/READY information or earlier history: these limits are
explicit, never inferred. Replay changes no labels or decisions.

Rollback: omit `--z-start-pose-shadow`. It defaults OFF. No required gate or
`--enable-z-start-pose-gate` is provided at this stage.

## Temporal evidence comparison (2026-10-14)

Legacy READY/state/attempt fields remain unchanged. The new
`temporal_evidence` snapshot is parallel SHADOW instrumentation, not input to
prediction, candidate validation, static suppression, HOLD or TextComposer.
No geometric/motion thresholds or 16x69 feature definitions changed.

The preparation token records creation frame/time, session/observed track
generation, last VALID frame/time/pose and tracking limitations. VALID
observations must continue to refresh it. If the current onset frame is
UNCERTAIN but the immediately preceding frame still carried valid preparation,
the token can be associated during that same frame transaction. If no onset
arrives in that transaction, it expires. This narrowly preserves pre-onset
evidence; it is NOT a configurable grace period or a guessed READY TTL.

INVALID pose, missing/ambiguous/invalid tracking, existing joint/palm/time
discontinuity, session change, attempt consumption/completion and rearm revoke
reusable evidence. During capture the frozen preparation is historical, not a
requirement to keep pose VALID through movement. Gaps still mark continuity
unverified. A new attempt needs a fresh preparation streak; physical hand
withdrawal is not required. No semantic Z pass is issued, and
`would_allow_z_candidate` remains false even when temporal association succeeds.

There is no age-only TTL yet. A long-held continuously reobserved VALID pose is
fresh evidence, not an unattended stale token. Measure READY-to-human-onset
timing on labelled trials before designing any longer uncertainty grace/TTL.
SHADOW events record association IDs, old/new linkage, delay, retention and
invalidation reasons. Snapshots are carried in existing attempt diagnostics.

Read-only replay of old reports (no video/MediaPipe/model execution or writes):

```powershell
python tools/audit_z_start_pose_references.py --replay-reports data/experiments/dynamic_jz_v1/z_start_pose_reference_audit/20261014T100404250906Z_434978d8
```

Report-level replay associates Z/380 READY f3 (0.10s) with detector onset f21
(0.70s), using last VALID f20. Old linkage remains false; new linkage is true
conditionally on continuity checks unavailable in old full-frame geometry.
This is not a rerun of recognition or evidence of actual Z onset/recall.
Z/3 and Z/68 have no VALID streak; Z/29 has only a two-frame streak and extensive
IMAGE detection loss. Do not call predicate failure proven false rejection
without human pose/occlusion labels. No geometry threshold adjustment follows.

The existing audit utility also supplies a user-only frame annotation mode:

```powershell
python tools/audit_z_start_pose_references.py --check-only
python tools/audit_z_start_pose_references.py --annotate --reference 380.avi --reviewer USER
```

It confirms before decoding one hash-matched reference. A/D step through source
frames without flipping; 1/2 mark preparation start/stability, 3/4 mark true Z
motion start/end, 5/6 mark inclusive uncertain/occluded ranges. S writes a unique
JSON under `z_start_pose_reference_audit/human_annotations/`; Q cancels. Missing
marks remain null/PARTIAL. Late preparation stability is allowed rather than
forcing READY-before-motion into the annotation. No detector mark is prefilled,
no source or historical audit is overwritten, and no training status changes.

## Phase-B plan / missing evidence

1. Resolve the ambiguous references and human-label a separate true-Z/negative
   comparison set, including different speed, hands, views, whole-hand movement,
   entry/exit, rotation, static transitions and consecutive J/Z/JACKZ.
2. Inspect pre-onset RAW frames and per-joint confidence/occlusion limitations.
   Measure false rejection at READY and pose preservation separately.
3. Develop Z-specific motion evidence on post-READY observations, with adaptive
   change points/order rather than source samples 0/5/10/15. Allow rigid-hand Z;
   measure translation/rotation alternatives and partial gestures. Preserve the
   original classifier feature builder unchanged.
4. Only after held-out recall/false-positive checks, add an opt-in acceptance
   flag for `linked start pose + motion evidence + model Z + existing endpoint`.
   UNKNOWN must not receive candidate/HOLD rights; J must bypass Z prerequisites.
5. Keep the existing static/digits/HOLD tests, positive J tests and no-release
   repeated-letter lifecycle tests. Synthetic unit tests are not real recall.

Do not claim this phase already rejects A/U/R/W/K/Y/V camera movements or
improves recognition. A static pointing pose can resemble Z's initial pose;
start pose by itself cannot solve false positives.
