# J/Z shadow diagnostics and exact-input replay

This is experimental instrumentation, not a new gesture validator. It changes
no classifier, normalization, 16x69 builder, motion/confirmation threshold,
candidate acceptance rule, HOLD algorithm or production entry point. The old
geometry rules are not asserted to be linguistic ASL standards. They remain
SHADOW, including when their predicates all fail.

## Start / stop diagnostic capture

From the project root, in the project's `nn` Python environment:

```powershell
python app/inference/unified_abc123_jz_auto_experiment.py --check-only --capture-jz-diagnostics
python app/inference/unified_abc123_jz_auto_experiment.py --debug --capture-jz-diagnostics
```

The first command does not create a directory, write files, run MediaPipe or
open the camera. The second command is for the USER to run. Press Q/Esc to close
the camera and finalize the session. J/Z commits remain OFF unless the user
also explicitly adds `--enable-experimental-jz` for a HOLD/commit experiment.

Omit `--capture-jz-diagnostics` to disable detailed capture. `--debug` alone
keeps event summaries, including per-predicate measured values and failures,
but does not save full landmark arrays, feature matrices or sampled XY series.
Nothing records pictures, video, names, signer identity, static text content or
personal identifiers in the capture artefacts. Existing `--debug` event logging
retains its existing text behavior; avoid spelling private information.
Landmarks should nevertheless be treated as sensitive temporary diagnostics.

Default output is `data/experiments/dynamic_jz_v1/diagnostics/`. An optional
`--jz-diagnostics-dir` must remain within `data/experiments/dynamic_jz_v1/`.
Every session has a new UTC/UUID ID, with exclusive paths; no file is replaced.
No production dependency on these files is added. They are NOT training data.

## Artefacts

```text
diagnostics/<session_id>/
    human_review_manifest.csv
    generation_000_attempt_00001/
        attempt.json
        replay_data.npz
```

`attempt.json` contains session/generation/attempt IDs, UTC save time,
monotonic onset/capture timestamps and original frame IDs, pre-roll membership,
gap frame IDs, technical validation, detector motion metrics, model result,
candidate/commit lifecycle, measured geometry predicates, raw trajectory
analysis, post-endpoint observations and stability, source code/config hashes,
model hash and package versions. Capture boundaries and post-endpoint evidence
are separate: the initial model input can end before the quiet endpoint.

`replay_data.npz` contains only numeric/bool arrays, read with
`allow_pickle=False`:

- `frame_indices`, `timestamps`, `frame_sizes` (width, height), `landmarks`
  (n,21,3): exact valid inference-bout entries in chronological order.
- `selected_frame_indices`: the actual model-result selection, never guessed.
- `features`: exact pre-scaler (16,69) matrix copied at `predict_proba` input;
  C-order flattening produces the actual 1104-vector.
- `observed_*`: valid observations from pre-roll through attempt resolution.
- `timeline_indices`, `timeline_timestamps`, `timeline_valid`: observed frame
  availability, including missing frames. Missing frames have no invented
  landmarks and are never inserted into the sequence.

The NPZ SHA256 is stored in JSON. Failed/rejected attempts are still reported,
but if the model was never invoked they have no exact model feature matrix and
model replay is explicitly unavailable. Interrupted/I/O-failed captures are
not silently considered reproducible. Errors are reported once on session exit.

## Geometry predicates

Common J/Z predicates, using the CURRENT historical source conditions:

| Name | Measured condition |
|---|---|
| `extended_and_folded_fraction` | At least .75 of 16 frames jointly satisfy extended tip norm > 1.1 x PIP norm and each other tip norm < 1.3 x its PIP norm. |
| `relative_excursion` | Maximum target-tip displacement relative to wrist >= .08 builder hand scales. |
| `trajectory_excursion` | Maximum target-tip path excursion >= .30 builder hand scales. |

J uses tip 20/PIP 18, other tips 8/12/16; Z uses tip 8/PIP 6,
other tips 12/16/20. Additional J predicates:

- `nonzero_chord`: endpoint chord norm > 1e-9.
- `first_downward`: first leg Y > .15 (image Y points down).
- `hook_horizontal`: absolute hook X > .12.
- `first_hook_dot`: first/hook dot product < .6 x their norm product.

Additional Z predicates, using source positions `[0,5,10,15]`:

- `horizontal_legs`: all three absolute X changes > .12.
- `first_direction_reversal`, `second_direction_reversal`: adjacent X products < 0.
- `middle_downward`: middle-leg Y > .10.
- `first_horizontal`, `last_horizontal`: absolute Y < .6 x absolute X.

Each predicate records its actual measurement, threshold, units, exact source
inequality, boolean result and named failure. Diagnostics measure all predicates
even if source short-circuit would skip later ones. The diagnostic combined
result is tested against the unchanged `geometry_candidates()` implementation.
In particular, the old relative-motion requirement can reject real rigid-hand Z;
these diagnostics do not fix or enforce that requirement.

Raw trajectory metrics separately report pixel XY, monotonic seconds,
pixel/second speeds, headings and signed turn angles (radians), axis reversal
frame IDs, canonical shape-step RMS and palm-scale ratios. No edge is measured
across a gap. Tiny raw direction reversals are NOT classified as real motion.
Detector metrics instead use its anchored scale and EMA. Model trajectory
features use the builder's fixed median selected-frame scale and displacement
from the first selected frame. These definitions are not interchangeable.

## Replay: USER commands, no camera

Replace `<attempt_dir>` with the printed session path plus its attempt folder:

```powershell
python tools/replay_jz_attempt.py --attempt-dir "<attempt_dir>" --check-only
python tools/replay_jz_attempt.py --attempt-dir "<attempt_dir>" --geometry-replay
python tools/replay_jz_attempt.py --attempt-dir "<attempt_dir>" --model-replay
python tools/replay_jz_attempt.py --attempt-dir "<attempt_dir>" --replay
```

Check-only validates hash, required fields, finite arrays, chronological order,
sizes, selected indices and shapes; no classifier invocation. Geometry replay
uses saved landmarks and the original sampler/builder, compares every feature
exactly, and explains geometry predicates. Model replay uses the exact saved
matrix and the existing hash-verified experimental model loader. It checks
label, exact probabilities and maximum probability difference, without fitting.
Results go to stdout; inputs are read-only. Mismatch returns nonzero, not a
synthetic replacement. Package/BLAS/code changes may affect exact reproducibility.
Code-hash matches are reported, not taken as semantic ground truth.

Historical event-only log attempts 5 and 7 lack raw landmarks and actual feature
arrays. They cannot be replayed exactly by this utility; do not reconstruct them
from aggregate metrics or label all historical commits as correct.

## Human review and next camera session

Edit only `human_label` and `note` in the NEW session's human review CSV after
observing the corresponding attempt. Allowed labels: `J_TRUE`, `Z_TRUE`,
`STATIC_TRANSITION`, `HAND_ENTRY_EXIT`, `OTHER_MOTION`, `UNCERTAIN`.
Unreviewed rows start as `UNREVIEWED` even if model/geometry/commit says J or Z.
Use the full `(session_id, session_generation, attempt_id)` key. No tool applies
these labels to acceptance, training or old audit decisions.

For a small comparison session, separately perform true J/Z (slow/fast and
consecutive), Z/J followed by A/U/R/W/K/Y, static-to-static transitions, hand
entry/exit, rotation, rigid translation, other finger motion and short tracking
gaps. Record uncertain truth as UNCERTAIN. Do not tune thresholds on a few
attempts; keep a separate held-out positive/negative comparison set.

## Resource cost / limitations

With capture OFF, no recording history, raw landmark copies or disk output are
added; summary geometry calculations run once per completed proposal. With
capture ON, per-frame copies and up to 512 history / 4096 attempt observations
are bounded, roughly 2.1 MiB for 4096 raw 21x3 float64 arrays plus Python/event
overhead. Attempt-finalization JSON/NPZ writes and trajectory calculations are
synchronous and can cause a brief frame-time spike; no camera FPS claim has
been measured. Capture overflow is explicit, not silently valid replay.

Synthetic tests establish plumbing equivalence, not real ASL recall, false
positive rates or same-hand identity. Capture/diagnostic errors never make a
recognition decision; the original recognition path remains authoritative.
