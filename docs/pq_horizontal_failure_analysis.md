# P horizontal-view failure analysis

## Confirmed facts

- Runtime loads the selected Baseline MLP checkpoint (epoch 143), TRAIN-fitted
  scaler, 24-label mapping and Hand Landmarker only after SHA-256 verification.
- Camera inference uses MediaPipe's 21 landmarks in their native index order and
  calls `AI/landmark_utils.py::normalize_landmarks`, the same function used by
  collection/extraction. It does not fit the scaler or remap P/Q.
- The canonical normalizer is wrist-relative and divides by maximum 3-D wrist
  distance. It is translation/scale normalized, but not rotation/viewpoint
  invariant. A horizontal view therefore produces a genuinely different feature
  vector that the MLP must learn from examples.
- P/Q overlap already exists in held-out results: validation has P→Q 2/491 and
  Q→P 6/501; TEST v1 has P→Q 4/491 and Q→P 2/502. High confidence is not semantic
  proof and does not rule out an out-of-distribution/viewpoint error.
- TRAIN has 2,290 P samples/1,563 groups and 2,341 Q samples/1,548 groups. All
  are tagged `internet`; no view-angle metadata or filename view tag exists.
- A group-distinct 64-image visual sample contains examples resembling both
  front/downward and horizontal/side views for P and Q. Thus total absence of
  horizontal P is not established. The visible capture styles, subjects and
  backgrounds are nevertheless concentrated, so coverage of the user's exact
  live pose is unknown.

## Unresolved cause

No saved frame exists yet for the reported live failure. Consequently it is not
yet possible to decide whether indices 4/8/12 follow the thumb/index/middle
fingertips correctly. Capture the same pose with the debug mode. If the indexed
overlay is wrong, investigate MediaPipe occlusion/correspondence. If the overlay
is anatomically correct while P probability remains below Q, classify this as a
model/data coverage failure. Do not use confidence alone for this decision.

## Four-condition camera challenge

Capture manually confirmed P-front, P-horizontal, Q-front and Q-horizontal in
separate sessions. Record person/session, hand, view, distance and lighting.
Include small rotations around each target view and true Q examples under the
same camera conditions. Keep related frames from one recording burst/session in
one group. Reserve people/sessions before model work:

1. development TRAIN captures;
2. development VALIDATION captures for model selection;
3. an untouched challenge TEST captured in separate sessions, preferably with
   different people.

Inspect every indexed overlay before admitting a sample. A frame with incorrect
landmark correspondence is excluded from the landmark corpus, not relabelled.

## Model-v2 gate

Do not create model v2 until debug captures establish a classifier/data issue.
Then start with the same architecture and add only verified P/Q development
captures to the v2 TRAIN partition, with group/session-aware splitting and a new
scaler fit only on v2 TRAIN. Recheck all 24 classes plus condition-level P/Q
recall and confusion. If data coverage is insufficient, collect more people and
sessions rather than introduce a P/Q angle rule.

TEST v1 has already been evaluated and must remain historical evidence only. It
must not tune model v2 and cannot serve as the sole untouched final estimate for
an iterative v2 workflow. Use the newly reserved person/session-aware challenge
TEST once, after the v2 procedure is frozen. A rotation-normalized representation
would be a separate preprocessing/model experiment requiring complete versioned
regeneration; it is not a patch to the current frozen pipeline.
