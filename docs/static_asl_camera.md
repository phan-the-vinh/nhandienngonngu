# Static ASL camera

The camera uses the frozen Baseline MLP selected on validation at epoch 143.
It recognizes only the 24 static classes `A-I, K-Y`; `J` and `Z` require a
separate dynamic-gesture pipeline.

Run on Windows from the project root:

```powershell
conda activate nn
python -B AI\static_asl_camera.py --check-only
python -B AI\static_asl_camera.py --camera-index 0
```

The live frame is mirrored by default, matching the existing data-collection
camera flow. Use `--no-mirror` only when the camera source is already mirrored.
Press `Q` or `Esc` to exit.

`Instant` is the current-frame prediction. `Stable` appears after temporal
agreement in the display smoother (default: 4 votes in a 7-frame window).
When the hand disappears or landmarks are invalid, both states are cleared.
No word composition, model fitting, scaler fitting, or threshold tuning occurs.

Camera check:

1. Hold one clear static ASL handshape inside the frame.
2. Confirm the 21-point overlay follows the hand.
3. Confirm `Instant` responds per frame and `Stable` changes less frequently.
4. Remove the hand and confirm the application shows `prediction cleared`.
5. Try `M`, `N`, and `T`, then quit with `Q`.

## P/Q debug capture

Debug images are saved only when `S` is pressed. Start four separately labelled
sessions; do not mix handshapes or views in one command:

```powershell
python -B AI\static_asl_camera.py --debug-dir AI\camera_debug_pq --debug-label P --debug-view front
python -B AI\static_asl_camera.py --debug-dir AI\camera_debug_pq --debug-label P --debug-view horizontal
python -B AI\static_asl_camera.py --debug-dir AI\camera_debug_pq --debug-label Q --debug-view front
python -B AI\static_asl_camera.py --debug-dir AI\camera_debug_pq --debug-label Q --debug-view horizontal
```

Each press of `S` stores the exact mirrored frame sent to MediaPipe, an indexed
landmark overlay, raw 21-landmark coordinates, canonical `f1..f63`, all class
probabilities, explicit P/Q probabilities, and the manual label/view. For each
condition, collect several independent poses while varying small rotations,
distance, lighting and background. Confirm the handshape manually before every
capture. Landmark indices 4, 8 and 12 are the thumb, index-finger and
middle-finger tips and should visibly remain attached to the correct fingers.

The model was evaluated with duplicate/variant `group_id` separation. Because
performer and session metadata are unavailable, this does not prove
generalization to entirely new people.

## U/V debug capture

Use a separate debug root so U/V captures cannot mix with the existing P/Q
collection. Each condition is stored under `U_front`, `U_horizontal`,
`V_front`, or `V_horizontal` beneath `AI/camera_debug_uv`:

```powershell
python -B AI\static_asl_camera.py --debug-dir AI\camera_debug_uv --debug-label U --debug-view front
python -B AI\static_asl_camera.py --debug-dir AI\camera_debug_uv --debug-label U --debug-view horizontal
python -B AI\static_asl_camera.py --debug-dir AI\camera_debug_uv --debug-label V --debug-view front
python -B AI\static_asl_camera.py --debug-dir AI\camera_debug_uv --debug-label V --debug-view horizontal
```

Press `S` once per intended sample. The saved JSON/CSV includes the manual
label probability as `probability_manual_label` as well as the full 24-class
probability map. Existing files in `AI/camera_debug_pq` are not read or changed.
