# Tests

Standalone test scripts. **No Gemini, no Haseef, no robot motion** — these
exercise local perception/cognition modules in isolation.

## `test_face_tracking.py`

Vision + audio test of the slimmed face-recognition + active-speaker design
from `some-ideas.md` §5–§6.

What it does:

- Opens the Reachy USB camera (auto-detects, falls back to index 0).
- Detects + tracks faces every frame (InsightFace + simple centroid tracker).
- Reads the mic and runs Silero VAD → "is anyone speaking right now?".
- When VAD says yes, computes per-face **lip-motion score** (frame-diff in
  the mouth region) and picks the **active speaker** (argmax with 3-frame
  hysteresis).
- The active speaker is **silently enrolled** into the IdentityManager
  gallery (anonymous-first, dual-threshold matching, multi-frame voting,
  duplicate-resistant per §6.4).
- Draws everything on screen:
  - Gray bbox = other people in view
  - **Green bbox** = the person Reachy should look at right now
  - Label above each face: `face_id` and `(name)` if assigned
  - HUD top-left: VAD state, FPS, total identities in gallery

Keys (while the window is focused):

```
q   quit
n   name the current speaker  (prompts in terminal)
c   clear all identities       (panic / privacy reset)
```

### Install (one-time)

```bash
.venv/bin/pip install insightface onnxruntime sounddevice
```

InsightFace will auto-download the `buffalo_sc` model (~70 MB) on first run.

### Run

```bash
.venv/bin/python tests/test_face_tracking.py
```

CLI options:

```
--camera N        camera index to use (default: auto-detect, then 0)
--no-audio        skip mic / VAD, run vision only
--det-size 640    InsightFace detection size (smaller = faster)
```
