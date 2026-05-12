# Test Files Analysis & Conclusions Summary

> This document summarizes all learnings from the test files that were previously scattered across the repository. All original test scripts have been removed and consolidated into this single reference.

---

## 1. HSAFA SDK / Cloud Platform Tests (Root Directory)

These files tested the robot's cloud brain (HSAFA Core) integration, skill registration, event flow, and tool-calling pipeline.

### 1.1 Core Architecture Validated
- **SDK Pattern**: `HsafaSDK` with `SdkOptions(core_url, api_key, skill)` is the canonical initialization pattern.
- **Event Flow**: Events are pushed via `sdk.push_event({type, data, haseefId})` and consumed by haseefs over SSE.
- **Tool Calling**: Skills register tools with `sdk.register_tools([...])`, then listen on SSE for `action` events. Handlers are bound via `sdk.on_tool_call(name, handler)`.
- **Lifecycle Events**: The system emits `run.started`, `run.completed`, and `tool.error` events over the SSE stream.

### 1.2 Key Tools Used in Tests
All SDK tests registered the same 4 general-purpose tools:

| Tool | Purpose | Input |
|------|---------|-------|
| `echo` | Pipeline health check | `{message: string}` |
| `get_current_time` | UTC time in ISO + human formats | `{}` |
| `calculate` | Safe math eval (math module only) | `{expression: string}` |
| `get_random_fact` | Returns a random trivia fact | `{}` |

### 1.3 Testing Patterns Discovered
- **Mock Core** (`test_hsafa_mock_core.py`): A full aiohttp mock of HSAFA Core v7 API was built for local dev. It simulated haseef CRUD, memory, runs, skill tool registration, SSE streaming, and even keyword-based tool dispatch (e.g., "time" in message -> `get_current_time`).
- **E2E Flow**: The standard test flow is: (1) start skill service, (2) push event, (3) connect SSE, (4) wait for tool calls, (5) verify runs via REST API.
- **Skill Service**: `test_hsafa_skill.py` is the canonical skill daemon — it registers tools and blocks on `sdk.connect()`.
- **Haseef Setup**: `test_hsafa_haseef.py` showed the full haseef lifecycle: create/update, inject OpenRouter API key for LLM, profile CRUD, memory write/read, skill attachment, and run listing.
- **Hardcoded API Key**: All tests used a shared hardcoded prod key (`sk_prod_7f2e8d9c...`) pointing to `https://core.hsafa.com`.
- **Target Haseef**: Most tests targeted haseef ID `b8f0ead5-036c-4a0b-8afb-e56314acdb9f` or name `TestHaseef`.

### 1.4 SDK API Surface Tested
- `sdk.haseef.list()` / `.get(id)` / `.create(payload)` / `.update(id, payload)` / `.delete(id)`
- `sdk.haseef.get_profile(id)` / `.update_profile(id, profile)`
- `sdk.haseef.status(id)` / `.add_skill(id, skill_name)`
- `sdk.memory.set(id, [{key, value, importance}])` / `.list(id)` / `.search(id, query, limit)`
- `sdk.runs.list(haseef_id=..., limit=...)`
- `sdk.register_tools([...])`
- `sdk.push_event({type, data, haseefId|target})`
- `sdk.connect()` (SSE) / `sdk.disconnect()`
- `sdk.on("run.started", ...)` / `sdk.on("run.completed", ...)` / `sdk.on("tool.error", ...)`
- `sdk.on_tool_call(name, handler)`

---

## 2. Robot Hardware / Vision Tests (Archive Directory)

These files tested physical robot behavior: face tracking, gyro stabilization, object detection, hand gesture recognition, and servo control.

### 2.1 Face Tracking & Head Control
Multiple tests explored face-following with the Reachy Mini robot:

- **Basic Face Follow** (`test_face_follow.py`): Uses YOLOv8-Pose for face detection, P-controller for head yaw/pitch, and supports both local and Reachy daemon cameras. Key tunables: `KP_YAW=0.6`, `KP_PITCH=0.4`, `STEP_SCALE=0.2`, deadzone `0.03`.
- **Gyro-Enhanced Face Tracking** (`test_gyro_face_stabilize.py`, `test_face_world_gyro.py`): BNO055 IMU data was fused with visual tracking to compensate for head motion. Two approaches were tested:
  - **Motion-adaptive smoothing**: Smooth more when head is moving fast, less when stable.
  - **World-coordinate tracking**: Convert visual error to world direction using fused euler angles, then EMA-smooth the world target. This keeps the robot looking at a fixed point in space even as its head rotates.
- **Gyro Calibration**: BNO055 axes had to be mapped empirically. Key calibrated values: `HFOV=90.0`, `VFOV=55.0`, `VERT_SIGN=-1.0` (roll decreases when head pitches down).

### 2.2 BNO055 Gyro Integration
- **Data Format** (`test_bno055_gyro.py`): ESP32 sends 93-byte packets over serial at 460800 baud. Protocol: `[0xAA][0x55][LEN=93][payload][XOR checksum]`. Payload contains quat(4f), euler(3f), acc(3f), lin(3f), grav(3f), gyro(3f), mag(3f), temp(1b), cal(4B).
- **Integration Patterns** (`test_gyro_integration.py`, `test_gyro_visual.py`): Gyro rates were integrated to estimate head pose. Rate integration drifts over time, so a decay factor (e.g., `0.999`) was applied. Fused euler angles (heading/roll/pitch) from BNO055 were preferred for absolute orientation.
- **Axis Mapping**: 8 different axis mappings were tested interactively to find the correct gyro-to-screen transformation. Default winning config: `yaw=Z- pitch=X+`.

### 2.3 Object Tracking (Vision Models)
Several approaches to object tracking were prototyped:

- **Qwen-VL + Optical Flow** (`test_object_tracker.py`): A sophisticated tracker combining:
  - Qwen-VL (via OpenRouter) for initial detection every N frames
  - Lucas-Kanade optical flow on Shi-Tomasi corners for frame-to-frame tracking
  - Forward-backward error filtering to reject bad points
  - RANSAC affine transform for rotation/scale handling
  - Color histogram drift detection (HS channels only, dropping V for lighting invariance)
  - NCC template matching for drift verification
  - Simple Kalman filter for occlusion handling (coasting up to 60 frames)
  - Multi-cue confidence fusion with hysteresis for occlusion state transitions
- **Periodic Qwen + EMA** (`test_periodic_qwen.py`): Simpler approach — call Qwen every 500ms and smooth bbox with EMA (`alpha=0.3`). Fades out bbox gradually if no detection for >10 frames.
- **SAM 3.1 + KLT** (`sam3-test2.py`): Uses Ultralytics SAM 3.1 for semantic segmentation, seeds KLT feature points inside the mask, and tracks at ~30 FPS. Direct pixel error -> head angle with `GAIN=0.6`, `DAMP=0.25`.
- **Qwen-VL + SAM 2** (`new-teck-test.py`): Two-stage pipeline — Qwen grounds a text description to a bbox, then SAM 2 tracks it frame-by-frame locally. Uses OpenRouter's OpenAI-compatible API.
- **Florence-2** (`florence2-test.py`): Fully local vision model (~500MB) for text-to-segmentation. Two-stage: `<CAPTION_TO_PHRASE_GROUNDING>` then `<REGION_TO_SEGMENTATION>` per bbox. No API key needed.

### 2.4 Optical Flow Tracking (LK)
- **Point Tracker** (`test_lk_points.py`): Minimal LK + forward-backward filter test. `FB_THRESHOLD_PX=1.0` was the standard. Points that fail FB filter flash red for one frame before being dropped.
- **BBox Tracker** (`test_lk_bbox.py`): MEDIANFLOW-style bbox tracker. Key techniques:
  - Per-pair distance ratios for scale estimation (unbiased under outliers)
  - Per-frame scale clamped to `[0.97, 1.03]` to prevent bbox blow-up
  - Spatial outlier rejection on displacement (MAD-based, `DISP_MAD_K=3.0`)
  - Replenish points only when confidence >= 0.5

### 2.5 Hand Gesture Detection
- **Object Held** (`test_object_held.py`): MediaPipe hand landmarks used to detect if hand is:
  - **OPEN** — fingers extended
  - **EMPTY GRASP** — compact fist (fingertips close to wrist)
  - **HOLDING** — grasping posture but not compact (object keeps fingers apart)
- Logic: `is_holding = is_showing(wrist high, hand large, stable) AND is_grasping AND NOT is_compact_fist`

### 2.6 Camera & Reachy Integration
- Both local OpenCV cameras and Reachy daemon cameras were supported.
- Reachy camera requires daemon running (`./scripts/daemon.sh start`).
- Camera probing with AVFoundation backend on macOS.
- Standard resolution: 640x480.

---

## 3. Conclusions & Design Decisions

### 3.1 What Worked Well
1. **HSAFA SDK Pattern**: The event-driven skill architecture (register tools -> push event -> SSE tool calls) is clean and testable. The mock core enabled full local E2E testing.
2. **World-Coordinate Face Follow**: Using BNO055 fused angles to maintain a world-fixed target proved more stable than raw visual tracking alone.
3. **LK + FB Filter**: Forward-backward error filtering at 1px threshold effectively removed drifting feature points.
4. **Multi-cue Tracking**: Fusing optical flow, template matching, color histogram, and Kalman prediction gave robust tracking with graceful degradation during occlusion.
5. **Safe Math Eval**: The `calculate` tool uses `eval()` with restricted globals (`__builtins__: None` + math module) — a secure pattern for LLM-triggered math.

### 3.2 What Was Learned / Tuned
1. **Gyro Axis Mapping**: Must be empirically calibrated per mounting orientation. The BNO055 "roll" axis corresponds to head nod, not the chip's pitch.
2. **Scale Clamping**: Without per-frame scale limits, optical-flow trackers suffer from "bbox grows on every frame" failure mode.
3. **Qwen-VL Latency**: Vision LLM calls take ~1-3s, so they cannot run every frame. Best pattern: periodic re-detection + fast local tracker (LK/SAM/Kalman) for inter-frame motion.
4. **Confidence Hysteresis**: Occlusion state transitions need hysteresis thresholds to avoid flickering between lost/tracking states.
5. **Color Histogram**: Dropping the V channel and using only H+S makes color matching robust to lighting changes.

### 3.3 Abandoned / Superseded Approaches
- Raw gyro rate integration for face tracking (drifts too much; fused euler is better).
- Simple EMA-only smoothing without gyro compensation (insufficient for moving head).
- Periodic Qwen-only tracking (too laggy without optical flow interleaving).
- Florence-2 was evaluated but likely superseded by Qwen-VL + SAM 2/3 for flexibility.

### 3.4 Key Configuration Values
| Parameter | Value | Context |
|-----------|-------|---------|
| HSAFA Core URL | `https://core.hsafa.com` | Production API |
| API Version | `/api/v7` | REST + SSE endpoints |
| Default Skill | `general_tester` | SDK test skill |
| Default Model | `openai/gpt-5.4-mini` (OpenRouter) | Haseef LLM |
| BNO055 Baud | 460800 | ESP32 serial |
| LK Win Size | `(21, 21)` | Optical flow |
| LK Pyramid | 3-4 levels | Scale handling |
| FB Threshold | 1.0 px | Point quality filter |
| H FOV | 90 deg | Reachy camera |
| V FOV | 55 deg | Reachy camera |
| Face P-control | KP_YAW=0.6, KP_PITCH=0.4 | Head servo |
| Head Limits | yaw +/-60, pitch +/-30 deg | Safety |
| Qwen Interval | 500ms | Periodic detection |
| EMA Alpha | 0.3 | Smoothing |

---

## 4. File Inventory (Deleted)

### Root Directory (9 files)
- `test_event_simple.py` — Simple event push + tool listen
- `test_hsafa_discovery.py` — API connectivity probe
- `test_hsafa_e2e.py` — Full E2E with background skill
- `test_hsafa_event_e2e.py` — Event E2E with OpenRouter injection
- `test_hsafa_final.py` — Clean listen-then-push pattern
- `test_hsafa_haseef.py` — Haseef CRUD + memory + runs
- `test_hsafa_mock_core.py` — Local mock HSAFA Core server
- `test_hsafa_skill.py` — General test skill daemon
- `test_push_event.py` — Quick event push utility

### Archive / Examples (17 files)
- `archive/test_bno055_gyro.py` — BNO055 packet decoder
- `archive/test_face_follow.py` — Face-follow with gyro toggle
- `archive/test_face_world_gyro.py` — World-coordinate face follow
- `archive/test_gyro_face_stabilize.py` — Gyro-stabilized tracking
- `archive/test_gyro_face_tracking.py` — Gyro integration test suite
- `archive/test_gyro_integration.py` — ESP + Reachy integration
- `archive/test_gyro_visual.py` — Gyro visual overlay test
- `archive/test_object_held.py` — Hand grasp/holding detection
- `archive/test_object_tracker.py` — Qwen + LK + Kalman tracker
- `archive/test_periodic_qwen.py` — Periodic Qwen + EMA
- `archive/hsafa-robot-v2/tests/test_lk_bbox.py` — MEDIANFLOW bbox tracker
- `archive/hsafa-robot-v2/tests/test_lk_points.py` — LK point tracker
- `archive/examples/florence2-test.py` — Florence-2 local segmentation
- `archive/examples/new-teck-test.py` — Qwen-VL + SAM 2 pipeline
- `archive/examples/sam3-test2.py` — SAM 3.1 + KLT head follow
- `archive/examples/sam3_camera_test.py` — SAM 3.1 remote client
- `archive/examples/sam3_test_client.py` — SAM 3.1 HTTP test client
- `archive/examples/sam3_ws_test.py` — SAM 3.1 WebSocket latency test
