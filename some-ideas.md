

# Hsafa / Reachy Mini — New Architecture: Socially-Aware Multi-Person Attention

The big upgrade is going from **"track one body"** to **"decide which person to look at, and why"** — that's what makes a robot feel alive instead of mechanical. This doc reflects the *current* Hsafa architecture (`main.py`) where **Haseef is the slow brain** (Hsafa Core, owns memory + tools) and **Gemini Live is the fast voice/eyes surface**. New tools below are added to **Haseef**, not Gemini.

---

## 1. The Core Concept: A Saliency / Attention Brain

Instead of one tracker locking onto a body, imagine a **mental scoreboard**. Every person in the scene gets a continuously updated **attention score**, and Reachy always looks at the highest scorer. Smooth transitions are blended so it doesn't snap robotically.

This is exactly how the research literature does it — it's called a **Gaze Control System (GCS)** with a competitive/saliency network. Each person accumulates points from multiple cues:

```
attention_score(person) =
      w_speak   * is_speaking          (the BIGGEST factor — ~40%)
    + w_addr    * is_addressing_robot  (head/face turned to camera)
    + w_prox    * proximity_score      (closer = more relevant)
    + w_motion  * motion_salience      (waving, gestures)
    + w_engage  * engagement_history   (have they been interacting?)
    - w_habit   * habituation          (looked too long → drift away)
    + w_new     * novelty_bonus        (new person just appeared)
```

The **highest score wins**, but with hysteresis (a stickiness margin) so Reachy doesn't flicker between two people who are close in score. This matches how humans actually do it.

### Why this is better than the old logic
The old `main.py` tracked one ID forever. The new logic means Reachy will:
- Look at whoever is **currently speaking**
- Glance at someone who **suddenly waves**
- Return to the **main speaker** after a quick check
- Slowly **disengage** from someone who stopped interacting
- Greet a **new person** entering the scene

> **Important:** this loop runs **locally** inside the robot process. It is a **reflex**, not a tool call. Sending every gaze decision through Haseef would be way too slow. Haseef only gets involved when a *decision* needs reasoning (who is this person, should we greet them by name, etc.).

---

## 2. The Speaker Question (Best Free Upgrade)

Reachy Mini has a **4-mic linear array** built on the Seeed reSpeaker XMOS XVF3800 (4 PDM mics, 16 kHz). This means we can get **DOA (Direction of Arrival)** — the azimuth angle of whoever is talking. The current stack only uses Silero VAD (yes/no speech), which leaves the mic array unused.

### The fusion trick
A single signal is unreliable. Combine three:

| Signal | What it tells you | Confidence |
|---|---|---|
| **Audio DOA** (mic array) | Roughly *where* the sound came from (±10–15°) | Coarse |
| **Visual lip motion / face activity** | *Who* is moving their mouth | Precise but needs visible face |
| **VAD** | *When* speech is happening | Gate |

→ When VAD says "speech now," check DOA angle, project that ray into the camera frame, find the nearest person bbox along that ray, and **confirm with visual lip motion**. The person whose lips move + who is along the DOA ray = **active speaker**, with high confidence. Same approach as SIG / ROBITA / RASA in the literature.

In the lightweight case, skip the heavy lip-motion model and just use **DOA + VAD + face-orientation-toward-camera** — already enough for 80%+ accuracy in 2-person scenes.

---

## 3. Proposed Architecture (Layered)

```
┌──────────────────────────────────────────────────────────────┐
│ L3 — DIALOGUE (fast surface)                                 │
│   • Gemini Live — voice, ears, real-time camera stream       │
│   • Local tools only: queue_thinker_task, get_current_time,  │
│     ping (kept exactly as in main.py)                        │
│   • Everything else → queue_thinker_task → Haseef            │
└──────────────────────────────────────────────────────────────┘
                            ▲   ▼
┌──────────────────────────────────────────────────────────────┐
│ L2.5 — HASEEF (slow brain on Hsafa Core)                     │
│   • Owns memory, schedules, naming, reasoning                │
│   • Existing tools (keep): create_schedule, list_schedules,  │
│     cancel_schedule, look_around, set_head_pose, say_this,   │
│     capture_image, show_expression                           │
│   • NEW tools (this doc):                                    │
│       look_at_person, set_attention_mode, set_mood,          │
│       name_face, rename_face, merge_faces, forget_face,      │
│       list_known_faces, describe_scene, ask_about_identity   │
└──────────────────────────────────────────────────────────────┘
                            ▲   ▼
┌──────────────────────────────────────────────────────────────┐
│ L2 — SOCIAL COGNITION (the new local brain)                  │
│   • AttentionManager (saliency scoreboard, hysteresis,       │
│     habituation, smooth target selection)                    │
│   • SpeakerFusion (DOA × VAD × visual → speaker_id)          │
│   • IdentityManager (face galleries, anonymous-first IDs)    │
│   • WorldState (one snapshot of "who is where, doing what")  │
│   • EventBus (typed pub/sub)                                 │
└──────────────────────────────────────────────────────────────┘
                            ▲   ▼
┌──────────────────────────────────────────────────────────────┐
│ L1 — PERCEPTION                                              │
│   • PersonTracker  → YOLO11n-pose + ByteTrack                │
│   • FaceTracker    → YOLO11n-face (5 landmarks)              │
│   • FaceEmbedder   → ArcFace / InsightFace 512-d             │
│   • AudioDOA       → SRP-PHAT on 4-mic array                 │
│   • VAD            → Silero VAD                              │
│   • GestureCue     → wave/point rules on pose keypoints      │
└──────────────────────────────────────────────────────────────┘
                            ▲   ▼
┌──────────────────────────────────────────────────────────────┐
│ L0 — MOTION                                                  │
│   • RobotControl: P-controller, head limit → body yaw        │
│   • Animation: idle drift, listening nod, talking overlay    │
│   • SmoothGaze: minimum-jerk trajectory between targets      │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. What to Keep, Drop, Add

### ✅ Keep from current stack (`main.py`)
- The Gemini ↔ Haseef bridge pattern (`UnifiedBridge`).
- All **8 existing Haseef tools**: `create_schedule`, `list_schedules`, `cancel_schedule`, `look_around`, `set_head_pose`, `say_this`, `capture_image`, `show_expression`.
- Gemini side: `queue_thinker_task`, `remember_fact`, `get_current_time`, `ping`.
- `asyncio` / `threading` / `signal` / `logging` skeleton.
- ByteTrack, Silero VAD, OpenCV camera, scipy `Rotation`.

### ❌ Drop / don't propose (already solved or not useful)
- ~~`play_animation`~~ → already covered by `show_expression` (full animated emotion clips with motion + sound).
- ~~`remember_fact` as a "new" Gemini tool~~ → already exists in `main.py`.
- ~~`point_at_object`~~ → Reachy Mini has **no arms**. Can't physically point. Use head + body yaw + antenna gesture instead.
- ~~Separate `search_web` tool~~ → Gemini Live has **built-in Google Search**; just enable the `googleSearch` tool, no custom plumbing.
- ~~MediaPipe~~ — heavy and redundant once we have YOLO faces.
- ~~`facenet-pytorch` as a runtime dep~~ — replace with InsightFace ONNX (lighter, ArcFace embeddings).
- ~~`speechbrain` voice ID~~ — the mic array tells you direction; voiceprints are overkill for short sessions.
- ~~MOG2 background subtraction~~ — YOLO11 + ByteTrack rarely loses targets now.
- ~~Standalone Kalman filter~~ — already inside ByteTrack.
- ~~"Spatial-memory-only, no face DB"~~ — we now explicitly want a face DB (see §6) so identity persists across sessions. The role-tag layer is *additional*, not a replacement.

### ➕ Add (new local modules)
| Module | What it does | Library |
|---|---|---|
| **YOLO11n-face** | Face bboxes + 5 landmarks | `ultralytics` + `akanametov/yolo-face` weights |
| **InsightFace ArcFace** | 512-d face embeddings | `insightface` (ONNX, CPU OK) |
| **AudioDOA** | Sound direction from 4-mic array | `pyroomacoustics` (SRP-PHAT) |
| **SpeakerFusion** | DOA ray + face activity + VAD → `speaker_id` | custom (~150 lines) |
| **AttentionManager** | Saliency scoreboard + hysteresis | custom (~200 lines) |
| **IdentityManager** | Anonymous-first face gallery + matching | custom (see §6) |
| **GestureCue** | Wave / point detection from pose keypoints | custom (geometric rule) |
| **SmoothGaze** | Minimum-jerk head trajectory | custom (~50 lines) |

---

## 5. Tool Surface (aligned with current `main.py`)

### Gemini Live — keep tiny, with one small extension
Exactly what is in `build_gemini_tools()` today, with **one new optional parameter** on `queue_thinker_task`:

```python
{"name": "queue_thinker_task",
 "parameters": {
     "task": "string",
     "what_i_told_user": "string",
     "with_image": "boolean?"   # NEW — default False
 }}
```

When `with_image=true`, the bridge grabs the current camera frame and attaches it to the event Haseef receives (`push_event` already supports `attachments` — see `_push_image_event` in `main.py`). Gemini sets this flag whenever the task is visually grounded: *"who is in front of me?"*, *"what is on the table?"*, *"does this person look like Sara?"*. **This replaces the previously-proposed `describe_scene` tool entirely** — Haseef just gets the pixels directly, on demand, through the channel that already exists.

If Haseef needs an image but Gemini didn't attach one, Haseef calls `say_this` to ask Gemini for one (*"send me what you see"*), and Gemini then re-issues `queue_thinker_task(..., with_image=true)`. No new tool needed — it's a natural two-turn handshake.

### 5.1 Pre-labelled scene images (how Haseef distinguishes people visually)

Haseef lives on the cloud and **never sees raw 512-d embeddings** (privacy + latency). To let Haseef reason about *which* face is Husam and *which* is Rayan, the robot **draws bounding boxes with names directly on the camera frame before attaching it**:

```
┌─ Robot local (pre-label) ─┐     ┌─ Haseef cloud ─┐
│  1. Run face detection    │     │                │
│  2. Match each track to   │     │                │
│     face_id + name        │     │                │
│  3. Draw on frame:        │ ──▶ │  Reads labels  │
│     red box  = "Husam"    │     │  on the image  │
│     blue box = "Rayan"    │     │  directly      │
│     yellow   = "unknown"  │     │                │
└───────────────────────────┘     └────────────────┘
```

**Why this works:**
- Haseef is a vision-language model. It can read text overlaid on an image just like a human reads sticky notes.
- No `face_id` ↔ embedding mapping needed in Haseef's memory. Haseef only needs the `face_id` string when calling tools like `name_face(face_id="face_a1b2", name="Husam")`.
- The robot handles all biometric matching locally. Haseef handles names, facts, and social reasoning.

**When to attach the labelled image:**
- Gemini sets `with_image=true` on `queue_thinker_task` whenever the topic involves visible people: *"who is in front of me?"*, *"did Sara leave?"*, *"the person on the left said what?"*.
- The robot draws all visible faces (named or anonymous) with their current `face_id` and name, then JPEG-encodes and attaches.

**Alternative (Option B):** If Haseef ever needs structured data instead of pixels, `list_visible_people` returns a JSON array of `{face_id, name?, bbox, is_speaking}` alongside the image. Option A is preferred because it requires zero new model capabilities — Haseef already sees images.

**No other new Gemini tools.** Everything else goes through `queue_thinker_task` to Haseef.

### Haseef — *new* tools to add to `setup_haseef()`'s `register_tools` list

```python
{"name": "look_at_person", "input": {"face_id": "string"}}
# Locks AttentionManager onto that face_id (raises its score).

{"name": "set_attention_mode", "input": {"mode": "string"}}
# mode: "focus" (lock current target), "scan" (round-robin everyone),
#       "idle" (biological saccades), "follow_speaker" (default).

# (Dropped per §7 over-engineering check: `set_mood` removed — use
# `show_expression` for emotional output.)

# --- Identity tools (see §6 for the data model) -----------------------
{"name": "list_known_faces", "input": {}}
# Returns [{face_id, name?, last_seen_at, embedding_count, in_view: bool}].
# This is metadata about the gallery — NOT a scene description. For the
# current scene, Haseef just looks at the image attached via
# queue_thinker_task(with_image=true).

{"name": "name_face", "input": {"face_id": "string", "name": "string"}}
# Assigns/updates a name on an existing face_id (was anonymous → "Sara").
# Overwrites silently if a name already exists. (Subsumes the dropped
# `rename_face` tool per §7.)

{"name": "merge_faces",
 "input": {"primary_id": "string", "secondary_id": "string"}}
# When the robot realizes two anonymous IDs are the same person, fold the
# secondary's gallery into the primary. Idempotent.

{"name": "forget_face", "input": {"face_id": "string"}}
# Privacy / cleanup. Deletes gallery + name.
```

> **No `ask_about_identity` tool.** When the IdentityManager is uncertain, it pushes an `identity.uncertain` event to Haseef with the candidate name(s). Haseef just calls `say_this("Have we met before? You look like Sara.")` using the existing tool — that's exactly what `say_this` is for. The listening window opens automatically because Gemini Live is always listening; the user's reply flows back as a normal turn, and Gemini either updates the name via `queue_thinker_task` (which then triggers `name_face`/`merge_faces`) or leaves it alone. **No new tool needed.**

**Key separation of duties (matches current code):**
- **Reflexes** (where to look right now, idle saccades, smooth gaze) → handled locally by `AttentionManager` + `SmoothGaze`. Never round-trip to Haseef.
- **Decisions** (greet by name? confirm identity? schedule a reminder?) → Haseef via `queue_thinker_task`, then Haseef calls the tools above.

---

## 6. Face Recognition: Human-Like, Anonymous-First, Robust to Mistakes

This is the section you specifically asked for. The goal: **every person who speaks is silently embedded into the gallery without needing a name. Haseef can name/rename/merge later. The system must NOT spawn a new identity every time the same face is seen at a slightly different angle.**

### 6.1 Design principles (how humans actually work)

1. **Tentative recognition first, commit later.** A human looks at a stranger and thinks *"maybe I've seen them before"* before deciding. The system mirrors this with three states: `provisional → confirmed → named`.
2. **Multi-view memory, not one face per person.** You don't remember a friend from one photo; your brain stores many views (lighting, angle, expression). Each `face_id` keeps a **gallery of N embeddings**, not one centroid.
3. **Continuity dominates recognition in the short term.** If you've been talking to someone for 10 seconds, you don't re-identify them every blink — you trust object permanence (the tracker). Re-ID only fires on track entry / track loss / periodic refresh.
4. **When in doubt, ask.** Humans go *"sorry, have we met?"* — so should the robot. This is done with the existing `say_this` tool, not a special one (see §6.8).
5. **Privacy-first defaults.** Anonymous IDs only; names only attach when the user volunteers them or Haseef explicitly assigns one. `forget_face` always works.

### 6.2 The pipeline

```
                ┌────────────────────────┐
camera frame ──▶│ FaceTracker (YOLO11n)  │── face crops + landmarks
                └────────────────────────┘
                          │
                          ▼
                ┌────────────────────────┐
                │  Quality filter        │
                │  ・min size (≥80 px)   │
                │  ・blur (Laplacian var)│
                │  ・|yaw|<35° |pit|<25° │
                │  ・not occluded        │
                └────────────────────────┘
                          │ (good crops only)
                          ▼
                ┌────────────────────────┐
                │  ArcFace 512-d embed   │
                └────────────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
   ┌────────────────────┐   ┌────────────────────┐
   │  Tracker bridge    │   │  Re-ID match       │
   │  (same track id?   │   │  (new track / lost │
   │  inherit face_id)  │   │  track / refresh)  │
   └────────────────────┘   └────────────────────┘
                                      │
                                      ▼
                ┌────────────────────────────────┐
                │ IdentityManager.match(emb)     │
                │  → high   → confirm + add      │
                │  → low    → new provisional    │
                │  → middle → multi-frame vote   │
                └────────────────────────────────┘
                          │
                          ▼
                ┌────────────────────────┐
                │ Speaker-gated enroll   │
                │ (only add to gallery   │
                │  while person speaks)  │
                └────────────────────────┘
```

### 6.3 The IdentityManager data model

```python
@dataclass
class FaceEntry:
    face_id: str                  # "fp_a1b2..." stable
    name: str | None              # None until Haseef calls name_face
    embeddings: list[Embedding]   # gallery, max 50
    pose_buckets: dict[str, int]  # "front"/"left"/"right" coverage counts
    first_seen_at: float
    last_seen_at: float
    confirmed: bool               # False = provisional
    seen_count: int               # how many distinct sessions
    notes: str | None             # Haseef-written ("wears glasses")

@dataclass
class Embedding:
    vec: np.ndarray               # 512-d L2-normalized
    quality: float                # composite score
    pose_bucket: str
    captured_at: float
    source: str                   # "speaker"|"passive"|"asked"
```

### 6.4 Matching algorithm (the anti-duplication core)

```
match(new_emb):
    candidates = []
    for entry in gallery:
        # Take cosine sim against the K=5 most-pose-similar gallery embs.
        sims = top_k_cosine(new_emb, entry.embeddings, k=5)
        score = mean(sims)
        candidates.append((entry, score))

    candidates.sort(by score desc)
    best, best_score = candidates[0]
    second_score    = candidates[1].score if len > 1 else 0

    if best_score >= 0.55 AND (best_score - second_score) >= 0.05:
        return MATCH(best)                      # confident
    elif best_score >= 0.40:
        return PROVISIONAL(best, best_score)    # ambiguous
    else:
        return NEW                              # confident new
```

**Hysteresis & voting** to avoid spawning duplicates:

- A `NEW` decision does **not** immediately create a `face_id`. It enters a **pending pool** and must be confirmed by **≥ 8 quality embeddings within the same track** before becoming a real (still anonymous) `face_id`. A passing stranger doesn't pollute the DB.
- A `PROVISIONAL` match accumulates votes across frames. If 70 % of frames over a 2-second window agree on the same candidate, **commit**. Otherwise stay provisional and trigger `ask_about_identity` if the person is engaged.
- A `MATCH` adds the new embedding to the gallery only if (a) its pose bucket is under-represented OR (b) its quality > median of the gallery. This keeps the gallery diverse but bounded.

### 6.5 Speaker-gated enrollment (the quality trick)

The single biggest win: **only enroll embeddings during confirmed speech from this person**. SpeakerFusion already gives `speaker_id`. While they speak:

- Their face is almost certainly oriented toward the robot (people look at who they talk to).
- They are stationary-ish.
- We get many frames in a few seconds.

This produces a **high-quality, frontal-leaning** anchor gallery. Passive (non-speaker) frames are added more conservatively — only if pose bucket is under-represented, and tagged `source="passive"` so we can downweight them later.

### 6.6 Tracker continuity bridge (don't re-embed every frame)

While ByteTrack keeps the same `track_id`:
- The face_id is **inherited** from the first commit on that track. Skip re-matching.
- Run a **refresh embedding** every 5 s (or on big yaw change > 30°) and append to the gallery if quality is good. This is how the gallery grows naturally.

When a track is lost or a new one appears within 0.5 s nearby:
- Run full re-ID match against the gallery before deciding it's a new person.

### 6.7 Online consolidation (background merge)

Every 60 s, or when `merge_faces` is called:
- Pairwise mean-cosine-sim between every pair of `confirmed` `face_id`s.
- If sim ≥ 0.50 AND pose bucket overlap is high, propose merge.
- If both are anonymous → auto-merge silently (most common case: same person enrolled twice in different sessions because lighting was very different).
- If one or both are named → push event to Haseef: *"face_id A (Sara) and face_id B (anonymous) might be the same. Confidence 0.62."* Haseef decides whether to call `merge_faces` directly or trigger the §6.8 "have we met?" dialogue (just `say_this` + Gemini's normal listening loop).

This is what stops the system from accumulating 30 "person_X" entries for the same friend over a month.

### 6.8 "Have we met?" loop — the human move (no new tool)

Borderline scores trigger a **dialogue resolution**, not a silent guess. This uses only existing tools:

1. IdentityManager flags a `PROVISIONAL` match with confidence 0.45.
2. Pushes event `identity.uncertain` to Haseef with `face_id`, `candidate_face_id`, `candidate_name` (using the existing `push_event` channel — same one `_push_image_event` uses).
3. Haseef reads the event and, if the user is engaged and not mid-task, calls `say_this("Have we met before? You look like Sara.")`.
4. Gemini Live speaks it. The microphone is always open, so the user's reply flows back naturally as a normal conversation turn.
5. Gemini interprets the answer:
   - **Confirm** → `queue_thinker_task(task="User confirmed they are Sara, face_id=X")` → Haseef calls `name_face` (and `merge_faces` if a candidate id existed).
   - **Deny** → `queue_thinker_task(task="User denied being Sara, face_id=X")` → Haseef marks the two `face_id`s as **anti-linked** (hard negative pair: future matching between them gets a penalty).
   - **Ignore / change subject** → Haseef keeps the id provisional and doesn't pester.

This is why no `ask_about_identity` tool is needed: "ask the user a question" is just `say_this` + Gemini doing its normal listening job.

### 6.9 Confidence decay & pruning

- Embeddings older than 30 days with no re-confirmation get a 0.5× weight in matching.
- Gallery cap = 50 per `face_id`. When full, evict by `(quality × recency × pose_diversity)` — keep the gallery a *spanning set*, not a recent set.
- `face_id`s with `seen_count == 1` and `last_seen_at > 14 days` are auto-pruned (was probably a one-off passerby).
- Named faces are **never auto-pruned**.

### 6.10 Failure modes handled explicitly

| Failure | Mitigation |
|---|---|
| Same person enrolled twice in different lighting | Online consolidation §6.7 catches it within a session. |
| Twins / siblings get merged | Anti-link from §6.8 user denial. Plus higher merge threshold (0.50 strict). |
| Bad crop creates a junk gallery entry | Quality filter §6.2 + `quality` weighting in match. |
| User wears glasses / hat → poor match | Pose bucket diversity in gallery + speaker-gated re-enrollment naturally captures the new look during conversation. After a few seconds of speaking, the gallery learns the new appearance. |
| Robot greets stranger by friend's name | Dual-threshold matching (0.55 + margin 0.05) makes false-positive named greeting rare. Borderline → `ask_about_identity` instead of greeting. |
| Privacy: someone wants out | `forget_face` deletes everything. `list_known_faces` is auditable. |
| Long-lost contact returns months later | Named faces never expire. Gallery diversity covers age/look drift. |

### 6.11 Two-layer identity (session role-tag ⟂ long-term face_id)

Independent of the face DB, the local `WorldState` always assigns each currently visible person a **role tag**: `"left_person"`, `"red_shirt_person"`, `"the_one_who_just_walked_in"`. This is what dialogue uses *immediately* — even before face matching has converged.

Mapping is many-to-one: a role tag points to an optional `face_id`. The dialogue layer uses role tags; the memory layer uses `face_id`s. This mirrors how humans say "the guy in the corner" while their long-term memory separately holds "oh that's Mike from accounting".

---

## 7. Over-engineering check — what to drop or defer

Going back through the doc, here are the parts that look smart on paper but are **probably not worth building first**. Cut these from v1; revisit only if a concrete use case appears.

### 🚫 Drop outright
- **`set_mood(valence, arousal)` continuous baseline.** Requires rewriting the animation system to multiplex a baseline mood with discrete clips. `show_expression` already covers emotional output; mood drift is invisible to users in short interactions. **Drop the `set_mood` tool entirely.**
- **`rename_face` as a separate tool.** It's just `name_face` with overwrite semantics. Collapse: one tool `name_face(face_id, name)` that overwrites silently. Saves a tool slot in the prompt.
- **Confidence decay (older embeddings 0.5× weight).** Adds time-aware weighting math that only matters over months. A robot used over days never hits the threshold. Just keep recent + diverse embeddings; let pruning (§6.9) do the rest.
- **Anti-linked pairs as hard negatives.** The whole "user said no, so penalize this pair forever" idea adds graph state to the matcher. Realistically, twins are rare and one wrong guess is recoverable. If the user denies, just don't auto-greet by that name again **for the session**. Skip the persistent anti-link store.

### ⏸ Defer to v2
- **Pose buckets in the gallery (front/left/right coverage counts).** Useful for diversity, but a much simpler rule — *"only add embedding if its cosine sim to every existing gallery entry is < 0.92"* — already prevents near-duplicates with no bucketing machinery.
- **Online consolidation as a 60 s background job.** Just run consolidation **on demand**: (a) when a new `face_id` is committed, check it against the rest once; (b) when Haseef calls `merge_faces`. No background daemon, no scheduler.
- **`GestureCue` (wave / point detection).** Cool, but the saliency-score boost from "motion_salience" already pulls attention to moving people. A dedicated gesture module is a v2 nice-to-have.
- **`SmoothGaze` minimum-jerk trajectory module.** The existing P-controller in `RobotControl` already produces smooth motion. Min-jerk is academically nicer but visually similar at robot speeds. Build only if motion looks robotic in practice.
- **Curiosity-gradient saliency map for idle saccades.** A simple "random small head movement every 3–6 s" looks 90% as alive at 5% of the code. Skip the OpenCV saliency map for now.
- **Whisper/shout reaction.** Tiny detail, requires per-direction RMS extraction. Build it after DOA is solid.
- **Theory of mind / joint attention.** Genuinely cool but needs robust 3D head-pose estimation and a shared spatial frame. Defer until the rest is stable.
- **Conversation-floor management ("invite the quiet one").** Requires longitudinal speech-time tracking per face_id. Defer.
- **Anticipatory "thinking" gaze.** Hookable later via the existing `tool.input.start` event without architectural changes.
- **Modular package split (`perception/`, `cognition/`, `motion/`, `dialogue/`).** Premature for the current size. Start flat in `hsafa_robot/` and split when files cross ~400 lines.

### ✅ Keep as essential to the redesign
- AttentionManager scoreboard + hysteresis (the heart of the new feel).
- SpeakerFusion: DOA + VAD + face activity.
- IdentityManager: speaker-gated enrollment, multi-view gallery, dual-threshold matching with margin, multi-frame voting, tracker-continuity bridge.
- Two-layer identity (role tag ⟂ `face_id`).
- "Have we met?" dialogue resolution via plain `say_this` (no new tool).
- `with_image` flag on `queue_thinker_task`.
- New Haseef tools, slimmed down: `look_at_person`, `set_attention_mode`, `list_known_faces`, `name_face`, `merge_faces`, `forget_face`.

### ❌ Already-removed (kept here so we don't re-propose them)
- ~~`play_animation`~~ — `show_expression` covers it.
- ~~`remember_fact` as new~~ — already exists.
- ~~`point_at_object`~~ — no arms.
- ~~Custom `search_web`~~ — use Gemini's built-in `googleSearch`.
- ~~`describe_scene`~~ — replaced by `queue_thinker_task(with_image=true)`.
- ~~`ask_about_identity`~~ — replaced by `say_this` + Gemini's normal listening loop.
- ~~Multi-robot social network~~ — premature.

---

## 8. Updated `requirements.txt` (deltas)

Additions on top of current `requirements.txt`:

```text
# Vision (face)
insightface>=0.7         # ArcFace ONNX embeddings, CPU-friendly
onnxruntime>=1.17        # backend for insightface

# Audio
pyroomacoustics>=0.7     # SRP-PHAT DOA
sounddevice>=0.4         # 4-channel capture from XMOS XVF3800
```

Everything else (`reachy-mini`, `numpy`, `scipy`, `opencv-python`, `ultralytics`, `google-genai`, `python-dotenv`, `silero-vad`, `croniter`) stays as is.

---

## 9. Module Map (active)

```
hsafa_robot/
├── perception/
│   ├── person_tracker.py     # YOLO11n-pose + ByteTrack
│   ├── face_tracker.py       # YOLO11n-face
│   ├── face_embedder.py      # InsightFace ArcFace
│   ├── audio_doa.py          # SRP-PHAT on 4-channel stream
│   ├── vad.py                # Silero
│   └── gesture_cue.py        # wave/point rules
├── cognition/
│   ├── speaker_fusion.py     # DOA × VAD × visual → speaker_id
│   ├── attention.py          # AttentionManager + scoreboard
│   ├── identity.py           # IdentityManager (see §6)
│   ├── world_state.py        # canonical scene snapshot
│   └── events.py             # EventBus
├── motion/
│   ├── robot_control.py      # P-controller (existing)
│   ├── smooth_gaze.py        # min-jerk trajectories
│   └── animation.py          # idle/listening/talking overlays
├── gemini_live.py            # existing
├── scheduler_skill.py        # existing
└── ...
```

`main.py` and `setup_haseef.py` stay in **repo root** (per the recent cleanup) and just wire these modules together.

---

## TL;DR

1. **Tools live on Haseef, not Gemini.** New Haseef tools (post over-engineering cut): `look_at_person`, `set_attention_mode`, `list_known_faces`, `name_face`, `merge_faces`, `forget_face`. Gemini gets exactly **one** new optional param: `with_image` on `queue_thinker_task`.
   Dropped/deferred (see §7): `set_mood`, `rename_face`, `describe_scene`, `ask_about_identity`, plus all the v2 "out-of-the-box" extras.
2. **Reflexes stay local.** Gaze decisions never round-trip to Haseef.
3. **Use the 4-mic array.** SRP-PHAT DOA + VAD + face position = robust active-speaker detection.
4. **Anonymous-first face memory.** Every speaker is silently embedded into a multi-view gallery. No name needed at first. Haseef adds names later.
5. **Don't spawn duplicate identities.** Tracker continuity + dual-threshold matching with margin + multi-frame voting + speaker-gated enrollment + background consolidation. When unsure, the robot **asks** via plain `say_this` — no special tool needed.
6. **Two-layer identity.** Role tags ("left person") for immediate dialogue; long-term `face_id`s for memory. They map many-to-one.

Want me to sketch the actual `IdentityManager` class with the matching/voting code, or the `AttentionManager` with the scoring math? Either one is a self-contained next step.
