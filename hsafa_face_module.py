"""hsafa_face_module.py — Face detection, recognition, and following.

A self-contained module wired into main.py. Public API: :class:`FaceModule`.

Design (matches the discipline in `some-ideas.md` §6):

* **InsightFace ArcFace (buffalo_s, ONNX, CPU)** for both detection and 512-d
  embeddings. Strong accuracy; lazy-loaded so importing is cheap.
* **Anonymous-first gallery**, persisted at ``~/.hsafa/faces.json``. A name
  only attaches when ``enroll(name)`` is called explicitly (Haseef does this
  ONLY after the user verbally confirms the name).
* **Dual-threshold matching** (high=0.55 with 0.05 margin, low=0.40). Borderline
  → "uncertain (maybe X?)" — never auto-named.
* **Annotation overlay** so every frame sent to the LLMs has names drawn on
  faces (green=known, amber=uncertain, yellow=unknown). The labels are the
  source of truth — the LLMs read them off the image instead of guessing.
* **Follow loop** moves the head to keep the chosen face centered, using a
  smoothed P-controller. Engages ``robot._expression_active`` so the idle/
  speaking animation yields cleanly.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("face_module")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
HIGH_THRESHOLD = 0.48       # confident named match (typical ArcFace ~0.45-0.50)
LOW_THRESHOLD = 0.28        # below this we don't even hint a candidate
MARGIN = 0.05               # min gap between best and second-best for confident match
MAX_GALLERY = 50            # max embeddings stored per name
DUP_THRESHOLD = 0.96        # skip enrollment of near-duplicate embeddings
DETECT_HZ = 8               # face detection cadence
DET_SIZE = 640              # SCRFD input size (was 320; bigger = better for small/distant faces)
FOLLOW_HZ = 15              # head-control cadence while following
UNKNOWN_NEW_SECS = 3.0      # stable unknown duration before emitting an event
UNCERTAIN_DEBOUNCE = 120.0  # min seconds between uncertain events for one candidate

EVENT_NEW_UNKNOWN = "face.new_unknown"
EVENT_IDENTITY_UNCERTAIN = "face.identity_uncertain"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FaceObservation:
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2 in raw frame coords
    name: Optional[str]               # None if not confidently matched
    confidence: float                 # cosine similarity of best match
    is_uncertain: bool                # True iff borderline match
    candidate_name: Optional[str]     # name suggestion when uncertain


@dataclass
class _GalleryEntry:
    name: str
    embeddings: List[List[float]] = field(default_factory=list)
    first_seen_at: float = 0.0
    last_seen_at: float = 0.0


# ---------------------------------------------------------------------------
# Persisted gallery
# ---------------------------------------------------------------------------
class _Gallery:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: Dict[str, _GalleryEntry] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            for raw in data.get("entries", []):
                e = _GalleryEntry(**raw)
                self.entries[e.name] = e
            log.info("[gallery] loaded %d names from %s", len(self.entries), self.path)
        except Exception as e:
            log.warning("[gallery] load failed: %s", e)

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {"entries": [asdict(e) for e in self.entries.values()]}
            self.path.write_text(json.dumps(data))
        except Exception as e:
            log.warning("[gallery] save failed: %s", e)

    @staticmethod
    def _norm(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-8 else v

    def add(self, name: str, embedding: np.ndarray) -> None:
        with self._lock:
            if name not in self.entries:
                self.entries[name] = _GalleryEntry(
                    name=name,
                    first_seen_at=time.time(),
                    last_seen_at=time.time(),
                )
            entry = self.entries[name]
            entry.last_seen_at = time.time()
            emb = self._norm(embedding)
            for existing in entry.embeddings:
                sim = float(np.dot(emb, self._norm(np.asarray(existing))))
                if sim > DUP_THRESHOLD:
                    return  # near-duplicate; skip
            entry.embeddings.append(emb.tolist())
            if len(entry.embeddings) > MAX_GALLERY:
                entry.embeddings = entry.embeddings[-MAX_GALLERY:]
            self._save()

    def remove(self, name: str) -> bool:
        with self._lock:
            if name in self.entries:
                del self.entries[name]
                self._save()
                return True
            return False

    def list_(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": e.name,
                    "embedding_count": len(e.embeddings),
                    "first_seen_at": e.first_seen_at,
                    "last_seen_at": e.last_seen_at,
                }
                for e in self.entries.values()
            ]

    def match(self, embedding: np.ndarray) -> Tuple[Optional[str], float, Optional[str]]:
        """Return ``(name | None, top_score, candidate_for_uncertain | None)``.

        * Confident match     → (name, score, None)
        * Borderline match    → (None, score, candidate_name)
        * No match at all     → (None, score, None)
        """
        with self._lock:
            entries = list(self.entries.values())
        if not entries:
            return None, 0.0, None
        emb = self._norm(embedding)
        scored: List[Tuple[str, float]] = []
        for e in entries:
            if not e.embeddings:
                continue
            arr = np.asarray(e.embeddings, dtype=np.float32)
            arr_norm = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
            sims = arr_norm @ emb
            top_k = np.sort(sims)[-min(5, len(sims)):]
            scored.append((e.name, float(top_k.mean())))
        if not scored:
            return None, 0.0, None
        scored.sort(key=lambda x: -x[1])
        best_name, best = scored[0]
        second = scored[1][1] if len(scored) > 1 else 0.0
        if best >= HIGH_THRESHOLD and (best - second) >= MARGIN:
            return best_name, best, None
        if best >= LOW_THRESHOLD:
            return None, best, best_name
        return None, best, None


# ---------------------------------------------------------------------------
# Embedder (lazy InsightFace)
# ---------------------------------------------------------------------------
class _InsightFaceEmbedder:
    def __init__(self, model_name: str = "buffalo_s") -> None:
        self._app = None
        self._model_name = model_name
        self._lock = threading.Lock()

    def ensure_ready(self) -> None:
        if self._app is not None:
            return
        with self._lock:
            if self._app is not None:
                return
            try:
                from insightface.app import FaceAnalysis
            except ImportError as e:
                raise RuntimeError(
                    "insightface is not installed. Add `insightface` and "
                    "`onnxruntime` to requirements.txt and `pip install`."
                ) from e
            log.info("[embedder] loading InsightFace '%s' (CPU)...", self._model_name)
            app = FaceAnalysis(
                name=self._model_name,
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
            self._app = app
            log.info("[embedder] InsightFace ready.")

    def detect(self, frame_bgr: np.ndarray) -> List[Any]:
        self.ensure_ready()
        return self._app.get(frame_bgr)


# ---------------------------------------------------------------------------
# FaceModule — public API
# ---------------------------------------------------------------------------
class FaceModule:
    """Detect faces, identify them, optionally follow one with the head."""

    def __init__(
        self,
        gallery_path: Optional[Path] = None,
        on_event: Optional[Callable[[str, Dict[str, Any], Optional[bytes]], None]] = None,
        model_name: str = "buffalo_s",
    ) -> None:
        self._gallery = _Gallery(
            gallery_path or (Path.home() / ".hsafa" / "faces.json")
        )
        self._embedder = _InsightFaceEmbedder(model_name=model_name)
        self._on_event = on_event

        self._frame_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._robot = None  # RobotController, set in start()

        self._latest_obs: List[FaceObservation] = []
        self._latest_lock = threading.Lock()

        self._stop = threading.Event()
        self._detect_thread: Optional[threading.Thread] = None

        # Follow state
        self._follow_target: Optional[str] = None  # name OR "_dominant_"
        self._follow_lock = threading.Lock()
        self._follow_thread: Optional[threading.Thread] = None

        # Event debouncing
        self._unknown_first_seen: Dict[str, float] = {}
        self._unknown_emitted: Dict[str, float] = {}
        self._uncertain_last: Dict[str, float] = {}

    # ---- lifecycle --------------------------------------------------------
    def start(
        self,
        frame_getter: Callable[[], Optional[np.ndarray]],
        robot,
    ) -> None:
        self._frame_getter = frame_getter
        self._robot = robot
        self._stop.clear()
        self._detect_thread = threading.Thread(
            target=self._detect_loop, name="face-detect", daemon=True,
        )
        self._detect_thread.start()
        log.info("FaceModule started.")

    def stop(self) -> None:
        self._stop.set()
        with self._follow_lock:
            self._follow_target = None
        if self._detect_thread:
            self._detect_thread.join(timeout=1.5)
        if self._follow_thread:
            self._follow_thread.join(timeout=1.0)
        log.info("FaceModule stopped.")

    # ---- detection loop ---------------------------------------------------
    def _detect_loop(self) -> None:
        period = 1.0 / DETECT_HZ
        while not self._stop.is_set():
            t0 = time.time()
            frame = self._frame_getter() if self._frame_getter else None
            if frame is not None:
                try:
                    self._tick(frame)
                except Exception as e:
                    log.warning("face detect tick failed: %s", e)
            elapsed = time.time() - t0
            time.sleep(max(0.0, period - elapsed))

    def _tick(self, frame_bgr: np.ndarray) -> None:
        try:
            faces = self._embedder.detect(frame_bgr)
        except Exception as e:
            log.warning("embedder detect failed: %s", e)
            return

        observations: List[FaceObservation] = []
        now = time.time()

        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            emb = getattr(f, "embedding", None)
            if emb is None:
                emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            name, score, candidate = self._gallery.match(np.asarray(emb))
            is_uncertain = (name is None and candidate is not None)
            obs = FaceObservation(
                bbox=(x1, y1, x2, y2),
                name=name,
                confidence=score,
                is_uncertain=is_uncertain,
                candidate_name=candidate,
            )
            observations.append(obs)

            # Debounced event emission
            tid = f"{x1 // 40}_{y1 // 40}_{(x2 - x1) // 20}"
            if name is None and not is_uncertain:
                first = self._unknown_first_seen.setdefault(tid, now)
                if (now - first >= UNKNOWN_NEW_SECS
                        and tid not in self._unknown_emitted):
                    self._unknown_emitted[tid] = now
                    self._latest_obs = observations  # fresh state for annotate
                    self._emit_event(EVENT_NEW_UNKNOWN, frame_bgr, {
                        "note": (
                            "An unfamiliar person has been visible for a few "
                            "seconds. The yellow 'unknown' box on the image is "
                            "this person."
                        ),
                        "visible": [_obs_to_dict(o) for o in observations],
                    })
            else:
                self._unknown_first_seen.pop(tid, None)

            if is_uncertain and candidate:
                last = self._uncertain_last.get(candidate, 0.0)
                if now - last >= UNCERTAIN_DEBOUNCE:
                    self._uncertain_last[candidate] = now
                    self._latest_obs = observations
                    self._emit_event(EVENT_IDENTITY_UNCERTAIN, frame_bgr, {
                        "note": (
                            f"Borderline face match — could be {candidate} "
                            f"(score={score:.2f}). Ask to confirm before "
                            "greeting by name."
                        ),
                        "candidate_name": candidate,
                        "score": score,
                        "visible": [_obs_to_dict(o) for o in observations],
                    })

        # Bound debounce maps
        if len(self._unknown_first_seen) > 50:
            self._unknown_first_seen.clear()
            self._unknown_emitted.clear()

        with self._latest_lock:
            self._latest_obs = observations

    # ---- events -----------------------------------------------------------
    def _emit_event(
        self, name: str, frame_bgr: np.ndarray, data: Dict[str, Any],
    ) -> None:
        if self._on_event is None:
            return
        annotated = self.annotate_frame(frame_bgr.copy())
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        jpeg_bytes = buf.tobytes() if ok else None
        try:
            self._on_event(name, data, jpeg_bytes)
        except Exception as e:
            log.warning("on_event(%s) raised: %s", name, e)

    # ---- annotation -------------------------------------------------------
    def annotate_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw labelled boxes on the frame in-place. Returns the same array."""
        with self._latest_lock:
            obs = list(self._latest_obs)
        for o in obs:
            x1, y1, x2, y2 = o.bbox
            if o.name:
                color = (0, 200, 0)            # green BGR
                label = f"{o.name} ({o.confidence:.2f})"
            elif o.is_uncertain and o.candidate_name:
                color = (0, 200, 220)          # amber
                label = f"unknown (maybe {o.candidate_name}? {o.confidence:.2f})"
            else:
                color = (0, 220, 240)          # yellow
                label = "unknown"
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            tw = max(80, len(label) * 9)
            cv2.rectangle(
                frame_bgr,
                (x1, max(0, y1 - 22)),
                (x1 + tw, y1),
                color, -1,
            )
            cv2.putText(
                frame_bgr, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )
        return frame_bgr

    def get_annotated_b64_jpeg(self, quality: int = 75) -> Optional[str]:
        if self._frame_getter is None:
            return None
        frame = self._frame_getter()
        if frame is None:
            return None
        annotated = self.annotate_frame(frame.copy())
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # ---- public state queries --------------------------------------------
    def describe_visible(self) -> List[Dict[str, Any]]:
        with self._latest_lock:
            obs = list(self._latest_obs)
        frame = self._frame_getter() if self._frame_getter else None
        w = frame.shape[1] if frame is not None else 640
        out = []
        for o in obs:
            cx = 0.5 * (o.bbox[0] + o.bbox[2])
            position = (
                "left" if cx < w / 3.0
                else "right" if cx > 2.0 * w / 3.0
                else "center"
            )
            out.append({
                "name": o.name,
                "confidence": round(o.confidence, 3),
                "uncertain_candidate": o.candidate_name if o.is_uncertain else None,
                "position": position,
            })
        return out

    def list_known(self) -> List[Dict[str, Any]]:
        return self._gallery.list_()

    # ---- enroll / forget -------------------------------------------------
    def enroll(self, name: str) -> Dict[str, Any]:
        """Capture several embeddings under ``name``.

        Multiple frames are sampled to build a small diverse gallery. Existing
        names are merged (new embeddings appended).

        Selection rule (avoids enrolling the wrong person when several faces
        are visible):
        1. If at least one face in view is currently UNKNOWN to the gallery
           (or matches a *different* name borderline), pick the largest such
           face. This is the common case: "remember Cady" while Husam is also
           in view → we want Cady's face, not Husam's.
        2. If ALL visible faces are already confidently named:
           - if one of them already matches ``name``, top up that entry
             (largest such face);
           - otherwise refuse with a clear error so we never overwrite
             someone else's identity.
        """
        name = (name or "").strip()
        if not name:
            return {"ok": False, "error": "name is required"}
        if self._frame_getter is None:
            return {"ok": False, "error": "module not started"}

        # Sample over ~3 s so we capture natural variation (angle, expression,
        # micro-lighting) instead of a single near-duplicate burst. This is
        # what makes recognition robust at later angles.
        target_captures = 8
        captured = 0
        attempts = 0
        last_error: Optional[str] = None
        while captured < target_captures and attempts < 25:
            attempts += 1
            frame = self._frame_getter()
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                faces = self._embedder.detect(frame)
            except Exception as e:
                return {"ok": False, "error": f"embed failed: {e}"}
            if not faces:
                time.sleep(0.15)
                continue

            # Score each visible face against the gallery so we know who's who.
            scored: List[Tuple[Any, np.ndarray, Optional[str], int]] = []
            for f in faces:
                emb = getattr(f, "embedding", None)
                if emb is None:
                    emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    continue
                arr = np.asarray(emb)
                matched_name, _score, _cand = self._gallery.match(arr)
                area = int((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                scored.append((f, arr, matched_name, area))

            if not scored:
                time.sleep(0.15)
                continue

            # 1. Prefer unknown faces (or the same name we are enrolling).
            candidates = [
                s for s in scored if s[2] is None or s[2] == name
            ]
            if not candidates:
                # 2. All visible faces are confidently OTHER people.
                others = sorted({s[2] for s in scored if s[2] is not None})
                last_error = (
                    f"all visible faces are already known as: {others}. "
                    "Refusing to overwrite — ask the new person to step in "
                    "front of the camera alone, or call forget_face first."
                )
                time.sleep(0.2)
                continue

            # Largest qualifying face.
            candidates.sort(key=lambda s: s[3], reverse=True)
            _f, emb_arr, _matched, _area = candidates[0]
            self._gallery.add(name, emb_arr)
            captured += 1
            time.sleep(0.4)

        if captured == 0:
            return {
                "ok": False,
                "error": last_error or "no clear face seen",
            }
        log.info("[enroll] %s: captured %d embedding(s)", name, captured)
        return {"ok": True, "name": name, "captured": captured}

    def forget(self, name: str) -> Dict[str, Any]:
        name = (name or "").strip()
        ok = self._gallery.remove(name)
        return {"ok": ok, "name": name}

    # ---- following -------------------------------------------------------
    def follow(self, name: Optional[str] = None) -> Dict[str, Any]:
        target = (name or "").strip() or "_dominant_"
        with self._follow_lock:
            self._follow_target = target
            if self._follow_thread is None or not self._follow_thread.is_alive():
                self._follow_thread = threading.Thread(
                    target=self._follow_loop, name="face-follow", daemon=True,
                )
                self._follow_thread.start()
        return {"ok": True, "target": target}

    def stop_following(self) -> Dict[str, Any]:
        with self._follow_lock:
            self._follow_target = None
        # Release the base so idle animation re-centers naturally.
        try:
            if self._robot is not None and hasattr(self._robot, "set_follow_base"):
                self._robot.set_follow_base(None, None)
        except Exception:
            pass
        return {"ok": True}

    def _follow_loop(self) -> None:
        """Drive the robot's follow-base yaw/pitch.

        We do NOT call ``set_target_head_pose`` directly — instead we push a
        smoothed yaw/pitch into ``RobotController.set_follow_base``. The
        controller's idle/speaking loop adds its expressive deltas (bobs,
        glances, antenna sway) ON TOP of this base, so face-following
        composes with animations rather than freezing them.
        """
        period = 1.0 / FOLLOW_HZ
        cur_yaw = 0.0
        cur_pitch = 0.0
        # Rough camera intrinsics (Reachy Mini): full HFOV ~60°, VFOV ~45°.
        FOV_H = 60.0
        FOV_V = 45.0
        while not self._stop.is_set():
            with self._follow_lock:
                target = self._follow_target
            if target is None:
                try:
                    if (self._robot is not None
                            and hasattr(self._robot, "set_follow_base")):
                        self._robot.set_follow_base(None, None)
                except Exception:
                    pass
                return

            with self._latest_lock:
                obs = list(self._latest_obs)
            frame = self._frame_getter() if self._frame_getter else None
            frame_w = frame.shape[1] if frame is not None else 640
            frame_h = frame.shape[0] if frame is not None else 480

            chosen: Optional[FaceObservation] = None
            if target == "_dominant_":
                if obs:
                    chosen = max(
                        obs,
                        key=lambda o: (o.bbox[2] - o.bbox[0])
                        * (o.bbox[3] - o.bbox[1]),
                    )
            else:
                for o in obs:
                    if o.name == target:
                        chosen = o
                        break

            if chosen is not None and self._robot is not None:
                cx = 0.5 * (chosen.bbox[0] + chosen.bbox[2])
                cy = 0.5 * (chosen.bbox[1] + chosen.bbox[3])
                err_x = (cx - frame_w / 2.0) / (frame_w / 2.0)
                err_y = (cy - frame_h / 2.0) / (frame_h / 2.0)
                # Robot frame: positive yaw = LEFT. Face on RIGHT side of
                # frame (err_x > 0) is physically to robot's right → need
                # NEGATIVE yaw delta.
                desired_yaw = cur_yaw - err_x * (FOV_H / 2.0) * 0.45
                desired_pitch = cur_pitch + err_y * (FOV_V / 2.0) * 0.45
                desired_yaw = max(-60.0, min(60.0, desired_yaw))
                desired_pitch = max(-30.0, min(30.0, desired_pitch))
                # Smooth motion (P-controller-ish)
                cur_yaw += (desired_yaw - cur_yaw) * 0.4
                cur_pitch += (desired_pitch - cur_pitch) * 0.4
                try:
                    if hasattr(self._robot, "set_follow_base"):
                        self._robot.set_follow_base(cur_yaw, cur_pitch)
                except Exception as e:
                    log.warning("follow base update failed: %s", e)
            # If chosen is None, we keep the last base (head holds position).
            time.sleep(period)


def _obs_to_dict(o: FaceObservation) -> Dict[str, Any]:
    return {
        "name": o.name,
        "confidence": round(o.confidence, 3),
        "uncertain_candidate": o.candidate_name if o.is_uncertain else None,
        "bbox": list(o.bbox),
    }
