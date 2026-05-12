#!/usr/bin/env python3
"""tests/test_face_tracking.py

Standalone vision + audio test for the face-recognition + active-speaker
design in ../some-ideas.md (§5 slimmed tool set, §6 IdentityManager).

NO Gemini, NO Haseef, NO robot motion.

Pipeline:
    camera frame -> InsightFace (bbox + 5 kps + 512-d embedding)
                 -> CentroidTracker (stable track_id)
                 -> LipMotionScorer (mouth-region frame diff)
                 -> ActiveSpeakerSelector (VAD-gated argmax + hysteresis)
                 -> IdentityManager (anonymous-first multi-view gallery)
                 -> overlay + display

Run:
    .venv/bin/python tests/test_face_tracking.py

Keys (window):
    q   quit
    n   name the current speaker (terminal prompt)
    c   clear identities
"""
from __future__ import annotations

import argparse
import collections
import logging
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional deps (give a clear error if missing)
# ---------------------------------------------------------------------------
try:
    from insightface.app import FaceAnalysis
except ImportError as e:
    print("ERROR: insightface is not installed.", file=sys.stderr)
    print("Install with: .venv/bin/pip install insightface onnxruntime", file=sys.stderr)
    raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("face_test")


# ===========================================================================
# 1. Camera helpers
# ===========================================================================
def open_camera(preferred_index: Optional[int] = None) -> cv2.VideoCapture:
    """Open Reachy / built-in camera. Tries preferred index first, then 0..3."""
    candidates: List[int] = []
    if preferred_index is not None:
        candidates.append(preferred_index)
    candidates.extend([i for i in (0, 1, 2, 3) if i != preferred_index])

    for idx in candidates:
        cap = cv2.VideoCapture(idx, getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY))
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, _ = cap.read()
            if ok:
                log.info("Opened camera index %d", idx)
                return cap
            cap.release()
    raise RuntimeError("Could not open any camera (tried indices 0-3).")


# ===========================================================================
# 2. Centroid tracker (simple, robust enough for a single-room test)
# ===========================================================================
class CentroidTracker:
    """Assigns persistent integer track_ids to bboxes across frames.

    Matches by nearest centroid; drops tracks after `max_disappeared` frames
    with no match. Distances above `max_distance` are not matched (would be
    teleportation).
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float = 120.0):
        self._next_id = 0
        self.tracks: Dict[int, Tuple[float, float]] = {}  # id -> centroid
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, bboxes: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        if not bboxes:
            # Age all tracks; drop stale ones.
            to_drop = []
            for tid in self.tracks:
                self.disappeared[tid] = self.disappeared.get(tid, 0) + 1
                if self.disappeared[tid] > self.max_disappeared:
                    to_drop.append(tid)
            for tid in to_drop:
                self.tracks.pop(tid, None)
                self.disappeared.pop(tid, None)
            return {}

        # Centroids of new detections.
        new_centroids = [((x1 + x2) / 2, (y1 + y2) / 2) for (x1, y1, x2, y2) in bboxes]

        assigned: Dict[int, Tuple[int, int, int, int]] = {}
        used_detection_idxs: set[int] = set()

        if self.tracks:
            existing_ids = list(self.tracks.keys())
            existing_centroids = [self.tracks[tid] for tid in existing_ids]

            # Pairwise distance matrix.
            dists = np.linalg.norm(
                np.array(existing_centroids)[:, None, :] - np.array(new_centroids)[None, :, :],
                axis=2,
            )

            # Greedy nearest-neighbour assignment.
            while True:
                if dists.size == 0:
                    break
                i, j = np.unravel_index(np.argmin(dists), dists.shape)
                if dists[i, j] > self.max_distance:
                    break
                tid = existing_ids[i]
                assigned[tid] = bboxes[j]
                self.tracks[tid] = new_centroids[j]
                self.disappeared[tid] = 0
                used_detection_idxs.add(j)
                dists[i, :] = np.inf
                dists[:, j] = np.inf

        # New tracks for unmatched detections.
        for j, bbox in enumerate(bboxes):
            if j in used_detection_idxs:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = new_centroids[j]
            self.disappeared[tid] = 0
            assigned[tid] = bbox

        # Age unmatched existing tracks.
        to_drop = []
        for tid in list(self.tracks.keys()):
            if tid in assigned:
                continue
            self.disappeared[tid] = self.disappeared.get(tid, 0) + 1
            if self.disappeared[tid] > self.max_disappeared:
                to_drop.append(tid)
        for tid in to_drop:
            self.tracks.pop(tid, None)
            self.disappeared.pop(tid, None)

        return assigned


# ===========================================================================
# 3. Lip-motion scorer (per track, frame-diff in mouth region)
# ===========================================================================
class LipMotionScorer:
    """Per track_id, computes mean abs frame-diff in the mouth crop.

    No landmark math beyond "use the 2 mouth-corner kps from InsightFace to
    locate the mouth region". Smoothed over a sliding window.
    """

    WINDOW = 10
    CROP_SIZE = (48, 24)  # w, h

    def __init__(self):
        self._prev_crops: Dict[int, np.ndarray] = {}
        self._scores: Dict[int, collections.deque] = {}

    @staticmethod
    def mouth_bbox(kps: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """kps = (5, 2) array: [l_eye, r_eye, nose, l_mouth, r_mouth]."""
        if kps is None or len(kps) < 5:
            return None
        lm, rm = kps[3], kps[4]
        cx = (lm[0] + rm[0]) / 2
        cy = (lm[1] + rm[1]) / 2
        mouth_w = abs(rm[0] - lm[0]) * 1.6 + 4
        mouth_h = mouth_w * 0.6
        x1 = int(cx - mouth_w / 2)
        y1 = int(cy - mouth_h / 2)
        x2 = int(cx + mouth_w / 2)
        y2 = int(cy + mouth_h / 2)
        H, W = frame_shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W - 1, x2); y2 = min(H - 1, y2)
        if x2 - x1 < 8 or y2 - y1 < 6:
            return None
        return (x1, y1, x2, y2)

    def update(self, track_id: int, frame: np.ndarray, kps: np.ndarray) -> float:
        bbox = self.mouth_bbox(kps, frame.shape)
        if bbox is None:
            return self.score(track_id)
        x1, y1, x2, y2 = bbox
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, self.CROP_SIZE)
        crop = cv2.GaussianBlur(crop, (3, 3), 0)

        score = 0.0
        if track_id in self._prev_crops:
            diff = cv2.absdiff(crop, self._prev_crops[track_id])
            score = float(diff.mean())
        self._prev_crops[track_id] = crop
        dq = self._scores.setdefault(track_id, collections.deque(maxlen=self.WINDOW))
        dq.append(score)
        return self.score(track_id)

    def score(self, track_id: int) -> float:
        dq = self._scores.get(track_id)
        if not dq:
            return 0.0
        return float(np.mean(dq))

    def forget(self, track_id: int) -> None:
        self._prev_crops.pop(track_id, None)
        self._scores.pop(track_id, None)


# ===========================================================================
# 4. Active speaker selector (VAD-gated, with hysteresis)
# ===========================================================================
class ActiveSpeakerSelector:
    """Picks one track_id as the active speaker, with stickiness.

    Rules:
      * Only nominate while VAD says speech.
      * Top track must have score >= MIN_SCORE AND lead 2nd by >= MARGIN.
      * Commit only after CONSECUTIVE consistent picks (avoids flicker).
    """

    MIN_SCORE = 2.5     # mean diff intensity (tune by eye)
    MARGIN = 1.0
    CONSECUTIVE = 3

    def __init__(self):
        self._candidate: Optional[int] = None
        self._streak: int = 0
        self._committed: Optional[int] = None

    def update(self, vad_active: bool, scores: Dict[int, float]) -> Optional[int]:
        if not vad_active or not scores:
            self._candidate = None
            self._streak = 0
            self._committed = None
            return None

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_score < self.MIN_SCORE or (top_score - second_score) < self.MARGIN:
            self._candidate = None
            self._streak = 0
            return self._committed  # keep last commit visible briefly

        if top_id == self._candidate:
            self._streak += 1
        else:
            self._candidate = top_id
            self._streak = 1

        if self._streak >= self.CONSECUTIVE:
            self._committed = top_id
        return self._committed


# ===========================================================================
# 5. Audio thread + Silero VAD
# ===========================================================================
class VADWorker:
    """Background thread that runs Silero VAD on the default microphone.

    Exposes `is_speaking()` returning the latest boolean.
    Gracefully no-ops if `sounddevice` or `silero-vad` are missing or fail.
    """

    SAMPLE_RATE = 16000
    CHUNK = 512  # 32 ms at 16 kHz, what Silero expects

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._speaking = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._enabled = True
        self._model = None

    def start(self) -> bool:
        try:
            import sounddevice as sd  # noqa: F401
            from silero_vad import load_silero_vad
        except Exception as e:
            log.warning("VAD disabled (missing dep): %s", e)
            self._enabled = False
            return False
        try:
            self._model = load_silero_vad()
        except Exception as e:
            log.warning("VAD disabled (model load failed): %s", e)
            self._enabled = False
            return False

        self._thread = threading.Thread(target=self._run, name="vad", daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        import sounddevice as sd
        import torch

        q: queue.Queue[np.ndarray] = queue.Queue(maxsize=20)

        def callback(indata, frames, time_info, status):
            if status:
                log.debug("audio status: %s", status)
            try:
                q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                blocksize=self.CHUNK,
                channels=1,
                dtype="float32",
                callback=callback,
            ):
                log.info("VAD listening (%d Hz, %d-sample chunks)", self.SAMPLE_RATE, self.CHUNK)
                # Smoothing window so brief silences in speech don't flip the flag.
                recent = collections.deque(maxlen=5)
                while not self._stop.is_set():
                    try:
                        chunk = q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if chunk.shape[0] != self.CHUNK:
                        continue
                    t = torch.from_numpy(chunk)
                    with torch.no_grad():
                        prob = float(self._model(t, self.SAMPLE_RATE).item())
                    recent.append(prob > self.threshold)
                    self._speaking = sum(recent) >= 2
        except Exception as e:
            log.warning("VAD thread crashed: %s", e)
            self._enabled = False

    def is_speaking(self) -> bool:
        return self._enabled and self._speaking

    def stop(self) -> None:
        self._stop.set()


# ===========================================================================
# 6. IdentityManager — slimmed v1 per some-ideas.md §6 + §7 critique
# ===========================================================================
@dataclass
class FaceEntry:
    face_id: str
    name: Optional[str] = None
    embeddings: List[np.ndarray] = field(default_factory=list)
    last_seen_at: float = field(default_factory=time.time)
    confirmed: bool = False
    pending_count: int = 0


class IdentityManager:
    """Anonymous-first face gallery.

    Implements (slimmed):
      * Multi-view gallery (max MAX_GALLERY embeddings per face_id).
      * Dual-threshold matching with margin (§6.4).
      * Anti-duplicate add rule: skip if cos sim > NEAR_DUPE_SIM vs any
        existing emb in that face_id's gallery (replaces pose bucketing).
      * Pending pool: a NEW emb must be reinforced PENDING_THRESHOLD times
        before becoming a real anonymous face_id (drops passers-by).
      * Speaker-gated enrollment is enforced by the *caller* — this class
        just exposes `match()` and `commit_speaker_embedding()`.

    Dropped (per §7): pose buckets, confidence decay, anti-link pairs,
    background consolidation, rename_face/set_mood.
    """

    HIGH = 0.55
    MARGIN = 0.05
    LOW = 0.40
    NEAR_DUPE_SIM = 0.92
    MAX_GALLERY = 20
    PENDING_THRESHOLD = 6

    def __init__(self):
        self.entries: Dict[str, FaceEntry] = {}
        # Pending pool: track_id -> list of embeddings not yet committed.
        self._pending: Dict[int, List[np.ndarray]] = {}

    # ---- matching --------------------------------------------------------
    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _best_match(self, emb: np.ndarray) -> List[Tuple[str, float]]:
        out = []
        for fid, entry in self.entries.items():
            if not entry.embeddings:
                continue
            sims = sorted((self._cos(emb, e) for e in entry.embeddings), reverse=True)
            top_k = sims[: min(5, len(sims))]
            out.append((fid, float(np.mean(top_k))))
        out.sort(key=lambda kv: kv[1], reverse=True)
        return out

    def match(self, emb: np.ndarray) -> Tuple[str, Optional[str], float]:
        """Return (state, face_id|None, score).

        state in {"match", "provisional", "new"}.
        """
        ranked = self._best_match(emb)
        if not ranked:
            return ("new", None, 0.0)
        best_id, best_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score >= self.HIGH and (best_score - second) >= self.MARGIN:
            return ("match", best_id, best_score)
        if best_score >= self.LOW:
            return ("provisional", best_id, best_score)
        return ("new", None, best_score)

    # ---- speaker-gated enrollment ---------------------------------------
    def commit_speaker_embedding(self, track_id: int, emb: np.ndarray) -> Optional[str]:
        """Called every frame the speaker is detected. Returns face_id when
        a binding is (re-)confirmed for the caller to associate with track."""
        state, fid, _ = self.match(emb)

        if state == "match":
            self._add_to_gallery(fid, emb)
            self.entries[fid].last_seen_at = time.time()
            return fid

        if state == "provisional":
            # Treat provisional as a match for enrollment but don't lock the
            # binding in stone — the gallery grows, and future frames may
            # promote it to "match" naturally.
            self._add_to_gallery(fid, emb)
            self.entries[fid].last_seen_at = time.time()
            return fid

        # state == "new" → pending pool until threshold reached.
        pool = self._pending.setdefault(track_id, [])
        pool.append(emb)
        if len(pool) >= self.PENDING_THRESHOLD:
            fid = self._create_entry(pool)
            self._pending.pop(track_id, None)
            return fid
        return None

    def _add_to_gallery(self, face_id: str, emb: np.ndarray) -> None:
        entry = self.entries[face_id]
        # Anti-duplicate: skip if too similar to any existing embedding.
        for existing in entry.embeddings:
            if self._cos(emb, existing) > self.NEAR_DUPE_SIM:
                return
        entry.embeddings.append(emb)
        # Cap gallery: drop the embedding most similar to its neighbours
        # (keeps the gallery a spanning set).
        if len(entry.embeddings) > self.MAX_GALLERY:
            redundancy = []
            for i, e in enumerate(entry.embeddings):
                other = [self._cos(e, x) for j, x in enumerate(entry.embeddings) if j != i]
                redundancy.append((i, max(other) if other else 0.0))
            drop_idx = max(redundancy, key=lambda kv: kv[1])[0]
            entry.embeddings.pop(drop_idx)
        entry.confirmed = True

    def _create_entry(self, pool: List[np.ndarray]) -> str:
        fid = f"face_{uuid.uuid4().hex[:6]}"
        # Seed gallery with the most diverse embeddings from the pool.
        seed = [pool[0]]
        for emb in pool[1:]:
            if all(self._cos(emb, s) < self.NEAR_DUPE_SIM for s in seed):
                seed.append(emb)
        self.entries[fid] = FaceEntry(
            face_id=fid,
            embeddings=seed,
            confirmed=True,
        )
        log.info("[Identity] NEW face committed: %s (%d seed embs)", fid, len(seed))
        return fid

    # ---- public API ------------------------------------------------------
    def name_face(self, face_id: str, name: str) -> bool:
        entry = self.entries.get(face_id)
        if not entry:
            return False
        old = entry.name
        entry.name = name
        log.info("[Identity] %s name: %r -> %r", face_id, old, name)
        return True

    def clear(self) -> None:
        self.entries.clear()
        self._pending.clear()
        log.info("[Identity] cleared all entries.")

    def label_for(self, face_id: Optional[str]) -> str:
        if not face_id:
            return "?"
        entry = self.entries.get(face_id)
        if not entry:
            return face_id
        return f"{face_id[:10]} ({entry.name})" if entry.name else face_id[:10]


# ===========================================================================
# 7. Main loop
# ===========================================================================
def run(args: argparse.Namespace) -> None:
    log.info("Loading InsightFace (this may download model on first run)...")
    app = FaceAnalysis(
        name="buffalo_sc",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))
    log.info("InsightFace ready.")

    cap = open_camera(args.camera)

    vad = VADWorker(threshold=0.5)
    if not args.no_audio:
        vad.start()

    tracker = CentroidTracker()
    lip_scorer = LipMotionScorer()
    speaker_selector = ActiveSpeakerSelector()
    identity = IdentityManager()

    # Bindings: track_id -> face_id (current best guess for this live track).
    bindings: Dict[int, str] = {}

    window = "Hsafa face/speaker test (q quit, n name speaker, c clear)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    fps_alpha, fps = 0.1, 0.0
    last_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.warning("camera read failed; retrying...")
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)  # mirror for natural feel

            # ---- face detection + embedding ---------------------------
            faces = app.get(frame)

            bboxes: List[Tuple[int, int, int, int]] = []
            face_data: List[Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]] = []
            for f in faces:
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                bboxes.append((x1, y1, x2, y2))
                emb = f.normed_embedding  # already L2-normalized 512-d
                kps = f.kps  # (5, 2)
                face_data.append(((x1, y1, x2, y2), emb, kps))

            # ---- track ------------------------------------------------
            assigned = tracker.update(bboxes)
            bbox_to_idx = {b: i for i, b in enumerate(bboxes)}

            # Drop bindings + scores for tracks that disappeared.
            active_ids = set(assigned.keys())
            for tid in list(bindings.keys()):
                if tid not in active_ids:
                    bindings.pop(tid, None)
            for tid in list(lip_scorer._scores.keys()):
                if tid not in active_ids:
                    lip_scorer.forget(tid)

            # ---- per-track lip motion + tentative match ---------------
            scores: Dict[int, float] = {}
            for tid, bbox in assigned.items():
                idx = bbox_to_idx.get(bbox)
                if idx is None:
                    continue
                _, emb, kps = face_data[idx]
                scores[tid] = lip_scorer.update(tid, frame, kps)
                # Cheap passive match (no enrollment) just to label bbox.
                if tid not in bindings:
                    state, fid, score = identity.match(emb)
                    if state == "match":
                        bindings[tid] = fid

            vad_on = vad.is_speaking()
            speaker_tid = speaker_selector.update(vad_on, scores)

            # ---- speaker-gated enrollment -----------------------------
            if speaker_tid is not None and speaker_tid in assigned:
                bbox = assigned[speaker_tid]
                idx = bbox_to_idx.get(bbox)
                if idx is not None:
                    _, emb, _ = face_data[idx]
                    # Optional quality gate: skip if face too small.
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w >= 80 and h >= 80:
                        fid = identity.commit_speaker_embedding(speaker_tid, emb)
                        if fid:
                            bindings[speaker_tid] = fid

            # ---- draw -------------------------------------------------
            for tid, bbox in assigned.items():
                x1, y1, x2, y2 = bbox
                is_speaker = tid == speaker_tid
                color = (0, 230, 0) if is_speaker else (180, 180, 180)
                thickness = 3 if is_speaker else 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                fid = bindings.get(tid)
                label = identity.label_for(fid)
                lip = scores.get(tid, 0.0)
                tag = f"#{tid} {label}  lip={lip:.1f}"
                if is_speaker:
                    tag = "[SPEAKER] " + tag

                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    frame, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0) if is_speaker else (20, 20, 20), 1, cv2.LINE_AA,
                )

            # HUD
            now = time.time()
            inst_fps = 1.0 / max(now - last_t, 1e-3)
            fps = (1 - fps_alpha) * fps + fps_alpha * inst_fps if fps > 0 else inst_fps
            last_t = now
            hud_lines = [
                f"FPS: {fps:4.1f}",
                f"VAD: {'SPEECH' if vad_on else 'silent'}",
                f"Tracks: {len(assigned)}   Identities: {len(identity.entries)}",
            ]
            for i, line in enumerate(hud_lines):
                cv2.putText(
                    frame, line, (10, 22 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, line, (10, 22 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
                )

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                identity.clear()
                bindings.clear()
            elif key == ord("n"):
                if speaker_tid is not None and speaker_tid in bindings:
                    fid = bindings[speaker_tid]
                    print(f"\n>> Name for current speaker (face_id={fid}): ", end="", flush=True)
                    try:
                        name = input().strip()
                    except EOFError:
                        name = ""
                    if name:
                        identity.name_face(fid, name)
                else:
                    print(">> No committed speaker right now; speak for a moment and try again.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        vad.stop()
        log.info("Done. Identities at exit: %d", len(identity.entries))
        for fid, entry in identity.entries.items():
            log.info("  %s  name=%r  embs=%d", fid, entry.name, len(entry.embeddings))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", type=int, default=None, help="camera index (default: auto)")
    p.add_argument("--no-audio", action="store_true", help="disable mic / VAD")
    p.add_argument("--det-size", type=int, default=640, help="InsightFace det_size (smaller = faster)")
    return p.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    run(parse_args())
