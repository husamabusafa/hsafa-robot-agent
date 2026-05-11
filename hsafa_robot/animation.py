"""animation.py - Overlay animations applied on top of the face-tracking pose.

Each animation returns a dict of deltas::

    {
        "roll":     float,          # radians, absolute roll offset
        "pitch":    float,          # radians, additive pitch offset
        "yaw":      float,          # radians, additive yaw offset
        "antennas": (right, left),  # radians, absolute antenna positions
    }

The controller blends these offsets onto the tracking pose so the robot looks
alive whether it is quietly staring or actively talking.
"""
from __future__ import annotations

import math
import time
from typing import Optional


class IdleAnimation:
    """Idle overlay - the head stays perfectly locked on the face so the
    stare is pristine; only the antennas "breathe" for a hint of life.

    Any head motion here would fight the P-controller and break the look-at
    behavior, so we zero all three axes (roll/pitch/yaw) on purpose.
    """

    def __init__(self) -> None:
        self.t0 = time.time()

    def offsets(self, now: Optional[float] = None) -> dict:
        now = now if now is not None else time.time()
        t = now - self.t0

        # Slow breath modulation (0.22 Hz) drives the antennas only.
        breath = math.sin(2.0 * math.pi * 0.22 * t)
        ant_base = math.radians(-8.0)
        ant_wiggle = math.radians(3.0) * breath
        right_ant = ant_base + ant_wiggle
        left_ant = ant_base - ant_wiggle

        return {
            "roll":     0.0,
            "pitch":    0.0,
            "yaw":      0.0,
            "antennas": (right_ant, left_ant),
        }


class TalkingAnimation:
    """Livelier overlay while the robot is speaking: gentle nod + antenna
    wiggle at speech-ish rhythms. Purely aesthetic - not phoneme-synced."""

    def __init__(self) -> None:
        self.t0 = time.time()

    def offsets(self, now: Optional[float] = None) -> dict:
        now = now if now is not None else time.time()
        t = now - self.t0

        # --- Head: gentle slow nod so the robot still "looks at you" ---
        # Single low-frequency wave for a calmer, less jittery feel.
        pitch = math.radians(0.8) * math.sin(2.0 * math.pi * 0.6 * t)
        roll = math.radians(0.5) * math.sin(2.0 * math.pi * 0.4 * t)
        yaw = math.radians(0.4) * math.sin(2.0 * math.pi * 0.5 * t + 1.0)

        # --- Antennas: soft, slow wiggle ---
        # Much smaller amplitude and lower frequency for a polite, attentive look.
        ant_base = math.radians(12.0)
        flick = math.radians(5.0) * math.sin(2.0 * math.pi * 1.5 * t)
        right_ant = ant_base + flick
        left_ant = ant_base - flick

        return {
            "roll":     roll,
            "pitch":    pitch,
            "yaw":      yaw,
            "antennas": (right_ant, left_ant),
        }


def blend_offsets(a: dict, b: dict, alpha: float) -> dict:
    """Crossfade between two animations so transitions don't snap.

    ``alpha == 0`` returns ``a``, ``alpha == 1`` returns ``b``.
    """
    inv = 1.0 - alpha
    ar, al = a["antennas"]
    br, bl = b["antennas"]
    return {
        "roll":     inv * a["roll"]  + alpha * b["roll"],
        "pitch":    inv * a["pitch"] + alpha * b["pitch"],
        "yaw":      inv * a["yaw"]   + alpha * b["yaw"],
        "antennas": (inv * ar + alpha * br, inv * al + alpha * bl),
    }
