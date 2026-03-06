"""
Gesture detector module – open hand swipe recognition.

Gesture: hold up an open palm and swipe horizontally.
  - Swipe LEFT  → push content to the left monitor  (moves right monitor's window left)
  - Swipe RIGHT → push content to the right monitor (moves left monitor's window right)

Detection is simple and reliable:
  1. Detect open hand (all fingers extended).
  2. Record starting position.
  3. Track horizontal displacement.
  4. When displacement exceeds threshold → fire.
  5. Cooldown prevents re-triggers.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional

import numpy as np

from hand_tracker import (
    HandData,
    INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
    WRIST,
)

INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18


# ── Gesture events ───────────────────────────────────────────────────────────

class GestureEvent(Enum):
    MOVE_TO_LEFT = auto()
    MOVE_TO_RIGHT = auto()


# ── Open hand detection ──────────────────────────────────────────────────────

_FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
_FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
_FINGER_NAMES = ["index", "middle", "ring", "pinky"]


def _finger_extended(landmarks_px: np.ndarray, tip_idx: int, pip_idx: int) -> bool:
    """A finger is extended when its tip is farther from the wrist than its PIP."""
    wrist = landmarks_px[WRIST]
    tip = landmarks_px[tip_idx]
    pip = landmarks_px[pip_idx]
    return float(np.linalg.norm(tip - wrist)) > float(np.linalg.norm(pip - wrist)) * 1.05


def get_finger_states(landmarks_px: np.ndarray) -> Dict[str, bool]:
    """Return finger_name → is_extended for the 4 main fingers."""
    return {
        name: _finger_extended(landmarks_px, tip, pip)
        for name, tip, pip in zip(_FINGER_NAMES, _FINGER_TIPS, _FINGER_PIPS)
    }


def is_open_hand(landmarks_px: np.ndarray) -> bool:
    """True when at least 3 of the 4 non-thumb fingers are extended."""
    states = get_finger_states(landmarks_px)
    return sum(states.values()) >= 3


# ── Hand state ───────────────────────────────────────────────────────────────

class HandState(Enum):
    IDLE = auto()
    SWIPING = auto()  # open hand detected, tracking displacement


@dataclass
class HandTrack:
    """Per-hand tracking state, all public for HUD debug."""

    state: HandState = HandState.IDLE
    position_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    time_history: Deque[float] = field(default_factory=lambda: deque(maxlen=30))

    is_open: bool = False
    finger_states: Dict[str, bool] = field(default_factory=dict)
    swipe_origin: Optional[np.ndarray] = None
    swipe_dx: float = 0.0
    velocity: Optional[np.ndarray] = None
    speed: float = 0.0

    def push_position(self, pos: np.ndarray, t: float):
        self.position_history.append(pos.copy())
        self.time_history.append(t)

    def recent_velocity(self, window: int = 4) -> Optional[np.ndarray]:
        if len(self.position_history) < window + 1:
            return None
        positions = list(self.position_history)
        times = list(self.time_history)
        dp = positions[-1] - positions[-window]
        dt = times[-1] - times[-window]
        if dt < 1e-6:
            return None
        return dp / dt

    def reset(self):
        self.state = HandState.IDLE
        self.swipe_origin = None
        self.swipe_dx = 0.0


# ── Debug state ──────────────────────────────────────────────────────────────

@dataclass
class DebugState:
    hand_tracks: Dict[str, HandTrack] = field(default_factory=dict)
    events: List[GestureEvent] = field(default_factory=list)


# ── Main detector ────────────────────────────────────────────────────────────

class GestureDetector:
    """
    Open-hand swipe detector.

    Hold up your open palm and push it left or right.
    """

    SWIPE_THRESHOLD: float = 80.0     # px of horizontal movement to trigger
    COOLDOWN_SECONDS: float = 2.5     # seconds between allowed triggers

    def __init__(self):
        self._trackers: Dict[str, HandTrack] = {
            "Left": HandTrack(),
            "Right": HandTrack(),
        }
        self._cooldown_until: float = 0.0
        self.debug_state = DebugState()

    def update(self, hands: List[HandData]) -> List[GestureEvent]:
        now = time.monotonic()
        events: List[GestureEvent] = []
        hand_map: Dict[str, HandData] = {h.label: h for h in hands}

        for label in ("Left", "Right"):
            tracker = self._trackers[label]
            hand = hand_map.get(label)

            if hand is None:
                tracker.reset()
                tracker.is_open = False
                tracker.finger_states = {}
                tracker.velocity = None
                tracker.speed = 0.0
                continue

            palm = hand.palm_center_px
            tracker.push_position(palm, now)

            tracker.finger_states = get_finger_states(hand.landmarks_px)
            tracker.is_open = is_open_hand(hand.landmarks_px)

            vel = tracker.recent_velocity()
            tracker.velocity = vel
            tracker.speed = float(np.linalg.norm(vel)) if vel is not None else 0.0

            # ── State machine ────────────────────────────────────────
            if tracker.state == HandState.IDLE:
                if tracker.is_open:
                    tracker.state = HandState.SWIPING
                    tracker.swipe_origin = palm.copy()

            elif tracker.state == HandState.SWIPING:
                if not tracker.is_open:
                    # Hand closed or lost — reset
                    tracker.reset()
                    continue

                dx = palm[0] - tracker.swipe_origin[0]
                tracker.swipe_dx = dx

                if now > self._cooldown_until and abs(dx) > self.SWIPE_THRESHOLD:
                    if dx < 0:
                        events.append(GestureEvent.MOVE_TO_LEFT)
                    else:
                        events.append(GestureEvent.MOVE_TO_RIGHT)
                    self._cooldown_until = now + self.COOLDOWN_SECONDS
                    tracker.reset()

        self.debug_state = DebugState(
            hand_tracks=dict(self._trackers),
            events=events,
        )
        return events

    def reset(self):
        for t in self._trackers.values():
            t.reset()
