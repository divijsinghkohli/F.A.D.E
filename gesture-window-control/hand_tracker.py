"""
Hand tracker module – MediaPipe hand tracking and landmark extraction.

Uses the MediaPipe Tasks API (HandLandmarker) to detect up to two hands
per frame and return normalised + pixel landmark coordinates together
with handedness labels.

The .task model file is auto-downloaded on first run.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List

# MediaPipe landmark indices for convenience
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

THUMB_IP = 3  # interphalangeal joint – used instead of MCP for thumb

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_FILENAME = "hand_landmarker.task"

# Connections between landmarks for drawing the hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle  (wrist→9 approximation)
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm cross-links
]


def _ensure_model(model_dir: str) -> str:
    """Download the hand_landmarker.task model if it doesn't exist yet."""
    path = os.path.join(model_dir, _MODEL_FILENAME)
    if os.path.isfile(path):
        return path
    print(f"[hand_tracker] downloading model to {path} ...")
    os.makedirs(model_dir, exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, path)
    print(f"[hand_tracker] model downloaded ({os.path.getsize(path)} bytes)")
    return path


@dataclass
class HandData:
    """Processed result for a single detected hand."""

    label: str  # "Left" or "Right"
    landmarks_norm: np.ndarray  # (21, 3) normalised [0..1]
    landmarks_px: np.ndarray  # (21, 2) in pixel coords
    score: float = 1.0

    @property
    def wrist_px(self) -> np.ndarray:
        return self.landmarks_px[WRIST]

    @property
    def palm_center_px(self) -> np.ndarray:
        """Approximate palm center (average of wrist + MCP landmarks)."""
        indices = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
        return self.landmarks_px[indices].mean(axis=0)


class HandTracker:
    """Runs MediaPipe HandLandmarker on each frame and returns structured results."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        model_dir: str | None = None,
    ):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))

        model_path = _ensure_model(model_dir)

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_ts_ms = 0

    def process(self, frame_bgr: np.ndarray) -> List[HandData]:
        """
        Detect hands in a BGR frame.

        Returns a list of HandData (0, 1, or 2 entries).
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_ts_ms += 33  # ~30 fps; must be monotonically increasing
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        hands: List[HandData] = []
        if not result.hand_landmarks:
            return hands

        for hand_lms, handedness_list in zip(
            result.hand_landmarks,
            result.handedness,
        ):
            label = handedness_list[0].category_name
            score = handedness_list[0].score

            norm = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_lms],
                dtype=np.float32,
            )
            px = np.array(
                [(lm.x * w, lm.y * h) for lm in hand_lms],
                dtype=np.float32,
            )

            hands.append(
                HandData(
                    label=label,
                    landmarks_norm=norm,
                    landmarks_px=px,
                    score=score,
                )
            )

        return hands

    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        hands: List[HandData],
        color: tuple = (0, 255, 0),
        thickness: int = 2,
        radius: int = 4,
    ) -> np.ndarray:
        """Draw hand skeletons on the frame using the already-detected HandData."""
        for hand in hands:
            pts = hand.landmarks_px.astype(int)
            for i, (x, y) in enumerate(pts):
                cv2.circle(frame_bgr, (x, y), radius, color, -1)
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame_bgr, tuple(pts[a]), tuple(pts[b]), color, thickness)
        return frame_bgr

    def close(self):
        self._landmarker.close()
