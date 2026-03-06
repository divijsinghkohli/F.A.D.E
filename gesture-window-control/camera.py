"""
Camera module – handles camera connection and frame capture.

Supports three input modes:
  1. Local webcam (device index)
  2. Network stream URL (e.g. http://<iphone-ip>:4747/video for EpocCam/Camo)
  3. Continuity Camera (appears as a normal device index on macOS)

The CameraSource class wraps OpenCV's VideoCapture with automatic
reconnection logic and resolution helpers.
"""

import sys
import cv2
import time
from typing import Optional, Tuple


class CameraSource:
    """Manages a single video input source."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        reconnect_delay: float = 2.0,
    ):
        """
        Args:
            source: Device index (int) for local cameras, or a URL string
                    for network streams (e.g. "http://192.168.1.5:4747/video").
            width:  Requested capture width.
            height: Requested capture height.
            reconnect_delay: Seconds to wait before retrying a failed connection.
        """
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open (or reopen) the video source. Returns True on success."""
        self.close()

        if isinstance(self.source, int) and sys.platform == "darwin":
            # On macOS, explicitly request the AVFoundation backend so that
            # Continuity Camera (iPhone) devices deliver frames reliably.
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_AVFOUNDATION)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if isinstance(self.source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self._cap.isOpened():
            print(f"[camera] failed to open source: {self.source}")
            return False

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[camera] opened {self.source} at {actual_w}x{actual_h}")
        return True

    def read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        """
        Read a single frame.

        Returns (success, frame). On failure the caller can decide
        whether to call `reconnect()`.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def reconnect(self) -> bool:
        """Close, wait, and reopen the source."""
        print(f"[camera] reconnecting in {self.reconnect_delay}s ...")
        self.close()
        time.sleep(self.reconnect_delay)
        return self.open()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
