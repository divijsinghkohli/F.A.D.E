#!/usr/bin/env python3
"""
F.A.D.E – gesture-window-control
Open hand swipe to push windows between monitors.

Usage:
    python main.py                     # iPhone camera, CV-only (dry run)
    python main.py --live              # enable actual window control
    python main.py --source 0          # use a different camera index
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np

from camera import CameraSource
from hand_tracker import HandTracker
from gesture_detector import GestureDetector, GestureEvent, HandState


# ── Event display ─────────────────────────────────────────────────────────────

_EVENT_LABELS = {
    GestureEvent.MOVE_TO_LEFT: "<< MOVE LEFT",
    GestureEvent.MOVE_TO_RIGHT: "MOVE RIGHT >>",
}

_EVENT_COLORS = {
    GestureEvent.MOVE_TO_LEFT: (255, 100, 0),
    GestureEvent.MOVE_TO_RIGHT: (0, 100, 255),
}


# ── HUD drawing ──────────────────────────────────────────────────────────────

def _draw_hud(
    frame: np.ndarray,
    hands,
    detector: GestureDetector,
    event_log: deque,
    fps: float,
):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    debug = detector.debug_state

    # FPS + hand count
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 28), font, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Hands: {len(hands)}", (10, 56), font, 0.7, (0, 255, 0), 2)

    # Cooldown indicator
    remaining = detector._cooldown_until - time.monotonic()
    if remaining > 0:
        cv2.putText(frame, f"Cooldown: {remaining:.1f}s", (10, 84), font, 0.6, (0, 0, 255), 2)

    # Per-hand debug
    panel_y = {"Left": 120, "Right": 260}
    for label in ("Left", "Right"):
        track = debug.hand_tracks.get(label)
        if track is None:
            continue
        y = panel_y[label]
        x = 10

        # State
        state_color = (0, 255, 255) if track.state == HandState.SWIPING else (180, 180, 180)
        cv2.putText(frame, f"{label} hand", (x, y), font, 0.6, state_color, 2)
        cv2.putText(frame, track.state.name, (x + 130, y), font, 0.5, state_color, 1)
        y += 24

        # Finger states
        if track.finger_states:
            parts = []
            for fname in ("index", "middle", "ring", "pinky"):
                ext = track.finger_states.get(fname, False)
                parts.append(f"{fname[0].upper()}:{'O' if ext else 'X'}")
            cv2.putText(frame, "  ".join(parts), (x, y), font, 0.5, (200, 200, 200), 1)
            y += 22

            # Open hand indicator
            open_color = (0, 255, 0) if track.is_open else (100, 100, 100)
            cv2.putText(frame, "OPEN" if track.is_open else "----", (x, y), font, 0.55, open_color, 2)
            y += 22

        # Swipe displacement bar
        if track.state == HandState.SWIPING:
            dx = track.swipe_dx
            thresh = detector.SWIPE_THRESHOLD
            bar_w = 200
            bar_h = 16
            bar_x = x
            bar_center = bar_x + bar_w // 2

            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h), (80, 80, 80), 1)

            # Threshold markers
            left_mark = int(bar_center - (thresh / 300) * (bar_w // 2))
            right_mark = int(bar_center + (thresh / 300) * (bar_w // 2))
            cv2.line(frame, (left_mark, y), (left_mark, y + bar_h), (0, 0, 200), 1)
            cv2.line(frame, (right_mark, y), (right_mark, y + bar_h), (0, 0, 200), 1)

            # Current displacement
            fill_x = int(bar_center + (dx / 300) * (bar_w // 2))
            fill_x = max(bar_x, min(bar_x + bar_w, fill_x))
            bar_color = (0, 255, 0) if abs(dx) > thresh else (200, 200, 0)
            cv2.line(frame, (bar_center, y + bar_h // 2), (fill_x, y + bar_h // 2), bar_color, 4)

            cv2.putText(frame, f"dx: {dx:+.0f}", (bar_x + bar_w + 8, y + 14), font, 0.45, (200, 200, 200), 1)
            y += 24

    # Palm dots + velocity arrows
    for hand in hands:
        cx, cy = hand.palm_center_px.astype(int)
        track = debug.hand_tracks.get(hand.label)
        base_color = (255, 200, 0) if hand.label == "Right" else (0, 200, 255)

        if track and track.state == HandState.SWIPING:
            cv2.circle(frame, (cx, cy), 14, (0, 255, 255), 3)
        else:
            cv2.circle(frame, (cx, cy), 8, base_color, -1)

        if track and track.velocity is not None and track.speed > 60:
            end_x = int(cx + track.velocity[0] * 0.12)
            end_y = int(cy + track.velocity[1] * 0.12)
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)

    # Event log (bottom-right)
    log_x = w - 260
    log_y = h - 30
    for entry_time, evt in reversed(list(event_log)):
        age = time.monotonic() - entry_time
        alpha = max(0.3, 1.0 - age / 5.0)
        lbl = _EVENT_LABELS.get(evt, evt.name)
        color = _EVENT_COLORS.get(evt, (255, 255, 255))
        faded = tuple(int(c * alpha) for c in color)
        cv2.putText(frame, f"{age:.1f}s  {lbl}", (log_x, log_y), font, 0.5, faded, 1)
        log_y -= 26

    # Big center flash
    for entry_time, evt in event_log:
        age = time.monotonic() - entry_time
        if age < 1.5:
            lbl = _EVENT_LABELS.get(evt, evt.name)
            color = _EVENT_COLORS.get(evt, (255, 255, 255))
            sz = cv2.getTextSize(lbl, font, 1.8, 3)[0]
            cv2.putText(frame, lbl, ((w - sz[0]) // 2, h // 2), font, 1.8, color, 3)
            break


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="F.A.D.E gesture window control")
    parser.add_argument(
        "--source", default="1",
        help="Camera source: device index (0,1,...) or stream URL.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable camera preview window")
    parser.add_argument(
        "--live", action="store_true",
        help="Enable actual window management (implies --no-preview)",
    )
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    camera = CameraSource(source=source, width=args.width, height=args.height)
    tracker = HandTracker()
    detector = GestureDetector()

    controller = None
    if args.live:
        from window_controller import WindowController
        controller = WindowController()
        args.no_preview = True

    if not camera.open():
        print("ERROR: could not open camera. Exiting.")
        sys.exit(1)

    print("\n── F.A.D.E running ──")
    print("  Open hand + swipe left/right to move windows between monitors.")
    print("  Press 'q' to quit  |  'r' to reset")
    if not args.live:
        print("  [CV ONLY] window actions disabled (use --live to enable)")
    print()

    prev_time = time.monotonic()
    fps = 0.0
    consecutive_failures = 0
    event_log: deque = deque(maxlen=8)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    if not camera.reconnect():
                        break
                    consecutive_failures = 0
                continue
            consecutive_failures = 0

            now = time.monotonic()
            dt = now - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = now

            frame = cv2.flip(frame, 1)

            hands = tracker.process(frame)
            events = detector.update(hands)

            for evt in events:
                lbl = _EVENT_LABELS.get(evt, evt.name)
                print(f"  >> {lbl}")
                event_log.append((time.monotonic(), evt))

                if controller is not None:
                    if evt == GestureEvent.MOVE_TO_LEFT:
                        controller.move_from_monitor("right", "left")
                    elif evt == GestureEvent.MOVE_TO_RIGHT:
                        controller.move_from_monitor("left", "right")

            if not args.no_preview:
                tracker.draw_landmarks(frame, hands)
                _draw_hud(frame, hands, detector, event_log, fps)
                cv2.imshow("F.A.D.E", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                detector.reset()
                event_log.clear()
                print("  [reset]")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        camera.close()
        tracker.close()
        cv2.destroyAllWindows()
        print("Goodbye.")


if __name__ == "__main__":
    main()
