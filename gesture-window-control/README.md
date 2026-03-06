# F.A.D.E – Gesture Window Control

A gesture-controlled window management system for macOS. Move and swap windows across monitors using hand gestures captured by a camera.

## Requirements

- macOS (tested on Ventura / Sonoma / Sequoia)
- Python 3.10+
- A camera source (built-in webcam, Continuity Camera, or an iPhone streaming app like Camo / EpocCam)
- Accessibility permissions granted to Terminal / your IDE (System Settings → Privacy & Security → Accessibility)

## Setup

```bash
cd gesture-window-control
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Default webcam (device index 0)
python main.py

# Specific device index (e.g. Continuity Camera)
python main.py --source 1

# iPhone via network stream (EpocCam, Camo, or custom)
python main.py --source http://192.168.1.5:4747/video

# Headless mode (no preview window)
python main.py --no-preview

# Dry run (print gestures without moving windows)
python main.py --dry-run
```

Press **q** in the preview window to quit, **r** to reset gesture state.

## Gestures

| Gesture | How to perform | Action |
|---------|---------------|--------|
| **Grasp** | Curl all fingers into a fist | Begins tracking (grab a window) |
| **Release** | Open hand from a grasp | Cancels grab if no throw detected |
| **Throw Left** | Grasp → fast hand swipe left → release | Move frontmost window one monitor left |
| **Throw Right** | Grasp → fast hand swipe right → release | Move frontmost window one monitor right |
| **Cross Hands** | Both hands cross each other horizontally | Swap all windows between monitor 0 and 1 |

## Project Structure

```
gesture-window-control/
├── main.py               # Main event loop
├── camera.py             # Camera connection and frame capture
├── hand_tracker.py       # MediaPipe hand tracking and landmark extraction
├── gesture_detector.py   # Gesture recognition state machine
├── window_controller.py  # macOS window movement (AppleScript / JXA)
├── requirements.txt      # Python dependencies
└── README.md
```

## Permissions

The first time you move a window, macOS will prompt you to grant **Accessibility** access to the process running the script (usually Terminal.app or your IDE). You must allow this or window repositioning will fail silently.

## Tuning

Key thresholds live in `gesture_detector.py`:

- `THROW_VELOCITY_THRESHOLD` – minimum hand speed (px/s) to count as a throw (default: 600)
- `THROW_MIN_DISPLACEMENT` – minimum horizontal travel (px) for directional detection (default: 80)
- `_CrossDetector.MIN_CROSS_DISTANCE` – minimum gap between hands for a cross event (default: 100)

Adjust these if gestures trigger too easily or not easily enough for your camera setup and distance.
