"""
Window controller – macOS window movement via AppleScript + JXA.

Key insight: System Events exposes ALL windows (including toolbars), so we
must find the main window by checking AXFullScreen or by largest area —
never assume ``window 1`` is the right one.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Monitor:
    x: int
    y: int
    width: int
    height: int
    name: str = ""


def _run_jxa(script: str, timeout: int = 15) -> str:
    result = subprocess.run(
        ["osascript", "-l", "JavaScript", "-e", script],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"JXA error: {result.stderr.strip()}")
    return result.stdout.strip()


def _run_applescript(script: str, timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=timeout,
    )


def detect_monitors() -> List[Monitor]:
    jxa = """
    ObjC.import('AppKit');
    var screens = $.NSScreen.screens;
    var result = [];
    for (var i = 0; i < screens.count; i++) {
        var s = screens.objectAtIndex(i);
        var frame = s.frame;
        result.push({
            name: s.localizedName.js,
            x: frame.origin.x, y: frame.origin.y,
            width: frame.size.width, height: frame.size.height
        });
    }
    JSON.stringify(result);
    """
    try:
        raw = _run_jxa(jxa)
        data = json.loads(raw)
        monitors = [Monitor(x=int(m["x"]), y=int(m["y"]),
                            width=int(m["width"]), height=int(m["height"]),
                            name=m["name"]) for m in data]
        monitors.sort(key=lambda m: m.x)
        return monitors
    except Exception as e:
        print(f"[wc] monitor detection failed ({e})")
        return [
            Monitor(x=0, y=0, width=1920, height=1080, name="Display 1"),
            Monitor(x=1920, y=0, width=1920, height=1080, name="Display 2"),
        ]


class WindowController:

    def __init__(self, monitors: Optional[List[Monitor]] = None):
        self.monitors = monitors or detect_monitors()
        print(f"[wc] {len(self.monitors)} monitor(s):")
        for i, m in enumerate(self.monitors):
            print(f"  [{i}] \"{m.name}\" x={m.x} {m.width}x{m.height}")

        self._monitor_apps: Dict[int, str] = {}
        self._cache_lock = threading.Lock()

        threading.Thread(target=self._scan_loop, daemon=True).start()
        threading.Thread(target=lambda: _run_jxa('"warm"'), daemon=True).start()

    def _scan_loop(self):
        while True:
            try:
                self._update_cache()
            except Exception:
                pass
            time.sleep(1.0)

    def _update_cache(self):
        """Find the topmost/biggest app window on each monitor."""
        monitor_ranges = [f"{{idx:{i},x:{m.x},xEnd:{m.x + m.width}}}"
                          for i, m in enumerate(self.monitors)]
        ranges_js = "[" + ",".join(monitor_ranges) + "]"
        jxa = f"""
        var se = Application("System Events");
        var procs = se.applicationProcesses.whose({{backgroundOnly: false}});
        var monitors = {ranges_js};
        var result = {{}};
        var areas = {{}};
        for (var i = 0; i < procs.length; i++) {{
            var proc = procs[i];
            try {{
                var name = proc.name();
                if (name.toLowerCase().indexOf("python") !== -1) continue;
                var wins = proc.windows();
                for (var j = 0; j < wins.length; j++) {{
                    var pos = wins[j].position();
                    var sz = wins[j].size();
                    var px = pos[0]; var area = sz[0] * sz[1];
                    for (var k = 0; k < monitors.length; k++) {{
                        var mon = monitors[k];
                        if (px >= mon.x && px < mon.xEnd) {{
                            if (!result[mon.idx] || area > (areas[mon.idx] || 0)) {{
                                result[mon.idx] = name;
                                areas[mon.idx] = area;
                            }}
                        }}
                    }}
                }}
            }} catch(e) {{}}
        }}
        JSON.stringify(result);
        """
        raw = _run_jxa(jxa)
        data = json.loads(raw)
        with self._cache_lock:
            self._monitor_apps = {int(k): v for k, v in data.items()}

    def move_from_monitor(self, source_side: str, direction: str):
        src_idx = 0 if source_side == "left" else len(self.monitors) - 1
        with self._cache_lock:
            app_name = self._monitor_apps.get(src_idx)
        if not app_name:
            print(f"[wc] no cached window on {source_side} monitor")
            return

        tgt_idx = (max(0, src_idx - 1) if direction == "left"
                   else min(len(self.monitors) - 1, src_idx + 1))
        if src_idx == tgt_idx:
            return

        src = self.monitors[src_idx]
        tgt = self.monitors[tgt_idx]

        threading.Thread(
            target=self._do_move,
            args=(app_name, src, tgt),
            daemon=True,
        ).start()

    def _do_move(self, app_name: str, src: Monitor, tgt: Monitor):
        """AppleScript: find correct window → exit FS → move → enter FS.

        System Events lists ALL windows including toolbars, so we iterate
        to find the one that is AXFullScreen or the biggest by area.
        """
        script = f'''
        tell application "{app_name}" to activate
        delay 0.3

        -- Phase 1: Exit fullscreen on the source app's MAIN window
        -- (iterate SE windows; don't assume window 1 is main)
        tell application "System Events"
            tell process "{app_name}"
                set wCount to count of windows
                repeat with i from 1 to wCount
                    try
                        set fs to value of attribute "AXFullScreen" of window i
                        if fs is true then
                            set value of attribute "AXFullScreen" of window i to false
                            delay 1.0
                            exit repeat
                        end if
                    end try
                end repeat
            end tell
        end tell

        -- Phase 2: Exit fullscreen on any window on the TARGET monitor
        tell application "System Events"
            set allProcs to every application process whose background only is false
            repeat with proc in allProcs
                try
                    set pName to name of proc
                    if pName is not "{app_name}" then
                        set ws to windows of proc
                        repeat with w in ws
                            try
                                set p to position of w
                                if item 1 of p >= {tgt.x} and item 1 of p < {tgt.x + tgt.width} then
                                    set fs2 to value of attribute "AXFullScreen" of w
                                    if fs2 is true then
                                        set value of attribute "AXFullScreen" of w to false
                                        delay 0.8
                                    end if
                                end if
                            end try
                        end repeat
                    end if
                end try
            end repeat
        end tell

        -- Phase 3: Move the window to the target monitor
        -- Use the app's own scripting dictionary (window 1 = main window)
        tell application "{app_name}" to activate
        delay 0.3
        try
            tell application "{app_name}"
                set bounds of window 1 to {{{tgt.x}, {tgt.y}, {tgt.x + tgt.width}, {tgt.y + tgt.height}}}
            end tell
        on error
            -- Fallback: use System Events to find and move the biggest window
            tell application "System Events"
                tell process "{app_name}"
                    set wCount to count of windows
                    set bestWin to 0
                    set bestArea to 0
                    repeat with i from 1 to wCount
                        try
                            set s to size of window i
                            set a to (item 1 of s) * (item 2 of s)
                            if a > bestArea then
                                set bestArea to a
                                set bestWin to i
                            end if
                        end try
                    end repeat
                    if bestWin > 0 then
                        set position of window bestWin to {{{tgt.x}, {tgt.y}}}
                        set size of window bestWin to {{{tgt.width}, {tgt.height}}}
                    end if
                end tell
            end tell
        end try

        delay 0.3

        -- Phase 4: Enter fullscreen on target
        tell application "{app_name}" to activate
        delay 0.2
        tell application "System Events"
            tell process "{app_name}"
                set wCount to count of windows
                set bestWin to 0
                set bestArea to 0
                repeat with i from 1 to wCount
                    try
                        set s to size of window i
                        set a to (item 1 of s) * (item 2 of s)
                        if a > bestArea then
                            set bestArea to a
                            set bestWin to i
                        end if
                    end try
                end repeat
                if bestWin > 0 then
                    set value of attribute "AXFullScreen" of window bestWin to true
                end if
            end tell
        end tell

        return "{app_name} moved"
        '''
        try:
            r = _run_applescript(script, timeout=25)
            if r.returncode == 0:
                print(f"[wc] {app_name} → monitor at x={tgt.x}")
            else:
                print(f"[wc] move failed: {r.stderr.strip()}")
        except Exception as e:
            print(f"[wc] move failed: {e}")
