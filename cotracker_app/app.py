"""Main window: state machine, tracking loop wiring, projects, exports.

Threading rules (the GUI thread never blocks):
- video decoding: SeekService thread (scrubbing) / TrackingWorker's own source
- model load + inference: TrackingWorker (QThread)
- torch is imported lazily in background threads (app startup stays instant)
"""

from __future__ import annotations

import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QTimer, QThread, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractSpinBox, QTextEdit, QAbstractItemView, QApplication, QDialog, QDialogButtonBox,
                               QDockWidget, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QProgressDialog,
                               QPushButton, QSizePolicy, QSpinBox, QSplitter, QTextBrowser,
                               QToolButton, QVBoxLayout, QWidget)

from cotracker_app import APP_NAME, APP_TAGLINE, APP_VERSION, theme
from cotracker_app import alltracker_backend
from cotracker_app.camerapanel import CameraPanel
from cotracker_app.view3d import CalibrationDialog, Scene3D, View3D
from cotracker_app.canvas import (ZOOM_STEP as CANVAS_ZOOM_STEP,
                                  TRAIL_FRAMES, DISPLAY_FILTERS, REGION_SHAPES)
from cotracker_app.project import MAX_VIEWS, REFERENCE_VIEW, Project
from cotracker_app.segmenter import (BACKENDS, DEFAULT_BACKEND, backend_status, has_token,
                                     model_is_cached as seg_is_cached, preferred_backend,
                                     save_token, working_size)
from cotracker_app.session import AUTOSAVE_SUFFIX, PROJECT_SUFFIX, TrackingSession
from cotracker_app.viewgrid import ViewGrid, caption_for
from cotracker_app.skeletons import all_templates, save_user_template
from cotracker_app.theme import apply_theme
from cotracker_app.timeline import TIME_ZOOM_STEP, TimelinePanel
from cotracker_app.video_source import DEFAULT_CACHE_BYTES, FrameCache, SeekService, VideoInfo, probe_video
from cotracker_app import icons
from cotracker_app.widgets import ManualDialog, OnboardingStrip, SettingsDialog, Toast

# duplicated from tracker.py so the GUI can check without importing torch
_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "scaled_online.pth"

IDLE, READY, TRACKING = range(3)
VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.webm *.mpg *.mpeg);;All files (*)"
# a companion camera only ever displays ONE frame, so it needs room for a couple
# of 4K frames and nothing more (see MainWindow._rebudget_caches)
COMPANION_CACHE_BYTES = 96 * 1024 ** 2

HOTKEYS_HTML = f"""
<style>
 h3 {{ margin: 10px 0 4px 0; color: {theme.TEXT}; }}
 table {{ border-collapse: collapse; margin-bottom: 6px; }}
 td {{ padding: 2px 10px 2px 0; vertical-align: top; color: {theme.TEXT_DIM}; }}
 td.k {{ white-space: nowrap; font-weight: bold; color: {theme.ACCENT_HOVER}; }}
</style>
<h3>Navigation</h3><table>
<tr><td class=k>F / B</td><td>frame forward / back (semi-automatic mode: F <i>tracks</i> one frame, then pauses)</td></tr>
<tr><td class=k>Shift+F / Shift+B</td><td>jump by the <i>step</i> size set in the control panel</td></tr>
<tr><td class=k>← / → (Shift: ±step)</td><td>step frames</td></tr>
<tr><td class=k>Home / End</td><td>first / last frame</td></tr>
<tr><td class=k>Space</td><td>play / pause preview (no tracking); pauses tracking while a run is live</td></tr>
<tr><td class=k>frame box</td><td>type a frame number + Enter</td></tr>
</table>
<h3>Tracking</h3><table>
<tr><td class=k>T</td><td>start tracking / pause (semi-automatic mode: one step)</td></tr>
<tr><td class=k>X / Space</td><td>pause a running track (during 3D → Re-track Disagreeing Stretches: stops the whole queue and asks whether to keep what was re-tracked)</td></tr>
<tr><td class=k>Track ▾</td><td>dropdown: Automatic (to the end) or Semi-automatic (F steps); <b>point model</b>: AllTracker (default, holds points on animals) or CoTracker3 (faster, sub-pixel on high-contrast markers)</td></tr>
<tr><td class=k>Auto-pause</td><td>stop the run when the model loses a point (a brief occlusion doesn't trigger it); the point's track is <b>cut</b> at the first unreliable frame and the playhead goes there</td></tr>
<tr><td class=k>ROI</td><td>track inside a crop around the points when it clearly helps</td></tr>
<tr><td class=k>Ctrl+Z</td><td>undo the last tracking run, bulk edit or hand edit (a click, drag, Ctrl+click, deleted point, Shift+X) — one step</td></tr>
</table>
<h3>Points &amp; regions (on the video)</h3><table>
<tr><td class=k>Body</td><td>keep tracked points on the segment: a point a few pixels off the silhouette is nudged back onto it; a point that <b>leaves</b> it stops the run at that frame and its track ends there — click it where it really is and Track again (right-click a point → "May leave the segment (free point)" to exempt it)</td></tr>
<tr><td class=k>N (or Add)</td><td>arm the crosshair: the next click places a point — stray clicks never edit</td></tr>
<tr><td class=k>click (armed)</td><td>place a new point — or continue the selected point where it has no data</td></tr>
<tr><td class=k>click (not armed)</td><td><b>annotate by hand</b>: place the point selected in the list here on this frame, replacing what the tracker put there. Nothing selected = nothing happens. One Ctrl+Z step. A landmark derived from the silhouette cannot be placed by hand (right-click → Data source → Track by appearance first)</td></tr>
<tr><td class=k>Shift+&lt; / Shift+&gt;</td><td>jump to the selected point's <b>first / last frame with data</b>; with nothing selected, the segment's first / last silhouette</td></tr>
<tr><td class=k>right-click a point (curve)</td><td><b>Fill its gaps between hand placements</b> / <b>Replace everything between its hand placements with that curve</b>: keyframe digitizing — a smooth curve through the frames you placed by hand fills the frames between (confidence 0.6); frames you marked hidden are left alone; Ctrl+Z undoes</td></tr>
<tr><td class=k>right-click a point</td><td>also: go to its first / last frame, its first / last hand-placed frame, its first doubtful stretch; clear its position on this frame, in the selected frame window, or its whole track; with a calibration, <b>Snap to the other cameras' rays here</b></td></tr>
<tr><td class=k>, / .</td><td>previous / next hand-placed frame of the selected point</td></tr>
<tr><td class=k>J / Shift+J</td><td>next / previous low-confidence stretch (the red runs) — of the selected points, or of all</td></tr>
<tr><td class=k>Shift+X</td><td>mark the selected point <b>hidden</b> on this frame (kept, not exported, not used for 3D); again to unmark. On the timeline: Shift+drag a window, right-click → Mark hidden</td></tr>
<tr><td class=k>Shift+N</td><td>note on this frame (▲ on the timeline; right-click it to edit)</td></tr>
<tr><td class=k>Add ▾</td><td>region shape for an armed drag: circle · rectangle · polygon (click corners, Enter closes) · <b>Ball marker</b>: click a ball, SAM outlines it every frame and the fitted circle's centre is the point (wand balls, markers, a dropped ball — add one per ball; they track together as long as they stay within about 740 px of each other — farther apart, the run stops and says to track each ball alone)</td></tr>
<tr><td class=k>O</td><td>onion skin: ghost markers of the previous (solid) and next (dashed) frame</td></tr>
<tr><td class=k>L</td><td>loupe: magnifier under the cursor with a crosshair on the exact pixel</td></tr>
<tr><td class=k>View → Trails</td><td>trail length (fading), optional upcoming path; View → Display filter: contrast / brighten / frame difference (display only)</td></tr>
<tr><td class=k>drag (armed)</td><td>outline a region in the Add ▾ shape (circle or rectangle; a polygon is clicked corner by corner) → tracked as one point (its fitted center)</td></tr>
<tr><td class=k>drag a marker</td><td>move / correct it on this frame (one Ctrl+Z step; not for a landmark derived from the silhouette)</td></tr>
<tr><td class=k>Ctrl+click</td><td>move the selected point here (one Ctrl+Z step)</td></tr>
<tr><td class=k>right-click</td><td>marker or list entry: rename · lock to seed appearance · data source · hidden on this frame · may leave the segment · delete</td></tr>
<tr><td class=k>Ctrl/Shift+click (list)</td><td>select several points</td></tr>
<tr><td class=k>Delete</td><td>delete selected point(s) — or clear the selected frame window (below)</td></tr>
<tr><td class=k>Esc</td><td>drops a drag or polygon in progress; then, one per press: the segment tool, the armed crosshair, a half-marked event, the frame-window selection; then deselects</td></tr>
</table>
<h3>Segment &amp; skeleton (optional)</h3><table>
<tr><td class=k>S (or Segment)</td><td>segment tool: <b>click the animal</b> — its silhouette appears within a second. Shift+click = "not the animal", drag = box around it. Right-click a click marker to remove it. S or Esc when done. Optional: points track without a segment. The ▾ on the Segment button picks the segmentation model; its last entry is Settings (Ctrl+,)</td></tr>
<tr><td class=k>Track ▶</td><td>with a segment defined, a run also segments every frame: the silhouette follows the animal, the crop follows the silhouette, and silhouette landmarks (tail tip, midline, feet) fill in</td></tr>
<tr><td class=k>Skeleton ▾</td><td>apply a named landmark set for your study animal; select a landmark, press N and click it. Landmarks marked <i>from silhouette</i> need no click — they come from the mask</td></tr>
<tr><td class=k>right-click a point</td><td>Data source: track by appearance, or derive from the silhouette (tail tip, midline %, extremities). Switching asks first, because it erases the point's track (Ctrl+Z brings it back). Feet and wing tips are named as seen from above; filmed from below, left and right swap</td></tr>
<tr><td class=k>Mask</td><td>show / hide the silhouette overlay (View menu: midline, bones; Settings: opacity, model)</td></tr>
<tr><td class=k>SEGMENT row (panel)</td><td>the segment has its own row in the right panel: checkbox = show / hide its silhouette, double-click = rename, <b>right-click</b> = jump to its first / last silhouette, clear it on this frame / in the selected frame window / everywhere (your clicks stay), or remove the segment</td></tr>
<tr><td class=k>fix a wrong mask</td><td>pause, go to the frame, press S, click again (Shift+click to exclude), press Track — the correction applies from there on</td></tr>
</table>
<h3>New to tracking?</h3><table>
<tr><td class=k>F1</td><td><b>Help → User Manual</b> — the full manual, written for someone who has never tracked anything before: what the ideas mean, a first session step by step, how to fix mistakes, and a glossary. This page is the quick reference; that one explains <i>why</i></td></tr>
</table>
<h3>Several cameras</h3><table>
<tr><td class=k>＋ Add video</td><td>CAMERAS panel (top of the right dock): put every camera of the same event in one project. Each view shows its own points and silhouette; up to 15</td></tr>
<tr><td class=k>offset  ◂ ▸</td><td><b>the frame this camera shows when camera 1 (the reference) is at its frame 0</b>: a camera switched on 12 frames after the reference has offset −12, one switched on earlier a positive offset. The first camera loaded is the reference: its offset is 0 by definition and locked, so every other number is measured against it and stays put when you switch cameras. Scrub to something every camera saw (a flash, a clap, first contact) and nudge until it lines up — or press <b>Align here</b> to take what you see as the match. The panel then reports the overlap window</td></tr>
<tr><td class=k>click a view</td><td>switch to that camera. Points, silhouette, events and the timeline are always the <i>working</i> camera's; the playhead keeps the same instant across the switch. Switching disarms N / S, drops a half-marked event and forgets the undo step; the tool toggles (Auto-pause, ROI, Body, Follow, the point model…) stay as you set them</td></tr>
<tr><td class=k>Ctrl+2</td><td>show only the working camera (the others stop decoding too)</td></tr>
<tr><td class=k>Ctrl+E</td><td><b>ALL CAMERAS — DLTdv8 xypts</b> writes one file for 3D reconstruction: row k = frame k of the reference camera (from 0), every camera sampled through its offset, landmarks matched across cameras <i>by name</i>, NaN where a camera has no data; top-left origin, first pixel = 1</td></tr>
<tr><td class=k>3D menu</td><td><b>Sync Cameras (Sound / Motion)</b> (whole-frame offsets from the sound tracks or the pictures, with a noise filter) · <b>Calibrate a Lens (checkerboard)</b> (how a wide-angle lens bends the picture; opens with or without a video — with nothing open, save the lens file and attach it later) · <b>Calibrate Cameras with a Wand</b> · <b>Import Calibration</b> (a Kinetrace .kcal.json, DLT coefficients from DLTdv / easyWand / Argus, K + R/t, or a DLTdv8 project with its lens undistortion) · <b>Export Calibration</b> · <b>Estimate Sub-frame Offsets</b> (fractional sync from the tracks) · <b>Re-track Disagreeing Stretches</b> (a camera's landmark put back on the other cameras' rays and re-tracked, before / after verdict) · <b>Export Mesh of This Frame</b>. Entries that need more (a video, a second camera, a calibration) open anyway and say what is missing</td></tr>
<tr><td class=k>Ctrl+3</td><td>reconstruct the 3D landmarks (matched by name across cameras) and open the 3D view</td></tr>
<tr><td class=k>Ctrl+4</td><td>carve the volume hull at this frame from every camera's silhouette</td></tr>
<tr><td class=k>Ctrl+5</td><td>show / hide the 3D view (drag to orbit, wheel to zoom, middle-drag to pan, double-click to reset)</td></tr>
<tr><td class=k>Ctrl+6</td><td>show / hide the body side-by-side view (footage + skeleton | the pose on its own | joint angles over time)</td></tr>
</table>
<h3>Events &amp; timeline</h3><table>
<tr><td class=k>E, then E</td><td>mark an event window — the name dropdown re-uses existing event types, so the same event can be tagged at many times (one color per type; the Events menu groups occurrences)</td></tr>
<tr><td class=k>scroll strip</td><td>while time-zoomed, the bar at the panel's bottom edge shows the view window — click/drag it to move through the video</td></tr>
<tr><td class=k>click / drag lanes</td><td>seek (the timeline IS the scrubber)</td></tr>
<tr><td class=k>click a ribbon</td><td>select the event's frame window AND jump to its start (right-click: rename, retime, delete)</td></tr>
<tr><td class=k>Shift+drag</td><td><b>select a frame window AND the lanes under the drag</b> → Delete clears exactly those: drag over the segment lane to remove the <i>silhouettes</i> in that stretch, over point lanes to remove <i>their</i> tracks, over both to remove both (one Ctrl+Z). Drag in the ruler to select the window without naming lanes — Delete then uses the point panel's selection, as before. Right-click inside the band for the same choices</td></tr>
<tr><td class=k>Shift+&#43; / Shift+− (or ⊕ ⊖ ⤢)</td><td>zoom the time axis around the playhead (also Ctrl+wheel); ⤢ shows the whole video; middle-drag pans; when zoomed, every frame gets its own pixels so you can click exactly the frame you want</td></tr>
<tr><td class=k>wheel</td><td>scroll point lanes when they overflow</td></tr>
<tr><td class=k>splitter bar</td><td>drag (above the panel) to give the timeline more lanes</td></tr>
</table>
<h3>View</h3><table>
<tr><td class=k>Ctrl+1</td><td>show / hide the right panel (Segment &amp; Points). Floated as its own window, it keeps the hotkeys</td></tr>
<tr><td class=k>narrow window</td><td>the tool buttons at the bottom show only their icons; hover one to see what it does. Menu entries explain themselves on hover too</td></tr>
<tr><td class=k>wheel</td><td>zoom the video under the cursor</td></tr>
<tr><td class=k>&#43; / − &nbsp;(or =)</td><td>zoom the video around the pointer</td></tr>
<tr><td class=k>R</td><td>reset view (fit the video)</td></tr>
<tr><td class=k>H (or Pan)</td><td>pan tool: left-drag pans instead of editing (does nothing until a video is open); middle-drag always pans, any time — even while tracking</td></tr>
<tr><td class=k>Follow</td><td>off by default; when on, keeps the selected point in view while zoomed in — or auto-frames all points when none is selected. R always fits the whole picture</td></tr>
<tr><td class=k>marker px</td><td>marker size on screen (shrink it to see exact placement)</td></tr>
<tr><td class=k>Shift+C</td><td>failsafe: clear the frame cache and re-decode this frame from the file (use if the picture ever looks stale or garbled; tracked data is untouched)</td></tr>
</table>
<h3>Files</h3><table>
<tr><td class=k>Ctrl+O / Ctrl+Shift+O</td><td>open video / project</td></tr>
<tr><td class=k>Ctrl+S / Ctrl+Shift+S</td><td>save project / save as (projects restore the exact working state)</td></tr>
<tr><td class=k>Ctrl+E</td><td>export tracks (CSV / TSV / MATLAB; events included)</td></tr>
<tr><td class=k>Ctrl+,</td><td>Settings: segmentation model, access token, silhouette opacity (also the last entry of the Segment ▾ menu)</td></tr>
</table>
"""


class _VideoProbe(QThread):
    done = Signal(object)  # VideoInfo | str(error)

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def run(self):
        try:
            self.done.emit(probe_video(self._path))
        except Exception as e:  # noqa: BLE001 — reported to the user verbatim
            self.done.emit(str(e))


class _DeviceProbe(QThread):
    """Imports torch in the background (slow) and reports the compute device."""
    got = Signal(str)

    def run(self):
        try:
            from cotracker_app.tracker import pick_device
            self.got.emit(pick_device()[1])
        except Exception as e:  # noqa: BLE001
            self.got.emit(f"device unavailable: {e}")


class _MaskPreviewWorker(QThread):
    """Segments ONE frame from the user's prompts so the silhouette shows right
    after a click (a tracking run re-segments every frame). Loads the model on
    first use — which downloads the weights once."""
    loading = Signal(str)
    done = Signal(object)      # mask summary dict (see segmenter.summarize_mask) + "frame"
    error = Signal(str)

    def __init__(self, video_path: str, frame: int, rgb, size: tuple[int, int],
                 clicks: list, box, backend: str, head_xy):
        super().__init__()
        self._video_path = video_path
        self._frame = frame
        self._rgb = rgb
        self._size = size
        self._clicks = list(clicks)
        self._box = box
        self._backend = backend
        self._head = None if head_xy is None else np.asarray(head_xy, np.float32)

    def run(self):
        try:
            from cotracker_app.segmenter import (MIDLINE_SAMPLES, Prompt, get_segmenter,
                                                 summarize_mask)
            from cotracker_app.silhouette import midline as silhouette_midline, resample
            self.loading.emit(self._backend)
            seg = get_segmenter(self._backend)
            rgb = self._rgb
            if rgb is None:
                from cotracker_app.video_source import VideoSource
                src = VideoSource(self._video_path, FrameCache(64 * 1024 ** 2))
                try:
                    rgb = src.get_frame(self._frame)
                finally:
                    src.close()
                if rgb is None:
                    raise RuntimeError(f"Could not decode frame {self._frame}")
            sess = seg.new_session(self._frame, self._size)
            pts = (np.array([[c[0], c[1]] for c in self._clicks], np.float32)
                   if self._clicks else None)
            labs = (np.array([c[2] for c in self._clicks], np.int64)
                    if self._clicks else None)
            prompt = Prompt(1, pts, labs, self._box)
            fm = sess.step(rgb, self._frame, [prompt])
            mask = fm.mask(1)
            # per axis: working size rounds each side on its own (I61)
            sx, sy = getattr(fm, "scale_xy", (fm.scale, fm.scale))
            summ = summarize_mask(mask, (sx, sy), float(fm.scores[fm.index(1)]))
            summ["frame"] = self._frame
            if summ["area"] > 0:
                anchor = None
                if self._head is not None:
                    anchor = ((self._head[0] + 0.5) / sx - 0.5, (self._head[1] + 0.5) / sy - 0.5)
                ml = silhouette_midline(mask, anchor=anchor)
                if ml is not None and len(ml.path) >= 2:
                    summ["midline"] = ((resample(ml.path, MIDLINE_SAMPLES) + 0.5)
                                       * np.array([sx, sy], np.float32) - 0.5)
            self.done.emit(summ)
        except Exception:  # noqa: BLE001 — reported to the user with guidance
            self.error.emit(traceback.format_exc())


def _model_error_hint(tb: str) -> str:
    """Turn a traceback into one actionable sentence for the user."""
    low = tb.lower()
    if "out of memory" in low:
        return ("The GPU ran out of memory. Close other GPU applications (or pick a smaller "
                "segmentation model in Settings) and try again.")
    if "gated" in low or "401" in low or "403" in low:
        return ("The model weights are gated on Hugging Face. Request access to the model, "
                "then paste a read token in Settings (it is stored inside this folder).")
    if any(k in low for k in ("download", "urlopen", "connection", "resolve", "name or service",
                              "max retries", "timed out", "offline")):
        return ("The model could not be downloaded. Check your internet connection and try "
                "again — the weights are only needed once, then they live in models/.")
    if "cuda" in low and "device" in low:
        return "The GPU could not be used. Check the NVIDIA driver, or run on CPU (slow)."
    if "no click, box or mask" in low:
        return "Press S, click the animal on the current frame, then start tracking again."
    return ""


class _ViewRuntime:
    """The DECODE side of one camera view: its probe result, its frame cache and
    its scrubbing thread. One per view; `MainWindow.info` / `.cache` / `.seek`
    are just the active view's. Tracking data lives in the view's
    `TrackingSession` (inside `Project`) — the two are deliberately separate, so
    adding a camera cannot disturb anyone's tracks."""

    def __init__(self, info: VideoInfo, cache_bytes: int):
        self.info = info
        self.n_frames = info.n_frames        # may shrink if the file ends early
        self.cache = FrameCache(cache_bytes)
        self.seek: SeekService | None = None
        self.want_frame: int | None = None   # frame last requested (stale-reply guard)

    def stop(self) -> None:
        if self.seek is not None:
            self.seek.stop()
            _retire(self.seek)      # still running after its wait: keep it (I112)
            self.seek = None


# QThreads that did not stop within their wait: dropping the last reference to
# a running QThread aborts the process (0xC0000409) when it finishes. They are
# kept here until their `finished` fires; closeEvent waits for them (I112).
_ORPHANS: list = []


def _retire(th) -> None:
    if th is None:
        return
    try:
        running = th.isRunning()
    except RuntimeError:
        return
    if running and th not in _ORPHANS:
        _ORPHANS.append(th)
        th.finished.connect(lambda t=th: _ORPHANS.remove(t) if t in _ORPHANS else None)


# ui_state entries that belong to the USER (tools, display), not to one camera:
# carried across a camera switch (I50). Frame, selection and zoom stay per camera.
GLOBAL_UI_KEYS = ("follow", "autopause", "roi", "track_mode", "marker_size", "show_mask", "mask_opacity",
                  "show_midline", "show_bones", "seg_backend", "on_body", "point_backend", "trail_len",
                  "trail_future", "onion", "loupe", "epipolar", "display_filter", "region_shape")


class _ResizeHook(QObject):
    """Calls `callback()` after every resize of the widget it filters."""

    def __init__(self, callback):
        super().__init__()
        self._cb = callback

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Resize:
            self._cb()
        return False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        apply_theme(QApplication.instance())  # idempotent; visual only
        self.setWindowTitle(f"{APP_NAME} — {APP_TAGLINE}")
        self.resize(1280, 860)

        self.state = IDLE
        # a project holds one or more camera views; `session`, `info`, `cache`,
        # `seek` and `n_frames` below are properties onto the ACTIVE one, so
        # every single-video code path reads exactly as it always did
        self.project: Project | None = None
        self._views: list[_ViewRuntime] = []
        self.view3d: View3D | None = None          # the floating 3D window (created on demand)
        self._hull_cache: dict[int, tuple] = {}    # reference frame -> (verts, faces, Hull)
        # hotkeys are offered to `_hotkey` before the focused widget sees them
        # (see eventFilter): a click on a spin box or a list must never kill F
        QApplication.instance().installEventFilter(self)
        self.worker = None
        self.current = 0
        self.selected: int | None = None
        self.project_path: Path | None = None
        self._undo_snap = None
        self._display_queue: deque = deque()
        self._fps_ema = 0.0
        self._last_emit_t = 0.0
        self._probe: _VideoProbe | None = None
        self._model_dialog: QProgressDialog | None = None
        self._pending_event: int | None = None      # E pressed once: start frame
        self._autopause_info: tuple[int, int] | None = None  # (frame, pid)
        self._member_frames: dict[int, dict] = {}   # frame -> {pid: (M,2)} overlay
        self._track_mode = "auto"                   # "auto" | "semi" (F = one tracked step)
        self._step_run = False                      # current run is a semi-auto step
        self._seg_backend = preferred_backend()     # segmentation model (Settings)
        self._point_backend = self._preferred_point_backend()   # Track dropdown
        self._mask_opacity = 0.35
        self._region_shape = "circle"               # Add ▾: circle | rect | polygon
        self._overlay = None                        # running OverlayRenderer, if any
        self.body_win = None                        # BodySideBySide, when open
        self._body_worker = None                    # running BodyPoseWorker
        self._body_progress = None
        self._body_video = None                     # running SideBySideRenderer
        self._body_backend = ""                      # last pose model used (Body dialog)
        self._wand_result = (None, None)            # (WandResult, gravity dict) of the last wizard run
        self._trail_len = TRAIL_FRAMES              # View → Trails
        self._trail_future = False
        self._preview: _MaskPreviewWorker | None = None
        self._preview_again = False                 # prompts changed while a preview ran
        self._loading_dialog: QProgressDialog | None = None
        self._animal_hint_shown = False

        self._build_ui()
        self._apply_state()
        self._fit_to_screen()

        self._device_label.setText("probing GPU…")
        self._dev_probe = _DeviceProbe()
        self._dev_probe.got.connect(self._device_label.setText)
        self._dev_probe.start()

        self._render_timer = QTimer(self, interval=33, timerType=Qt.CoarseTimer)
        self._render_timer.timeout.connect(self._render_tick)
        self._autosave_timer = QTimer(self, interval=30_000)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)

    # ----------------------------------------------- the ACTIVE view, as attrs
    # Everything below the project layer works on one view at a time. Exposing
    # the active view through properties means the whole single-video app —
    # tracking, exports, undo, the timeline — needed no changes at all when
    # multi-camera support arrived.

    @property
    def _rt(self) -> _ViewRuntime | None:
        i = self.project.active if self.project is not None else 0
        return self._views[i] if 0 <= i < len(self._views) else None

    @property
    def session(self) -> TrackingSession | None:
        return self.project.session if self.project is not None else None

    @session.setter
    def session(self, s: TrackingSession | None) -> None:
        if s is None:
            self.project = None
        elif self.project is None or self.project.n_views == 0:
            self.project = Project([s])
        else:
            self.project.sessions[self.project.active] = s

    @property
    def canvas(self):
        """The active view's canvas. In a one-camera project this IS the old
        single canvas — the grid adds no chrome around a lone view."""
        return self.grid.active_canvas

    @property
    def info(self) -> VideoInfo | None:
        rt = self._rt
        return rt.info if rt is not None else None

    @property
    def cache(self) -> FrameCache | None:
        rt = self._rt
        return rt.cache if rt is not None else None

    @property
    def seek(self) -> SeekService | None:
        rt = self._rt
        return rt.seek if rt is not None else None

    @property
    def n_frames(self) -> int:
        rt = self._rt
        return rt.n_frames if rt is not None else 0

    @n_frames.setter
    def n_frames(self, value: int) -> None:
        if self._rt is not None:
            self._rt.n_frames = int(value)

    # ------------------------------------------------------------------- UI

    def _build_ui(self):
        self.grid = ViewGrid()
        self.grid.view_activated.connect(self._set_active_view)
        self._wire_canvas(self.grid.canvases[0])
        self.toast = Toast(self.grid)   # important notices float over the video
        self._build_ui_rest()

    def _wire_canvas(self, canvas) -> None:
        """Every view's canvas reports to the same handlers. Only the active one
        is interactive, so an edit can never arrive from the camera the user is
        not working in."""
        canvas.add_requested.connect(self._on_add)
        canvas.annotate_requested.connect(self._on_annotate)
        canvas.group_requested.connect(self._on_add_group)
        canvas.point_selected.connect(self._on_select)
        canvas.point_moved.connect(lambda pid, x, y: None)  # live marker already moves
        canvas.move_committed.connect(self._on_place)
        canvas.reposition_requested.connect(self._on_reposition)
        canvas.delete_requested.connect(self._on_delete)
        canvas.rename_requested.connect(self._on_rename)
        canvas.anchor_toggled.connect(self._on_anchor_toggled)
        canvas.source_change_requested.connect(self._on_source_change)
        canvas.free_toggled.connect(self._on_free_toggled)
        canvas.region_requested.connect(self._on_add_region)
        canvas.occluded_toggled.connect(self._on_occluded_toggled)
        canvas.menu_extra = self._extend_point_menu
        canvas.menu_extra_action = self._point_menu_extra_action
        canvas.animal_click.connect(self._on_animal_click)
        canvas.animal_box.connect(self._on_animal_box)
        canvas.prompt_remove_requested.connect(self._on_prompt_remove)
        canvas.view_clicked.connect(
            lambda c=canvas: self._on_canvas_clicked(c))
        self._init_canvas_state(canvas)

    def _init_canvas_state(self, canvas) -> None:
        """Give a NEWLY created view the toolbar state everything else already
        has. Without this a camera added later keeps the class defaults —
        Follow on, so it auto-frames the silhouette while the toggle says off."""
        if getattr(self, "btn_follow", None) is not None:
            canvas.set_follow(self.btn_follow.isChecked())
        if getattr(self, "btn_pan", None) is not None:
            canvas.set_pan_mode(self.btn_pan.isChecked())
        if getattr(self, "marker_spin", None) is not None:
            canvas.set_marker_size(self.marker_spin.value())
        canvas.set_region_shape(self._region_shape)
        canvas.set_trails(self._trail_len, self._trail_future)
        if getattr(self, "act_onion", None) is not None:
            canvas.set_onion(self.act_onion.isChecked())
            canvas.set_loupe(self.act_loupe.isChecked())
            canvas.set_display_filter(self._display_filter_key())

    def _on_canvas_clicked(self, canvas) -> None:
        """Clicking a companion view switches to that camera (it is view-only
        until then, so nothing is lost)."""
        for i, cv in enumerate(self.grid.canvases):
            if cv is canvas and self.project is not None and i != self.project.active:
                self._set_active_view(i)
                return

    def _build_ui_rest(self):
        # (no shortcut strip: the full reference lives in Help → Keyboard &
        # Mouse Reference; chrome stays out of the content's way)

        # transport bar — text glyphs, not QStyle bitmap icons: those ignore
        # the dark palette and vanish black-on-black
        self.btn_prev = QToolButton()
        self.btn_prev.setIcon(icons.prev())
        self.btn_next = QToolButton()
        self.btn_next.setIcon(icons.next_())
        self.btn_play = QToolButton()
        self.btn_play.setIcon(icons.play())
        self.btn_play.setCheckable(True)
        self.btn_prev.clicked.connect(lambda: self._goto(self.current - 1))
        self.btn_next.clicked.connect(lambda: self._goto(self.current + 1))
        self.btn_play.toggled.connect(self._toggle_play)
        self.btn_prev.setToolTip("Previous frame (Left)")
        self.btn_next.setToolTip("Next frame (Right)")
        self.btn_play.setToolTip("Play / pause preview playback (Space) — display only, "
                                 "no tracking")

        self.spin = QSpinBox()
        self.spin.setToolTip("Jump to a frame number (type and press Enter)")
        self.spin.setFixedWidth(118)
        # time-axis zoom: precise frame navigation on long videos
        self.btn_tz_in = QToolButton()
        self.btn_tz_in.setIcon(icons.zoom_in())
        self.btn_tz_out = QToolButton()
        self.btn_tz_out.setIcon(icons.zoom_out())
        self.btn_tz_fit = QToolButton()
        self.btn_tz_fit.setIcon(icons.zoom_fit())
        self.btn_tz_in.setToolTip("Zoom the timeline in around the playhead (Shift++ or Ctrl+wheel over the timeline)")
        self.btn_tz_out.setToolTip("Zoom the timeline out (Shift+−)")
        self.btn_tz_fit.setToolTip("Show the whole video on the timeline")
        self.btn_tz_in.clicked.connect(lambda: self.timeline.zoom_time(TIME_ZOOM_STEP))
        self.btn_tz_out.clicked.connect(lambda: self.timeline.zoom_time(1 / TIME_ZOOM_STEP))
        self.btn_tz_fit.clicked.connect(lambda: self.timeline.zoom_fit())
        self.spin.editingFinished.connect(self._on_spin_seek)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 100_000)
        self.step_spin.setValue(10)
        self.step_spin.setPrefix("±")
        self.step_spin.setFixedWidth(64)
        self.step_spin.setToolTip("Jump size (frames) for Shift+F / Shift+B and Shift+arrows")

        self.marker_spin = QSpinBox()
        self.marker_spin.setRange(2, 24)
        self.marker_spin.setValue(3)           # 3 px by default (was 7)
        self.marker_spin.setPrefix("● ")
        self.marker_spin.setSuffix("px")
        self.marker_spin.setFixedWidth(70)
        self.marker_spin.setToolTip(
            "Marker size on screen. Make it small to see the exact pixel under a "
            "target whose apparent size changes as it nears or leaves the camera.")
        # every camera view, not just the active one (the property would bind
        # the bound method of whichever canvas happened to be active here)
        self.marker_spin.valueChanged.connect(
            lambda px: [cv.set_marker_size(px) for cv in self.grid.canvases])

        self.btn_follow = QToolButton()
        self.btn_follow.setText("Follow")
        self.btn_follow.setIcon(icons.follow())
        self.btn_follow.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_follow.setCheckable(True)
        self.btn_follow.setChecked(False)      # OFF by default: the view moves only when asked
        self.btn_follow.setToolTip(
            "Follow (off by default). When on, keeps tracked points in view while zoomed in: pans to\n"
            "follow the selected point (holds still where its track has a gap); with nothing\n"
            "selected, re-frames ALL points automatically. R always fits the whole frame.")
        self.btn_follow.toggled.connect(
            lambda on: [cv.set_follow(on) for cv in self.grid.canvases])
        self.canvas.set_follow(self.btn_follow.isChecked())

        self.btn_autopause = QToolButton()
        self.btn_autopause.setText("Auto-pause")
        self.btn_autopause.setIcon(icons.autopause())
        self.btn_autopause.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_autopause.setCheckable(True)
        self.btn_autopause.setChecked(True)
        self.btn_autopause.setToolTip(
            "Auto-pause: stop the run when the model has lost a point: its confidence stays low\n"
            "for 16 frames in a row (a point hidden for a moment keeps its confidence, so ordinary\n"
            "occlusion does not trigger this). Its track is cut back to the first unreliable frame,\n"
            "the playhead jumps there and the point is selected: click where it really is, then\n"
            "Track. Also stops when the segment is lost for 16 frames, or a ball marker is lost\n"
            "inside the picture. Uncheck to always run to the end. (With Body on, a landmark that\n"
            "leaves the segment stops the run either way.)")

        self.btn_roi = QToolButton()
        self.btn_roi.setText("ROI")
        self.btn_roi.setIcon(icons.roi())
        self.btn_roi.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_roi.setCheckable(True)
        self.btn_roi.setChecked(True)
        self.btn_roi.setToolTip(
            "ROI zoom: when the tracked points sit in a small part of a high-res frame,\n"
            "track inside a crop around them so small objects keep real detail at the\n"
            "model's internal resolution. Engages only when it clearly helps (≥2× zoom).")

        self.btn_add = QToolButton()
        self.btn_add.setText("Add")
        self.btn_add.setIcon(icons.add())
        self.btn_add.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_add.setCheckable(True)
        self.btn_add.setToolTip(
            "Add a point (N): arms the crosshair — the next click places a point\n"
            "(drag instead to outline a region: the ▾ picks circle / rectangle / polygon).\n"
            "Stray clicks never edit anything. One placement per press; Esc cancels.")
        self.btn_add.toggled.connect(self._on_add_mode)
        self.btn_add.setPopupMode(QToolButton.MenuButtonPopup)
        menu_add = QMenu(self.btn_add)
        self._shape_group = QActionGroup(self)
        self._shape_acts: dict[str, QAction] = {}
        for key, label, tip in (
                ("circle", "Region shape: circle (drag)",
                 "Drag from the centre outward: the circle is tracked as ONE point (its fitted centre)"),
                ("rect", "Region shape: rectangle (drag)",
                 "Drag a box around the object"),
                ("polygon", "Region shape: polygon (click corners, Enter closes)",
                 "Click each corner in turn; Enter or a double-click closes it, Esc cancels")):
            act = QAction(label, self, checkable=True)
            act.setToolTip(tip)
            act.triggered.connect(lambda _=False, k=key: self._set_region_shape(k))
            self._shape_group.addAction(act)
            menu_add.addAction(act)
            self._shape_acts[key] = act
        self._shape_acts["circle"].setChecked(True)
        menu_add.addSeparator()
        self.act_add_ball = QAction("Ball marker (SAM circle): click a ball", self)
        self.act_add_ball.setToolTip(
            "For wand balls, reflective markers, a dropped ball: SAM segments the ball you click,\n"
            "a circle is fitted to it on every frame and the circle's centre is the tracked point.\n"
            "Add one per ball; they are tracked together in one 960-pixel window, so balls more than\n"
            "about 740 px apart cannot share a run: select one ball in the POINTS list and Track it\n"
            "alone. Choosing this arms the crosshair like N.")
        self.act_add_ball.triggered.connect(self._arm_ball)
        menu_add.addAction(self.act_add_ball)
        self._place_kind = "point"          # what the next armed click creates: "point" | "ball"
        self._retrack = None                # an automatic epipolar re-track in progress (see _retrack_start)
        self._retrack_last = None
        self.btn_add.setMenu(menu_add)

        self.btn_animal = QToolButton()
        self.btn_animal.setText("Segment")
        self.btn_animal.setIcon(icons.segment())
        self.btn_animal.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_animal.setCheckable(True)
        self.btn_animal.setToolTip(
            "Segment tool (S): click the segment and its silhouette appears. Shift+click = "
            "\"not the segment\", drag = box around it. Tracking then follows the silhouette "
            "and derives tail tip / midline / feet from it. S or Esc when done.\n"
            "Optional: points placed with N track without it (a target a few pixels across has no\n"
            "outline worth drawing). The ▾ arrow picks the segmentation model.")
        self.btn_animal.toggled.connect(self._on_animal_mode)
        # segmentation model lives on the button itself, like the Track ▾ point
        # model — Settings keeps the same choice plus the token / opacity
        self.btn_animal.setPopupMode(QToolButton.MenuButtonPopup)
        self._seg_menu = QMenu(self.btn_animal)
        self._seg_group = QActionGroup(self)
        self._seg_acts: dict[str, QAction] = {}
        for key in BACKENDS:
            act = QAction("", self, checkable=True)
            act.triggered.connect(lambda _=False, k=key: self._set_seg_backend(k))
            self._seg_group.addAction(act)
            self._seg_menu.addAction(act)
            self._seg_acts[key] = act
        self._seg_menu.addSeparator()
        self._seg_menu.aboutToShow.connect(self._refresh_seg_menu)
        self.btn_animal.setMenu(self._seg_menu)

        self.btn_mask = QToolButton()
        self.btn_mask.setText("Mask")
        self.btn_mask.setIcon(icons.mask())
        self.btn_mask.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_mask.setCheckable(True)
        self.btn_mask.setChecked(True)
        self.btn_mask.setToolTip("Mask: show / hide the segment's silhouette overlay")
        self.btn_mask.toggled.connect(lambda _on: (self._refresh_overlay(),
                                                   self._refresh_animal_panel()))

        self.btn_onbody = QToolButton()
        self.btn_onbody.setText("Body")
        self.btn_onbody.setIcon(icons.body())
        self.btn_onbody.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_onbody.setCheckable(True)
        self.btn_onbody.setChecked(True)
        self.btn_onbody.setToolTip(
            "Body: keep tracked points ON the segment (only when there is a segment, S). A point\n"
            "that strays a few pixels past the silhouette's edge is nudged back onto it.\n"
            "A point that clearly LEAVES the silhouette stops the run at that frame and its\n"
            "track ends there: it is never pulled onto some other spot of the animal.\n"
            "Right-click a point → \"May leave the segment\" to exempt it (markers on\n"
            "the ground, reference objects).")
        self.btn_onbody.toggled.connect(self._on_onbody_toggled)

        self.btn_pan = QToolButton()
        self.btn_pan.setText("Pan")
        self.btn_pan.setIcon(icons.pan())
        self.btn_pan.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_pan.setCheckable(True)
        self.btn_pan.setToolTip(
            "Pan tool (H): left-drag moves the view instead of editing points.\n"
            "Middle-drag always pans, in any mode — even while tracking runs.")
        self.btn_pan.toggled.connect(
            lambda on: [cv.set_pan_mode(on) for cv in self.grid.canvases])

        # Track button with a mode dropdown: automatic (run to end) or
        # semi-automatic (each F tracks exactly one frame, then pauses)
        self.btn_track = QToolButton()
        self.btn_track.setObjectName("primary")  # the one filled accent button
        self.btn_track.setMinimumWidth(150)
        self.btn_track.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_track.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_track.clicked.connect(self._toggle_tracking)
        menu_track = QMenu(self.btn_track)
        self._mode_group = QActionGroup(self)
        self.act_mode_auto = QAction("Automatic — track to the end of the video", self,
                                     checkable=True, checked=True)
        self.act_mode_semi = QAction("Semi-automatic — each F tracks ONE frame, then pauses",
                                     self, checkable=True)
        for act, mode in ((self.act_mode_auto, "auto"), (self.act_mode_semi, "semi")):
            self._mode_group.addAction(act)
            menu_track.addAction(act)
            act.triggered.connect(lambda _=False, m=mode: self._set_track_mode(m))
        # point-tracking model: AllTracker (robust on animals) or CoTracker3 (fast)
        menu_track.addSeparator()
        self._point_group = QActionGroup(self)
        self.act_pm_alltracker = QAction(
            "Point model: AllTracker — holds points on animals (default)", self, checkable=True)
        self.act_pm_cotracker = QAction(
            "Point model: CoTracker3 — faster; sub-pixel on high-contrast markers", self,
            checkable=True)
        self.act_pm_alltracker.setToolTip(
            "Dense high-resolution tracker (ICCV 2025, MIT). Measured on a real lizard clip: keeps head "
            "and hip on the body through 270 frames where CoTracker3 drifts off; halves the error on "
            "a small textured feature. About 2x slower.")
        self.act_pm_cotracker.setToolTip(
            "Meta's online point tracker with the app's LK sub-pixel refinement: 0.74 px at 4K on "
            "high-contrast dots; drifts along textureless bodies.")
        for act, key in ((self.act_pm_alltracker, "alltracker"), (self.act_pm_cotracker, "cotracker3")):
            self._point_group.addAction(act)
            menu_track.addAction(act)
            act.triggered.connect(lambda _=False, k=key: self._set_point_backend(k))
        self.act_pm_alltracker.setEnabled(alltracker_backend.available())
        (self.act_pm_alltracker if self._point_backend == "alltracker"
         else self.act_pm_cotracker).setChecked(True)
        self.btn_track.setMenu(menu_track)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(220)
        self.progress.setTextVisible(False)  # thin quiet bar; the timeline
        # playhead, frame box and status fps/ETA already carry the numbers

        # The timeline panel IS the scrubber (click/drag it to seek): a separate
        # QSlider can never align with the lanes — its handle center is inset
        # from the groove edge by half the handle width, so frame 0 lands at
        # different x positions. One seek surface, no alignment problem.
        # control panel: frame box + transport | tools | toggles ... progress + Track
        # grouped by hairlines, spaced on the 4/8 grid — no frames, no bevels
        def hairline():
            s = QWidget()
            s.setFixedSize(1, 18)
            s.setStyleSheet(f"background: {theme.HAIRLINE};")
            return s

        controls = QWidget()
        tl = QHBoxLayout(controls)
        tl.setContentsMargins(6, 6, 6, 6)
        tl.setSpacing(3)
        tl.addWidget(self.spin)
        tl.addSpacing(8)
        for w in (self.btn_prev, self.btn_play, self.btn_next):
            tl.addWidget(w)
        tl.addSpacing(4)
        for w in (self.btn_tz_out, self.btn_tz_in, self.btn_tz_fit):
            tl.addWidget(w)
        tl.addSpacing(8)
        tl.addWidget(hairline())
        tl.addSpacing(8)
        tl.addWidget(self.step_spin)
        tl.addWidget(self.marker_spin)
        tl.addSpacing(8)
        tl.addWidget(hairline())
        tl.addSpacing(8)
        tl.addWidget(self.btn_add)
        tl.addWidget(self.btn_animal)
        tl.addWidget(self.btn_pan)
        tl.addSpacing(8)
        tl.addWidget(hairline())
        tl.addSpacing(8)
        tl.addWidget(self.btn_follow)
        tl.addWidget(self.btn_mask)
        tl.addWidget(self.btn_onbody)
        tl.addWidget(self.btn_autopause)
        tl.addWidget(self.btn_roi)
        tl.addStretch(1)
        tl.addWidget(self.progress)
        tl.addSpacing(8)
        tl.addWidget(self.btn_track)
        # With every tool button showing its label the bar needed ~1490 px, so
        # the window could not be narrower than ~1720 px and ran off a 1366 /
        # 1440 px screen or a 1080p laptop at 125 % scaling, Track button and
        # all (release sweep G2). When the bar is narrower than its full width
        # the labelled tool buttons show their icon only (each keeps its
        # tooltip); the Track button always keeps its text.
        # The first version folded ALL eight labels as soon as the bar was a few pixels
        # short (1487 px wanted, 1322 px on a 1600 px window). Now the
        # icon-only transport / zoom buttons are always tight, labelled buttons
        # use a slightly tighter padding, and labels fold ONE BUTTON AT A TIME,
        # least-needed first, only as far as the width requires.
        self._controls = controls
        self._compact_btns = [self.btn_add, self.btn_animal, self.btn_pan, self.btn_follow,
                              self.btn_mask, self.btn_onbody, self.btn_autopause, self.btn_roi]
        self._compact_order = [self.btn_pan, self.btn_roi, self.btn_mask, self.btn_follow,
                               self.btn_onbody, self.btn_autopause, self.btn_animal, self.btn_add]
        self._compact_icons = [self.btn_prev, self.btn_play, self.btn_next,
                               self.btn_tz_out, self.btn_tz_in, self.btn_tz_fit]
        for b in self._compact_icons:
            b.setProperty("compact", True)
        for b in self._compact_btns:
            b.setProperty("bar", True)
        for b in self._compact_icons + self._compact_btns:
            b.style().unpolish(b)
            b.style().polish(b)
        self._controls_gaps = []            # the fixed 8 px gaps between groups: 4 px when compact
        for i in range(tl.count()):
            sp = tl.itemAt(i).spacerItem()
            if sp is not None and sp.sizeHint().width() == 8:
                self._controls_gaps.append(sp)
        self._label_saving = {}             # px each button gives back when it folds to its icon
        for b in self._compact_btns:
            w_lab = b.sizeHint().width()
            b.setToolButtonStyle(Qt.ToolButtonIconOnly)
            b.setProperty("compact", True)
            b.style().unpolish(b)
            b.style().polish(b)
            self._label_saving[b] = max(0, w_lab - b.sizeHint().width())
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.setProperty("compact", False)
            b.style().unpolish(b)
            b.style().polish(b)
        tl.invalidate()
        self._controls_full_w = tl.minimumSize().width()
        self._set_compact_controls(len(self._compact_order))
        controls.setMinimumWidth(tl.minimumSize().width())
        self._set_compact_controls(0)
        self._controls_level = 0            # how many labels are folded (0 = all shown)
        self._controls_compact = False
        self._controls_hook = _ResizeHook(self._fit_controls)
        controls.installEventFilter(self._controls_hook)

        self.timeline = TimelinePanel()
        self.timeline.seek_requested.connect(self._goto)
        self.timeline.point_selected.connect(self._on_select)
        self.timeline.events_changed.connect(self._on_events_changed)
        self.timeline.clear_requested.connect(self._clear_tracked_window)
        self.timeline.clear_masks_requested.connect(self._clear_masks_window)
        self.timeline.clear_both_requested.connect(self._clear_window_both)
        self.timeline.occlude_requested.connect(self._occlude_window)
        self.timeline.note_requested.connect(self._edit_note)

        # vertical splitter: drag the boundary above the timeline to give it
        # more lanes (the visible-lane count follows the panel height)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.addWidget(self.timeline, 1)
        bl.addWidget(controls)
        split = QSplitter(Qt.Vertical)
        self.onboarding = OnboardingStrip(self._onboarding_step)
        top = QWidget()
        top_l = QVBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)
        top_l.setSpacing(0)
        top_l.addWidget(self.onboarding)
        top_l.addWidget(self.grid, 1)
        split.addWidget(top)
        split.addWidget(bottom)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setCollapsible(0, False)
        split.setCollapsible(1, False)
        self.setCentralWidget(split)
        self._split = split
        self._bottom = bottom

        # point list dock
        self.point_list = QListWidget()
        self.point_list.setToolTip(
            "Tracked points — double-click to rename, checkbox toggles display.\n"
            "Select one, then a plain click on the video places it on this frame.\n"
            "Right-click: rename, data source, hidden here, jump to its first / last\n"
            "frame, clear a stretch, fill gaps between hand placements, delete.\n"
            "Track follows the selection: with some points selected it tracks only\n"
            "those (select all, or none, to track everything).\n"
            "Ctrl/Shift+click selects several: Delete removes them all; with a\n"
            "frame window selected on the timeline, Delete clears just that window\n"
            "(Shift+drag the timeline over the lanes you want cleared).")
        self.point_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.point_list.itemChanged.connect(self._on_item_changed)
        self.point_list.currentRowChanged.connect(self._on_list_select)
        # the run scope follows the selection: keep the Track button's wording current
        self.point_list.itemSelectionChanged.connect(self._update_track_button)
        self.point_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.point_list.customContextMenuRequested.connect(self._point_list_menu)

        # right panel: cameras, then the segment (silhouette), then the points
        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(8, 6, 8, 6)
        pl.setSpacing(6)
        self.cameras = CameraPanel()
        self.cameras.add_requested.connect(self._add_video_dialog)
        self.cameras.activate_requested.connect(self._set_active_view)
        self.cameras.offset_changed.connect(self._on_view_offset)
        self.cameras.align_requested.connect(self._align_view_here)
        self.cameras.remove_requested.connect(self._remove_view)
        pl.addWidget(self.cameras)
        sep0 = QWidget()
        sep0.setFixedHeight(1)
        sep0.setStyleSheet(f"background: {theme.HAIRLINE};")
        pl.addWidget(sep0)
        head = QLabel("SEGMENT")
        head.setStyleSheet(f"color: {theme.TEXT_DIM}; font-weight: 600; letter-spacing: 1px;")
        pl.addWidget(head)
        # the segment gets a ROW of its own, like a point: the user sees it the
        # moment it exists and can rename / hide / clear / delete it from here
        # instead of hunting through menus
        self.animal_list = QListWidget()
        self.animal_list.setFixedHeight(26)   # one row; re-measured once it has an item
        self.animal_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.animal_list.setToolTip(
            "The tracked segment. The checkbox shows / hides its silhouette; double-click to "
            "rename; right-click to clear a stretch of silhouettes or remove it entirely.")
        self.animal_list.itemChanged.connect(self._on_animal_item_changed)
        self.animal_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.animal_list.customContextMenuRequested.connect(self._animal_menu)
        self.animal_list.setVisible(False)
        pl.addWidget(self.animal_list)
        self.animal_label = QLabel("")
        self.animal_label.setWordWrap(True)
        self.animal_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.animal_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        pl.addWidget(self.animal_label)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.btn_skeleton = QToolButton()
        self.btn_skeleton.setText("Skeleton ▾")
        self.btn_skeleton.setPopupMode(QToolButton.InstantPopup)
        self.btn_skeleton.setToolTip("Apply a named landmark set for your study segment")
        self.m_skeleton_btn = QMenu(self.btn_skeleton)
        self.btn_skeleton.setMenu(self.m_skeleton_btn)
        self.btn_clear_animal = QToolButton()
        self.btn_clear_animal.setText("Clear segment")
        self.btn_clear_animal.setToolTip("Remove the segment's clicks and silhouettes (points stay)")
        self.btn_clear_animal.clicked.connect(self._clear_animal)
        row.addWidget(self.btn_skeleton)
        row.addWidget(self.btn_clear_animal)
        row.addStretch(1)
        pl.addLayout(row)
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {theme.HAIRLINE};")
        pl.addWidget(sep)
        head2 = QLabel("POINTS")
        head2.setStyleSheet(f"color: {theme.TEXT_DIM}; font-weight: 600; letter-spacing: 1px;")
        pl.addWidget(head2)
        pl.addWidget(self.point_list, 1)
        dock = QDockWidget("Segment && Points", self)   # && = a literal ampersand (not a mnemonic)
        dock.setWidget(panel)
        dock.setMinimumWidth(230)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.dock = dock

        # menu / toolbar actions
        m_file = self.menuBar().addMenu("&File")
        self.act_open = QAction("Open &Video…", self, shortcut=QKeySequence("Ctrl+O"),
                                triggered=self._open_video_dialog)
        self.act_open_proj = QAction("Open &Project…", self, shortcut=QKeySequence("Ctrl+Shift+O"),
                                     triggered=self._open_project_dialog)
        self.act_save = QAction("&Save Project", self, shortcut=QKeySequence("Ctrl+S"),
                                triggered=self._save_project)
        self.act_save_as = QAction("Save Project &As…", self, shortcut=QKeySequence("Ctrl+Shift+S"),
                                   triggered=self._save_project_as)
        self.act_export = QAction("&Export Tracks…", self, shortcut=QKeySequence("Ctrl+E"),
                                  triggered=self._export_dialog)
        self.act_overlay = QAction("Export Overlay &Video…", self, triggered=self._export_overlay)
        self.act_overlay.setToolTip("An MP4 with markers, names, skeleton, silhouette, trails, frame "
                                    "counter, events and notes drawn on it — for talks and for checking "
                                    "a result without the app")
        for a in (self.act_open, self.act_open_proj, None, self.act_save, self.act_save_as,
                  None, self.act_export, self.act_overlay):
            m_file.addSeparator() if a is None else m_file.addAction(a)

        m_edit = self.menuBar().addMenu("&Edit")
        self.act_undo = QAction("&Undo Last Run / Edit", self,
                                shortcut=QKeySequence("Ctrl+Z"),
                                triggered=self._undo_run, enabled=False)
        m_edit.addAction(self.act_undo)
        m_edit.addSeparator()
        self.act_note = QAction("&Note at This Frame…   (Shift+N)", self, triggered=lambda: self._edit_note())
        self.act_note.setToolTip("Attach a free-text note to the current frame (shown on the timeline, "
                                 "exported with the events)")
        self.act_hidden = QAction("Mark Selected Point &Hidden Here   (Shift+X)", self,
                                  triggered=self._toggle_hidden_here)
        self.act_hidden.setToolTip("The selected point is not really visible on this frame: keep its "
                                   "position but leave it blank in exports and out of 3D. Press again "
                                   "to unmark.")
        self.act_annotator = QAction("&Annotator Name…", self, triggered=self._set_annotator)
        self.act_annotator.setToolTip("Recorded on the events and notes you add")
        for a in (self.act_note, self.act_hidden, self.act_annotator):
            m_edit.addAction(a)

        m_view = self.menuBar().addMenu("&View")
        act_panel = self.dock.toggleViewAction()
        act_panel.setText("Segment && Points panel")
        act_panel.setShortcut(QKeySequence("Ctrl+1"))
        act_panel.setIcon(icons.panel())
        m_view.addAction(act_panel)
        self.act_onboarding = QAction("Getting started strip", self, checkable=True, checked=True)
        self.act_onboarding.toggled.connect(lambda on: self.onboarding.setVisible(on))
        m_view.addAction(self.act_onboarding)
        self.act_solo = QAction("Only the &working camera", self, checkable=True, checked=False,
                                shortcut=QKeySequence("Ctrl+2"))
        self.act_solo.setToolTip("Hide the other cameras — they also stop decoding, "
                                 "which gives scrubbing the whole disk back")
        self.act_solo.toggled.connect(self._on_solo_toggled)
        self.act_solo.setEnabled(False)
        m_view.addAction(self.act_solo)
        m_view.addSeparator()
        self.act_show_midline = QAction("Show segment &midline", self, checkable=True, checked=True,
                                        triggered=lambda _=False: self._refresh_overlay())
        self.act_show_midline.setToolTip("Also available by right-clicking the segment in the panel")
        self.act_show_bones = QAction("Show skeleton &bones", self, checkable=True, checked=True,
                                      triggered=lambda _=False: self._refresh_overlay())
        self.act_settings = QAction("&Settings… (segmentation model, token, opacity)", self,
                                    shortcut=QKeySequence("Ctrl+,"), triggered=self._show_settings)
        # the Segment ▾ menu ends with the full Settings dialog (token, opacity)
        self._seg_menu.addAction(self.act_settings)
        self._refresh_seg_menu()
        m_view.addAction(self.act_show_midline)
        m_view.addAction(self.act_show_bones)
        m_view.addSeparator()
        # trajectory trails: length + upcoming path
        m_trails = m_view.addMenu("&Trails")
        self._trail_group = QActionGroup(self)
        self._trail_acts: dict[int, QAction] = {}
        for n in (0, 15, 30, 60, 120, 300):
            act = QAction("Off" if n == 0 else f"Last {n} frames", self, checkable=True)
            act.triggered.connect(lambda _=False, k=n: self._set_trail_len(k))
            self._trail_group.addAction(act)
            m_trails.addAction(act)
            self._trail_acts[n] = act
        self._trail_acts[TRAIL_FRAMES].setChecked(True)
        m_trails.addSeparator()
        self.act_trail_future = QAction("Also show the &upcoming path (dashed)", self, checkable=True)
        self.act_trail_future.setToolTip("Draws where each point goes AFTER this frame, so a wrong "
                                         "step ahead is visible before you get there")
        self.act_trail_future.toggled.connect(self._on_trail_future)
        m_trails.addAction(self.act_trail_future)
        self.act_onion = QAction("&Onion skin — ghosts of the previous / next frame  (O)", self,
                                 checkable=True)
        self.act_onion.setToolTip("Hollow ghost markers where every point was one frame ago (solid) "
                                  "and will be one frame on (dashed): a wrong step shows at a glance")
        self.act_onion.toggled.connect(lambda on: [cv.set_onion(on) for cv in self.grid.canvases]
                                       + [self._refresh_overlay()])
        m_view.addAction(self.act_onion)
        self.act_loupe = QAction("&Loupe — magnifier under the cursor  (L)", self, checkable=True)
        self.act_loupe.setToolTip("A magnified inset follows the cursor over the video, with a crosshair "
                                  "on the exact pixel: for sub-pixel placement at 4K without zooming in")
        self.act_loupe.toggled.connect(lambda on: [cv.set_loupe(on) for cv in self.grid.canvases])
        m_view.addAction(self.act_loupe)
        self.act_epipolar = QAction("&Epipolar guides from the other cameras", self, checkable=True)
        self.act_epipolar.setChecked(True)
        self.act_epipolar.setToolTip("With a calibration: dashed lines showing where the SELECTED landmark, as "
                                     "the other cameras see it at this instant, can lie in this picture. "
                                     "Place it on the line (or right-click it → Snap to the other cameras' rays).")
        self.act_epipolar.toggled.connect(lambda _on: self._refresh_overlay())
        m_view.addAction(self.act_epipolar)
        # display-only filters
        m_filter = m_view.addMenu("&Display filter (display only)")
        self._filter_group = QActionGroup(self)
        self._filter_acts: dict[str, QAction] = {}
        for key, label, tip in (
                ("none", "None", "The video as filmed"),
                ("contrast", "Enhance contrast (CLAHE)",
                 "Local contrast equalisation: brings out a dim animal against water or foliage"),
                ("bright", "Brighten dark footage (gamma)", "Lifts the shadows without clipping highlights"),
                ("diff", "Frame difference (what moved since the previous frame)",
                 "Shows only what changed since the previous frame you looked at: a small moving "
                 "animal stands out from still background. Step with F/B for a true difference.")):
            act = QAction(label, self, checkable=True)
            act.setToolTip(tip)
            act.triggered.connect(lambda _=False, k=key: self._set_display_filter(k))
            self._filter_group.addAction(act)
            m_filter.addAction(act)
            self._filter_acts[key] = act
        self._filter_acts["none"].setChecked(True)
        m_filter.menuAction().setToolTip("These change only what you SEE. The tracker always works on the "
                            "original pixels, and nothing is written to the video.")
        # Settings is NOT in this menu: it is the last entry of the Segment ▾
        # dropdown, next to the model choice it configures. The window owns the
        # action so Ctrl+, still works from anywhere.
        self.addAction(self.act_settings)

        # skeleton templates (also under the panel's Skeleton ▾ button)
        self.m_skeleton = self.menuBar().addMenu("&Skeleton")
        self._refresh_skeleton_menu()

        # E is handled in keyPressEvent (a menu shortcut would steal the letter
        # from rename editors); the menu wording documents it
        self.m_events = self.menuBar().addMenu("E&vents")
        self.act_mark_event = QAction("Mark Event Start / End  (E)", self,
                                      triggered=self._mark_event)
        self._refresh_events_ui()

        # 3D: calibration -> sub-frame sync -> triangulation -> volume hull
        m_3d = self.menuBar().addMenu("&3D")
        self.act_lens = QAction("Calibrate a &Lens (checkerboard)…", self, triggered=self._lens_wizard)
        self.act_lens.setToolTip("Measure how a lens bends the picture from a video of a printed checkerboard. "
                                 "Needed for wide-angle / action cameras before a wand calibration; explains "
                                 "when you need it and when you do not.")
        self.act_wand = QAction("Calibrate Cameras with a &Wand…", self, triggered=self._wand_wizard)
        self.act_wand.setToolTip("Work out where the cameras are from a wand of known length waved in "
                                 "front of them — no MATLAB, no easyWand. Explains every step.")
        self.act_calib = QAction("Import &Calibration…", self, triggered=self._import_calibration)
        self.act_calib.setToolTip("A Kinetrace calibration (.kcal.json), DLT coefficients from DLTdv / "
                                  "easyWand / Argus (dltCoefs.csv), an easyWandData.mat, a DLTdv8 project, "
                                  "or OpenCV-style camera matrices (K + R/t, JSON or text)")
        self.act_export_cal = QAction("&Export Calibration…", self, triggered=self._export_calibration)
        self.act_export_cal.setToolTip("Save this project's calibration for other projects (.kcal.json) and "
                                       "for DLTdv (dltCoefs.csv), with the report")
        self.act_sync = QAction("S&ync Cameras (Sound / Motion)…", self, triggered=self._sync_dialog)
        self.act_sync.setToolTip("Whole-frame offsets for cameras that were started by hand: lines up the\n"
                                 "cameras' sound tracks (claps, voices; with a noise filter) or how much the\n"
                                 "picture changes in every camera. Needs no tracking, no calibration.")
        self.act_offsets3d = QAction("Estimate &Sub-frame Offsets…", self,
                                     triggered=self._estimate_offsets_dialog)
        self.act_offsets3d.setToolTip("Refine every camera's offset to a fraction of a frame from the\n"
                                      "tracks themselves (the triangulation residual is minimised).\n"
                                      "Needs a calibration and the same named landmarks tracked in two\n"
                                      "or more cameras; says whether the result can be trusted.")
        self.act_recon = QAction("&Reconstruct 3D Landmarks", self, shortcut=QKeySequence("Ctrl+3"),
                                 triggered=self._reconstruct_3d)
        self.act_recon.setToolTip("Turn the landmarks tracked in two or more cameras (matched by name) into\n"
                                  "3D positions with the calibration, and say how well the cameras agree\n"
                                  "(GOOD / USABLE / NOT TRUSTWORTHY). Needs every camera calibrated.")
        self.act_retrack = QAction("Re-track &Disagreeing Stretches…", self, triggered=self._retrack_dialog)
        self.act_retrack.setToolTip("Where one camera's landmark disagrees with the others (the magenta band on\n"
                                    "the timeline), put it back on the other cameras' rays at the start of the\n"
                                    "stretch and re-track it through; then reconstruct again and report before / after")
        self.act_hull = QAction("Carve &Volume at This Frame", self, shortcut=QKeySequence("Ctrl+4"),
                                triggered=self._carve_hull_here)
        self.act_hull.setToolTip("Visual hull: the volume every camera's silhouette agrees on, at this frame.\n"
                                 "Needs a silhouette (S) in at least three cameras here and the 3D landmarks\n"
                                 "(Reconstruct, Ctrl+3) to know where in space to look.")
        self.act_view3d = QAction("Show 3D &View", self, checkable=True, shortcut=QKeySequence("Ctrl+5"),
                                  triggered=self._toggle_view3d)
        self.act_view3d.setToolTip("The reconstructed landmarks, the cameras and the volume in 3D, orbitable.\n"
                                   "Needs a calibration first (Calibrate Cameras with a Wand, or Import\n"
                                   "Calibration), then Reconstruct 3D Landmarks (Ctrl+3).")
        self.act_export_mesh = QAction("Export &Mesh of This Frame…", self, triggered=self._export_mesh)
        self.act_export_mesh.setToolTip("Writes the volume carved at this frame (Carve Volume, Ctrl+4) as an OBJ / PLY\n"
                                        "mesh. Carve a volume at this frame first.")
        for a in (self.act_sync, None, self.act_lens, self.act_wand, self.act_calib, self.act_export_cal,
                  self.act_offsets3d, None, self.act_recon, self.act_retrack, self.act_hull, None, self.act_view3d,
                  self.act_export_mesh):
            m_3d.addSeparator() if a is None else m_3d.addAction(a)

        # Body: human joints and joint angles from the footage itself
        m_body = self.menuBar().addMenu("&Body")
        self.act_body_run = QAction("Find People && Measure &Joints…", self,
                                    triggered=self._body_run)
        self.act_body_run.setToolTip(
            "Run a human pose model over a range of frames. SAM 3D Body gives real 3D "
            "joints and joint angles from a single camera; the 2D model gives joints in "
            "the picture. Both then give angles, plots and exports.")
        self.act_body_view = QAction("&Side-by-side View", self, checkable=True,
                                     shortcut=QKeySequence("Ctrl+6"),
                                     triggered=self._toggle_body_view)
        self.act_body_view.setToolTip("The footage with the skeleton on it, the pose on its "
                                      "own, and the joint angles against time")
        self.act_body_video = QAction("Export Side-by-side &Video…", self,
                                      triggered=self._export_body_video)
        self.act_body_joints = QAction("Export Joint &Positions…", self,
                                       triggered=self._export_body_joints)
        self.act_body_joints.setToolTip("One row per frame and person: every joint in pixels, "
                                        "and in metres when the model gives 3D")
        self.act_body_angles = QAction("Export Joint &Angles…", self,
                                       triggered=self._export_body_angles)
        self.act_body_angles.setToolTip("Joint angles in degrees with their rate of change, "
                                        "and a report saying what 0 means for each one")
        self.act_body_clear = QAction("&Remove Body Pose", self, triggered=self._clear_body)
        for a in (self.act_body_run, None, self.act_body_view, self.act_body_video, None,
                  self.act_body_joints, self.act_body_angles, None, self.act_body_clear):
            m_body.addSeparator() if a is None else m_body.addAction(a)

        m_help = self.menuBar().addMenu("&Help")
        self.act_manual = QAction("&User Manual…", self, shortcut=QKeySequence("F1"),
                                  triggered=self._show_manual)
        self.act_manual.setToolTip("The full manual, written for someone new to tracking")
        m_help.addAction(self.act_manual)
        m_help.addAction(QAction("&Keyboard && Mouse Reference…", self,
                                 triggered=self._show_hotkeys))
        # QMenu hides action tooltips unless told otherwise: every explanation
        # written on a menu entry (what a wizard needs, what an export holds)
        # was invisible until this (release sweep G1, 2026-09-22)
        for m in self.findChildren(QMenu):
            m.setToolTipsVisible(True)

        # combo shortcuts only — single-letter keys (F/B/N/T/X/E/R/H/Space/
        # Delete/Esc/±) are handled in keyPressEvent so they never steal
        # keystrokes from the point-rename editor or other text fields
        QShortcut(QKeySequence("Right"), self, lambda: self._goto(self.current + 1))
        QShortcut(QKeySequence("Left"), self, lambda: self._goto(self.current - 1))
        QShortcut(QKeySequence("Shift+Right"), self, lambda: self._goto(self.current + self.step_spin.value()))
        QShortcut(QKeySequence("Shift+Left"), self, lambda: self._goto(self.current - self.step_spin.value()))
        QShortcut(QKeySequence("Home"), self, lambda: self._goto(0))
        QShortcut(QKeySequence("End"), self, lambda: self._goto(self.n_frames - 1))

        # Keyboard focus discipline. Toolbar buttons never
        # take the focus: a click is a click, not a place for Space to land.
        # Spin boxes take it only by click and give it back the moment an edit
        # is done, so the field does not stay highlighted and the hotkeys
        # (which the event filter routes anyway) feel immediate.
        for b in self.findChildren(QToolButton):
            b.setFocusPolicy(Qt.NoFocus)
        for sb in self.findChildren(QAbstractSpinBox):
            sb.setFocusPolicy(Qt.ClickFocus)
            if sb is not self.spin:           # the frame box seeks on Enter, then clears itself
                sb.editingFinished.connect(sb.clearFocus)

        # status bar
        # eliding labels: a plain QLabel's minimum width is its whole text, so a
        # longer frame / fps / ETA text widened the WINDOW (it can be as narrow
        # as a laptop screen since G2) and refitted the video (G8)
        from cotracker_app.widgets import ElidedLabel
        self._frame_label = ElidedLabel("no video", pad=16)
        self._device_label = ElidedLabel("", pad=16)
        self._track_label = ElidedLabel("", pad=16)
        for w in (self._frame_label, self._device_label, self._track_label):
            self.statusBar().addPermanentWidget(w)
            w.setStyleSheet("padding: 0 8px;")

    def _need_calibration(self, what: str) -> bool:
        """True when 3D can run; otherwise explain the missing step and return
        False (the entries stay clickable so a novice is TOLD what to do, I115)."""
        p = self.project
        if p is None or p.n_views < 2:
            QMessageBox.information(
                self, what,
                f"{what} works on two or more cameras filming the same moment.\n\n"
                "1. Add the other cameras' videos with ＋ Add video in the CAMERAS panel.\n"
                "2. Line them up in time (3D → Sync Cameras).\n"
                "3. Track the same landmarks, with the same names, in every camera.\n"
                "4. Calibrate the cameras (3D → Calibrate Cameras with a Wand), or import a "
                "calibration you already have (3D → Import Calibration).")
            return False
        if p.calibration is None:
            QMessageBox.information(
                self, what,
                f"{what} needs a camera calibration first: it tells the program where each camera "
                "stands.\n\nMake one with 3D → Calibrate Cameras with a Wand, or bring one you already "
                "have (DLTdv, easyWand, Argus or a Kinetrace .kcal.json) with 3D → Import Calibration.")
            return False
        if len(p.calibration) != p.n_views:
            QMessageBox.information(
                self, what,
                f"The calibration covers {len(p.calibration)} camera(s) but the project has {p.n_views}. "
                "Calibrate all cameras (3D → Calibrate Cameras with a Wand), import a calibration that "
                "includes every camera, or remove the extra camera with its × in the CAMERAS panel.")
            return False
        return True

    def _reference_instant(self) -> int:
        """The playhead's instant in REFERENCE frames: what the hull cache and
        the 3D rows are keyed by (I114: the working camera's own frame number
        greyed Export Mesh out right after a carve in camera 2+)."""
        p = self.project
        if p is None or p.n_views == 0:
            return int(self.current)
        return int(np.floor(p.reference_time(p.active, self.current) + 0.5))

    def _fit_to_screen(self):
        """Open wide enough for the labelled control bar when the screen has
        room, and never larger than the screen's free area: a fixed 1280 x 860
        hid the labels on a big screen and ran off a 768 px one (G2)."""
        scr = QApplication.primaryScreen()
        if scr is None or QApplication.platformName() == "offscreen":
            return                           # the suites keep their historical 1280 x 860
        av = scr.availableGeometry()
        want_w = self._controls_full_w + self.dock.minimumWidth() + 40
        w = max(self.minimumSizeHint().width(), min(want_w, av.width() - 24))
        h = max(self.minimumSizeHint().height(), min(960, av.height() - 48))
        self.resize(w, h)
        self.move(av.x() + max(0, (av.width() - w) // 2), av.y() + max(0, (av.height() - h - 32) // 2))

    def _fit_controls(self):
        """Fold as FEW tool-button labels as the control bar's width requires,
        least-needed first (G2); labels come back as soon as there is room."""
        avail = self._controls.width()
        need = self._controls_full_w
        level = 0
        while need > avail and level < len(self._compact_order):
            need -= self._label_saving.get(self._compact_order[level], 0)
            level += 1
        if level == self._controls_level:
            return
        self._set_compact_controls(level)

    def _set_compact_controls(self, level):
        """Fold the labels of the first `level` buttons of `_compact_order`
        (True = all, False = none); with every label folded the group gaps
        shrink from 8 to 4 px too."""
        n = len(self._compact_order)
        level = n if level is True else (0 if level is False else int(level))
        folded = set(self._compact_order[:level])
        for b in self._compact_btns:
            on = b in folded
            b.setToolButtonStyle(Qt.ToolButtonIconOnly if on else Qt.ToolButtonTextBesideIcon)
            b.setProperty("compact", on)             # theme: tighter padding when icon-only
            b.style().unpolish(b)
            b.style().polish(b)
        for sp in self._controls_gaps:
            sp.changeSize(4 if level >= n else 8, 0)
        self._controls_level = level
        self._controls_compact = level > 0
        self._controls.layout().invalidate()

    # ----------------------------------------------------------- state gates

    def _apply_state(self):
        has_video = self.state != IDLE
        tracking = self.state == TRACKING
        for w in (self.spin, self.btn_prev, self.btn_next, self.btn_play):
            w.setEnabled(has_video and not tracking)
        self.point_list.setEnabled(has_video and not tracking)
        self.animal_list.setEnabled(has_video and not tracking)
        # a run belongs to ONE camera: switching views or retiming mid-run would
        # pull the session out from under the worker
        self.cameras.setEnabled(has_video and not tracking)
        self._refresh_cameras()
        s = self.session
        exportable = s is not None and (s.n_points > 0 or (s.masks is not None and s.masks.n_masked() > 0))
        self.act_export.setEnabled(has_video and not tracking and exportable)
        self.act_overlay.setEnabled(has_video and not tracking and self._overlay is None)
        # 3D layer: needs several cameras, then a calibration, then a result
        p = self.project
        multi = p is not None and p.n_views > 1 and has_video and not tracking
        has_cal = multi and p.calibration is not None and len(p.calibration) == p.n_views
        # Import Calibration and Sync open with one video too and explain what
        # a further step needs (I115)
        live3d = has_video and not tracking and p is not None
        self.act_calib.setEnabled(live3d)
        # the two wizards stay clickable: a first-time user must be able to open
        # them and be TOLD what to prepare, not meet a greyed-out entry
        self.act_wand.setEnabled(not tracking)
        self.act_lens.setEnabled(not tracking)
        self.act_export_cal.setEnabled(live3d)
        self.act_offsets3d.setEnabled(live3d)
        self.act_sync.setEnabled(live3d)
        self.act_recon.setEnabled(live3d)
        has_rec = bool(has_cal and self.project is not None and self.project.reconstruction is not None
                       and self.project.reconstruction.per_cam is not None)
        self.act_retrack.setEnabled(has_rec and not tracking)
        self.act_hull.setEnabled(live3d)
        self.act_view3d.setEnabled(bool(has_cal or self.act_view3d.isChecked()))
        self.act_export_mesh.setEnabled(bool(has_cal and self._reference_instant() in self._hull_cache))
        # Body layer: running needs a video; everything else needs a result
        busy_body = self._body_worker is not None and self._body_worker.isRunning()
        has_body = s is not None and s.has_body()
        self.act_body_run.setEnabled(has_video and not tracking and not busy_body)
        self.act_body_view.setEnabled(has_video)
        self.act_body_video.setEnabled(has_body and not tracking and self._body_video is None)
        self.act_body_joints.setEnabled(has_body and not tracking)
        self.act_body_angles.setEnabled(has_body and not tracking)
        self.act_body_clear.setEnabled(has_body and not tracking)
        self.act_save.setEnabled(has_video and not tracking)
        self.act_save_as.setEnabled(has_video and not tracking)
        self.act_open.setEnabled(not tracking)
        self.act_open_proj.setEnabled(not tracking)
        self.act_undo.setEnabled(has_video and not tracking and self._undo_snap is not None)
        self.act_mark_event.setEnabled(has_video and not tracking)
        self.timeline.setEnabled(has_video and not tracking)  # still paints progress live
        # ONLY the working camera takes edits — a click on a companion view
        # switches to it instead, so an edit can never land in the wrong session
        active_i = self.project.active if self.project is not None else 0
        for k, cv in enumerate(self.grid.canvases):
            cv.set_interactive(has_video and not tracking and k == active_i)
        self.btn_add.setEnabled(has_video and not tracking)
        if tracking and self.btn_add.isChecked():
            self.btn_add.setChecked(False)
        self.btn_animal.setEnabled(has_video and not tracking)
        if tracking and self.btn_animal.isChecked():
            self.btn_animal.setChecked(False)
        self.btn_mask.setEnabled(has_video)
        self.btn_onbody.setEnabled(has_video and not tracking)
        self.btn_skeleton.setEnabled(has_video and not tracking)
        self.m_skeleton.setEnabled(has_video and not tracking)
        self.btn_clear_animal.setEnabled(has_video and not tracking and s is not None
                                         and s.animal is not None)
        self.btn_pan.setEnabled(has_video)  # panning is view-only: fine mid-run
        for act in (self.act_mode_auto, self.act_mode_semi):
            act.setEnabled(not tracking)
        self.progress.setVisible(tracking and not self._step_run)  # a 1-frame bar is noise
        self._update_track_button()
        self._refresh_onboarding()

    def _onboarding_step(self, k: int):
        if k == 0:
            if self.state != TRACKING:
                self._open_video_dialog()
            return
        if self.state != READY or self.session is None:
            return
        if k == 1:
            self.btn_animal.setChecked(True)
        elif k == 2:
            self.btn_skeleton.showMenu()
        else:
            self._toggle_tracking()

    def _refresh_onboarding(self):
        s = self.session
        if s is None or self.state == IDLE:
            self.onboarding.set_state(
                [False, False, False, False],
                "Open a video (Ctrl+O) or a project (Ctrl+Shift+O) to begin  ·  F1 = manual")
            return
        seg = s.animal is not None and (s.animal.n_prompts() > 0 or (s.masks is not None and s.masks.n_masked() > 0))
        # the skeleton step ticks as soon as a skeleton is applied or any landmark exists
        landmarks = s.skeleton is not None or s.n_points > 0
        tracked = int(s.tracked.sum()) > sum(1 for i in range(s.n_points) if s.tracked[:, i].any())
        done = [True, seg, landmarks, tracked]
        # Segmenting is OPTIONAL. An animal a few pixels across (seen from
        # far away) has no silhouette worth outlining: place a point
        # with N and Track — the tracker zooms into its own crop around it.
        # The strip therefore never waits for a segment; Track is the goal.
        if not tracked:
            hint = ("Press N and click the animal (a few pixels is enough), then Track ▶ — or S to "
                    "outline a larger animal first for silhouette landmarks"
                    if not (seg or landmarks) else
                    "Press Track ▶ now, or Skeleton ▾ first for named landmarks (select one, press N, click it)")
        else:
            hint = "All set — refine, mark events, export (Ctrl+E)"
        self.onboarding.set_state(done, hint)
        if tracked and self.onboarding.isVisible() and not getattr(self, "_onboarding_auto_hidden", False):
            self._onboarding_auto_hidden = True
            self.onboarding.hide()
            self.act_onboarding.setChecked(False)

    def _run_scope(self):
        """Which points a run covers: the panel's multi-selection, or every point
        when all (or none) are selected. Returns (set of pids | None = all, n_selected)."""
        s = self.session
        if s is None:
            return None, 0
        sel = set(self._selected_pids())
        if not sel or len(sel) >= s.n_points:
            return None, len(sel)
        return sel, len(sel)

    def _update_track_button(self):
        semi = self._track_mode == "semi"
        if self.state == TRACKING:
            self.btn_track.setText("Pause ■  (X)")
            self.btn_track.setEnabled(True)
            self.btn_track.setToolTip("Stop tracking (X or Space). Corrections are made while paused.")
            return
        s = self.session
        scope, n_sel = self._run_scope()
        # the button itself says what a run covers, so a scoped run never surprises
        if scope is not None:
            self.btn_track.setText(f"Step {n_sel} sel. ▶  (F)" if semi else f"Track {n_sel} sel. ▶")
        else:
            self.btn_track.setText("Step ▶  (F)" if semi else "Track ▶")
        n = len([p for p in s.seedable_at(self.current) if scope is None or p in scope]) if s is not None else 0
        animal_ok = s is not None and s.animal_seedable_at(self.current)
        if self.state == IDLE:
            self.btn_track.setEnabled(False)
            self.btn_track.setToolTip("Open a video first")
        elif s is None or (n == 0 and not animal_ok):
            self.btn_track.setEnabled(False)
            if s is not None and s.animal is not None:
                self.btn_track.setToolTip(
                    "Nothing to track from this frame: place a point (N), press S and click the "
                    "segment here, or move to a frame where the points/segment exist")
            else:
                self.btn_track.setToolTip("No point has a position at this frame — press N and "
                                          "click the animal to add one (a few pixels is enough); "
                                          "S outlines a larger animal, optionally")
        else:
            self.btn_track.setEnabled(True)
            what = []
            if n:
                what.append(f"the {n} selected point(s)" if scope is not None else f"all {n} point(s)")
            if animal_ok:
                what.append("the segment (silhouette + derived landmarks)")
            what = " and ".join(what)
            if semi:
                self.btn_track.setToolTip(
                    f"Semi-automatic: track {what} ONE frame forward from frame "
                    f"{self.current} and pause (F or T). Fix anything, press F again — "
                    "each step re-seeds from what you see.")
            else:
                self.btn_track.setToolTip(f"Track {what} forward from frame {self.current} (T)")

    @staticmethod
    def _preferred_point_backend() -> str:
        return "alltracker" if alltracker_backend.available() else "cotracker3"

    def _set_point_backend(self, key: str):
        self._point_backend = key
        if self.session is not None:
            self.session.ui_state["point_backend"] = key
            self.session.dirty = True
        self.statusBar().showMessage(
            "Point model: AllTracker — applies to the next Track run" if key == "alltracker"
            else "Point model: CoTracker3 — applies to the next Track run", 5000)
        self._update_track_button()

    def _set_track_mode(self, mode: str):
        self._track_mode = mode
        self._update_track_button()
        if mode == "semi":
            self.statusBar().showMessage(
                "Semi-automatic mode: F tracks one frame forward and pauses; correct "
                "anything, then F again — B still just steps back", 8000)

    def _show_manual(self):
        """Help → User Manual (F1). Non-modal and reused, so it can stay open
        beside the video while you follow it."""
        dlg = getattr(self, "_manual_dlg", None)
        if dlg is None:
            dlg = ManualDialog(self)
            self._manual_dlg = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _show_hotkeys(self):
        dlg = getattr(self, "_hotkeys_dlg", None)
        if dlg is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Keyboard & mouse reference")
            lay = QVBoxLayout(dlg)
            tb = QTextBrowser()
            tb.setHtml(HOTKEYS_HTML)
            tb.setOpenExternalLinks(False)
            lay.addWidget(tb)
            dlg.resize(620, 680)
            self._hotkeys_dlg = dlg
        dlg.show()          # non-modal: keep it open next to the video
        dlg.raise_()
        dlg.activateWindow()

    # ------------------------------------------------------------ open video

    def _open_video_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", VIDEO_FILTER)
        if path:
            self._open_video(path)

    def _open_video(self, path: str, then=None):
        """Probe in a worker thread, then wire everything up. `then(info)` runs
        after success (used by project loading)."""
        self.statusBar().showMessage(f"Opening {Path(path).name}…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        probe = _VideoProbe(path)
        self._probe = probe  # latest request wins; older probes are ignored below

        def done(result):
            probe.wait(1000)
            QApplication.restoreOverrideCursor()  # each open pushed exactly one
            if self._probe is not probe:
                return  # a newer open superseded this one while it was probing
            self.statusBar().clearMessage()
            if isinstance(result, str):
                QMessageBox.critical(self, "Could not open video", result)
                return
            self._attach_video(result, then)

        probe.done.connect(done)
        probe.start()

    def _attach_video(self, info: VideoInfo, then=None):
        """Open `info` as a brand-new single-camera project (extra cameras are
        added afterwards with `_add_video_dialog`)."""
        self._autosave()  # don't lose the previous session when switching videos
        self._teardown_video()
        self._wand_result = (None, None)      # a new project: no wand run belongs to it (I33)
        self.project = Project([TrackingSession(info.path, info.n_frames, info.fps,
                                                info.width, info.height)])
        self._views = [_ViewRuntime(info, DEFAULT_CACHE_BYTES)]
        self.grid.set_count(1)
        self.grid.set_active(0)
        self._start_seek_service()

        self.canvas.set_video_size(info.width, info.height)
        self.spin.setRange(0, info.n_frames - 1)
        self.spin.setSuffix(f" / {info.n_frames - 1}")
        self._play_timer.setInterval(max(10, round(1000 / info.fps)))

        if then is None:
            self.project_path = None
            autosave = Path(info.path + AUTOSAVE_SUFFIX)
            if autosave.exists():
                self._maybe_resume(autosave)
        else:
            then(info)

        self.state = READY
        self._refresh_cameras()
        self._undo_snap = None
        self.selected = None
        self._animal_hint_shown = False
        self._refresh_point_list()
        self.timeline.set_session(self.session)
        self._refresh_events_ui()
        self._refresh_skeleton_menu()
        self._refresh_animal_panel()
        self._goto(self.session.current_frame if self.session else 0, force=True)
        self._apply_ui_state()   # restored sessions reopen EXACTLY as saved
        self._apply_state()
        if self.session is not None and self.session.n_points > 3:
            QTimer.singleShot(0, self._fit_timeline_height)
        self.setWindowTitle(f"{APP_NAME} — {Path(info.path).name}")
        if self.session is not None and self.session.n_points == 0 and self.session.animal is None:
            self.toast.show_message(
                "Press <b>S</b> and click the segment — or <b>N</b> and click a point. "
                "New to tracking? <b>F1</b> opens the manual, written from scratch.",
                "info", 9000)
        if getattr(info, "header_overcount", 0) > 0:
            # said once, at open: otherwise the user wonders why the timeline
            # is shorter than the file's own frame count
            self.toast.show_message(
                f"This file says it has {info.header_frames} frames but only {info.n_frames} can be "
                f"decoded; the last {info.header_overcount} are missing from the file itself. "
                "Kinetrace uses the frames that exist.", "warn", 10000)
        if getattr(info, "fps_note", ""):
            # every time, speed and camera-rate number depends on it (I37)
            self.toast.show_message(info.fps_note, "warn", 12000)
        if info.vfr_suspected:
            QMessageBox.warning(
                self, "Variable frame rate detected",
                "This video appears to have a variable frame rate. Frame numbers in the "
                "export may not map cleanly to timestamps.\n\nFor reliable results, "
                "re-encode to constant frame rate first:\n\n"
                f'ffmpeg -i "{Path(info.path).name}" -vsync cfr -r {info.fps:.6g} '
                '-c:v libx264 -crf 18 fixed.mp4')

    def _start_seek_service(self, index: int | None = None):
        """(Re)create one view's scrubbing decoder thread on its own cache.
        Every view has its own — one `cv2.VideoCapture` per thread is a hard
        rule, and companion views decode at the same time as the active one."""
        i = self.project.active if index is None else index
        rt = self._views[i]
        rt.stop()
        rt.seek = SeekService(rt.info.path, rt.cache, n_frames=rt.info.n_frames)
        # The view's index is looked up when a frame ARRIVES, not bound here:
        # reopening a project saved in camera 2+ moves the first runtime to
        # another index, and removing a camera renumbers the ones after it; a
        # bound index then painted the working camera's frames into another
        # tile and truncated the wrong camera at EOF (I103, I111).
        rt.seek.frame_ready.connect(lambda f, rgb, r=rt: self._on_seek_frame(self._view_index(r), f, rgb))
        rt.seek.seek_slow.connect(lambda f, r=rt: self._on_seek_slow(self._view_index(r), f))
        rt.seek.eof_truncated.connect(lambda f, r=rt: self._on_eof_truncated(self._view_index(r), f))
        rt.seek.decode_failed.connect(lambda f, msg, r=rt: self._on_decode_failed(self._view_index(r), f, msg))
        rt.seek.start()

    def _view_index(self, rt) -> int:
        """Current index of a view runtime (-1 once it has been removed)."""
        for k, r in enumerate(self._views):
            if r is rt:
                return k
        return -1

    def _rebudget_caches(self):
        """The WORKING camera keeps the whole frame-cache budget; companions get
        a small fixed allowance each.

        An even split would be wrong: the working view is the one being scrubbed,
        played and prefetched, while a companion only ever displays a single
        frame. At 15 cameras an even split would leave the working view with a
        fifteenth of the cache and make scrubbing crawl, while the companions
        sat on memory they never re-read."""
        active = self.project.active if self.project is not None else 0
        for i, rt in enumerate(self._views):
            rt.cache.max_bytes = (DEFAULT_CACHE_BYTES if i == active
                                  else COMPANION_CACHE_BYTES)
            if i != active:
                rt.cache.trim()   # hand the memory back now, not on its next decode

    def _reset_frame_cache(self):
        """Shift+C failsafe: drop every cached frame and re-decode this one
        from scratch. A stale cache entry and a desynced decoder look the
        same from the outside (wrong/garbled picture), so this rebuilds the
        scrubbing decoder too. Tracked data is untouched — this only affects
        decoded pixels."""
        if self.state != READY or self.info is None:
            return
        n_frames, n_bytes = self.cache.stats()
        for i, rt in enumerate(self._views):
            rt.cache.clear()
            if i == self.project.active or rt.seek is not None:
                self._start_seek_service(i)   # fresh VideoCapture: fixes decoder desync too
        self._goto(self.current, force=True)
        self.statusBar().showMessage(
            f"Frame cache cleared ({n_frames} frames, {n_bytes / 1024**2:.0f} MB) "
            f"— frame {self.current} re-decoded from the file", 5000)

    # ------------------------------------------------------- multi-camera views

    def _add_video_dialog(self):
        """Add another camera's video of the same event."""
        if self.project is None or self.state != READY:
            QMessageBox.information(self, "Open a video first",
                                    "Open the first camera's video (Ctrl+O), then add the others.")
            return
        if self.project.n_views >= MAX_VIEWS:
            QMessageBox.information(self, "Too many cameras",
                                    f"A project holds up to {MAX_VIEWS} cameras.")
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Add camera video(s)", "", VIDEO_FILTER)
        added = [p for p in paths if self._add_view(p)]
        if added:
            self._refresh_cameras()
            self._refresh_companions()
            self.toast.show_message(
                f"Added {len(added)} camera(s). Line them up in time: <b>3D → Sync Cameras (Sound / "
                "Motion)</b> does it from the sound tracks or the pictures, or scrub to a shared event and nudge each "
                "camera's <b>offset</b> (or press <b>Align here</b>).",
                "info", 10000)

    def _add_view(self, path: str) -> bool:
        """Probe `path` and append it as a view. Probing is synchronous here —
        the user is explicitly waiting for this one file, and chaining async
        probes for a multi-select would buy nothing."""
        if self.project is None or self.project.n_views >= MAX_VIEWS:
            return False
        if any(rt.info.path == path for rt in self._views):
            self.statusBar().showMessage(f"{Path(path).name} is already in this project", 5000)
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            info = probe_video(path)
        except (ValueError, OSError) as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Could not open video", str(e))
            return False
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
        s = TrackingSession(info.path, info.n_frames, info.fps, info.width, info.height)
        # a second camera of the same animal wants the same landmark names —
        # matching names across views is exactly what the 3D export joins on
        ref = self.project.session
        if ref is not None and ref.skeleton:
            s.apply_skeleton(ref.skeleton)
        i = self.project.add_view(s, Path(path).stem[:24] or None)
        self._views.append(_ViewRuntime(info, DEFAULT_CACHE_BYTES))
        self._rebudget_caches()
        for cv in self.grid.set_count(self.project.n_views):
            self._wire_canvas(cv)
        cv = self.grid.canvas(i)
        cv.set_video_size(info.width, info.height)
        cv.set_interactive(False)          # only the active view takes edits
        cv.set_marker_size(self.marker_spin.value())
        self.grid.set_active(self.project.active)
        self._apply_state()            # multi-camera actions (sync, import calibration) light up
        if getattr(info, "fps_note", ""):
            self.toast.show_message(f"{self.project.name(i)}: {info.fps_note}", "warn", 12000)
        cal = self.project.calibration
        if cal is not None and 0 < len(cal) < self.project.n_views:
            # the 3D menu greys out (it needs every camera calibrated): say why,
            # and that the existing calibration is kept, not lost (I16)
            QMessageBox.information(
                self, "The calibration covers the first cameras",
                f"This project's calibration was made for {len(cal)} camera(s); {Path(path).name} is "
                f"camera {i + 1} and is not in it.\n\n"
                "The calibration stays in the project and in the saved file, but 3D (Reconstruct, "
                "Volume, the 3D view) needs every camera calibrated. Either calibrate all "
                f"{self.project.n_views} cameras (3D → Calibrate Cameras with a Wand, or Import "
                "Calibration with a file that includes this camera), or remove this camera again "
                "with its × in the CAMERAS panel to use 3D again.")
        return True

    def _remove_view(self, i: int):
        p = self.project
        if p is None or self.state != READY or not (0 <= i < p.n_views) or p.n_views <= 1:
            return
        n_tracked = int(p.sessions[i].tracked.any(axis=1).sum())     # FRAMES with data, not cells
        if n_tracked and QMessageBox.question(
                self, "Remove camera",
                f"Remove {p.name(i)} and its {n_tracked} tracked frames from the project?\n\n"
                "This cannot be undone.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        target = None
        if i == p.active:                 # the next camera takes over at the SAME instant (I111)
            nxt = i + 1 if i + 1 < p.n_views else i - 1
            target = p.map_frame(i, nxt, self.current)
        self._views[i].stop()
        del self._views[i]
        p.remove_view(i)
        self._rebudget_caches()
        self.grid.set_count(p.n_views)
        for k, rt in enumerate(self._views):          # tiles after i now show another camera
            cv = self.grid.canvas(k)
            if cv is not None:
                cv.set_video_size(rt.info.width, rt.info.height)
            rt.want_frame = None
        self.grid.set_active(p.active)
        self._apply_active_view(target if target is not None else p.session.current_frame)

    def _set_active_view(self, i: int):
        """Switch the camera being worked on. The playhead follows through the
        offsets, so the picture stays on the same instant."""
        p = self.project
        if p is None or self.state != READY or not (0 <= i < p.n_views) or i == p.active:
            return
        self.canvas.cancel_gesture()
        # The tools arm ONE canvas; left armed across a switch, the new view took
        # plain clicks as hand placements while S / N still looked on (I47), and a
        # half-marked event was finished with the other camera's frame number (I65)
        for b in (self.btn_add, self.btn_animal):
            if b.isChecked():
                b.setChecked(False)
        if self._pending_event is not None:
            self._pending_event = None
            self.timeline.set_pending_event(None)
        # the camera being left keeps its exact state; the tool / display toggles
        # travel with the user: re-applying the other camera's stored copy silently
        # turned auto-pause and ROI back on and swapped the point model (I50)
        self._sync_ui_state()
        carried = {k: self.session.ui_state[k] for k in GLOBAL_UI_KEYS if k in self.session.ui_state}
        target = p.set_active(i)
        p.session.ui_state.update(carried)
        self.grid.set_active(i)
        self._apply_active_view(target if target is not None else p.session.current_frame)
        self._update_disagreement()            # the band is per camera: recompute for this one
        self.statusBar().showMessage(
            f"Working in {p.name(i)} — points, silhouette and timeline are this camera's", 5000)

    def _apply_active_view(self, frame: int):
        """Re-point every widget at the active view's session and frame. This is
        the ONE place a view switch takes effect, so nothing can be left showing
        the previous camera's data."""
        p = self.project
        if p is None or p.session is None:
            return
        s, rt = p.session, self._rt
        info = rt.info
        self._rebudget_caches()   # the cache follows the working camera
        if rt.seek is None:       # it may have been a lazily-started companion
            self._start_seek_service(p.active)
        self.spin.setRange(0, rt.n_frames - 1)
        self.spin.setSuffix(f" / {rt.n_frames - 1}")
        self._play_timer.setInterval(max(10, round(1000 / max(info.fps, 1e-6))))
        for k, cv in enumerate(self.grid.canvases):
            cv.set_interactive(k == p.active and self.state == READY)
        self.canvas.set_video_size(info.width, info.height)
        self.selected = None
        self._undo_snap = None
        self.act_undo.setEnabled(False)
        self._refresh_point_list()
        self.timeline.set_session(s)
        self._refresh_events_ui()
        self._refresh_skeleton_menu()
        self._refresh_animal_panel()
        self._goto(int(np.clip(frame, 0, rt.n_frames - 1)), force=True)
        self._apply_ui_state()
        self._refresh_cameras()
        self._apply_state()
        self.setWindowTitle(f"{APP_NAME} — {Path(info.path).name}")

    def _on_solo_toggled(self, on: bool):
        self.grid.set_solo(on)
        self._refresh_companions()

    def _on_view_offset(self, i: int, offset: float):
        if self.project is None:
            return
        had_3d = self.project.reconstruction is not None
        self.project.set_offset(i, float(offset))
        self._after_retime(had_3d)
        self._refresh_companions()
        self._refresh_cameras()

    def _after_retime(self, had_3d: bool) -> None:
        """A changed offset / rate makes the triangulation stale (I23): the
        project drops it; here the volumes, the band and the menus follow."""
        if had_3d and self.project is not None and self.project.reconstruction is None:
            self._hull_cache.clear()
            self._update_disagreement()
            self._refresh_view3d()
            self._apply_state()
            self.statusBar().showMessage(
                "Camera timing changed: the 3D result made with the old timing was cleared — "
                "press Ctrl+3 (3D → Reconstruct) again", 8000)

    def _align_view_here(self, i: int):
        """Take what camera `i` is showing right now as the match for the active
        camera's current frame (the flash / clap alignment step)."""
        p = self.project
        if p is None or not (0 <= i < p.n_views) or i == p.active:
            return
        shown = self._views[i].want_frame
        if shown is None:
            self.statusBar().showMessage(
                f"{p.name(i)} has no frame at this instant — nudge its offset first", 5000)
            return
        had_3d = p.reconstruction is not None
        off = p.align_to(i, shown, self.current)
        self._after_retime(had_3d)
        self._refresh_companions()
        self._refresh_cameras()
        self.statusBar().showMessage(
            f"{p.name(i)} aligned: offset {off:+g} frames against {p.name(p.active)}", 6000)

    def _refresh_cameras(self):
        """Repaint the CAMERAS panel from the project."""
        p = self.project
        if p is None:
            self.cameras.update_rows([], [], 0, [])
            self.cameras.setVisible(False)
            return
        self.cameras.setVisible(True)
        self.act_solo.setEnabled(p.n_views > 1)
        # the live frame number lives in each view's caption, which updates on
        # every scrub; this panel carries the things that only change on edits
        statuses = [f"{s.n_points} pts · {int(s.tracked.any(axis=1).sum())} tracked frames "
                    f"· {Path(s.video_path).name}" for s in p.sessions]
        note = ""
        if p.n_views > 1:
            lo, hi = p.coverage()
            note = (f"Overlap: frames {lo}–{hi} of {p.name(p.active)}"
                    if hi >= lo else "These cameras never overlap at the current offsets.")
            if p.fps_mismatch():
                note += ("  Frame rates differ: the ×n cameras step several frames per "
                         "reference frame (their offsets are in their own frames).")
            if p.has_fractional_offsets():
                note += "  Sub-frame offsets are set — the 3D layer interpolates at them."
        self.cameras.update_rows(list(p.names), list(p.offsets), p.active, statuses, note,
                                 rates=list(p.rates))

    def _set_aside(self, autosave: Path, why: str, quiet: bool = False) -> None:
        """Move an autosave the app cannot resume out of the way (it would be
        overwritten by the next autosave) and say where it went (I113).
        `quiet`: the user declined it -- a status-bar line, no dialog (I129)."""
        aside = autosave.with_name(autosave.name.replace(AUTOSAVE_SUFFIX, "")
                                   + f".cotracker.{time.strftime('%Y%m%d-%H%M%S')}.bak.npz")
        base, k = aside, 2
        while aside.exists():            # two set-asides in one second must not collide
            aside = base.with_name(base.name.replace(".bak.npz", f"-{k}.bak.npz"))
            k += 1
        try:
            autosave.rename(aside)
            where = f"It was kept as {aside.name} next to the video."
        except OSError as e:
            where = f"It could not be moved aside ({e}); copy it somewhere safe before working on."
        if quiet:
            self.statusBar().showMessage(f"{why} {where}", 10000)
            return
        QMessageBox.warning(self, "Earlier session not resumed", f"{why}\n\n{where}")

    def _maybe_resume(self, autosave: Path):
        try:
            proj = Project.load_npz(autosave)   # also reads pre-v4 single-view files
        except Exception as e:  # noqa: BLE001
            self._set_aside(autosave, f"An earlier session for this video was found but could not be read ({e}).")
            return
        if proj.n_views == 0:
            return
        n_saved = proj.sessions[proj.active].n_frames
        if n_saved != self.info.n_frames:
            header = getattr(self.info, "header_frames", 0)
            if not (header == n_saved and self.info.n_frames < n_saved):
                self._set_aside(
                    autosave, f"An earlier session for this video was found, but it was saved for a video "
                              f"of {n_saved} frames and this one has {self.info.n_frames}, so its tracks "
                              "would sit on the wrong frames.")
                return
            # saved when the header's frame count was trusted: the extra frames
            # never had a picture, the tracks are fine (the project path says so too)
        restored = proj.sessions[proj.active]
        tracked_n = int(restored.tracked.any(axis=1).sum())
        cams = (f", {proj.n_views} cameras" if proj.n_views > 1 else "")
        if QMessageBox.question(
                self, "Resume session?",
                f"A previous session for this video was found "
                f"({restored.n_points} points, {tracked_n} tracked frames{cams}).\n\nResume it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
            # the next autosave would overwrite it: keep it as a dated copy (I129)
            self._set_aside(autosave, "Earlier session not resumed.", quiet=True)
            return
        restored.video_path = self.info.path
        if proj.n_views == 1:
            self.session = restored
        else:
            self._adopt_project(proj, self.info)

    def _teardown_video(self):
        for th in (self._body_worker, self._body_video):
            if th is not None and th.isRunning():
                th.request_cancel()
                th.wait(10000)
            _retire(th)
        self._body_worker = self._body_video = None
        if self.worker is not None:
            self.worker.request_pause()
            self.worker.wait(5000)
            _retire(self.worker)
            self.worker = None
        if self._preview is not None:
            self._preview.wait(15000)
            _retire(self._preview)
            self._preview = None
        if self._loading_dialog is not None:
            self._loading_dialog.close()
            self._loading_dialog = None
        for rt in self._views:
            rt.stop()
        self._views = []
        self._hull_cache.clear()
        if self.view3d is not None:
            self.view3d.hide()
            self.act_view3d.setChecked(False)
        self._end_run_cleanup()
        self._pending_event = None
        self.timeline.set_pending_event(None)
        self._render_timer.stop()
        self._play_timer.stop()
        self.btn_play.setChecked(False)

    def _on_seek_slow(self, view: int, idx: int):
        if self.project is not None and view == self.project.active:
            self.statusBar().showMessage(f"Seeking frame {idx}…", 3000)

    def _on_decode_failed(self, view: int, idx: int, msg: str):
        """A frame inside the video could not be decoded (twice): say so once
        per frame, and never shorten the video over it (I40)."""
        if not (0 <= view < len(self._views)):
            return
        seen = self._views[view].__dict__.setdefault("_bad_frames", set())
        if idx in seen:
            return
        seen.add(idx)
        name = self.project.name(view) if self.project is not None else "The video"
        if idx < 0:
            text = (f"{name} could not be opened for display ({msg}). Check that the file is still "
                    "reachable; Shift+C retries.")
        else:
            text = (f"{name}: frame {idx} could not be decoded ({msg}) — the file is damaged there or "
                    "ends early. The frame shows blank; Shift+C retries.")
        self.toast.show_message(text, "warn", 10000)
        self.statusBar().showMessage(text, 10000)

    def _on_eof_truncated(self, view: int, idx: int):
        if not (0 <= view < len(self._views)):
            return
        rt = self._views[view]
        if not 0 < idx < rt.n_frames:
            return
        rt.n_frames = idx
        if self.project is not None and view == self.project.active:
            self.spin.setMaximum(idx - 1)
            self.spin.setSuffix(f" / {idx - 1}")
            self.statusBar().showMessage(
                f"Note: video ends at frame {idx - 1} (metadata reported more)", 6000)
            if self.current >= idx:              # the playhead sat on a frame that does not exist
                self._goto(idx - 1)
        else:
            # a companion's end is ITS frame number: comparing it with the working
            # camera's playhead moved the playhead for no reason (I43); the
            # companion simply shows nothing past its end
            name = self.project.name(view) if self.project else f"view {view}"
            self.statusBar().showMessage(
                f"Note: {name} ends at frame {idx - 1} (metadata reported more)", 6000)

    # ------------------------------------------------------------ navigation

    # Hotkeys must work wherever the keyboard focus happens to sit. Before
    # this filter, clicking a spin box, the point list or any toolbar button
    # moved the focus there and the letter keys died: a spin box swallows them
    # as rejected text, a list uses them for type-ahead search, a button takes
    # Space as a click. So the
    # application event filter hands hotkeys to `_hotkey` FIRST, unless the
    # focus is in a real text field (the rename editor, a token field), where
    # letters are text. Spin boxes keep only what they type with (digits,
    # sign, arrows, Enter, Backspace, Delete); lists keep their navigation.
    _TEXT_WIDGETS = (QLineEdit, QTextEdit, QPlainTextEdit)
    _SPIN_KEYS = {Qt.Key_Delete, Qt.Key_Plus, Qt.Key_Minus, Qt.Key_Equal, Qt.Key_Underscore,
                  Qt.Key_BracketLeft, Qt.Key_BracketRight, Qt.Key_Period, Qt.Key_Comma,
                  Qt.Key_Return, Qt.Key_Enter}

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.KeyPress and self._hotkeys_apply(ev):
            if self._hotkey(ev):
                return True
        return super().eventFilter(obj, ev)

    def _hotkeys_apply(self, ev) -> bool:
        """Should this key press be offered to the hotkey table before the
        focused widget sees it?"""
        if QApplication.activeModalWidget() is not None:
            return False
        if QApplication.activePopupWidget() is not None:
            # an open menu owns the keyboard: Escape closes it, arrows / letters
            # navigate it. Offered to the hotkey table first, Escape deselected the
            # point and left the menu open, and F / Delete acted behind it (G6)
            return False
        # the Segment & Points panel may be floated (its own top-level window):
        # it is still this window's panel, so its focus keeps the hotkeys (I68)
        mine = (self, self.view3d, getattr(self, "dock", None), getattr(self, "body_win", None))
        active = QApplication.activeWindow()
        if active is None or active not in mine:
            return False
        fw = QApplication.focusWidget()
        if fw is None or fw.window() not in mine:
            return False
        spin = isinstance(fw, QAbstractSpinBox) or isinstance(fw.parent(), QAbstractSpinBox)
        if isinstance(fw, self._TEXT_WIDGETS) and not spin:
            return False                    # typing a name / a token: letters are text
        if spin and ev.key() in self._SPIN_KEYS:
            return False                    # editing keys stay with the spin box
        return True

    def keyPressEvent(self, ev):
        """Single-letter hotkeys reaching the window by propagation (the
        application filter already handled the focused-widget cases)."""
        if not self._hotkey(ev):
            super().keyPressEvent(ev)

    def _hotkey(self, ev) -> bool:
        """The hotkey table. Returns True when the key was one of ours."""
        key, mods = ev.key(), ev.modifiers()
        shift_only = mods == Qt.ShiftModifier
        plain = mods in (Qt.NoModifier, Qt.KeypadModifier)
        if key == Qt.Key_F and plain:
            # semi-automatic mode: F tracks one frame forward instead of just
            # moving (falls back to plain navigation when nothing is seedable)
            if (self._track_mode == "semi" and self.state == READY
                    and self.session is not None
                    and self.current < self.n_frames - 1
                    and (self.session.seedable_at(self.current)
                         or self.session.animal_seedable_at(self.current))):
                self._track_step()
            else:
                self._goto(self.current + 1)
        elif key == Qt.Key_B and plain:
            self._goto(self.current - 1)
        elif key == Qt.Key_F and shift_only:
            self._goto(self.current + self.step_spin.value())
        elif key == Qt.Key_B and shift_only:
            self._goto(self.current - self.step_spin.value())
        elif key == Qt.Key_Space and plain:
            # Space = play/pause the preview — but while a run is live it is
            # the natural "stop": it pauses tracking
            if self.state == TRACKING:
                self._pause_tracking()
            else:
                self.btn_play.toggle()
        elif key == Qt.Key_T and plain:
            self._toggle_tracking()
        elif key == Qt.Key_X and plain:
            self._pause_tracking()
        elif key == Qt.Key_N and plain:
            if self.state == READY:
                self.btn_add.toggle()
        elif key == Qt.Key_S and plain:
            if self.state == READY:
                self.btn_animal.toggle()
        elif key == Qt.Key_Delete and plain:
            self._delete_selected()
        elif key == Qt.Key_Escape and plain:
            self.canvas.cancel_gesture()
            if self.btn_animal.isChecked():
                self.btn_animal.setChecked(False)
                self.statusBar().showMessage("Segment tool off", 3000)
            elif self.btn_add.isChecked():
                self.btn_add.setChecked(False)
                self.statusBar().showMessage("Point placement cancelled", 3000)
            elif self._pending_event is not None:
                self._pending_event = None
                self.timeline.set_pending_event(None)
                self.statusBar().showMessage("Event mark cancelled", 3000)
            elif self.timeline.sel_range is not None:
                self.timeline.clear_selection()
                self.statusBar().showMessage("Frame-window selection cleared", 3000)
            else:
                self._deselect()
        elif key == Qt.Key_E and plain:
            self._mark_event()
        elif key == Qt.Key_R and plain:
            self.canvas.fit()
        elif key == Qt.Key_H and plain:
            # the button is disabled without a video: toggling it anyway armed
            # an invisible pan tool that later ate the first clicks (G4)
            if self.btn_pan.isEnabled():
                self.btn_pan.toggle()
        elif key in (Qt.Key_Plus, Qt.Key_Equal) and plain:
            self.canvas.zoom_step(CANVAS_ZOOM_STEP)   # video zoom, around the pointer
        elif key == Qt.Key_Minus and plain:
            self.canvas.zoom_step(1 / CANVAS_ZOOM_STEP)
        elif key == Qt.Key_C and shift_only:
            self._reset_frame_cache()
        elif key in (Qt.Key_Greater, Qt.Key_Period) and shift_only:
            self._goto_clicked_frame(last=True)     # Shift+> : last hand-placed frame
        elif key in (Qt.Key_Less, Qt.Key_Comma) and shift_only:
            self._goto_clicked_frame(last=False)    # Shift+< : first hand-placed frame
        elif key == Qt.Key_Period and plain:
            self._goto_manual_step(forward=True)    # . : next hand-placed frame
        elif key == Qt.Key_Comma and plain:
            self._goto_manual_step(forward=False)   # , : previous hand-placed frame
        elif key == Qt.Key_J and plain:
            self._goto_low_conf(forward=True)       # J : next low-confidence stretch
        elif key == Qt.Key_J and shift_only:
            self._goto_low_conf(forward=False)
        elif key == Qt.Key_O and plain:
            self.act_onion.toggle()
        elif key == Qt.Key_L and plain:
            self.act_loupe.toggle()
            self.statusBar().showMessage("Loupe " + ("on" if self.act_loupe.isChecked() else "off"), 2000)
        elif key == Qt.Key_X and shift_only:
            self._toggle_hidden_here()
        elif key == Qt.Key_N and shift_only:
            self._edit_note()
        elif key in (Qt.Key_Return, Qt.Key_Enter) and plain and self.canvas.polygon_in_progress():
            if not self.canvas.finish_polygon():
                self.statusBar().showMessage("A polygon needs at least 3 corners", 3000)
        elif key in (Qt.Key_Plus, Qt.Key_Equal) and shift_only:
            self.timeline.zoom_time_keyboard(TIME_ZOOM_STEP)   # timeline time-axis
        elif key in (Qt.Key_Minus, Qt.Key_Underscore) and shift_only:
            self.timeline.zoom_time_keyboard(1 / TIME_ZOOM_STEP)
        else:
            return False
        ev.accept()
        return True

    def _on_spin_seek(self):
        self._goto(self.spin.value())
        # release focus so single-letter and Shift+± hotkeys work right away
        # (a focused spinbox swallows them as rejected text input)
        self.spin.clearFocus()

    def _goto(self, idx: int, force: bool = False):
        if self.state != READY or self.session is None:
            return
        idx = max(0, min(self.n_frames - 1, idx))
        if idx == self.current and not force:
            return
        self.current = idx
        self.session.current_frame = idx
        self.spin.blockSignals(True)
        self.spin.setValue(idx)
        self.spin.blockSignals(False)
        # instant preview from cache while the real frame decodes
        if self.cache is not None and self.cache.get(idx) is None:
            near = self.cache.nearest(idx)
            if near is not None:
                self.canvas.set_frame(near[1])
        if self.seek is not None and self.state != TRACKING:
            self.seek.request(idx)
        self._refresh_overlay()
        self._refresh_companions()
        self.timeline.set_current(idx)
        self._update_frame_label()
        self._update_track_button()
        self._refresh_view3d()
        self._refresh_body_view()

    def _on_seek_frame(self, view: int, idx: int, rgb: np.ndarray):
        if self.project is None or not (0 <= view < len(self._views)):
            return
        if view == self.project.active:
            if idx == self.current and self.state != TRACKING:
                self.canvas.set_frame(rgb)
                # the side-by-side window shows the same instant, and the
                # decoded frame only exists here -- a seek is what fills it
                self._refresh_body_view()
            return
        # a companion view: only the frame we last asked it for (a scrub leaves
        # older replies in flight, and they would show the wrong instant)
        if idx == self._views[view].want_frame:
            cv = self.grid.canvas(view)
            if cv is not None:
                cv.set_frame(rgb)

    def _refresh_companions(self):
        """Put every other camera on the instant the playhead is on, with its
        own tracked points drawn. Skipped during a run — the tracking worker
        needs the decode bandwidth, and the companion views would only show
        frames nobody is looking at."""
        p = self.project
        if p is None or p.n_views < 2 or self.state == TRACKING:
            return
        visible = set(self.grid.visible_indices())
        for i in p.others():
            rt = self._views[i]
            cv = self.grid.canvas(i)
            if cv is None:
                continue
            if i not in visible:
                # hidden (solo): give the decoder thread and its VideoCapture
                # back — at 15 cameras that is 14 open 4K captures for nothing
                rt.want_frame = None
                rt.stop()
                continue
            f = p.map_frame(p.active, i, self.current)
            self.grid.set_caption(i, caption_for(p.name(i), f, p.offsets[i], rt.n_frames))
            s = p.sessions[i]
            if f is None:                    # this camera was not recording yet
                rt.want_frame = None
                cv.set_points(np.zeros((0, 2), np.float32), np.zeros(0, bool), [], None)
                cv.set_mask(None)
                cv.set_midline(None)
                continue
            f = min(f, rt.n_frames - 1)
            if rt.want_frame != f:
                rt.want_frame = f
                cached = rt.cache.get(f)
                if cached is not None:
                    cv.set_frame(cached)
                else:
                    # decoder threads are started on FIRST NEED, not up front: at
                    # 15 cameras that is 15 idle threads each holding a 4K
                    # VideoCapture open for views the user may never look at
                    if rt.seek is None:
                        self._start_seek_service(i)
                    rt.seek.request(f)
            if s.animal is not None and s.masks is not None and self.btn_mask.isChecked():
                cv.set_mask(s.masks.contours.get(f), s.animal.color, self._mask_opacity)
                cv.set_midline(s.masks.midline.get(f)
                               if self.act_show_midline.isChecked() else None, s.animal.color)
            else:
                cv.set_mask(None)
                cv.set_midline(None)
            cv.set_bones(s.bones() if self.act_show_bones.isChecked() else [])
            cv.set_points(s.positions_at(f), s.visibility[f], s.points, None,
                          occluded=s.occluded[f])
        rt_a = self._rt
        if rt_a is not None:
            self.grid.set_caption(p.active, caption_for(p.name(p.active), self.current,
                                                        p.offsets[p.active], rt_a.n_frames))

    def _toggle_play(self, on: bool):
        self.btn_play.setIcon(icons.pause() if on else icons.play())
        if on and self.state == READY:
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def _play_tick(self):
        if self.current >= self.n_frames - 1 or self.state != READY:
            self.btn_play.setChecked(False)
            return
        self._goto(self.current + 1)

    def _update_frame_label(self):
        last = self.session.last_tracked_frame() if self.session else None
        extra = f"   last tracked: {last}" if last is not None else ""
        self._frame_label.setText(f"frame {self.current} / {self.n_frames - 1}{extra}")

    # ---------------------------------------------------------------- points

    def _on_add_mode(self, on: bool):
        self.canvas.set_place_mode(on)
        # a ball is placed with ONE click, whatever the region shape (I48)
        self.canvas.set_click_only(on and self._place_kind == "ball")
        if not on:
            self._place_kind = "point"
        elif self._place_kind == "ball":
            self.statusBar().showMessage(
                "Ball marker: click the middle of the ball. SAM outlines it on every frame and the "
                "fitted circle's centre becomes the point. Esc cancels", 9000)
        elif self._region_shape == "polygon":
            self.statusBar().showMessage(
                "Polygon region: click each corner, Enter (or a double-click) closes it; a single "
                "point needs Add ▾ → Region shape: circle first. Esc cancels", 9000)
        else:
            self.statusBar().showMessage(
                "Place a point: click where it should track — or drag to outline a "
                "region (circle or rectangle, chosen under Add ▾). Esc cancels (N toggles)", 8000)

    def _arm_ball(self):
        """Add ▾ → Ball marker: the next click segments a ball with SAM."""
        if self.session is None or self.state != READY:
            return
        self._place_kind = "ball"
        if self.btn_animal.isChecked():
            self.btn_animal.setChecked(False)
        if self.btn_add.isChecked():
            self._on_add_mode(True)        # already armed: just re-word the hint
        else:
            self.btn_add.setChecked(True)

    def _on_add(self, x: float, y: float):
        if self.session is None:
            return
        kind = self._place_kind
        self.btn_add.setChecked(False)  # one placement per arm (also resets _place_kind)
        s = self.session
        if kind == "ball":
            # a selected ball with no data here is CONTINUED by the click (the
            # same rule as points): the click is a new SAM prompt on this frame
            if (self.selected is not None and self.selected < s.n_points and s.points[self.selected].is_ball
                    and not s.tracked[self.current, self.selected]):
                self._undo_snap = s.snapshot()  # (I123)
                self.act_undo.setEnabled(True)
                s.add_ball_prompt(self.selected, self.current, x, y)
                self._refresh_overlay()
                self._update_track_button()
                self.statusBar().showMessage(
                    f"{s.points[self.selected].name}: SAM will be prompted here on frame {self.current} — "
                    "press Track to continue it", 7000)
                return
            self._undo_snap = s.snapshot()      # adding is one undo step too (I123)
            self.act_undo.setEnabled(True)
            pid = s.add_ball(self.current, x, y)
            self.selected = pid
            self._refresh_point_list()
            self._refresh_overlay()
            self._apply_state()
            self.statusBar().showMessage(
                f"Added ball marker {s.points[pid].name} at ({x:.0f}, {y:.0f}) on frame {self.current}. "
                "Add the other balls the same way (Add ▾ → Ball marker), then press Track: SAM outlines "
                "each ball and the fitted circle's centre is its point", 9000)
            return
        # If the selected point has no position on this frame (it left the
        # frame, or this frame is beyond its track), a click CONTINUES that
        # point here instead of creating a new one — select a point in the
        # panel, click where it reappeared, then Track. Never a silhouette-
        # derived landmark: it has no data before a run, so N + click placed it
        # by hand instead of adding the point asked for (I69).
        if (self.selected is not None and self.selected < s.n_points
                and not s.points[self.selected].derived
                and not s.tracked[self.current, self.selected]):
            self._on_place(self.selected, x, y)
            self.statusBar().showMessage(
                f"{s.points[self.selected].name} placed here (same point, no new point created) — "
                "press Track to continue it, or Esc first if you wanted a new point", 7000)
            return
        # (I123) without a snapshot here, Ctrl+Z right after adding restored an OLDER
        # snapshot: the new point vanished AND the previous run or edit was undone
        self._undo_snap = s.snapshot()
        self.act_undo.setEnabled(True)
        derived_sel = (self.selected is not None and self.selected < s.n_points
                       and s.points[self.selected].derived)
        pid = s.add_point(self.current, x, y)
        self.selected = pid
        self._refresh_point_list()
        self._refresh_overlay()
        self._apply_state()
        self.statusBar().showMessage(
            f"Added {s.points[pid].name} at ({x:.0f}, {y:.0f}) on frame {self.current}"
            + (" — the landmark that was selected is derived from the silhouette and cannot be placed by "
               "hand, so a NEW point was added" if derived_sel else ""), 6000 if derived_sel else 4000)

    def _on_add_group(self, cx: float, cy: float, radius: float):
        """Circle drag: track a region as one point (robust center of a member
        constellation). Future region shapes (rectangle, polygon) plug in here."""
        if self.session is None:
            return
        self.btn_add.setChecked(False)  # one placement per arm
        s = self.session
        self._undo_snap = s.snapshot()          # (I123)
        self.act_undo.setEnabled(True)
        pid = s.add_point(self.current, cx, cy, kind="group", radius=radius)
        self.selected = pid
        self._refresh_point_list()
        self._refresh_overlay()
        self._apply_state()
        self.statusBar().showMessage(
            f"Added region {s.points[pid].name} (r={radius:.0f} px) at frame "
            f"{self.current} — it tracks as one point: the region's fitted center", 6000)

    # ---------------------------------------------------------------- animal

    def _on_animal_mode(self, on: bool):
        if on and self.btn_add.isChecked():
            self.btn_add.setChecked(False)
        self.canvas.set_animal_mode(on)
        if on:
            self.statusBar().showMessage(
                "Segment tool: click the segment (Shift+click = not the segment, drag = box). "
                "S or Esc when done", 10000)
            if not self._animal_hint_shown:
                self._animal_hint_shown = True
                self.toast.show_message(
                    "Click the segment once. Its silhouette appears in a moment — add a "
                    "Shift+click on anything wrongly included, or another click on a "
                    "missed part. Then press <b>Track</b>.", "info", 9000)

    def _animal_status_text(self) -> str:
        s = self.session
        if s is None or s.animal is None:
            return ("No segment yet. Press <b>S</b> and click it on the video; the silhouette "
                    "then follows it when you track, and tail tip / midline / feet landmarks "
                    "can be derived from it.")
        a, m = s.animal, s.masks
        n_masked = m.n_masked() if m is not None else 0
        pf = a.prompt_frames()
        here = " (this frame)" if self.current in pf else ""
        # the name lives in the row above; this line is the numbers
        txt = (f"{a.n_prompts()} click(s)/box(es) on {len(pf)} frame(s){here}; "
               f"silhouette on {n_masked:,} of {s.n_frames:,} frames.")
        if s.derived_pids():
            txt += f" {len(s.derived_pids())} landmark(s) derive from it."
        if n_masked == 0 and pf:
            txt += " Press Track to segment the video."
        return txt

    def _refresh_animal_panel(self):
        """The SEGMENT section: one row for the segment itself (when there is
        one) plus the status line. Rebuilt like the point list, signals blocked
        so the rebuild is not mistaken for a user edit."""
        self.animal_label.setText(self._animal_status_text())
        s = self.session
        self.animal_list.blockSignals(True)
        self.animal_list.clear()
        has = s is not None and s.animal is not None
        self.animal_list.setVisible(has)
        if has:
            a = s.animal
            n_masked = s.masks.n_masked() if s.masks is not None else 0
            item = QListWidgetItem(a.name)
            pm = QPixmap(14, 14)
            pm.fill(Qt.transparent)
            from PySide6.QtGui import QBrush, QPainter, QPen
            painter = QPainter(pm)
            painter.setRenderHint(QPainter.Antialiasing)
            # a filled blob, not the points' square: the segment is an area
            col = QColor(*a.color)
            painter.setPen(QPen(col, 1.5))
            painter.setBrush(QBrush(QColor(a.color[0], a.color[1], a.color[2], 140)))
            painter.drawEllipse(1, 2, 12, 10)
            painter.end()
            item.setIcon(QIcon(pm))
            item.setFlags(item.flags() | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if self.btn_mask.isChecked() else Qt.Unchecked)
            item.setToolTip(
                f"{a.name}: silhouette on {n_masked:,} of {s.n_frames:,} frames, from "
                f"{a.n_prompts()} click(s)/box(es).\n"
                "Checkbox: show / hide the silhouette. Double-click: rename. "
                "Right-click: clear a stretch, jump to the silhouette, or remove the segment.")
            if n_masked == 0:
                item.setForeground(QColor(theme.TEXT_DIM))
            self.animal_list.addItem(item)
            # exactly one row tall at whatever font / DPI this machine uses
            self.animal_list.setFixedHeight(
                self.animal_list.sizeHintForRow(0) + 2 * self.animal_list.frameWidth() + 2)
        self.animal_list.blockSignals(False)

    def _on_animal_item_changed(self, item):
        """The segment row was edited: the checkbox toggles the silhouette
        overlay, the text renames the segment."""
        s = self.session
        if s is None or s.animal is None:
            return
        want = item.checkState() == Qt.Checked
        if want != self.btn_mask.isChecked():
            self.btn_mask.setChecked(want)       # drives _refresh_overlay + this panel
            return
        desired = item.text().strip()
        if desired and desired != s.animal.name:
            name = s.rename_animal(desired)
            self.timeline.refresh()
            self.statusBar().showMessage(f"Segment renamed to {name}", 4000)
        self._refresh_animal_panel()
        self._refresh_overlay()

    # ---- the segment row's context menu ----------------------------------

    def _build_animal_menu(self):
        """Menu + action map, built separately so tests can drive the entries
        (the canvas point menu uses the same pattern)."""
        s = self.session
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        acts = {}
        n_masked = s.masks.n_masked() if (s is not None and s.masks is not None) else 0
        acts["rename"] = menu.addAction("Rename segment…")
        a_mask = menu.addAction("Show its silhouette")
        a_mask.setCheckable(True)
        a_mask.setChecked(self.btn_mask.isChecked())
        acts["show_mask"] = a_mask
        a_mid = menu.addAction("Show its midline")
        a_mid.setCheckable(True)
        a_mid.setChecked(self.act_show_midline.isChecked())
        acts["show_midline"] = a_mid
        menu.addSeparator()
        acts["first"] = menu.addAction("Jump to its first silhouette")
        acts["last"] = menu.addAction("Jump to its last silhouette")
        for k in ("first", "last"):
            acts[k].setEnabled(bool(n_masked > 0))
        menu.addSeparator()
        a_here = menu.addAction(f"Clear the silhouette on frame {self.current}")
        a_here.setEnabled(bool(s is not None and s.masks is not None and s.masks.has(self.current)))
        acts["clear_here"] = a_here
        sel = self.timeline.sel_range
        a_win = menu.addAction(
            f"Clear the silhouettes in frames {sel[0]}–{sel[1]}" if sel else
            "Clear the silhouettes in the selected frame window")
        a_win.setEnabled(bool(sel is not None and n_masked > 0))
        a_win.setToolTip("Shift+drag across the timeline to select a frame window first")
        acts["clear_window"] = a_win
        a_all = menu.addAction(f"Clear ALL {n_masked:,} silhouettes (keep the clicks)")
        a_all.setEnabled(bool(n_masked > 0))
        a_all.setToolTip("The clicks stay, so pressing Track segments the video again")
        acts["clear_all"] = a_all
        menu.addSeparator()
        acts["remove"] = menu.addAction("Remove the segment (clicks and silhouettes)")
        return menu, acts

    def _animal_menu(self, pos):
        if self.session is None or self.session.animal is None or self.state != READY:
            return
        menu, acts = self._build_animal_menu()
        chosen = menu.exec(self.animal_list.viewport().mapToGlobal(pos))
        menu.deleteLater()          # release the shown menu (see canvas._context_menu)
        self._animal_menu_action(chosen, acts)

    def _animal_menu_action(self, chosen, acts):
        """Apply one entry of the segment menu (the exec result, or an action
        picked by a test)."""
        s = self.session
        if chosen is None or s is None or s.animal is None:
            return
        if chosen is acts["rename"]:
            name, ok = QInputDialog.getText(self, "Rename segment", "Name:", text=s.animal.name)
            if ok and name.strip():
                applied = s.rename_animal(name)
                self.timeline.refresh()
                self._refresh_animal_panel()
                self.statusBar().showMessage(f"Segment renamed to {applied}", 4000)
        elif chosen is acts["show_mask"]:
            self.btn_mask.setChecked(acts["show_mask"].isChecked())
        elif chosen is acts["show_midline"]:
            self.act_show_midline.setChecked(acts["show_midline"].isChecked())
            self._refresh_overlay()
        elif chosen is acts["first"] or chosen is acts["last"]:
            fr = s.mask_frames()
            if len(fr):
                self._goto(int(fr[-1] if chosen is acts["last"] else fr[0]))
        elif chosen is acts["clear_here"]:
            self._clear_masks_window(self.current, self.current)
        elif chosen is acts["clear_window"]:
            sel = self.timeline.sel_range
            if sel is not None:
                self._clear_masks_window(sel[0], sel[1])
        elif chosen is acts["clear_all"]:
            n = s.masks.n_masked() if s.masks is not None else 0
            if QMessageBox.question(
                    self, "Clear silhouettes",
                    f"Remove {s.animal.name}'s silhouettes on {n:,} frames?\n\nIts clicks stay, so "
                    "pressing Track segments the video again. Ctrl+Z undoes this.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
                self._clear_masks_window(0, s.n_frames - 1)
        elif chosen is acts["remove"]:
            self._clear_animal()
        if self.session is not None:
            self.btn_clear_animal.setEnabled(self.state == READY and self.session.animal is not None)

    def _current_rgb(self):
        return self.cache.get(self.current) if self.session is not None else None

    def _on_animal_click(self, x: float, y: float, positive: bool):
        s = self.session
        if s is None or self.state != READY:
            return
        s.ensure_animal()
        s.animal.add_click(self.current, x, y, positive)
        self._after_prompt_change("excluded" if not positive else "marked")

    def _on_animal_box(self, x0: float, y0: float, x1: float, y1: float):
        s = self.session
        if s is None or self.state != READY:
            return
        s.ensure_animal()
        s.animal.set_box(self.current, (x0, y0, x1, y1))
        self._after_prompt_change("boxed")

    def _on_prompt_remove(self, index: int):
        s = self.session
        if s is None or s.animal is None:
            return
        clicks = s.animal.prompts.get(self.current, [])
        if 0 <= index < len(clicks):
            del clicks[index]
            if not clicks:
                s.animal.prompts.pop(self.current, None)
            s._touch()
            if s.animal.has_prompt(self.current):
                self._after_prompt_change("updated")
            else:
                s.clear_masks(self.current, self.current)
                self._refresh_overlay()
                self._refresh_animal_panel()
                self._apply_state()
                self.statusBar().showMessage("Click removed — no prompt left on this frame", 4000)

    def _after_prompt_change(self, verb: str):
        self._refresh_overlay()
        self._refresh_animal_panel()
        self._apply_state()
        self.statusBar().showMessage(f"Segment {verb} on frame {self.current} — computing its "
                                     "silhouette…", 5000)
        self._preview_mask()

    def _preview_mask(self):
        """Segment the current frame from its prompts in the background."""
        s = self.session
        if s is None or s.animal is None or not s.animal.has_prompt(self.current):
            return
        if self._preview is not None and self._preview.isRunning():
            self._preview_again = True   # rerun with the newest prompts when it finishes
            return
        head = None
        hp = s.head_pid()
        if hp is not None and s.tracked[self.current, hp]:
            head = s.tracks[self.current, hp]
        clicks = list(s.animal.prompts.get(self.current, []))
        box = s.animal.boxes.get(self.current)
        w = _MaskPreviewWorker(self.info.path, self.current, self._current_rgb(),
                               (self.info.width, self.info.height), clicks, box,
                               self._seg_backend, head)
        w.target_session = s            # the camera may change while SAM works (I66)
        w.loading.connect(self._on_seg_loading)
        w.done.connect(self._on_preview_done)
        w.error.connect(self._on_preview_error)
        self._preview = w
        self._preview_again = False
        QApplication.setOverrideCursor(Qt.BusyCursor)
        w.start()

    def _on_seg_loading(self, backend: str):
        from cotracker_app.segmenter import loaded_backends
        if backend in loaded_backends() or self._loading_dialog is not None:
            return
        label = BACKENDS.get(backend, BACKENDS[DEFAULT_BACKEND])[2]
        if seg_is_cached(backend):
            text = f"Loading the segmentation model ({label}) onto the GPU…"
        else:
            text = (f"Downloading the segmentation model ({label}) — first use only.\n"
                    "It is stored inside the tool folder (models/hf).")
        dlg = QProgressDialog(text, None, 0, 0, self)
        dlg.setWindowTitle("Preparing model")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(300)
        dlg.setValue(0)
        self._loading_dialog = dlg

    def _close_loading_dialog(self):
        if self._loading_dialog is not None:
            self._loading_dialog.close()
            self._loading_dialog = None

    def _on_preview_done(self, summ: dict):
        QApplication.restoreOverrideCursor()
        self._close_loading_dialog()
        target = getattr(self._preview, "target_session", None) if self._preview is not None else None
        if self._preview is not None:
            self._preview.wait(2000)
        self._preview = None
        s = target if target is not None else self.session
        if s is None or s.animal is None:
            return
        f = int(summ["frame"])
        s.write_mask_summaries([summ])
        if s is not self.session:     # finished after a camera switch: stored in ITS camera (I66)
            self.statusBar().showMessage(
                f"The silhouette for frame {f} was stored in the camera it was clicked in", 6000)
            return
        if f == self.current:
            self._refresh_overlay()
        self._refresh_animal_panel()
        self._apply_state()
        if summ["area"] > 0:
            self.statusBar().showMessage(
                f"Silhouette on frame {f}: {summ['area']:,} px². Looks right? Press Track. "
                "Wrong part included? Shift+click it; missed part? click it.", 8000)
        else:
            self.toast.show_message(
                "No silhouette found from that click. Try clicking nearer the middle of the "
                "segment, or drag a box around it.", "warn", 7000)
        if self._preview_again:
            self._preview_mask()

    def _on_preview_error(self, tb: str):
        QApplication.restoreOverrideCursor()
        self._close_loading_dialog()
        if self._preview is not None:
            self._preview.wait(2000)
        self._preview = None
        hint = _model_error_hint(tb)
        self.toast.show_message("Segmentation failed. " + (hint or "See the details dialog."),
                                "error", 10000)
        QMessageBox.critical(self, "Segmentation failed",
                             (hint + "\n\nDetails:\n\n" if hint else "") + tb[-1500:])

    def _clear_animal(self):
        s = self.session
        if s is None or s.animal is None or self.state != READY:
            return
        n = s.masks.n_masked() if s.masks is not None else 0
        if QMessageBox.question(
                self, "Clear segment",
                f"Remove the segment's {s.animal.n_prompts()} click(s)/box(es) and its "
                f"silhouettes on {n:,} frames?\n\nPoints and their tracks are kept; "
                "silhouette-derived landmarks keep their data until you track again.\n\n"
                "This cannot be undone with Ctrl+Z.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        s.clear_animal()
        # an older snapshot would restore something else and not the segment (I124)
        self._undo_snap = None
        self.act_undo.setEnabled(False)
        self.btn_animal.setChecked(False)
        self.timeline.set_session(s)   # lane layout changes
        self._refresh_overlay()
        self._refresh_animal_panel()
        self._apply_state()
        self.statusBar().showMessage("Segment cleared", 4000)

    def _clear_masks_window(self, f0: int, f1: int):
        """Drop the segment's silhouettes inside [f0, f1], leaving every tracked
        point alone (timeline clear_masks_requested)."""
        self._clear_window(f0, f1, [], do_points=False, do_masks=True)
        self._refresh_animal_panel()

    def _on_onbody_toggled(self, on: bool):
        if self.session is None:
            return
        if on:
            self.statusBar().showMessage(
                "Body constraint ON: small slips past the silhouette's edge are nudged back; a tracked point "
                "that clearly leaves the segment stops the run there (applies to the next run; right-click a "
                "point → May leave the segment to exempt it)", 8000)
        else:
            self.statusBar().showMessage("Body constraint off: points may drift off the segment", 5000)

    # ------------------------------------------------ view options (display)

    def _display_filter_key(self) -> str:
        for k, act in self._filter_acts.items():
            if act.isChecked():
                return k
        return "none"

    def _set_display_filter(self, key: str, announce: bool = True):
        key = key if key in DISPLAY_FILTERS else "none"
        self._filter_acts[key].setChecked(True)
        for cv in self.grid.canvases:
            cv.set_display_filter(key)
        if announce:
            label = self._filter_acts[key].text()
            self.statusBar().showMessage(
                f"Display filter: {label} — what you see only; tracking and exports use the original "
                "pixels", 5000)
        if self.session is not None:
            self.session.dirty = True

    def _set_trail_len(self, n: int, announce: bool = True):
        self._trail_len = int(n)
        if n in self._trail_acts:
            self._trail_acts[n].setChecked(True)
        for cv in self.grid.canvases:
            cv.set_trails(self._trail_len, self._trail_future)
        self._refresh_overlay()
        if announce and self.session is not None:
            self.session.dirty = True

    def _on_trail_future(self, on: bool):
        self._trail_future = bool(on)
        for cv in self.grid.canvases:
            cv.set_trails(self._trail_len, self._trail_future)
        self._refresh_overlay()

    def _set_region_shape(self, key: str, announce: bool = True):
        key = key if key in REGION_SHAPES else "circle"
        self._region_shape = key
        self._shape_acts[key].setChecked(True)
        for cv in self.grid.canvases:
            cv.set_region_shape(key)
        if announce:
            how = {"circle": "drag from the centre outward",
                   "rect": "drag a box around the object",
                   "polygon": "click each corner, then Enter (double-click also closes; Esc cancels)"}[key]
            self.statusBar().showMessage(f"Region shape: {key} — press N, then {how}", 6000)

    def _on_add_region(self, shape: str, outline):
        """Rectangle / polygon region from an armed gesture: tracked as ONE
        point (the fitted centre of a member constellation sampled inside it)."""
        if self.session is None:
            return
        self.btn_add.setChecked(False)  # one placement per arm
        pts = np.asarray(outline, np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return
        c = pts.mean(axis=0)
        radius = float(np.max(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])))
        s = self.session
        self._undo_snap = s.snapshot()          # (I123)
        self.act_undo.setEnabled(True)
        pid = s.add_point(self.current, float(c[0]), float(c[1]), kind="group", radius=radius,
                          shape=shape, outline=pts.tolist())
        self.selected = pid
        self._refresh_point_list()
        self._refresh_overlay()
        self._apply_state()
        self.statusBar().showMessage(
            f"Added {shape} region {s.points[pid].name} ({len(pts)} corners) at frame {self.current} "
            "— it tracks as one point: the region's fitted centre", 6000)

    # ------------------------------------------------ hidden (occluded) marks

    def _on_occluded_toggled(self, pid: int, on: bool):
        s = self.session
        if s is None or pid >= s.n_points or self.state != READY:
            return
        self._undo_snap = s.snapshot()          # Ctrl+Z takes the mark back, nothing older (I71)
        self.act_undo.setEnabled(True)
        s.set_occluded(self.current, pid, on)
        self._refresh_overlay()
        name = s.points[pid].name
        self.statusBar().showMessage(
            f"{name}: marked HIDDEN on frame {self.current} — kept, not exported, not used for 3D "
            "(Shift+X again to unmark)" if on else f"{name}: shown again on frame {self.current}", 5000)

    def _toggle_hidden_here(self):
        s = self.session
        if s is None or self.state != READY:
            return
        pid = self.selected
        if pid is None or pid >= s.n_points:
            self.statusBar().showMessage("Select a point first, then Shift+X marks it hidden on this frame", 4000)
            return
        self._on_occluded_toggled(pid, not bool(s.occluded[self.current, pid]))

    def _occlude_window(self, f0: int, f1: int, pids, on: bool):
        """Timeline: mark / unmark a frame window of the covered lanes as hidden."""
        s = self.session
        if s is None or self.state != READY:
            return
        use = self._resolve_window_pids(
            f0, f1, pids,
            verb="mark as hidden (kept, but not exported or used for 3D)" if on else "unmark as hidden",
            title="Mark all points hidden?" if on else "Unmark all points?")
        if not use:
            return
        self._undo_snap = s.snapshot()
        n = s.set_occluded_window(use, f0, f1, on)
        self.act_undo.setEnabled(True)
        self.timeline.clear_selection()
        self._refresh_overlay()
        names = ", ".join(s.points[p].name for p in use[:4]) + (" …" if len(use) > 4 else "")
        self.statusBar().showMessage(
            f"{'Marked' if on else 'Unmarked'} {n} cells hidden: {names}, frames {f0}–{f1} "
            "(Ctrl+Z undoes)", 6000)

    # ------------------------------------------------ notes + annotator

    def _default_annotator(self) -> str:
        try:
            return str(QSettings("Kinetrace", "Kinetrace").value("annotator", "") or "")
        except Exception:      # noqa: BLE001
            return ""

    def _apply_annotator(self, name: str):
        name = (name or "").strip()
        try:
            QSettings("Kinetrace", "Kinetrace").setValue("annotator", name)
        except Exception:      # noqa: BLE001
            pass
        if self.session is not None:
            self.session.annotator = name
            self.session.dirty = True

    def _set_annotator(self):
        cur = self.session.annotator if self.session is not None else self._default_annotator()
        name, ok = QInputDialog.getText(self, "Annotator", "Your name (recorded on events and notes):",
                                        text=cur)
        if ok:
            self._apply_annotator(name)
            self.statusBar().showMessage(f"Annotator: {name.strip() or '(none)'}", 4000)

    def _edit_note(self, frame: int | None = None):
        """Shift+N / timeline: add, edit or clear the note on a frame."""
        s = self.session
        if s is None or self.state != READY:
            return
        f = self.current if frame is None else int(frame)
        cur = s.notes.get(f, {}).get("text", "")
        text, ok = QInputDialog.getMultiLineText(
            self, "Frame note", f"Note for frame {f} (empty removes it):", cur)
        if not ok:
            return
        s.set_note(f, text)
        self._on_events_changed()
        self.statusBar().showMessage(
            f"Note {'saved' if text.strip() else 'removed'} at frame {f}" +
            (f" ({s.notes[f]['author']})" if text.strip() and s.notes.get(f, {}).get("author") else ""),
            4000)

    # ------------------------------------------------ navigation helpers

    def _goto_manual_step(self, forward: bool):
        """. / , : the next / previous hand-placed frame of the selected point."""
        s = self.session
        if s is None or self.state != READY:
            return
        pid = self.selected
        if pid is None or pid >= s.n_points:
            self.statusBar().showMessage("Select a point first — . and , walk its hand-placed frames", 3000)
            return
        target = s.next_manual_frame(pid, self.current, forward)
        if target is None:
            self.statusBar().showMessage(
                f"{s.points[pid].name}: no hand-placed frame {'after' if forward else 'before'} this one", 3000)
            return
        self._goto(target)
        fr = s.manual_frames(pid)
        k = int(np.searchsorted(fr, target)) + 1
        self.statusBar().showMessage(f"{s.points[pid].name}: hand-placed frame {k} of {len(fr)}", 3000)

    def _goto_low_conf(self, forward: bool):
        """J / Shift+J: jump to the next / previous low-confidence stretch (the
        red runs on the timeline) of the selected points, or of every point."""
        s = self.session
        if s is None or self.state != READY:
            return
        pids = self._selected_pids() or ([self.selected] if self.selected is not None else None)
        run = s.next_low_conf(self.current, forward, pids)
        if run is None:
            self.statusBar().showMessage(
                f"No low-confidence stretch {'after' if forward else 'before'} this frame"
                + (" for the selected points" if pids else ""), 3000)
            return
        a, b = run
        self._goto(a)
        self.statusBar().showMessage(
            f"Low-confidence stretch: frames {a}–{b} ({b - a + 1} frames) — check and correct here, "
            "then Track", 5000)

    def _on_free_toggled(self, pid: int, free: bool):
        s = self.session
        if s is None or pid >= s.n_points:
            return
        s.points[pid].free = bool(free)
        s.dirty = True
        name = s.points[pid].name
        self.statusBar().showMessage(
            f"{name}: may leave the segment (not constrained to the silhouette)" if free
            else f"{name}: kept on the segment while the Body toggle is on (if it clearly leaves it, "
                 "the run stops there)", 6000)

    def _on_source_change(self, pid: int, spec: str):
        s = self.session
        if s is None or pid >= s.n_points or self.state != READY:
            return
        meta = s.points[pid]
        # a switch to a silhouette spec, or away from one, erases the point's
        # track (set_source clears it): that used to happen silently, with no
        # undo, and autosave wrote the loss to disk (I45)
        will_clear = (bool(spec) and not (meta.source == "silhouette" and meta.spec == spec)) or \
            (not spec and meta.source == "silhouette")
        n_data = int(s.tracked[:, pid].sum())
        if will_clear and n_data:
            n_hand = int((s.manual[:, pid] & s.tracked[:, pid]).sum())
            what = (f"derive {meta.name} from the silhouette ({spec})" if spec
                    else f"track {meta.name} by its appearance")
            if QMessageBox.question(
                    self, "Replace this point's track?",
                    f"To {what}, its current track is erased: {n_data} frame(s)"
                    + (f", {n_hand} of them placed by hand" if n_hand else "") + ".\n\n"
                    "Ctrl+Z brings it back. Go ahead?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                self._refresh_point_list()
                return
        if will_clear and n_data:
            self._undo_snap = s.snapshot()
            self.act_undo.setEnabled(True)
        if spec:
            if s.animal is None:
                self.toast.show_message(
                    f"<b>{meta.name}</b> will derive from the animal's silhouette — press "
                    "<b>S</b> and click the animal first, then Track.", "info", 8000)
            s.set_source(pid, "silhouette", spec)
            self.statusBar().showMessage(
                f"{meta.name}: derived from the silhouette ({spec}) — fills in when you track", 6000)
        else:
            s.set_source(pid, "track")
            self.statusBar().showMessage(
                f"{meta.name}: tracked by appearance — select it, press N and click to seed it", 6000)
        self._refresh_point_list()
        self._refresh_overlay()
        self._apply_state()

    # -------------------------------------------------------------- skeleton

    def _refresh_skeleton_menu(self):
        for menu in (self.m_skeleton, self.m_skeleton_btn):
            menu.clear()
            for t in all_templates():
                act = menu.addAction(f"Use template: {t['name']}  ({len(t['landmarks'])} landmarks)")
                act.setToolTip(t.get("note", "") or ", ".join(t["landmarks"]))
                act.triggered.connect(lambda _=False, tt=t: self._apply_skeleton_template(tt))
            menu.addSeparator()
            menu.addAction("Custom skeleton…", self._custom_skeleton_dialog)
            from cotracker_app.skeletons import user_template_problems
            probs = user_template_problems()
            if probs:           # a broken file in skeletons/ is said, not silently skipped (I62)
                menu.addAction(f"{len(probs)} problem(s) in the skeletons/ folder…",
                               lambda pr=tuple(probs): QMessageBox.warning(
                                   self, "Skeleton files", "\n\n".join(pr)))
            if self.session is not None and self.session.skeleton:
                menu.addSeparator()
                menu.addAction(f"Forget skeleton “{self.session.skeleton.get('name', '')}” "
                               "(keep the points)", self._clear_skeleton)
            menu.setToolTipsVisible(True)

    def _apply_skeleton_template(self, t: dict):
        s = self.session
        if s is None or self.state != READY:
            return
        new = s.apply_skeleton(t)
        self._refresh_point_list()
        self._refresh_overlay()
        self._refresh_skeleton_menu()
        self._refresh_animal_panel()
        self._apply_state()
        QTimer.singleShot(0, self._fit_timeline_height)
        derived = [s.points[p].name for p in new if s.points[p].derived]
        placed = [s.points[p].name for p in new if not s.points[p].derived]
        msg = f"Skeleton <b>{t['name']}</b>: {len(new)} landmark(s) added."
        if placed:
            msg += (f" Select one in the panel, press <b>N</b> and click it on the video "
                    f"(start with <b>{t.get('head', placed[0])}</b>).")
        if derived:
            msg += f" {len(derived)} come from the animal's silhouette when you track."
        self.toast.show_message(msg, "success", 12000)
        self.statusBar().showMessage(t.get("note", ""), 10000)

    def _custom_skeleton_dialog(self):
        if self.session is None or self.state != READY:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Custom skeleton")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        name = QLineEdit("my skeleton")
        head = QLineEdit("head")
        marks = QPlainTextEdit("head\nneck\ntail_base\ntail_tip = tip")
        marks.setToolTip("One landmark per line. Add '= tip' (tail tip), '= midline:0.5' (0 = head, 1 = tail "
                         "tip), '= centroid', '= ext:L' / '= ext:R' (wing tips) or '= ext:FL' / FR / HL / HR "
                         "(feet) to derive it from the silhouette instead of tracking. Left and right are the "
                         "animal's own as seen from ABOVE: filmed from below they come out swapped.")
        bones = QPlainTextEdit("head - neck\nneck - tail_base\ntail_base - tail_tip")
        bones.setToolTip("Optional: one 'a - b' pair per line, drawn as lines between landmarks.")
        form.addRow("Name", name)
        form.addRow("Head landmark", head)
        form.addRow("Landmarks", marks)
        form.addRow("Bones", bones)
        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        dlg.resize(460, 480)
        if dlg.exec() != QDialog.Accepted:
            return
        landmarks, derived = [], {}
        for line in marks.toPlainText().splitlines():
            line = line.strip()
            if not line:
                continue
            if "=" in line:
                nm, spec = (p.strip() for p in line.split("=", 1))
                if nm:
                    landmarks.append(nm)
                    if spec:
                        derived[nm] = spec
            else:
                landmarks.append(line)
        if not landmarks:
            self.toast.show_message("A skeleton needs at least one landmark.", "warn")
            return
        pairs = []
        for line in bones.toPlainText().splitlines():
            if "-" in line:
                a, b = (p.strip() for p in line.split("-", 1))
                if a in landmarks and b in landmarks:
                    pairs.append([a, b])
        t = {"name": name.text().strip() or "custom", "head": head.text().strip() or landmarks[0],
             "landmarks": landmarks, "bones": pairs, "derived": derived, "note": ""}
        from cotracker_app.skeletons import validate_template
        t, problems = validate_template(t)
        if problems:
            # a typo such as midline:50 used to become a landmark that never fills (I62)
            QMessageBox.warning(self, "Skeleton rules corrected",
                                "Some of the rules could not be used and were left out:\n\n• "
                                + "\n• ".join(problems)
                                + "\n\nThose landmarks are tracked by appearance instead (select, N, click).")
        try:
            save_user_template(t)
        except OSError as e:
            self.statusBar().showMessage(f"Template not saved to skeletons/: {e}", 6000)
        self._apply_skeleton_template(t)

    def _clear_skeleton(self):
        if self.session is None:
            return
        self.session.clear_skeleton()
        self._refresh_skeleton_menu()
        self._refresh_overlay()
        self.statusBar().showMessage("Skeleton forgotten — the points remain", 4000)

    # ---------------------------------------------------------------- events

    def _mark_event(self):
        """Two-press flow: first E marks the start, second E marks the end."""
        if self.state != READY or self.session is None:
            return
        if self._pending_event is None:
            self._pending_event = self.current
            self.timeline.set_pending_event(self.current)
            self.statusBar().showMessage(
                f"Event start marked at frame {self.current} — scrub to the end frame "
                "and press E again (Esc cancels)", 8000)
            return
        start, end = self._pending_event, self.current
        lo, hi = min(start, end), max(start, end)
        s = self.session
        # existing event TYPES, most recent first: one click marks a repeat
        # occurrence; typing a new name creates a new type
        names = list(dict.fromkeys(e.name for e in reversed(s.events)))
        if names:
            name, ok = QInputDialog.getItem(
                self, "Name event",
                f"Frames {lo}–{hi} — pick an event to mark another occurrence, "
                "or type a new name:", names, 0, True)
        else:
            name, ok = QInputDialog.getText(
                self, "Name event", f"Event window frames {lo}–{hi}:",
                text=f"event {len(s.events) + 1}")
        self._pending_event = None
        self.timeline.set_pending_event(None)
        if not ok:
            self.statusBar().showMessage("Event cancelled", 3000)
            return
        s.add_event(name, start, end)
        self._on_events_changed()
        e = s.events[-1]
        n_occ = sum(1 for ev in s.events if ev.name == e.name)
        occ = f" — occurrence {n_occ}" if n_occ > 1 else ""
        self.statusBar().showMessage(
            f"“{e.name}” marked: frames {e.start}–{e.end}{occ}. Click its ribbon "
            "(or use the Events menu) to jump back", 6000)

    def _on_events_changed(self):
        self._refresh_events_ui()
        self.timeline.refresh()

    def _refresh_events_ui(self):
        self.m_events.clear()
        self.m_events.addAction(self.act_mark_event)
        if self.session is None or not (self.session.events or self.session.notes):
            return
        self.m_events.addSeparator()

        def jump(index):
            # jumping = the same as clicking the ribbon: select + seek
            def go(_=False, i=index):
                if i < len(self.session.events):
                    self.timeline.select_event_window(i)
                    self._goto(self.session.events[i].start)
            return go

        groups: dict[str, list[int]] = {}
        for i, ev in enumerate(self.session.events):
            groups.setdefault(ev.name, []).append(i)
        for name, idxs in groups.items():  # one entry per TYPE
            if len(idxs) == 1:
                ev = self.session.events[idxs[0]]
                act = self.m_events.addAction(f"{name}   [{ev.start}–{ev.end}]")
                act.triggered.connect(jump(idxs[0]))
            else:
                sub = self.m_events.addMenu(f"{name}   ({len(idxs)}×)")
                for k, i in enumerate(idxs, 1):
                    ev = self.session.events[i]
                    act = sub.addAction(f"#{k}   [{ev.start}–{ev.end}]")
                    act.triggered.connect(jump(i))
        if self.session.notes:
            self.m_events.addSeparator()
            sub = self.m_events.addMenu(f"Notes   ({len(self.session.notes)})")
            for f in self.session.note_frames():
                txt = self.session.notes[f]["text"].replace("\n", " ")
                act = sub.addAction(f"frame {f}: {txt[:50]}{'…' if len(txt) > 50 else ''}")
                act.triggered.connect(lambda _=False, fr=f: self._goto(fr))

    def _on_select(self, pid: int):
        self.selected = pid
        self.point_list.blockSignals(True)
        self.point_list.setCurrentRow(pid)
        self.point_list.blockSignals(False)
        self._refresh_overlay()

    def _on_list_select(self, row: int):
        if row >= 0:
            self.selected = row
            self._refresh_overlay()
            s = self.session
            if s is not None and row < s.n_points and not s.tracked[self.current, row]:
                self.statusBar().showMessage(
                    (f"{s.points[row].name} is derived from the silhouette: it fills in when you track "
                     "with a segment (it cannot be placed by hand)" if s.points[row].derived else
                     f"{s.points[row].name} has no position on this frame — click on the video "
                     "to place it here and continue the same point"), 6000)

    def _on_place(self, pid: int, x: float, y: float):
        """Drag release or Ctrl+click: a manual correction at the current frame."""
        if self.session is None or pid >= self.session.n_points or self.state != READY:
            return  # the point can vanish mid-drag (Delete while holding)
        if self.session.points[pid].derived:        # the same refusal as a plain click (I69)
            self._refresh_overlay()                 # put the dragged marker back where it is
            self.statusBar().showMessage(
                f"{self.session.points[pid].name} is derived from the silhouette — to place it by hand, "
                "right-click it → Data source → Track by appearance", 6000)
            return
        # a drag / Ctrl+click / N-continue is ONE undo step, like a plain click:
        # without it Ctrl+Z threw away the whole previous tracking run (I64)
        self._undo_snap = self.session.snapshot()
        self.act_undo.setEnabled(True)
        self.session.set_position(self.current, pid, x, y)
        self._refresh_overlay()
        self._update_track_button()
        name = self.session.points[pid].name
        self.statusBar().showMessage(
            f"{name} set to ({x:.1f}, {y:.1f}) at frame {self.current} — press Track to "
            "re-track forward from here (the Track button names the points it will track); "
            "Ctrl+Z undoes the move", 7000)

    def _on_reposition(self, x: float, y: float):
        if self.selected is not None and self.session is not None \
                and self.selected < self.session.n_points:
            self._on_place(self.selected, x, y)
        else:
            self.statusBar().showMessage("Select a point first (click it), then Ctrl+Click to move it", 4000)

    def _on_annotate(self, x: float, y: float):
        """Plain (unarmed) click on the video: annotate the SELECTED point at
        the current frame by hand — the way to digitize frames the tracker
        gets wrong. Overwrites whatever the tracker put there; the frame is
        flagged manual (Shift+< / Shift+> walk to the first / last one).
        Nothing selected = nothing edited."""
        if self.state != READY or self.session is None:
            return
        s = self.session
        pid = self.selected
        if pid is None or pid >= s.n_points:
            self.statusBar().showMessage(
                "Nothing placed: select a point in the POINTS list to annotate it here, "
                "or press N to add a new point", 5000)
            return
        if s.points[pid].derived:
            self.statusBar().showMessage(
                f"{s.points[pid].name} is derived from the silhouette — to annotate it by hand, "
                "right-click it → Data source → Track by appearance", 6000)
            return
        self._undo_snap = s.snapshot()   # Ctrl+Z takes the click back
        s.set_position(self.current, pid, x, y)
        self._refresh_overlay()
        self._update_track_button()
        self.act_undo.setEnabled(True)
        n = int(s.manual[:, pid].sum())
        plural = "s" if n != 1 else ""
        self.statusBar().showMessage(
            f"{s.points[pid].name} annotated at ({x:.1f}, {y:.1f}) on frame {self.current} "
            f"— {n} hand-placed frame{plural} (Shift+< / Shift+> jump to first / last; "
            "Ctrl+Z undoes)", 6000)

    def _goto_clicked_frame(self, last: bool):
        """Shift+> / Shift+<: jump to the LAST / FIRST frame the selected point
        has data on — tracked or hand-placed, whichever the point actually has
        (on an ordinary tracked point the hand-placed-only rule looked like a
        dead key). With no point selected it walks the
        segment's silhouette instead. `,` / `.` still step between hand-placed
        frames."""
        if self.state != READY or self.session is None:
            return
        s = self.session
        pid = self.selected
        which = "last" if last else "first"
        if pid is not None and pid < s.n_points:
            frames = s.data_frames(pid)
            if len(frames) == 0:
                self.statusBar().showMessage(
                    f"{s.points[pid].name} has no position on any frame yet — select it, press N "
                    "and click it on the video", 5000)
                return
            n_manual = len(s.manual_frames(pid))
            extra = f", {n_manual} of them hand-placed" if n_manual else ""
            self._goto(int(frames[-1] if last else frames[0]))
            self.statusBar().showMessage(
                f"{s.points[pid].name}: {which} frame with data ({len(frames):,} frames"
                f"{extra}). , and . step between hand-placed frames", 5000)
            return
        frames = s.mask_frames()
        if len(frames):
            name = s.animal.name if s.animal is not None else "the segment"
            self._goto(int(frames[-1] if last else frames[0]))
            self.statusBar().showMessage(
                f"{name}: {which} frame with a silhouette ({len(frames):,} frames). "
                "Select a point in the list to walk that point instead", 5000)
            return
        self.statusBar().showMessage(
            "Nothing to jump to: select a point in the POINTS list (Shift+< / Shift+> go to its "
            "first / last frame), or outline the segment with S and Track", 6000)

    def _on_delete(self, pid: int):
        if self.session is None or pid >= self.session.n_points:
            return
        self.canvas.cancel_gesture()  # a live drag would hold a stale pid
        name = self.session.points[pid].name
        n_tracked = int(self.session.tracked[:, pid].sum())
        if n_tracked > 1 and QMessageBox.question(
                self, "Delete point",
                f"Delete {name} and its {n_tracked} tracked frames?\n\nCtrl+Z restores it.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        # one undo step: without it Ctrl+Z restored an OLDER snapshot and took
        # back the last tracking run instead (I71)
        self._undo_snap = self.session.snapshot()
        self.act_undo.setEnabled(True)
        self.session.remove_point(pid)
        self.timeline.clear_selection()   # its lane rows would now point elsewhere
        if self.selected is not None:
            if self.selected == pid:
                self.selected = None
            elif self.selected > pid:
                self.selected -= 1
        self._refresh_point_list()
        self._refresh_overlay()
        self._apply_state()

    def _selected_pids(self) -> list[int]:
        """Rows multi-selected in the point panel (Ctrl/Shift+click)."""
        return sorted(self.point_list.row(it) for it in self.point_list.selectedItems())

    def _delete_selected(self):
        """Delete key: clear the timeline's selected frame window — the lanes the
        marquee covered decide whether that is the segmentation, points, or both
        — else delete the selected point(s)."""
        if self.state != READY or self.session is None:
            return
        if self.timeline.request_delete_selection():
            return
        pids = self._selected_pids()
        if len(pids) > 1:
            self._delete_points(pids)
        elif self.selected is not None:
            self._on_delete(self.selected)

    def _delete_points(self, pids: list[int]):
        """Bulk point deletion (multi-select + Delete). Undoable."""
        s = self.session
        pids = [p for p in pids if p < s.n_points]
        if len(pids) < 2:
            if pids:
                self._on_delete(pids[0])
            return
        n_cells = int(s.tracked[:, pids].sum())
        names = ", ".join(s.points[p].name for p in pids[:5])
        if len(pids) > 5:
            names += f", … ({len(pids)} points)"
        if QMessageBox.question(
                self, "Delete points",
                f"Delete {len(pids)} points ({names}) and their {n_cells} tracked "
                "frames?\n\nCtrl+Z restores them.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._undo_snap = s.snapshot()
        self.canvas.cancel_gesture()
        for pid in sorted(pids, reverse=True):  # descending keeps indices valid
            s.remove_point(pid)
        self.timeline.clear_selection()   # its lane rows would now point elsewhere
        self.selected = None
        self.act_undo.setEnabled(True)
        self._refresh_point_list()
        self._refresh_overlay()
        self._apply_state()
        self.statusBar().showMessage(
            f"Deleted {len(pids)} points — Ctrl+Z restores them", 6000)

    def _resolve_window_pids(self, f0: int, f1: int, pids, verb: str = "",
                             title: str = "") -> list[int] | None:
        """Which points a window action applies to: the lanes the timeline
        marquee covered, else the point panel's selection, else — after a
        confirmation — every point. None means the user backed out. `verb` /
        `title` word the question for actions other than a clear (I70)."""
        s = self.session
        if pids:
            return [p for p in pids if 0 <= p < s.n_points]
        panel = self._selected_pids()
        if panel:
            return panel
        if s.n_points == 0:
            return []
        if QMessageBox.question(
                self, title or "Clear all points?",
                f"The drag named no point lane and no point is selected in the panel "
                f"— {verb or 'clear the tracked data of'} ALL {s.n_points} points in frames "
                f"{f0}–{f1}?\n\nCtrl+Z restores it.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return None
        return list(range(s.n_points))

    def _clear_window(self, f0: int, f1: int, pids, do_points: bool, do_masks: bool):
        """Core of the timeline-window deletes: tracked points and / or the
        segment's silhouettes inside [f0, f1], as ONE undoable step. The points
        themselves survive — only that stretch of their tracks is removed."""
        s = self.session
        if s is None or self.state != READY:
            return
        do_masks = do_masks and s.animal is not None and s.masks is not None
        use: list[int] = []
        if do_points:
            resolved = self._resolve_window_pids(f0, f1, pids)
            if resolved is None:
                return                      # confirmation declined
            use = resolved
        if not use and not do_masks:
            self.statusBar().showMessage(
                "Nothing to clear in that window — Shift+drag across the lanes you "
                "want to delete (the segment lane clears silhouettes)", 6000)
            return
        self._undo_snap = s.snapshot()
        n_pts = s.clear_window(use, f0, f1) if use else 0
        n_msk = s.clear_masks(f0, f1) if do_masks else 0
        self.act_undo.setEnabled(True)
        self.timeline.clear_selection()
        self._refresh_overlay()
        self._refresh_animal_panel()
        self._update_frame_label()
        self._apply_state()
        parts = []
        if use:
            names = ", ".join(s.points[p].name for p in use[:5])
            if len(use) > 5:
                names += f", … ({len(use)} points)"
            parts.append(f"{n_pts} tracked frames for {names}")
        if do_masks:
            parts.append(f"{n_msk} silhouettes")
        self.statusBar().showMessage(
            f"Cleared {' and '.join(parts)} in frames {f0}–{f1} — Ctrl+Z restores "
            "the data", 8000)

    def _clear_tracked_window(self, f0: int, f1: int, pids=None):
        """Blank the tracked data inside [f0, f1] (timeline clear_requested)."""
        self._clear_window(f0, f1, pids, do_points=True, do_masks=False)

    def _clear_window_both(self, f0: int, f1: int, pids=None):
        """Points AND silhouettes inside [f0, f1], as one undo step."""
        self._clear_window(f0, f1, pids, do_points=True, do_masks=True)

    def _deselect(self):
        if self.selected is not None:
            self.selected = None
            self.point_list.blockSignals(True)
            self.point_list.clearSelection()
            self.point_list.setCurrentRow(-1)
            self.point_list.blockSignals(False)
            self._refresh_overlay()
            self.statusBar().showMessage("Selection cleared — a plain click on the video now edits nothing; "
                                         "press N, then click, to add a new point", 5000)

    def _apply_rename(self, pid: int, desired: str) -> str:
        """Rename a point with collision protection; explains the auto-suffix
        in the status bar. Returns the name actually applied. Single collision
        UX for both the dialog and the in-list editor."""
        applied = self.session.rename_point(pid, desired)
        if applied != desired:
            self.statusBar().showMessage(
                f"“{desired}” is already another point's name — renamed to "
                f"“{applied}” so nothing gets overwritten", 6000)
        return applied

    def _on_rename(self, pid: int):
        if self.session is None:
            return
        name, ok = QInputDialog.getText(self, "Rename point", "Name:",
                                        text=self.session.points[pid].name)
        if ok and name.strip():
            self._apply_rename(pid, name.strip())
            self._refresh_point_list()
            self._refresh_overlay()

    def _point_list_menu(self, pos):
        """Same menu as right-clicking the marker — findable even when markers
        overlap or the point currently has no position on screen."""
        item = self.point_list.itemAt(pos)
        if item is None or self.session is None or self.state != READY:
            return
        pid = self.point_list.row(item)
        menu, acts = self.canvas._build_context_menu(pid)
        chosen = menu.exec(self.point_list.viewport().mapToGlobal(pos))
        menu.deleteLater()          # a shown menu otherwise lingers as a child of the canvas
        if self._point_menu_extra_action(chosen, acts, pid):
            return
        if chosen == acts["rename"]:
            self._on_rename(pid)
        elif chosen == acts["anchor"]:
            self._on_anchor_toggled(pid, acts["anchor"].isChecked())
        elif chosen == acts["delete"]:
            self._on_delete(pid)
        elif chosen in acts["source"]:
            self._on_source_change(pid, acts["source"][chosen])
        elif chosen == acts["free"]:
            self._on_free_toggled(pid, acts["free"].isChecked())
        elif chosen == acts["occluded"]:
            self._on_occluded_toggled(pid, acts["occluded"].isChecked())

    def _extend_point_menu(self, menu, acts: dict, pid: int) -> None:
        """Append the frame-aware entries to a point's context menu: where its
        track starts and ends, where you hand-placed it, where it looks
        doubtful, and the three ways to clear it. The canvas builds the rest
        (it has no session of its own)."""
        s = self.session
        if s is None or not (0 <= pid < s.n_points):
            return
        name = s.points[pid].name
        frames = s.data_frames(pid)
        manual = s.manual_frames(pid)
        low = s.low_conf_runs([pid])
        menu.addSeparator()
        a_first = menu.addAction(
            f"Go to its first frame  ({frames[0]})" if len(frames) else "Go to its first frame")
        a_last = menu.addAction(
            f"Go to its last frame  ({frames[-1]})" if len(frames) else "Go to its last frame")
        for a in (a_first, a_last):
            a.setEnabled(bool(len(frames)))
            a.setToolTip("Shift+< and Shift+> do this for the selected point")
        acts["go_first"], acts["go_last"] = a_first, a_last
        a_mf = menu.addAction(
            f"Go to its first hand-placed frame  ({manual[0]})" if len(manual)
            else "Go to its first hand-placed frame")
        a_ml = menu.addAction(
            f"Go to its last hand-placed frame  ({manual[-1]})" if len(manual)
            else "Go to its last hand-placed frame")
        for a in (a_mf, a_ml):
            a.setEnabled(bool(len(manual)))
            a.setToolTip("The frames you placed by hand (the white diamonds on its timeline row); "
                         ", and . step between them")
        acts["go_manual_first"], acts["go_manual_last"] = a_mf, a_ml
        a_low = menu.addAction(
            f"Go to its first doubtful stretch  ({low[0][0]}\u2013{low[0][1]})" if low
            else "Go to its first doubtful stretch")
        a_low.setEnabled(bool(low))
        a_low.setToolTip("The red stretches on its timeline row, where the tracker was unsure "
                         "(J jumps to the next one)")
        acts["go_low"] = a_low
        menu.addSeparator()
        a_here = menu.addAction(f"Clear its position on frame {self.current}")
        a_here.setEnabled(bool(s.tracked[self.current, pid]))
        a_here.setToolTip("Delete this one frame of its track (Ctrl+Z undoes it). To keep the "
                          "position but leave it out of the exports, use \u201cHidden on this "
                          "frame\u201d instead.")
        acts["clear_here"] = a_here
        sel = self.timeline.sel_range
        a_win = menu.addAction(
            f"Clear its track in frames {sel[0]}\u2013{sel[1]}" if sel
            else "Clear its track in the selected frame window")
        a_win.setEnabled(bool(sel is not None and len(frames)))
        a_win.setToolTip("Shift+drag across the timeline to select a frame window first")
        acts["clear_window"] = a_win
        a_all = menu.addAction(f"Clear its whole track  ({len(frames):,} frame{'' if len(frames) == 1 else 's'})")
        a_all.setEnabled(bool(len(frames)))
        a_all.setToolTip(f"{name} itself stays in the list, ready to be placed again")
        acts["clear_all"] = a_all
        # keyframe digitizing: a curve through the hand-placed frames fills the rest
        mf = s.manual_frames(pid)
        menu.addSeparator()
        a_fill = menu.addAction(
            f"Fill its gaps between hand placements (curve through {len(mf)} placed frames, "
            f"{int(mf[0])}–{int(mf[-1])})" if len(mf) >= 2 else "Fill its gaps between hand placements")
        a_fill.setEnabled(len(mf) >= 2 and not s.points[pid].derived)
        a_fill.setToolTip("Keyframe digitizing: place the part by hand on a few frames, then let a smooth curve "
                          "through those frames fill the frames between them that have no data. Filled frames "
                          "show at confidence 0.6 on the timeline. Ctrl+Z undoes it."
                          if len(mf) >= 2 else "Place this part by hand on at least two frames first (click it "
                          "while it is selected, or drag its marker)")
        acts["interp_fill"] = a_fill
        a_repl = menu.addAction("Replace everything between its hand placements with that curve")
        a_repl.setEnabled(len(mf) >= 2 and not s.points[pid].derived)
        a_repl.setToolTip("The same curve, but it also overwrites tracked frames between the placements - for a "
                          "stretch where the tracker drifted and you have re-placed the part by hand at both "
                          "ends (and anywhere in between). Ctrl+Z undoes it.")
        acts["interp_replace"] = a_repl
        # with a calibration, the other cameras' rays say where this landmark
        # can be here: an epipolar-constrained re-seed the user can see
        guides = self._epipolar_guides(pid) if (self.project is not None and self.project.calibration) else []
        menu.addSeparator()
        n_other = len(guides)
        a_snap = menu.addAction(
            f"Snap to the other cameras' rays here  ({n_other} camera{'s' if n_other != 1 else ''} see it)"
            if n_other else "Snap to the other cameras' rays here")
        a_snap.setEnabled(bool(n_other) and not s.points[pid].derived)
        a_snap.setToolTip("Moves this landmark onto the dashed epipolar line(s) drawn from the other cameras' "
                          "positions at this instant (the nearest point with one camera, the crossing with "
                          "two or more; cameras standing in a line give only a line, so check the place along "
                          "it by eye), marks the frame hand-placed, and Track re-seeds from it. Ctrl+Z undoes it."
                          if n_other else
                          "Needs a calibration and this landmark tracked in another camera at this instant")
        acts["snap_epipolar"] = a_snap

    def _point_menu_extra_action(self, chosen, acts: dict, pid: int) -> bool:
        """Apply one of the frame-aware point entries. Returns True when it
        handled the choice, so the canvas leaves its own dispatch alone."""
        s = self.session
        if chosen is None or s is None or not (0 <= pid < s.n_points):
            return False
        name = s.points[pid].name
        if chosen is acts.get("snap_epipolar"):
            self._snap_to_epipolar(pid)
            return True
        if chosen is acts.get("interp_fill") or chosen is acts.get("interp_replace"):
            self._interpolate_keyframes(pid, replace=chosen is acts.get("interp_replace"))
            return True
        if chosen is acts.get("go_first") or chosen is acts.get("go_last"):
            fr = s.data_frames(pid)
            if len(fr):
                self._goto(int(fr[-1] if chosen is acts.get("go_last") else fr[0]))
            return True
        if chosen is acts.get("go_manual_first") or chosen is acts.get("go_manual_last"):
            fr = s.manual_frames(pid)
            if len(fr):
                self._goto(int(fr[-1] if chosen is acts.get("go_manual_last") else fr[0]))
            return True
        if chosen is acts.get("go_low"):
            low = s.low_conf_runs([pid])
            if low:
                self._goto(int(low[0][0]))
                self.statusBar().showMessage(
                    f"{name}: doubtful from frame {low[0][0]} to {low[0][1]} \u2014 check it here "
                    "(J jumps to the next stretch)", 5000)
            return True
        if chosen is acts.get("clear_here"):
            self._clear_tracked_window(self.current, self.current, [pid])
            return True
        if chosen is acts.get("clear_window"):
            sel = self.timeline.sel_range
            if sel is not None:
                self._clear_tracked_window(sel[0], sel[1], [pid])
            return True
        if chosen is acts.get("clear_all"):
            n = int(s.tracked[:, pid].sum())
            if QMessageBox.question(
                    self, "Clear track",
                    f"Clear {name}'s positions on all {n:,} frames?\n\nThe point itself stays in "
                    "the list, so you can place it again. Ctrl+Z undoes this.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
                self._clear_tracked_window(0, s.n_frames - 1, [pid])
            return True
        return False

    def _interpolate_keyframes(self, pid: int, replace: bool) -> None:
        """Point menu: a curve through the hand-placed frames fills the gaps
        (or replaces everything) between them; one undo step."""
        s = self.session
        if s is None or self.state != READY or not (0 <= pid < s.n_points):
            return
        name = s.points[pid].name
        snap = s.snapshot()
        n, span = s.interpolate_keyframes(pid, replace=replace, window=self.timeline.sel_range)
        if not n:
            self.statusBar().showMessage(
                f"{name}: nothing to fill — it needs at least two hand-placed frames"
                + (" with empty frames between them" if not replace else "")
                + (" inside the selected window" if self.timeline.sel_range else ""), 7000)
            return
        self._undo_snap = snap
        self.act_undo.setEnabled(True)
        self._refresh_overlay()
        self.timeline.refresh()
        self._update_track_button()
        hidden = int(getattr(s, "last_interp_hidden_skipped", 0))
        self.statusBar().showMessage(
            f"{name}: {n} frame(s) between {span[0]} and {span[1]} filled from the curve through its hand "
            f"placements (confidence {s.INTERP_CONF:.1f} on the timeline) — Ctrl+Z undoes"
            + (f". {hidden} frame(s) marked hidden were left as they are (unmark them to fill them)"
               if hidden else ""), 9000)

    def _on_anchor_toggled(self, pid: int, on: bool):
        if self.session is None or pid >= self.session.n_points:
            return
        self.session.points[pid].anchor = bool(on)
        self.session.dirty = True
        name = self.session.points[pid].name
        msg = (f"{name}: appearance lock ON — future runs snap back to how it "
               f"looked at its seed frame" if on
               else f"{name}: appearance lock off — plain tracking")
        self.statusBar().showMessage(msg, 6000)

    def _on_item_changed(self, item: QListWidgetItem):
        pid = self.point_list.row(item)
        if self.session is None or pid >= self.session.n_points:
            return
        meta = self.session.points[pid]
        new_name = item.text().strip()
        if new_name and new_name != meta.name:
            applied = self._apply_rename(pid, new_name)
            if applied != new_name:
                self.point_list.blockSignals(True)
                item.setText(applied)
                self.point_list.blockSignals(False)
            self._refresh_overlay()
        want = item.checkState() == Qt.Checked
        if want != meta.display:
            meta.display = want
            self.session.dirty = True
            self._refresh_overlay()

    def _fit_timeline_height(self):
        """Give the timeline enough height for all lanes (up to its default
        lane cap) when points appear in bulk — a skeleton template or a
        reopened project — so nothing hides below the panel edge."""
        if self.session is None or not hasattr(self, "_split"):
            return
        want = self.timeline.sizeHint().height() + max(0, self._bottom.height() - self.timeline.height())
        sizes = self._split.sizes()
        total = sum(sizes)
        if total > 0 and sizes[1] < want and total - want >= 300:
            self._split.setSizes([total - want, want])
            self.timeline.refresh()

    def _refresh_point_list(self):
        if getattr(self, "timeline", None) is not None and self.project is not None:
            self._update_disagreement()        # columns follow the point list
        self.point_list.blockSignals(True)
        self.point_list.clear()
        if self.session is not None:
            s = self.session
            for pid, meta in enumerate(s.points):
                item = QListWidgetItem(meta.name)
                pm = QPixmap(14, 14)
                pm.fill(Qt.transparent)
                from PySide6.QtGui import QPainter, QPen
                painter = QPainter(pm)
                painter.setRenderHint(QPainter.Antialiasing)
                if meta.derived:   # hollow dotted square = computed from the silhouette
                    pen = QPen(QColor(*meta.color), 1.5, Qt.DotLine)
                    painter.setPen(pen)
                    painter.drawRect(2, 2, 10, 10)
                elif meta.is_ball:  # ring = a ball SAM outlines, tracked as its circle centre
                    pen = QPen(QColor(*meta.color), 2.0)
                    painter.setPen(pen)
                    painter.drawEllipse(2, 2, 10, 10)
                else:
                    painter.fillRect(1, 1, 12, 12, QColor(*meta.color))
                painter.end()
                item.setIcon(QIcon(pm))
                item.setFlags(item.flags() | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if meta.display else Qt.Unchecked)
                has_any = bool(s.tracked[:, pid].any())
                if meta.derived:
                    item.setToolTip(f"{meta.name}: derived from the segment's silhouette ({meta.spec}) "
                                    "— fills in when you track with a segment defined (S)")
                elif not has_any:
                    item.setForeground(QColor(theme.TEXT_DIM))
                    item.setToolTip(f"{meta.name}: not placed yet — select it, press N, click it "
                                    "on the video")
                elif meta.is_ball:
                    item.setToolTip(f"{meta.name}: a ball marker — SAM outlines it each frame and the "
                                    "fitted circle's centre is the point (its circle is drawn on the video)")
                else:
                    item.setToolTip(f"{meta.name}: tracked by appearance")
                self.point_list.addItem(item)
            if self.selected is not None and self.selected < s.n_points:
                self.point_list.setCurrentRow(self.selected)
        self.point_list.blockSignals(False)

    # ------------------------------------------------ epipolar guides (3D)

    GUIDE_COLORS = [(255, 120, 200), (120, 220, 255), (255, 200, 90), (170, 255, 140),
                    (255, 140, 120), (200, 170, 255)]

    def _epipolar_guides(self, pid: int, frame: int | None = None) -> list:
        """[(polyline (M, 2) native px of the ACTIVE view, colour, camera name), ...]:
        where landmark `pid`, as each OTHER camera sees it at this instant, can
        lie in the active picture. Empty without a calibration, without a
        selection, or when no other camera has it here."""
        from cotracker_app.calib import epipolar_polyline, working_probe
        p = self.project
        s = self.session
        if p is None or p.calibration is None or s is None or not (0 <= pid < s.n_points):
            return []
        if len(p.calibration.cameras) != p.n_views:
            return []
        f = self.current if frame is None else int(frame)
        name = s.points[pid].name
        probe = None
        r = p.reconstruction
        k = int(np.floor(p.reference_time(p.active, f) + 0.5)) - r.t0 if r is not None else -1
        if r is not None and r.n_frames and 0 <= k < r.n_frames:
            # the row is the REFERENCE instant: comparing the working camera's own
            # frame number with the reference range raised IndexError (or wrapped
            # to another row) whenever the working camera is not camera 1 (I67)
            row = r.xyz[k]
            if row is not None:
                row = row[np.isfinite(row).all(axis=1)]
                probe = row.mean(axis=0) if len(row) else None
        if probe is None:
            probe = working_probe(p.calibration)
        dst = p.calibration.cameras[p.active]
        out = []
        for c in p.others():
            sc = p.sessions[c]
            j = sc.pid_by_name(name)
            if j is None:
                continue
            fc = p.map_frame(p.active, c, f)
            if fc is None or not sc.exportable[fc, j]:
                continue
            uv = sc.tracks[fc, j]
            if not np.isfinite(uv).all():
                continue
            try:
                pts = epipolar_polyline(p.calibration.cameras[c], dst, uv, probe)
            except Exception:       # noqa: BLE001 -- a degenerate calibration must not break the overlay
                continue
            if len(pts) >= 2:
                out.append((pts, self.GUIDE_COLORS[c % len(self.GUIDE_COLORS)], p.name(c)))
        return out

    def _refresh_guides(self) -> None:
        pid = self.selected
        show = (self.act_epipolar.isChecked() and pid is not None and self.project is not None
                and self.project.calibration is not None and self.state == READY)
        self.canvas.set_guides(self._epipolar_guides(pid) if show else [])

    def _snap_to_epipolar(self, pid: int) -> None:
        """Move landmark `pid` at this frame onto the other cameras' epipolar
        line(s): the nearest point with one camera, their crossing with two or
        more. Flagged hand-placed, one undo step; Track re-seeds from it."""
        from cotracker_app.calib import intersect_polylines
        s = self.session
        if s is None or self.state != READY or not (0 <= pid < s.n_points) or s.points[pid].derived:
            return
        guides = self._epipolar_guides(pid)
        if not guides:
            self.statusBar().showMessage(
                f"{s.points[pid].name}: no other camera has it at this instant (or no calibration)", 6000)
            return
        cur = s.tracks[self.current, pid]
        near = cur if np.isfinite(cur).all() else np.array([s.width / 2.0, s.height / 2.0])
        info: dict = {}
        target = intersect_polylines([g[0] for g in guides], near, info)
        if target is None or not np.isfinite(target).all():
            return
        from cotracker_app.retrack import edge_tolerance, outside_by
        off = outside_by(target, s.width, s.height)
        if off > edge_tolerance(s.width):
            # the rays cross OUTSIDE this picture: clipping that onto the edge stored
            # a hand placement where the part is not (I9)
            self.statusBar().showMessage(
                f"{s.points[pid].name}: the other cameras put it {off:.0f} px outside this camera's picture at "
                f"frame {self.current} — out of the picture means no data here; clear it on this frame instead",
                8000)
            return
        x, y = float(np.clip(target[0], 0, s.width - 1)), float(np.clip(target[1], 0, s.height - 1))
        self._undo_snap = s.snapshot()
        s.set_position(self.current, pid, x, y)
        self._refresh_overlay()
        self._update_track_button()
        self.act_undo.setEnabled(True)
        moved = float(np.linalg.norm(np.array([x, y]) - cur)) if np.isfinite(cur).all() else float("nan")
        self.statusBar().showMessage(
            f"{s.points[pid].name} snapped onto {len(guides)} camera{'s' if len(guides) != 1 else ''}' rays at "
            f"frame {self.current}" + (f" (moved {moved:.1f} px)" if np.isfinite(moved) else "")
            + (" — the cameras stand in a line, so their rays coincide here: it was moved onto the line "
               "only; check its place ALONG the line by eye" if info.get("mode") == "parallel" else "")
            + " — press Track to re-seed from here; Ctrl+Z undoes", 10000)

    # ------------------------------------------- automatic epipolar re-tracking

    def _disagree_thresholds(self) -> list[float]:
        """The timeline's band threshold per camera: 5 px at 1920 wide, scaled."""
        return [5.0 * max(1.0, (s.width or 1920) / 1920.0) for s in self.project.sessions]

    def _retrack_dialog(self):
        """3D → Re-track Disagreeing Stretches: explain, list, ask, run."""
        from cotracker_app import retrack
        p = self.project
        if p is None or p.reconstruction is None or p.reconstruction.per_cam is None or self.state != READY:
            return
        if self._retrack is not None:
            return
        thr = self._disagree_thresholds()
        stretches = retrack.plan(p, thr)
        if not stretches:
            QMessageBox.information(
                self, "Nothing to re-track",
                "No camera disagrees with the others for 3 frames or more (no magenta band on any "
                f"timeline; the band starts at {thr[0]:.1f} px). Nothing to fix.")
            return
        doable = [st for st in stretches if st.target is not None]
        skipped = [st for st in stretches if st.target is None]
        lines = []
        for st in doable[:14]:
            lines.append(f"  {p.name(st.view)}: {st.name}, frames {st.local0}–{st.local1} "
                         f"({st.median_px:.1f} px off; {st.n_rays} other camera{'s' if st.n_rays != 1 else ''} see it)")
        if len(doable) > 14:
            lines.append(f"  … and {len(doable) - 14} more")
        for st in skipped[:6]:
            lines.append(f"  (skipped) {p.name(st.view)}: {st.name}, frames {st.local0}–{st.local1} — {st.reason}")
        n_cams = len({st.view for st in doable})
        if not doable:
            QMessageBox.information(
                self, "Cannot re-track", "\n".join(lines) + "\n\nNo disagreeing stretch can be blamed on one "
                "camera that the others could correct: either no other camera has the landmark at the start "
                "of the stretch, or the cameras disagree among themselves (check the offsets and the calibration).")
            return
        if QMessageBox.question(
                self, f"Re-track {len(doable)} disagreeing stretch{'es' if len(doable) != 1 else ''}?",
                f"Where one camera's landmark disagrees with the others by more than the band ({thr[0]:.1f} px), "
                "the landmark probably slid onto the wrong spot in THAT camera. For each stretch below the program "
                "will:\n\n  1. move the landmark, on the stretch's first frame, to where the other cameras' rays say "
                "it is (a hand placement),\n  2. re-track it through the stretch with the ordinary tracker,\n"
                "  3. reconstruct again and tell you whether the disagreement went down.\n\n"
                f"{len(doable)} stretch{'es' if len(doable) != 1 else ''} in {n_cams} camera{'s' if n_cams != 1 else ''}:\n"
                + "\n".join(lines) + "\n\nX or Space stops it at any time; either way you are asked at the end "
                "whether to keep the result, and can undo all of it. Run it now?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
            return
        self._retrack_start(doable, thr)

    def _retrack_start(self, stretches, thr):
        from cotracker_app import retrack
        p = self.project
        self._retrack = {
            "jobs": list(stretches), "done": [], "thr": thr, "stretches": list(stretches),
            "before": retrack.cells_summary(p, stretches),
            "snaps": {}, "prev_active": p.active, "prev_frame": self.current,
        }
        self.statusBar().showMessage(f"Re-tracking {len(stretches)} stretch(es)…", 5000)
        self._retrack_next()

    def _retrack_next(self):
        """Run the next queued stretch, or finish."""
        from cotracker_app import retrack
        st = self._retrack
        if st is None:
            return
        if self.state != READY:
            QTimer.singleShot(100, self._retrack_next)
            return
        p = self.project
        while st["jobs"]:
            job = st["jobs"].pop(0)
            s = p.sessions[job.view]
            pid = s.pid_by_name(job.name)
            if pid is None or s.points[pid].derived:
                continue
            if job.view not in st["snaps"]:
                st["snaps"][job.view] = s.snapshot()
            # the target is recomputed now: an earlier job may have changed the other cameras
            got = retrack.ray_target(p, job.view, job.name, job.local0)
            if got is None:
                continue
            target, n_rays = got
            if p.active != job.view:
                self._set_active_view(job.view)
            self._goto(job.local0, force=True)
            s.set_position(job.local0, pid, float(target[0]), float(target[1]))
            self._on_select(pid)
            self._refresh_overlay()
            self._start_tracking(stop_after=job.local1, only_pids=[pid], quiet=True)
            if self.state == TRACKING:
                st["done"].append(job)
                self.statusBar().showMessage(
                    f"Re-tracking {job.name} in {p.name(job.view)}, frames {job.local0}–{job.local1} "
                    f"({len(st['jobs'])} stretch(es) to go)…", 8000)
                return          # _on_track_finished brings us back
        self._retrack_finish()

    def _retrack_restore(self, st) -> None:
        p = self.project
        for view, snap in st["snaps"].items():
            if view < p.n_views:
                p.sessions[view].restore(snap)
        if p.calibration is not None:
            self._reconstruct_3d(quiet=True)
        self._refresh_point_list()
        self._refresh_overlay()
        self.timeline.refresh()
        self._apply_state()

    def _retrack_finish(self):
        from cotracker_app import retrack
        st = self._retrack
        self._retrack = None
        p = self.project
        if st is None or p is None:
            return
        self._reconstruct_3d(quiet=True)
        after = retrack.cells_summary(p, st["stretches"])
        # each camera judged against its own band, cameras named (I7)
        v, why = retrack.verdict(st["before"], after, list(st["thr"]), p)
        self._retrack_last = {"verdict": v, "before": st["before"], "after": after, "n_done": len(st["done"])}
        if st["prev_active"] < p.n_views and p.active != st["prev_active"]:
            self._set_active_view(st["prev_active"])
        self._goto(min(st["prev_frame"], self.n_frames - 1), force=True)
        self._update_disagreement()
        self.timeline.refresh()
        word = {"better": "BETTER", "same": "NO CHANGE", "worse": "WORSE"}[v]
        b, a = st["before"], after
        default = QMessageBox.Yes if v == "better" else QMessageBox.No
        keep = QMessageBox.question(
            self, f"Re-tracking: {word}",
            f"{why}\n\n{len(st['done'])} stretch(es) re-tracked. On those frames this camera disagreed with the others "
            f"by {b['median_px']:.1f} px (median, max {b['max_px']:.1f}) before and {a['median_px']:.1f} px "
            f"(max {a['max_px']:.1f}) after.\n\nKeep the re-tracked stretches? (No puts every camera back as it was.)",
            QMessageBox.Yes | QMessageBox.No, default)
        if keep != QMessageBox.Yes:
            self._retrack_restore(st)
            self.statusBar().showMessage("Re-tracking undone: every camera is back as it was", 8000)
            return
        p.dirty = True
        self._refresh_overlay()
        self._apply_state()
        self.toast.show_message(f"Re-tracking kept ({word.lower()}). Save the project (Ctrl+S).",
                                "success" if v == "better" else "warn", 8000)

    def _update_disagreement(self) -> None:
        """Per-cell 'this camera disagrees with the others' for the active view,
        from the last reconstruction's per-camera reprojection errors, mapped
        onto the view's own frames and point columns; handed to the timeline
        (NaN = no 3D there). Threshold and paint live in the timeline."""
        p = self.project
        s = self.session
        if p is None or s is None or p.reconstruction is None or p.reconstruction.per_cam is None:
            self.timeline.set_disagreement(None)
            return
        r = p.reconstruction
        C = r.per_cam.shape[2]
        if p.active >= C:
            self.timeline.set_disagreement(None)
            return
        out = np.full((s.n_frames, s.n_points), np.nan, np.float32)
        t = np.arange(r.t0, r.t0 + r.n_frames)
        local = np.round(p.rates[p.active] * t + p.offsets[p.active]).astype(int)
        ok = (local >= 0) & (local < s.n_frames)
        for j, nm in enumerate(r.names):
            pid = s.pid_by_name(nm)
            if pid is None:
                continue
            out[local[ok], pid] = r.per_cam[ok, j, p.active]
        self.timeline.set_disagreement(out)

    def _refresh_overlay(self):
        if self.session is None:
            return
        s = self.session
        f = self.current
        n_tr = self._trail_len
        lo = max(0, f - n_tr)
        trails = [s.tracks[lo:f + 1, i] for i in range(s.n_points)] if n_tr > 0 else None
        future = ([s.tracks[f:min(s.n_frames, f + n_tr + 1), i] for i in range(s.n_points)]
                  if (n_tr > 0 and self._trail_future) else None)
        onion = self.act_onion.isChecked()
        ghost_prev = s.tracks[f - 1] if (onion and f > 0) else None
        ghost_next = s.tracks[f + 1] if (onion and f + 1 < s.n_frames) else None
        positions = s.positions_at(f)
        # animal overlays first (they sit under the markers)
        if s.animal is not None and s.masks is not None and self.btn_mask.isChecked():
            self.canvas.set_mask(s.masks.contours.get(f), s.animal.color, self._mask_opacity)
            self.canvas.set_midline(s.masks.midline.get(f) if self.act_show_midline.isChecked()
                                    else None, s.animal.color)
        else:
            self.canvas.set_mask(None)
            self.canvas.set_midline(None)
        if s.animal is not None and self.state == READY:
            self.canvas.set_prompts(s.animal.prompts.get(f, []), s.animal.boxes.get(f), s.animal.color)
        else:
            self.canvas.set_prompts([], None)
        self.canvas.set_bones(s.bones() if self.act_show_bones.isChecked() else [])
        # set_points also applies follow when it is enabled — one code path
        # for scrubbing, tracking ticks, selection changes, and zooming
        self.canvas.set_points(positions, s.visibility[f], s.points, self.selected, trails,
                               future, ghost_prev, ghost_next, s.occluded[f], s.radius[f])
        self.timeline.set_selected(self.selected)
        self._refresh_guides()
        self.timeline.refresh()

    # -------------------------------------------------------------- tracking

    def _toggle_tracking(self):
        if self.state == TRACKING:
            self._pause_tracking()
        elif self.state == READY:
            if self._track_mode == "semi":
                self._track_step()
            else:
                self._start_tracking()

    def _track_step(self):
        """Semi-automatic: track exactly one frame forward, then pause. Each
        step re-seeds from the current (possibly corrected) positions, so
        F → check → fix → F walks the video with the user in the loop."""
        if self.state != READY or self.session is None:
            return
        if self.current >= self.n_frames - 1:
            self.statusBar().showMessage("Already at the last frame", 3000)
            return
        if not (self.session.seedable_at(self.current) or self.session.animal_seedable_at(self.current)):
            # a segment alone is enough, as for the Track button (I128)
            self.statusBar().showMessage(
                "Nothing to track from this frame — press N and click to place a point, or S and click the "
                "animal (or select a point in the list and click where it is)", 5000)
            return
        self._start_tracking(stop_after=self.current + 1)

    def _pause_tracking(self):
        if self.state == TRACKING and self.worker is not None:
            self._user_paused = True
            self.btn_track.setText("Pausing…")
            self.btn_track.setEnabled(False)
            self.worker.request_pause()

    def _start_tracking(self, stop_after: int | None = None, only_pids=None, quiet: bool = False):
        """`only_pids` overrides the panel selection (automatic runs); `quiet`
        skips the overwrite guard and keeps the caller's undo snapshot."""
        s = self.session
        scope, _n_sel = self._run_scope()   # panel selection limits the run
        if only_pids is not None:
            scope = set(int(q) for q in only_pids)
        pids = [p for p in s.seedable_at(self.current) if scope is None or p in scope]
        animal_ok = s.animal_seedable_at(self.current)
        if not pids and not animal_ok:
            if scope is not None:
                self.toast.show_message(
                    f"The selected point(s) have no position on frame {self.current}. Select a "
                    "point that exists here, or clear the selection (Esc) to track everything.",
                    "warn", 7000)
            elif s.animal is not None:
                self.toast.show_message(
                    f"Nothing to start from on frame {self.current}: place a point with <b>N</b> "
                    "(a few pixels is enough), or press <b>S</b> and click the segment here.",
                    "warn", 7000)
            return
        if self._preview is not None and self._preview.isRunning():
            self.toast.show_message("One moment — the silhouette preview is still computing.",
                                    "info", 4000)
            return
        self._step_run = stop_after is not None
        skipped = sum(1 for i in range(s.n_points) if not s.points[i].derived
                      and (scope is None or i in scope)) - len(pids)
        if scope is not None and not self._step_run:
            self.statusBar().showMessage(
                f"Tracking only the {len(pids)} selected point(s) — select all (or nothing) to track everything", 6000)
        if skipped and not self._step_run:
            self.statusBar().showMessage(
                f"{skipped} point(s) have no position at frame {self.current} and will be "
                "skipped (their existing tracks are kept)", 6000)

        # accidental-overwrite guard: starting far before a large body of work
        # (irrelevant for a one-frame step, which would nag on every F)
        last = s.last_tracked_frame() or 0
        frames_after = int(s.tracked[self.current + 1:].any(axis=1).sum())
        if not quiet and not self._step_run and frames_after > 2000 and self.current < 0.5 * last:
            if QMessageBox.question(
                    self, "Re-track from here?",
                    f"Tracking from frame {self.current} will progressively overwrite "
                    f"{frames_after} already-tracked frames ahead of it.\n\n"
                    "You can undo with Ctrl+Z after it finishes. Continue?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
                return

        if not quiet:
            self._undo_snap = s.snapshot()
        self.act_undo.setEnabled(False)  # re-enabled when the run ends

        from cotracker_app.tracker import (AnimalSpec, DerivedSpec, PointSpec,  # lazy: imports torch
                                           TrackingWorker)
        seeds = s.positions_at(self.current)
        def _offsets(meta):
            if meta.kind != "group" or meta.outline is None or len(meta.outline) < 3:
                return None
            o = np.asarray(meta.outline, np.float32).reshape(-1, 2)
            return o - o.mean(axis=0)
        ball_pids = [pid for pid in pids if s.points[pid].is_ball]
        pids = [pid for pid in pids if not s.points[pid].is_ball]
        specs = [PointSpec(pid, seeds[pid].astype(np.float32).copy(),
                           s.points[pid].kind, s.points[pid].radius,
                           s.points[pid].anchor, _offsets(s.points[pid]))
                 for pid in pids]
        # ball markers: SAM segments them, a circle is fitted, the centre is the
        # point (balls.py) - prompts from this frame on, seeded from the ball's
        # current position, the last known radius as the size hint
        from cotracker_app.tracker import BallSpec
        balls = []
        for pid in ball_pids:
            q = s.points[pid]
            prompts = {f: cs for f, cs in (q.ball_prompts or {}).items() if f >= self.current}
            rad = s.radius[:, pid]
            known = rad[np.isfinite(rad)]
            balls.append(BallSpec(pid, prompts, seeds[pid], float(known[-1]) if len(known) else None,
                                  self._seg_backend))
        animal = derived = None
        head_pid = None
        on_body: list[int] = []
        constrain: list[int] = []
        if animal_ok:
            seed_mask = None
            if not s.animal.has_prompt(self.current) and s.masks.has(self.current):
                ww, wh = working_size(self.info.width, self.info.height)
                seed_mask = s.masks.rasterize(self.current, wh, ww)
            animal = AnimalSpec({f: list(v) for f, v in s.animal.prompts.items()},
                                dict(s.animal.boxes), seed_mask, self._seg_backend)
            derived = [DerivedSpec(pid, s.points[pid].spec) for pid in s.derived_pids()
                       if scope is None or pid in scope]
            hp = s.head_pid()
            head_pid = hp if hp in pids else None
            if s.skeleton:
                on_body = [p for p in pids if s.points[p].name in s.skeleton.get("landmarks", [])]
            if self.btn_onbody.isChecked():
                constrain = [p for p in pids if not s.points[p].free]
        if not specs and not balls and animal is None:
            self.toast.show_message(
                f"Nothing to start from on frame {self.current}: place a point with <b>N</b>, a ball "
                "with Add ▾ → Ball marker, or press <b>S</b> and click the segment here.", "warn", 7000)
            return
        end = self.n_frames if stop_after is None else min(self.n_frames, stop_after + 1)
        self.worker = TrackingWorker(self.info.path, self.current, None, None,
                                     self.cache, end, refine=True,
                                     specs=specs,
                                     roi=self.btn_roi.isChecked(),
                                     autopause=self.btn_autopause.isChecked(),
                                     animal=animal, derived=derived, head_pid=head_pid,
                                     on_body_pids=on_body, constrain_pids=constrain,
                                     point_backend=self._point_backend, balls=balls)
        self.worker.balls_ready.connect(self._on_ball_radii)
        self.worker.model_loading.connect(self._on_model_loading)
        self.worker.started_ok.connect(self._on_track_started)
        self.worker.masks_ready.connect(self._on_masks)
        self.worker.chunk_ready.connect(self._on_chunk)
        self.worker.autopaused.connect(self._on_autopaused)
        self.worker.finished_ok.connect(self._on_track_finished)
        self.worker.error.connect(self._on_track_error)
        self._track_pids = list(self.worker.point_ids)
        self._run_had_animal = animal is not None
        self._autopause_info = None
        self._member_frames.clear()
        self._run_start = self.current
        self._fps_ema = 0.0
        self._last_emit_t = 0.0
        self.state = TRACKING
        self._apply_state()
        self.progress.setRange(0, self.n_frames - 1)
        self.progress.setValue(self.current)
        self._track_label.setText("starting…")
        self.worker.start()

    def _on_model_loading(self):
        from cotracker_app.segmenter import loaded_backends
        parts = []
        if self._track_pids and any(not self.session.points[p].derived for p in self._track_pids
                                    if p < self.session.n_points):
            parts.append("Downloading the CoTracker3 model (~100 MB, first run only)"
                         if not _CHECKPOINT.exists() else "")
        if getattr(self, "_run_had_animal", False) and self._seg_backend not in loaded_backends():
            label = BACKENDS.get(self._seg_backend, BACKENDS[DEFAULT_BACKEND])[2]
            parts.append(f"Loading the segmentation model ({label})" if seg_is_cached(self._seg_backend)
                         else f"Downloading the segmentation model ({label}) — first use only")
        parts = [p for p in parts if p]
        if parts:
            self._model_dialog = QProgressDialog(
                "\n".join(parts) + "\n\nModels are stored inside the tool folder.", None, 0, 0, self)
            self._model_dialog.setWindowTitle("Preparing models")
            self._model_dialog.setWindowModality(Qt.WindowModal)
            self._model_dialog.setCancelButton(None)
            self._model_dialog.setMinimumDuration(200)
            self._model_dialog.setValue(0)
        else:
            self.statusBar().showMessage("Loading models onto the GPU…", 8000)

    def _on_masks(self, summaries: list):
        if self.session is not None:
            self.session.write_mask_summaries(summaries)

    def _on_ball_radii(self, rows: list):
        if self.session is not None:
            self.session.write_ball_radii(rows)

    def _on_track_started(self):
        if self._model_dialog is not None:
            self._model_dialog.close()
            self._model_dialog = None
        self.statusBar().showMessage(f"Tracking from frame {self.current}… press X or the "
                                     "Pause button to stop at any time", 5000)
        self._render_timer.start()

    def _on_chunk(self, w0: int, tracks: np.ndarray, vis: np.ndarray,
                  conf: np.ndarray, members: dict, new_frames: list):
        if self.session is None:
            return
        self.session.write_segment(w0, tracks, vis, self._track_pids, conf)
        for idx, wrgb in new_frames:
            self._display_queue.append((idx, wrgb))
        if members:
            L = tracks.shape[0]
            for i in range(L):
                self._member_frames[w0 + i] = {
                    self._track_pids[k]: pts[i] for k, pts in members.items()}
            for f in [f for f in self._member_frames if f < w0 - 48]:
                del self._member_frames[f]
        self.timeline.refresh()
        head = w0 + tracks.shape[0] - 1
        now = time.perf_counter()
        if self._last_emit_t:
            inst = len(new_frames) / max(now - self._last_emit_t, 1e-6)
            self._fps_ema = inst if not self._fps_ema else 0.9 * self._fps_ema + 0.1 * inst
        self._last_emit_t = now
        self.progress.setValue(head)
        if self._fps_ema and not self._step_run:  # a 1-frame step has no useful ETA
            remaining = (self.n_frames - 1 - head) / max(self._fps_ema, 1e-6)
            mins, secs = divmod(round(remaining), 60)
            eta = f"{mins} min {secs:02d} s" if mins else f"{secs} s"
            self._track_label.setText(f"tracking {self._fps_ema:.1f} fps · ETA {eta}")

    def _render_tick(self):
        if not self._display_queue:
            if self.state != TRACKING:
                self._render_timer.stop()
            return
        # never fall behind inference: drop to the newest frame if backed up
        while len(self._display_queue) > 16:
            self._display_queue.popleft()
        idx, wrgb = self._display_queue.popleft()
        self.current = idx
        self.session.current_frame = idx
        self.canvas.set_frame(wrgb)
        self._refresh_overlay()
        overlay = self._member_frames.get(idx)
        if overlay is not None:
            self.canvas.set_group_members(overlay)
        self.timeline.set_current(idx)
        self.spin.blockSignals(True)
        self.spin.setValue(idx)
        self.spin.blockSignals(False)
        self._update_frame_label()

    def _on_autopaused(self, frame: int, pid: int):
        self._autopause_info = (frame, pid)

    def _end_run_cleanup(self):
        """Drop everything scoped to one tracking run. Called from BOTH the
        finished and error paths (and video teardown) so a future per-run
        cache can't leak into the next run by being cleared in only one."""
        self._display_queue.clear()
        self._member_frames.clear()
        self.canvas.clear_group_members()

    def _on_track_finished(self, last: int, was_paused: bool):
        user_pause = bool(getattr(self, "_user_paused", False))
        self._user_paused = False
        self._on_track_finished_impl(last, was_paused)
        if self._retrack is not None:
            if user_pause and was_paused:
                # X / Space during an automatic re-track stops the WHOLE queue and
                # goes to the Keep / Undo question; it used to start the next
                # stretch at once (I107)
                self._retrack["jobs"] = []
            QTimer.singleShot(0, self._retrack_next)

    def _on_track_finished_impl(self, last: int, was_paused: bool):
        self.worker.wait(2000)
        reason = getattr(self.worker, "_autopause_reason", "")
        ball_ended = dict(getattr(self.worker, "_ball_ended", {}) or {})
        self.worker = None
        self.state = READY
        step_run = self._step_run
        self._step_run = False
        self._end_run_cleanup()
        self.act_undo.setEnabled(self._undo_snap is not None)
        self._autosave()  # before the guidance message — 'Saved ✓' must not clobber it
        self._refresh_animal_panel()
        if self._autopause_info is not None:
            fail_frame, pid = self._autopause_info
            self._autopause_info = None
            self._goto(min(fail_frame, self.n_frames - 1), force=True)
            self._apply_state()
            self._track_label.setText("")
            aname = (self.session.animal.name
                     if self.session and self.session.animal else "the segment")
            if pid < 0:  # the animal itself
                self.toast.show_message(
                    f"Auto-paused: <b>{aname}</b> was lost (or left the frame) around frame "
                    f"{fail_frame}. If it is visible here, press <b>S</b>, click it, then Track. "
                    "Or turn <b>Auto-pause</b> off (bottom bar) to run through.", "warn", 15000)
                self.statusBar().showMessage(
                    f"Auto-paused: the segment was lost around frame {fail_frame}", 12000)
                return
            name = (self.session.points[pid].name
                    if self.session and pid < self.session.n_points else f"point {pid}")
            # The run stops at the first unreliable frame, not a window later:
            # whatever the worker emitted for this point
            # from that frame to the end of its last window is cut, so the
            # track ends exactly where it stopped being trustworthy.
            if self.session and pid < self.session.n_points:
                cut = self.session.clear_window([pid], fail_frame, max(fail_frame, last))
                if cut:
                    self._refresh_overlay()
                    self.timeline.refresh()
                self._on_select(pid)
            if reason == "lost" and self.session and pid < self.session.n_points and self.session.points[pid].is_ball:
                self.toast.show_message(
                    f"Stopped at frame {fail_frame}: SAM could not find the ball <b>{name}</b> any more, and it "
                    f"had not reached the picture edge, so its track ends at frame {fail_frame - 1}. If the ball "
                    "is visible here, select it, Add ▾ → Ball marker, click it, then Track. A ball that "
                    "leaves the picture never stops the run.", "warn", 15000)
                self.statusBar().showMessage(f"Stopped: the ball {name} was lost at frame {fail_frame}", 12000)
                return
            if reason == "apart" and self.session and pid < self.session.n_points:
                from cotracker_app.balls import CROP as _BALL_CROP
                self.toast.show_message(
                    f"Stopped at frame {fail_frame}: the ball <b>{name}</b> is too far from the other ball "
                    f"markers to be followed with them (the balls of one run share a {_BALL_CROP}-pixel window). "
                    "Clicking it again will not help: select this ball alone in the POINTS panel and press "
                    "Track, then do the same for each other ball.", "warn", 15000)
                self.statusBar().showMessage(f"Stopped: the ball {name} is too far from the others at frame "
                                             f"{fail_frame}", 12000)
                return
            if reason == "exit":
                self.toast.show_message(
                    f"Stopped at frame {fail_frame}: <b>{name}</b> left the segment's silhouette, so its "
                    f"track ends at frame {fail_frame - 1}. Click it where it really is and press Track. "
                    "If it is allowed off the segment, right-click it → <i>May leave the segment</i>.",
                    "warn", 15000)
                self.statusBar().showMessage(
                    f"Stopped: {name} left the segment at frame {fail_frame}", 12000)
                return
            self.toast.show_message(
                f"Auto-paused: <b>{name}</b> became unreliable at frame {fail_frame}, so its track is cut "
                "there (its timeline lane is empty from this frame on). It is selected: click where it really "
                "is on the video and press Track — or turn <b>Auto-pause</b> off (bottom bar) to run through.",
                "warn", 12000)
            self.statusBar().showMessage(
                f"Auto-paused: the model lost {name} at frame {fail_frame} "
                f"(low confidence; its track is cut there). Click where it really is and press Track — "
                f"or turn Auto-pause off to run through", 12000)
            return
        if last < getattr(self, "_run_start", 0):
            # paused during model load / warm-up: nothing was emitted, stay put
            self._apply_state()
            self._track_label.setText("")
            self.statusBar().showMessage(
                "Paused before any frames were tracked — press Track to try again", 6000)
            return
        self._goto(min(last, self.n_frames - 1), force=True)
        self._apply_state()
        self._track_label.setText("")
        if step_run:
            self.statusBar().showMessage(
                f"Stepped to frame {last} — press F for the next frame, or fix a "
                "point first (each step re-seeds from what you see; Ctrl+Z undoes it)",
                6000)
            return
        msg = (f"Paused at frame {last} — drag or Ctrl+click points to correct them, then press "
               "Track to re-track from here"
               if was_paused else f"Tracking complete (through frame {last})")
        self.statusBar().showMessage(msg, 8000)
        s = self.session
        ended = [(pid, f, why) for pid, (f, why) in sorted(ball_ended.items()) if s is not None and pid < s.n_points]
        if ended:
            # auto-pause off: the run went on, but these tracks stopped inside the
            # picture and nothing said so (I127)
            parts = [f"<b>{s.points[pid].name}</b> at frame {f}"
                     + (" (too far from the other balls: track it on its own)" if why == "apart" else "")
                     for pid, f, why in ended]
            self.toast.show_message(
                "Ball markers lost inside the picture (auto-pause is off, so the run went on): "
                + ", ".join(parts) + ". Their tracks end there. Select one, Add ▾ → Ball marker, click it "
                "where it is and press Track.", "warn", 15000)

    def _on_track_error(self, tb: str):
        decode_at = getattr(self.worker, "decode_failed_at", None) if self.worker is not None else None
        if self._retrack is not None:
            st = self._retrack
            self._retrack = None
            self._retrack_restore(st)
            self.statusBar().showMessage("Re-tracking stopped on an error; everything was put back", 8000)
        if self._model_dialog is not None:
            self._model_dialog.close()
            self._model_dialog = None
        if self.worker is not None:
            self.worker.wait(2000)
        self.worker = None
        self.state = READY
        self._step_run = False
        self._end_run_cleanup()  # else stale member dots / queued frames leak
        self._apply_state()
        self._track_label.setText("")
        self._refresh_animal_panel()
        if decode_at is not None:
            # a damaged frame mid-video, not a failure of the model: everything
            # before it was emitted and is kept (I40)
            self._autosave()
            self._goto(max(0, min(decode_at - 1, self.n_frames - 1)), force=True)
            self.toast.show_message(f"Tracking stopped: frame {decode_at} of the video could not be decoded. "
                                    "Everything before it is kept.", "warn", 12000)
            self.statusBar().showMessage(f"Stopped: frame {decode_at} could not be decoded", 12000)
            QMessageBox.warning(self, "Damaged frame in the video", tb[-1500:])
            return
        hint = _model_error_hint(tb)
        self.toast.show_message("Tracking stopped. " + (hint or "See the details dialog."),
                                "error", 10000)
        QMessageBox.critical(self, "Tracking failed",
                             (hint + "\n\n" if hint else "") + "Details:\n\n" + tb[-1500:])

    def _undo_run(self):
        if self._undo_snap is None or self.session is None or self.state != READY:
            return
        self.session.restore(self._undo_snap)
        self._undo_snap = None
        self.act_undo.setEnabled(False)
        self._refresh_point_list()
        self._refresh_overlay()
        self._update_frame_label()
        self._apply_state()  # Track/Export gates depend on the restored data
        self.statusBar().showMessage(
            "Restored tracks from before the last tracking run / edit", 5000)

    # ------------------------------------------------------- project / saves

    def _sync_ui_state(self):
        """Record the exact working state into the session before it is saved."""
        if self.session is None:
            return
        st = self.session.ui_state
        st.update(self.canvas.view_state())
        st["selected"] = -1 if self.selected is None else int(self.selected)
        st["follow"] = self.btn_follow.isChecked()
        st["autopause"] = self.btn_autopause.isChecked()
        st["roi"] = self.btn_roi.isChecked()
        st["track_mode"] = self._track_mode
        st["marker_size"] = self.marker_spin.value()
        st["show_mask"] = self.btn_mask.isChecked()
        st["mask_opacity"] = float(self._mask_opacity)
        st["show_midline"] = self.act_show_midline.isChecked()
        st["show_bones"] = self.act_show_bones.isChecked()
        st["seg_backend"] = self._seg_backend
        st["on_body"] = self.btn_onbody.isChecked()
        st["point_backend"] = self._point_backend
        st["trail_len"] = int(self._trail_len)
        st["trail_future"] = bool(self._trail_future)
        st["onion"] = self.act_onion.isChecked()
        st["loupe"] = self.act_loupe.isChecked()
        st["epipolar"] = self.act_epipolar.isChecked()
        st["display_filter"] = self._display_filter_key()
        st["region_shape"] = self._region_shape

    def _apply_ui_state(self):
        """Restore the exact working state after a project/autosave load."""
        if self.session is None:
            return
        st = self.session.ui_state
        self.btn_follow.setChecked(bool(st.get("follow", False)))
        self.btn_autopause.setChecked(bool(st.get("autopause", True)))
        self.btn_roi.setChecked(bool(st.get("roi", True)))
        self.btn_mask.setChecked(bool(st.get("show_mask", True)))
        self.btn_onbody.setChecked(bool(st.get("on_body", True)))
        pb = str(st.get("point_backend", "") or "")
        if pb == "alltracker" and not alltracker_backend.available():
            pb = "cotracker3"
        self._point_backend = pb if pb in ("alltracker", "cotracker3") else self._preferred_point_backend()
        (self.act_pm_alltracker if self._point_backend == "alltracker"
         else self.act_pm_cotracker).setChecked(True)
        self._mask_opacity = float(np.clip(st.get("mask_opacity", 0.35), 0.05, 0.9))
        self.act_show_midline.setChecked(bool(st.get("show_midline", True)))
        self.act_show_bones.setChecked(bool(st.get("show_bones", True)))
        backend = str(st.get("seg_backend", "") or "")
        # silently — restoring a project is not the user picking a model
        self._set_seg_backend(backend if backend in BACKENDS else preferred_backend(),
                              announce=False)
        mode = str(st.get("track_mode", "auto"))
        self._track_mode = mode if mode in ("auto", "semi") else "auto"
        (self.act_mode_semi if self._track_mode == "semi"
         else self.act_mode_auto).setChecked(True)
        self.marker_spin.setValue(int(np.clip(st.get("marker_size", 3), 2, 24)))
        tl = int(st.get("trail_len", TRAIL_FRAMES))
        self._set_trail_len(tl if tl in self._trail_acts else TRAIL_FRAMES, announce=False)
        self.act_trail_future.setChecked(bool(st.get("trail_future", False)))
        self.act_onion.setChecked(bool(st.get("onion", False)))
        self.act_loupe.setChecked(bool(st.get("loupe", False)))
        self.act_epipolar.setChecked(bool(st.get("epipolar", True)))
        self._set_display_filter(str(st.get("display_filter", "none")), announce=False)
        self._set_region_shape(str(st.get("region_shape", "circle")), announce=False)
        if not self.session.annotator:
            self.session.annotator = self._default_annotator()
        sel = int(st.get("selected", -1))
        if 0 <= sel < self.session.n_points:
            self._on_select(sel)
        # view restore runs after the pending layout pass, else fit() wins
        zoom = float(st.get("zoom", 0.0))
        cx, cy = float(st.get("center_x", 0.0)), float(st.get("center_y", 0.0))
        uz = bool(st.get("user_zoomed", False))
        QTimer.singleShot(0, lambda: self.canvas.set_view_state(zoom, cx, cy, uz))

    def _autosave_path(self) -> Path | None:
        if self.project_path is not None:
            return self.project_path
        # keyed on the ACTIVE camera's video, so reopening that video finds it
        if self.info is not None:
            return Path(self.info.path + AUTOSAVE_SUFFIX)
        return None

    def _autosave(self):
        # also fires mid-tracking (same GUI thread as write_segment — no race),
        # so a crash never loses more than 30 s of tracking work
        if self.project is not None and self.project.dirty:
            self._sync_ui_state()
            path = self._autosave_path()
            if path is not None:
                try:
                    # always the project (schema v5): a one-camera project is byte-wise
                    # a single view, and Project.load_npz still reads v3 files
                    self.project.save_npz(path)
                    self.statusBar().showMessage(f"Saved ✓  {time.strftime('%H:%M:%S')}", 2000)
                except OSError as e:
                    # a project file that cannot be written: fall back to the
                    # video-side autosave once and say so (I104)
                    side = Path(self.info.path + AUTOSAVE_SUFFIX) if self.info is not None else None
                    if path == self.project_path and side is not None and side != path:
                        try:
                            self.project.save_npz(side)
                            if not getattr(self, "_autosave_fallback_said", False):
                                self._autosave_fallback_said = True
                                self.toast.show_message(
                                    f"Could not write {path.name} ({e}). Your work is autosaved next to the "
                                    f"video instead ({side.name}); use File → Save Project As… to choose a "
                                    "place you can write to.", "warn", 15000)
                            return
                        except OSError:
                            pass
                    self.statusBar().showMessage(f"Autosave failed: {e}", 10000)

    def _save_project(self) -> bool:
        if self.project_path is None:
            return self._save_project_as()
        if self.project is None:
            return False
        self._sync_ui_state()
        try:
            self.project.save_npz(self.project_path)
        except Exception as e:  # noqa: BLE001 - a failed save used to be silent (I104)
            QMessageBox.critical(
                self, "Could not save the project",
                f"{self.project_path}\n\n{e}\n\nNothing was written there. Check that the folder exists, "
                "that you may write to it and that the file is not open elsewhere, then use "
                "File → Save Project As… to save somewhere else. Your work is still in the program.")
            return False
        extra = (f" ({self.project.n_views} cameras)" if self.project.n_views > 1 else "")
        self.statusBar().showMessage(
            f"Project saved ✓  {self.project_path.name}{extra}", 4000)
        # the video-side autosave written before the project had a file is now an
        # OLDER copy of what was just saved; left in place it offered to "resume"
        # that older state the next time the video was opened on its own (I129)
        side = Path(self.info.path + AUTOSAVE_SUFFIX) if self.info is not None else None
        if side is not None and side != self.project_path and side.exists():
            try:
                side.unlink()
            except OSError:
                pass
        return True

    def _save_project_as(self) -> bool:
        if self.session is None:
            return False
        start = str(Path(self.info.path).with_suffix(PROJECT_SUFFIX)) if self.info else ""
        path, _ = QFileDialog.getSaveFileName(self, "Save project", start,
                                              f"Kinetrace project (*{PROJECT_SUFFIX})")
        if not path:
            return False
        if not path.endswith(PROJECT_SUFFIX):
            path += PROJECT_SUFFIX
        before = self.project_path
        self.project_path = Path(path)
        if not self._save_project():
            self.project_path = before     # autosave must not keep aiming at a path that failed (I104)
            return False
        return True

    def _open_project_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "",
                                              f"Kinetrace project (*{PROJECT_SUFFIX})")
        if path:
            self._open_project_from_path(path)

    def _locate_video(self, want: Path, label: str) -> Path | None:
        """Ask the user where a project's video went (projects are portable;
        the footage next to them often is not)."""
        if want.exists():
            return want
        QMessageBox.information(
            self, "Locate video",
            f"{label}'s video was not found at:\n{want}\n\nPlease locate it.")
        vpath, _ = QFileDialog.getOpenFileName(self, f"Locate video for {label}", "", VIDEO_FILTER)
        return Path(vpath) if vpath else None

    def _open_project_from_path(self, path: str):
        try:
            proj = Project.load_npz(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open project", f"{path}\n\n{e}")
            return
        if proj.n_views == 0:
            QMessageBox.critical(self, "Could not open project", f"{path}\n\nNo camera views.")
            return
        master = proj.sessions[proj.active]
        video = self._locate_video(Path(master.video_path), proj.name(proj.active))
        if video is None:
            self.statusBar().showMessage(
                f"Project not opened: the video of {proj.name(proj.active)} was not located. Open the project "
                "again and point to the video when asked (the project file is unchanged)", 10000)
            return

        def adopt(info: VideoInfo):
            if info.n_frames != master.n_frames:
                if getattr(info, "header_frames", 0) == master.n_frames and info.n_frames < master.n_frames:
                    # the project was saved when the header's count was trusted;
                    # the extra rows never had a picture behind them
                    QMessageBox.information(
                        self, "Shorter than its header says",
                        f"This video's file header claims {info.header_frames} frames but only "
                        f"{info.n_frames} can be decoded, and the project was saved with the header's "
                        f"count. Nothing is lost: the last {info.header_frames - info.n_frames} frame(s) "
                        "never had a picture. They stay blank in the timeline.")
                else:
                    QMessageBox.warning(
                        self, "Video mismatch",
                        f"The selected video has {info.n_frames} frames but the project "
                        f"expects {master.n_frames}. Loading anyway — verify your tracks. If this is "
                        "a different cut of the same recording, the tracks will sit on the wrong frames.")
            master.video_path = info.path
            self.project_path = Path(path)
            self._adopt_project(proj, info)

        # _attach_video opens the ACTIVE view; the rest are attached in adopt
        self._open_video(str(video), then=adopt)

    def _adopt_project(self, proj: Project, active_info: VideoInfo):
        """Replace the one-view project `_attach_video` just built with the
        loaded one, opening every other camera's video."""
        active = proj.active
        self.project = proj
        self._wand_result = (None, None)      # (I33)
        runtimes: list[_ViewRuntime | None] = [None] * proj.n_views
        runtimes[active] = self._views[0]
        drop: list[int] = []
        for i in range(proj.n_views):
            if i == active:
                continue
            s = proj.sessions[i]
            vid = self._locate_video(Path(s.video_path), proj.name(i))
            if vid is None:
                drop.append(i)
                continue
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                info = probe_video(str(vid))
            except (ValueError, OSError) as e:
                QMessageBox.warning(self, "Could not open camera",
                                    f"{proj.name(i)}: {e}\n\nIt is left out of this session.")
                drop.append(i)
                continue
            finally:
                if QApplication.overrideCursor() is not None:
                    QApplication.restoreOverrideCursor()
            s.video_path = info.path
            runtimes[i] = _ViewRuntime(info, DEFAULT_CACHE_BYTES)
            if info.n_frames != s.n_frames:
                # the scrubber must never index past the session's arrays (I106)
                runtimes[i].n_frames = min(info.n_frames, s.n_frames)
                QMessageBox.warning(
                    self, "Camera video length differs",
                    f"{proj.name(i)}: the video has {info.n_frames} frames but the project remembers "
                    f"{s.n_frames}. If it is a different cut or another take, its tracks sit on the wrong "
                    "frames — locate the original video. The camera is limited to "
                    f"{runtimes[i].n_frames} frames meanwhile.")
        lost = [proj.name(i) for i in drop]
        for i in reversed(drop):       # descending keeps the indices valid
            del runtimes[i]
            proj.remove_view(i)
        if lost:
            # The project FILE still holds these cameras. Autosave used to write
            # the reduced project straight back over it within 30 s, deleting
            # their tracks and calibration for good (I14): from here on autosave
            # goes next to the video, and Save asks for a file name.
            kept = self.project_path
            self.project_path = None
            QMessageBox.warning(
                self, "Opened without some cameras",
                f"{', '.join(lost)}: the video was not found or could not be opened, so "
                f"{'this camera is' if len(lost) == 1 else 'these cameras are'} left out of this session.\n\n"
                f"Your project file is NOT changed: {Path(kept).name if kept else 'it'} still holds "
                f"{'that camera' if len(lost) == 1 else 'those cameras'} with every point, track and the "
                "calibration. Nothing will be saved over it automatically; Save asks for a file name.\n\n"
                "To work with all cameras, close without saving, make the videos reachable (connect the "
                "drive, or move them next to the project), and open the project again.")
        self._views = [rt for rt in runtimes if rt is not None]
        self._rebudget_caches()
        for cv in self.grid.set_count(proj.n_views):
            self._wire_canvas(cv)
        for i, rt in enumerate(self._views):
            cv = self.grid.canvas(i)
            cv.set_video_size(rt.info.width, rt.info.height)
            if i != proj.active:
                cv.set_interactive(False)
        self.grid.set_active(proj.active)
        self.act_solo.setEnabled(proj.n_views > 1)
        QTimer.singleShot(0, self._update_disagreement)   # a saved reconstruction paints its disagreement

    # -------------------------------------------------------------------- 3D

    def _view_sizes(self) -> list[tuple[int, int]]:
        return [(s.width, s.height) for s in self.project.sessions] if self.project else []

    def _import_calibration(self):
        p = self.project
        if p is None or p.n_views < 2:
            QMessageBox.information(self, "Add cameras first",
                                    "A calibration describes several cameras filming the same moment, so it is "
                                    "imported into a project that has them all: add the other cameras' videos "
                                    "first with ＋ Add video in the CAMERAS panel (in the order the calibration "
                                    "lists them), then 3D → Import Calibration again.")
            return
        start = str(Path(self.project_path or self.info.path).parent) if self.info else ""
        dlg = CalibrationDialog(self, list(p.names), self._view_sizes(), start)
        if dlg.exec() != QDialog.Accepted or dlg.result_calibration is None:
            return
        p.calibration = dlg.result_calibration
        self._wand_result = (None, None)   # the last wand run's report is not this calibration's (I33)
        p.reconstruction = None
        self._hull_cache.clear()
        p.dirty = True
        self._apply_state()
        self.toast.show_message(
            f"Calibration loaded for {p.n_views} cameras ({p.calibration.source}). "
            "Next: <b>3D → Reconstruct 3D Landmarks</b> (Ctrl+3).", "info", 8000)

    def _lens_wizard(self):
        """3D → Calibrate a Lens: intrinsics + distortion of ONE camera from a
        checkerboard video, attached to that camera of this project."""
        from cotracker_app.lenswizard import LensWizard
        p = self.project
        if self.state == TRACKING:
            return
        if p is None or self.state != READY:
            # no video open: the wizard still opens (it has its own board-video
            # picker); the result is a lens FILE to attach later. It used to refuse
            # with "Open the checkerboard video first" (G9)
            wiz = LensWizard(self, None, "", None, str(Path.home()))
            if wiz.exec() == QDialog.Accepted and wiz.result_profile is not None \
                    and not getattr(wiz, "saved_path", ""):
                if QMessageBox.question(
                        self, "Save the lens profile?",
                        "No video is open, so the profile is not attached to a camera and would be lost "
                        "when this window closes. Save it as a lens file now?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes:
                    path, _ = QFileDialog.getSaveFileName(self, "Save lens profile", str(Path.home() / "camera.klens.json"),
                                                          "Kinetrace lens (*.klens.json)")
                    if path:
                        saved = wiz.result_profile.save(path)
                        self.toast.show_message(f"Lens profile saved: {Path(saved).name}. Load it for its camera "
                                                "with 3D → Calibrate a Lens → I already have a lens file…",
                                                "success", 9000)
            return
        start = str(Path(self.project_path or self.info.path).parent) if self.info else ""
        wiz = LensWizard(self, p, self.info.path if self.info else "", p.active, start)
        if wiz.exec() != QDialog.Accepted or wiz.result_profile is None or wiz.result_view is None:
            return
        from cotracker_app.calibwizard import lens_size_mismatch
        sv = p.sessions[wiz.result_view]
        bad = lens_size_mismatch(wiz.result_profile, sv.width, sv.height, p.name(wiz.result_view))
        if bad:                                  # a profile of another picture size (I31)
            QMessageBox.warning(self, "Lens profile not attached", bad[0].upper() + bad[1:])
            return
        while len(p.lenses) < p.n_views:
            p.lenses.append(None)
        p.lenses[wiz.result_view] = wiz.result_profile
        p.dirty = True
        v = str(wiz.result_profile.report.get("verdict", "loaded")).upper()
        self.toast.show_message(
            f"Lens profile attached to {p.name(wiz.result_view)} ({v}: {wiz.result_profile.summary()}). "
            "The wand calibration will use it; save the project to keep it.", "success", 9000)

    def _wand_wizard(self):
        """3D → Calibrate Cameras with a Wand: the native easyWand replacement,
        explained for a first-time user. The result becomes this project's
        calibration (and can be exported for the animal projects)."""
        from cotracker_app.calibwizard import WandWizard
        p = self.project
        if self.state != READY:
            return
        # one video is enough to OPEN the wizard (the wand may be tracked in
        # one camera for other reasons); the pages say what a calibration
        # still needs. Only "no video at all" is refused here.
        if p is None or p.n_views < 1:
            QMessageBox.information(
                self, "Open a wand video first",
                "Wand calibration works on the wand recording of your cameras, in ONE project:\n\n"
                "1. File → Open Video… — the wand video of the first camera.\n"
                "2. In the CAMERAS panel on the right, ＋ Add video — the wand video of each other "
                "camera; set their offsets so they show the same instant (3D → Sync Cameras).\n"
                "3. In every camera, track the two wand ends (Add ▾ → Ball marker on each ball, or N "
                "on a mark; the SAME names in every camera, e.g. 'wand A' and 'wand B'), then Track ▶.\n"
                "4. Then 3D → Calibrate Cameras with a Wand… again.\n\n"
                "Help → User Manual (F1), section 10, walks through all of it.")
            return
        start = str(Path(self.project_path or self.info.path).parent) if self.info else ""
        wiz = WandWizard(self, p, start)
        if wiz.exec() != QDialog.Accepted or wiz.result_calibration is None:
            return
        p.calibration = wiz.result_calibration
        self._wand_result = (wiz.result, wiz.gravity)
        p.reconstruction = None
        self._hull_cache.clear()
        p.dirty = True
        self._apply_state()
        v = str(wiz.result.report.get("verdict", "?")).upper() if wiz.result is not None else "?"
        self.toast.show_message(
            f"Wand calibration ({v}) is now this project's calibration. Save the project (Ctrl+S); "
            "<b>3D → Export Calibration</b> writes it for your animal projects.", "success", 9000)

    def _export_calibration(self):
        from cotracker_app.calibwizard import save_calibration_files
        p = self.project
        if not self._need_calibration("Export Calibration"):
            return
        base = Path(self.project_path or self.info.path)
        start = str(base.with_suffix("")) + ".kcal.json"
        path, _ = QFileDialog.getSaveFileName(self, "Export calibration", start,
                                              "Kinetrace calibration (*.kcal.json)")
        if not path:
            return
        res, grav = getattr(self, "_wand_result", (None, None))
        try:
            written = save_calibration_files(res, grav, path, cal=p.calibration)
        except Exception as e:      # noqa: BLE001
            QMessageBox.critical(self, "Export calibration", str(e))
            return
        names = ", ".join(Path(w).name for w in written)
        from cotracker_app.calibwizard import dlt_csv_caveat
        note = dlt_csv_caveat(p.calibration, list(p.names))
        # with lens corrections the dltCoefs.csv is for lens-corrected pixels (I32)
        self.toast.show_message(f"Calibration exported: {names}" + (f"<br>{note}" if note else ""),
                                "warn" if note else "success", 12000 if note else 8000)
        self.statusBar().showMessage(f"Calibration exported: {names}", 8000)

    def _t_range_3d(self) -> tuple[int, int]:
        """Reference-frame window to reconstruct: every instant any camera
        tracked, clipped to the instants at least two cameras recorded.

        In REFERENCE frames, because that is what `calib.reconstruct` and
        `estimate_offsets` take: this used to be computed in the working
        camera's numbering, so working in camera 2 (offset / rate) shifted and
        stretched the triangulated window (I13), and it demanded that EVERY
        camera had a frame, so one short overview clip shrank all 3D (I15)."""
        p = self.project
        span = p.reference_span(min_views=2)
        if span is None:
            return (0, -1)
        # the instants at least two cameras TRACKED (each camera's tracked span in
        # reference time): only there can anything be triangulated
        tracked = []
        for c, s in enumerate(p.sessions):
            rows = np.nonzero(s.tracked.any(axis=1))[0]
            if len(rows):
                tracked.append((p.reference_time(c, int(rows[0])), p.reference_time(c, int(rows[-1]))))
        both = p.span_of(tracked, 2)
        if both is not None:
            span = (max(span[0], both[0]), min(span[1], both[1]))
        lo = int(np.ceil(span[0] - 1e-9))
        hi = int(np.floor(span[1] + 1e-9))
        return (lo, hi) if lo <= hi else (0, -1)

    def _sync_dialog(self):
        """3D -> Sync Cameras (Sound / Motion): whole-frame offsets from the sound tracks or the pictures."""
        from cotracker_app.syncdialog import SyncDialog
        p = self.project
        if p is None or p.n_views < 2:
            QMessageBox.information(
                self, "Add cameras first",
                "Syncing lines up two or more cameras of the same event. Open the first camera's "
                "video, then add the others with ＋ Add video in the CAMERAS panel.")
            return
        if self.state != READY:
            return
        dlg = SyncDialog(self, p, [rt.info.path for rt in self._views], self.current)
        if dlg.exec() != QDialog.Accepted:
            return
        n = getattr(dlg, "applied", 0)
        p.dirty = True
        p.reconstruction = None
        self._hull_cache.clear()
        self._refresh_companions()
        self._refresh_cameras()
        self._apply_state()
        if n:
            self.toast.show_message(
                f"Offsets applied to {n} camera(s). Check by eye: step to a moment every camera saw and "
                "confirm it shows at the same time in each view. Later, after tracking, "
                "<b>3D → Estimate Sub-frame Offsets</b> refines them to a fraction of a frame.",
                "success", 12000)

    def _estimate_offsets_dialog(self):
        from cotracker_app.calib import estimate_offsets
        p = self.project
        if not self._need_calibration("Estimate Sub-frame Offsets"):
            return
        t0, t1 = self._t_range_3d()
        if t1 < t0:
            self.toast.show_message("Nothing tracked in the overlap yet — track the cameras first.", "warn", 6000)
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        rep: dict = {}
        try:
            offs, before, after = estimate_offsets(p.sessions, p.calibration, p.rates, p.offsets, (t0, t1),
                                                   report=rep)
        except Exception as e:      # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Sub-frame offsets", str(e))
            return
        QApplication.restoreOverrideCursor()
        if not np.isfinite(before) or not np.isfinite(after):
            QMessageBox.information(self, "Sub-frame offsets",
                                    "No landmark is seen by two cameras at the same instant, so there is "
                                    "nothing to align. Track the same named landmarks in at least two cameras.")
            return
        per_view = rep.get("per_view", {})
        words = {"sharp": "well determined", "weak": "weakly determined",
                 "flat": "NOT determined -- do not trust", "none": "unchanged: no shared tracks here"}
        rows = "\n".join(f"  {p.name(i)}: {p.offsets[i]:+.3f} → {offs[i]:+.3f}   ({words.get(per_view.get(i), '')})"
                         for i in range(p.n_views) if i)
        # The estimator always returns SOME number; the report says whether the
        # tracks actually support it (a flat residual surface once produced
        # -1.26 then -0.32 frames on a real stereo pair). A flat or weak
        # result defaults the question to No.
        verdict = rep.get("verdict", "weak")
        word = {"reliable": "RELIABLE", "weak": "WEAK", "flat": "NOT SUPPORTED"}.get(verdict, verdict)
        default = QMessageBox.Yes if verdict == "reliable" else QMessageBox.No
        if QMessageBox.question(
                self, f"Sub-frame offsets: {word}",
                f"{rep.get('why', '')}\n\n"
                f"Camera disagreement {before:.3f} px → {after:.3f} px (mean) over frames {t0}–{t1}, "
                f"scored on {rep.get('n_cells', 0)} landmark-frames.\n\n"
                f"Offsets (frames of each camera):\n{rows}\n\n"
                + ("Apply them?" if verdict == "reliable" else
                   "Apply them anyway? (No keeps the current offsets.)"),
                QMessageBox.Yes | QMessageBox.No, default) != QMessageBox.Yes:
            return
        p.offsets = [float(o) for o in offs]
        p.dirty = True
        p.reconstruction = None
        self._hull_cache.clear()
        self._refresh_companions()
        self._refresh_cameras()
        self._reconstruct_3d(quiet=True)

    def _reconstruct_3d(self, quiet: bool = False):
        from cotracker_app.calib import reconstruct
        p = self.project
        if not self._need_calibration("Reconstruct 3D Landmarks"):
            return
        t0, t1 = self._t_range_3d()
        if t1 < t0:
            self.toast.show_message("Nothing tracked in the overlap yet — track the cameras first.", "warn", 6000)
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            p.reconstruction = reconstruct(p.sessions, p.calibration, p.rates, p.offsets, (t0, t1))
        except Exception as e:      # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "3D reconstruction", str(e))
            return
        QApplication.restoreOverrideCursor()
        p.dirty = True
        r = p.reconstruction
        solved = int(np.isfinite(r.residual).sum())
        med = float(np.nanmedian(r.residual)) if solved else float("nan")
        self._apply_state()
        if not self.act_view3d.isChecked():
            self.act_view3d.setChecked(True)
            self._toggle_view3d(True)
        else:
            self._refresh_view3d(force=True)
        # A number without a verdict is what let a real stereo test export
        # 38 px of camera disagreement as if it were data: say what the residual
        # means and what to do, every time, and keep the verdict with the result.
        from cotracker_app.calib import reconstruction_report
        widths = [s.width for s in p.sessions if s.width] or [1920]
        rep = reconstruction_report(r, p.n_views, float(max(widths)))
        self._recon_report = rep
        self._update_disagreement()             # per-camera disagreement onto the timeline
        self._refresh_guides()
        v = rep["verdict"]
        if not quiet:
            word = {"good": "GOOD", "ok": "USABLE, with care", "poor": "NOT TRUSTWORTHY",
                    "empty": "NOTHING TO SHOW"}.get(v, v)
            body = "\n\n".join(f"• {x}" for x in rep["reasons"])
            self.toast.show_message(
                f"3D {word}: {solved} landmark positions over frames {t0}–{t1}, cameras disagree by "
                f"{med:.2f} px (median). Export with Ctrl+E (choose \"3D landmarks\" or \"3D kinematics\"); "
                "carve the volume with Ctrl+4.",
                "success" if v == "good" else ("warn" if v in ("ok", "empty") else "error"), 9000)
            if v != "good":
                QMessageBox.information(
                    self, f"3D reconstruction: {word}",
                    f"Frames {t0}–{t1}, {p.n_views} cameras.\n\n{body}\n\n"
                    "What the number means: for every landmark, the 3D point is projected back into "
                    "each camera; the residual is how far (in pixels) that lands from where you tracked "
                    "it. Zero would mean every camera agrees perfectly.")
            # the offer: stretches where ONE camera disagrees can be re-seeded on the
            # other cameras' rays and re-tracked automatically
            from cotracker_app import retrack
            n_st = len([q for q in retrack.plan(p, self._disagree_thresholds()) if q.target is not None])
            if n_st and self._retrack is None:
                self.toast.show_message(
                    f"{n_st} stretch{'es' if n_st != 1 else ''} where one camera disagrees with the others "
                    "(magenta band on the timeline). <b>3D → Re-track Disagreeing Stretches</b> puts the landmark "
                    "back on the other cameras' rays and re-tracks it, with a before / after verdict.",
                    "warn", 12000)

    def _masks_at_instant(self, t: int):
        """(cameras, raw masks) at reference instant t — None where a camera
        has no silhouette there."""
        p = self.project
        masks = []
        for c, s in enumerate(p.sessions):
            f = int(round(p.local_frame(c, t)))
            masks.append(s.masks.rasterize(f, s.height, s.width)
                         if s.masks is not None and s.masks.has(f) else None)
        return masks

    def _carve_hull_here(self, quiet: bool = False):
        from cotracker_app.hull import bounds_from_points, carve, hull_mesh, mesh_volume
        p = self.project
        if not self._need_calibration("Carve Volume"):
            return
        t = self._reference_instant()         # the hull cache is keyed by the reference instant (I114)
        masks = self._masks_at_instant(t)
        n_masks = sum(1 for m in masks if m is not None)
        if n_masks < 3:
            # Two silhouettes carve a sliver the full depth of the intersection
            # (a real stereo test: a 1.2-1.7 m deep "body"); the volume
            # means nothing until a third, well-separated camera bounds it.
            if not quiet:
                QMessageBox.information(
                    self, "Volume needs three cameras",
                    f"Only {n_masks} camera(s) have a silhouette at this instant. A volume carved from "
                    "two silhouettes is a long sliver along the line between the cameras -- its size "
                    "is not the animal's. Segment the animal in at least THREE cameras that look at it "
                    "from clearly different directions, then carve again.")
            return
        pts = None
        r = p.reconstruction
        if r is not None and r.t0 <= t < r.t0 + r.n_frames:
            row = r.xyz[t - r.t0]
            row = row[np.isfinite(row).all(axis=1)]
            pts = row if len(row) else None
        if pts is None:
            self.toast.show_message(
                "Reconstruct the landmarks first (Ctrl+3): they tell the carver where in space "
                "to look.", "warn", 7000)
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            cams = p.calibration.cameras
            span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) if len(pts) > 1 else 0.0
            margin = max(2.0 * span, 0.05)
            cut_off = False
            for _grow in range(5):
                lo, hi = bounds_from_points(pts, margin)
                coarse = carve(cams, masks, lo, hi, margin / 25.0, dilate_px=3)
                ext = coarse.extent()
                if ext is None:
                    raise ValueError("the silhouettes do not intersect anywhere near the landmarks — "
                                     "check the calibration's pixel convention and the camera order")
                # the box comes from the LANDMARKS: when they cover part of the body
                # the volume ran into its walls and was cut off in silence (I105)
                tol = margin / 25.0 * 1.5
                cut_off = bool((ext[0] <= np.asarray(lo) + tol).any() or (ext[1] >= np.asarray(hi) - tol).any())
                if not cut_off:
                    break
                margin *= 2.0
            voxel = max(float(np.linalg.norm(ext[1] - ext[0])) / 120.0, 1e-6)
            h = carve(cams, masks, ext[0] - 8 * voxel, ext[1] + 8 * voxel, voxel, dilate_px=2)
            verts, faces = hull_mesh(h)
        except Exception as e:      # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Volume hull", str(e))
            return
        QApplication.restoreOverrideCursor()
        if cut_off:
            QMessageBox.warning(
                self, "Volume hull cut off",
                "The volume still reaches the edge of the space searched around the landmarks after "
                "enlarging it several times, so the reported volume is cut off (too small). Usually a "
                "silhouette includes the background here; check each camera's silhouette at this frame.")
        self._hull_cache[t] = (verts, faces, h)
        self._apply_state()
        if not self.act_view3d.isChecked():
            self.act_view3d.setChecked(True)
            self._toggle_view3d(True)
        else:
            self._refresh_view3d(force=True)
        if not quiet:
            unit = p.calibration.unit or "unit"
            self.toast.show_message(
                f"Volume hull at frame {t} of {p.name(0)}: {h.n_views} cameras, {h.volume():.3g} {unit}³ "
                f"({mesh_volume(verts, faces):.3g} {unit}³ as a mesh), voxel {voxel:.3g} {unit}. "
                "3D → Export Mesh writes it as OBJ/PLY.", "info", 9000)

    def _toggle_view3d(self, on: bool):
        if on:
            if self.view3d is None:
                self.view3d = View3D(self)
                self.view3d.closed.connect(lambda: self.act_view3d.setChecked(False))
            self.view3d.show()
            self._refresh_view3d(force=True)
        elif self.view3d is not None:
            self.view3d.hide()

    def _refresh_view3d(self, force: bool = False):
        """Point the 3D window at the current instant (cheap when hidden)."""
        p = self.project
        if self.view3d is None or not self.view3d.isVisible() or p is None:
            return
        scene = Scene3D()
        r = p.reconstruction
        t = self.current if p.active == REFERENCE_VIEW else int(round(
            p.map_frame_exact(p.active, REFERENCE_VIEW, self.current)))
        info = f"reference frame {t}"
        if r is not None:
            scene.points_all = r.xyz
            scene.names = list(r.names)
            if r.t0 <= t < r.t0 + r.n_frames:
                scene.points = r.xyz[t - r.t0]
                n_ok = int(np.isfinite(scene.points).all(axis=1).sum())
                res = r.residual[t - r.t0]
                med = float(np.nanmedian(res)) if n_ok else float("nan")
                info += f" · {n_ok}/{len(r.names)} landmarks · residual {med:.2f} px"
            idx = {nm: i for i, nm in enumerate(r.names)}
            s = p.session
            scene.bones = [(idx[a], idx[b]) for a, b in
                           (p.session.skeleton.get("bones", []) if s and s.skeleton else [])
                           if a in idx and b in idx]
            scene.unit = r.unit
        if p.calibration is not None:
            scene.cameras = np.array([c.center() for c in p.calibration.cameras])
            scene.camera_names = list(p.names)
        if t in self._hull_cache:
            verts, faces, h = self._hull_cache[t]
            scene.mesh = (verts, faces)
            unit = p.calibration.unit if p.calibration else ""
            info += f" · hull {h.volume():.3g} {unit}³ from {h.n_views} cameras"
        self.view3d.set_scene(scene, info)

    # ------------------------------------------------------------------ body

    def _lens_focal_px(self) -> float | None:
        """This camera's measured focal length in pixels, when it has one. A
        3D body model otherwise has to guess how wide the lens is."""
        p = self.project
        if p is None or not getattr(p, "lenses", None):
            return None
        prof = p.lenses[p.active] if p.active < len(p.lenses) else None
        if prof is None or self.info is None:
            return None
        if (int(prof.width), int(prof.height)) != (int(self.info.width), int(self.info.height)):
            return None             # a profile of another picture size has another focal length (I31)
        return float(prof.f_square)            # a property: calling it crashed the Body run (I122)

    def _body_run(self):
        """Body → Find People && Measure Joints: run a pose backend over a
        frame range on a worker thread, with a cancellable progress dialog."""
        from cotracker_app.bodyview import BodyPoseWorker, BodyRunDialog
        s = self.session
        if s is None or self.state != READY or self._body_worker is not None:
            return
        has_masks = s.masks is not None and s.masks.n_masked() > 0
        # existing=: the dialog asks before a run that cannot be merged (I82)
        dlg = BodyRunDialog(self, self.n_frames, self.current, self.timeline.sel_range,
                            has_masks, self._body_backend, self._lens_focal_px(),
                            existing=s.body)
        if dlg.exec() != QDialog.Accepted or dlg.result_options is None:
            return
        opts = dlg.result_options
        self._body_backend = opts.backend
        self._undo_snap = s.snapshot()      # Ctrl+Z takes the whole run back
        self.act_undo.setEnabled(True)
        # target=: the result belongs to THIS camera, whatever is active when it ends (I83)
        w = BodyPoseWorker(self.info.path, self.n_frames, opts, s.masks, float(s.fps or 0.0),
                           target=s)
        self._body_worker = w
        total = max(1, (opts.end - opts.start) // max(1, opts.step) + 1)
        dl = QProgressDialog("Looking for people…", "Stop", 0, total, self)
        dl.setWindowTitle("Body pose")
        dl.setWindowModality(Qt.NonModal)
        dl.setMinimumDuration(0)
        dl.setAutoClose(False)
        dl.setAutoReset(False)
        dl.canceled.connect(w.request_cancel)
        self._body_progress = dl
        w.progress.connect(self._on_body_progress)
        w.finished_ok.connect(self._on_body_done)
        w.error.connect(self._on_body_error)
        w.finished.connect(self._end_body_run)
        dl.show()
        w.start()
        self._apply_state()

    def _on_body_progress(self, done: int, total: int, note: str):
        if self._body_progress is not None:
            self._body_progress.setMaximum(max(1, total))
            self._body_progress.setValue(done)
            self._body_progress.setLabelText(f"Looking for people…  {note}")

    def _on_body_done(self, track):
        from cotracker_app.body import merge_run
        w = self.sender()
        if w is None or w is not self._body_worker:
            return                      # a run torn down with its video (I83)
        s = getattr(w, "target", None) or self.session
        views = list(self.project.sessions) if self.project is not None else []
        k = next((i for i, v in enumerate(views) if v is s), -1)
        if k < 0 or not w.result_fits(s):
            self.toast.show_message(
                "The body run finished for a video or camera that is no longer open, so its "
                "result was not stored.", "warn", 8000)
            return
        # merged, not swapped in: a re-run over part of the video, or a stopped
        # run, keeps every pose outside the frames it processed (I82)
        merged, note = merge_run(s.body, track)
        s.body = merged
        s._touch()
        if s is not self.session:
            self.toast.show_message(
                f"The body run for {self.project.name(k)} finished and was stored in that "
                f"camera. {note}".strip(), "info", 8000)
            self.timeline.update()
            return
        if not track.n_posed():
            self.toast.show_message(
                "No person was found anywhere in that range"
                + ("; the earlier poses are unchanged. " if merged.n_posed() else ". ")
                + "Try the segment tool (S) to draw round the person, then run again using "
                "the silhouette.", "warn", 8000)
        else:
            self.toast.show_message(f"{merged.summary()}. {note}".strip(), "info", 8000)
            if self.body_win is None:
                self.act_body_view.setChecked(True)
                self._toggle_body_view(True)
        self.timeline.update()
        self._refresh_body_view(force=True)

    def _on_body_error(self, msg: str):
        QMessageBox.warning(self, "Body pose", msg)

    def _end_body_run(self):
        if self._body_progress is not None:
            self._body_progress.close()
            self._body_progress = None
        self._body_worker = None
        self._apply_state()

    def _toggle_body_view(self, on: bool):
        from cotracker_app.bodyview import BodySideBySide
        if on:
            if self.body_win is None:
                self.body_win = BodySideBySide(self)
                self.body_win.closed.connect(lambda: self.act_body_view.setChecked(False))
                self.body_win.export_requested.connect(self._export_body_video)
            self.body_win.show()
            self._refresh_body_view(force=True)
        elif self.body_win is not None:
            self.body_win.hide()

    def _refresh_body_view(self, force: bool = False):
        """Point the side-by-side window at the current frame (cheap when it
        is hidden, which is why every seek may call it)."""
        w = self.body_win
        if w is None or not w.isVisible():
            return
        s = self.session
        if s is None:
            return
        if force or w.track is not s.body:
            w.set_track(s.body, float(s.fps or 0.0))
        import cv2
        rgb = self._current_rgb()
        if rgb is None and self.cache is not None:
            # not decoded yet: show the nearest cached frame rather than an
            # empty panel, exactly as `_goto` does for the canvas. The real
            # frame arrives through `_on_seek_frame` a moment later.
            near = self.cache.nearest(self.current)
            rgb = None if near is None else near[1]
        bgr = None if rgb is None else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        w.set_frame(self.current, bgr)

    def _body_default_path(self, tail: str) -> str:
        base = Path(self.project_path or self.info.path)
        return str(base.with_suffix("")) + tail

    def _export_body_joints(self):
        s = self.session
        if s is None or not s.has_body():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export joint positions",
                                              self._body_default_path("_joints.csv"),
                                              "CSV (*.csv)")
        if not path:
            return
        try:
            s.export_body_joints_csv(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Export", str(exc))
            return
        self.toast.show_message(f"Joint positions written to {Path(path).name}.", "info", 5000)

    def _export_body_angles(self):
        """The angles CSV plus a plain-language report beside it -- the numbers
        are useless without the convention each one uses."""
        from cotracker_app.body import angle_report
        s = self.session
        if s is None or not s.has_body():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export joint angles",
                                              self._body_default_path("_angles.csv"),
                                              "CSV (*.csv)")
        if not path:
            return
        try:
            s.export_body_angles_csv(path)
            rep = Path(path).with_suffix("").as_posix() + "_report.txt"
            Path(rep).write_text(angle_report(s.body, float(s.fps or 0.0)), encoding="utf-8")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Export", str(exc))
            return
        self.toast.show_message(
            f"Joint angles written to {Path(path).name}, with the conventions and the range "
            f"of motion in {Path(rep).name}.", "info", 7000)

    def _export_body_video(self):
        """Write exactly what the side-by-side window shows to an mp4."""
        from cotracker_app.bodyview import PoseDrawOptions, SideBySideRenderer
        s = self.session
        if s is None or not s.has_body() or self._body_video is not None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export side-by-side video",
                                              self._body_default_path("_body.mp4"),
                                              "MP4 video (*.mp4)")
        if not path:
            return
        fr = s.body.frames()
        f0, f1 = (int(fr[0]), int(fr[-1])) if len(fr) else (0, self.n_frames - 1)
        if self.timeline.sel_range is not None:
            a, b = self.timeline.sel_range
            f0, f1 = max(f0, int(a)), min(f1, int(b))
            if f1 < f0:
                # the selected window and the posed frames do not overlap; an
                # empty range would otherwise write a 0-frame file and call it
                # a success
                self.toast.show_message(
                    f"The frames selected on the timeline have no body pose on them "
                    f"(the pose covers {int(fr[0])}-{int(fr[-1])}). Clear the selection "
                    f"with Esc, or select a window inside that range.", "warn", 8000)
                return
        w = self.body_win
        person = w.person() if w is not None else 0
        opts = w.draw_options() if w is not None else PoseDrawOptions(person=0)
        az, el = (w.azimuth, w.elevation) if w is not None else (25.0, 12.0)
        r = SideBySideRenderer(self.info.path, s.body, path, f0, f1, person, opts,
                               float(s.fps or 30.0), 1600, az, el,
                               w.chk_plot.isChecked() if w is not None else True,
                               w.chk_upright.isChecked() if w is not None else True,
                               w.chk_shape.isChecked() if w is not None else True)
        self._body_video = r
        dl = QProgressDialog("Rendering the side-by-side video…", "Cancel", 0,
                             max(1, f1 - f0 + 1), self)
        dl.setWindowTitle("Export")
        dl.setWindowModality(Qt.NonModal)
        dl.setMinimumDuration(0)
        dl.setAutoClose(False)
        dl.canceled.connect(r.request_cancel)
        r.progress.connect(lambda d, tt: (dl.setMaximum(max(1, tt)), dl.setValue(d)))
        r.finished_ok.connect(lambda p, c: self.toast.show_message(
            f"Side-by-side video written to {Path(p).name} ({c})."
            + (f" {r.note}" if getattr(r, "note", "") else ""), "info", 8000))
        r.error.connect(lambda m: None if m == "cancelled"
                        else QMessageBox.warning(self, "Export", m))
        r.finished.connect(lambda: (dl.close(), setattr(self, "_body_video", None),
                                    self._apply_state()))
        dl.show()
        r.start()
        self._apply_state()

    def _clear_body(self):
        s = self.session
        if s is None or s.body is None:
            return
        if QMessageBox.question(
                self, "Remove body pose",
                "Remove every body pose from this view? Ctrl+Z takes it back.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._undo_snap = s.snapshot()
        s.clear_body()
        self.act_undo.setEnabled(True)
        self._refresh_body_view(force=True)
        self.timeline.update()
        self._apply_state()

    def _export_mesh(self):
        from cotracker_app.hull import save_obj, save_ply
        p = self.project
        t = self._reference_instant()          # the same key as the carve (I114)
        if p is None or t not in self._hull_cache:
            self.toast.show_message("Carve the volume at this frame first (Ctrl+4).", "warn", 5000)
            return
        base = Path(self.project_path or self.info.path)
        start = str(base.with_suffix("")) + f"_hull_f{t}.obj"
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export mesh", start, "Wavefront OBJ (*.obj);;Stanford PLY (*.ply)")
        if not path:
            return
        verts, faces, h = self._hull_cache[t]
        try:
            if chosen.startswith("Stanford") or path.lower().endswith(".ply"):
                if not path.lower().endswith(".ply"):
                    path += ".ply"
                save_ply(path, verts, faces)
            else:
                if not path.lower().endswith(".obj"):
                    path += ".obj"
                unit = p.calibration.unit if p.calibration else ""
                save_obj(path, verts, faces, f"Kinetrace visual hull, reference frame {t}, unit {unit}")
        except OSError as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        self.toast.show_message(f"Mesh written: {Path(path).name} ({len(faces)} triangles)", "info", 6000)

    # ---------------------------------------------------------------- export

    EXPORT_FORMATS = [
        # (filter label, suffix, key)
        ("Wide CSV — one row per frame (*.csv)", ".csv", "wide"),
        ("DeepLabCut CSV — scorer/bodyparts/coords header, likelihood (*.csv)", ".csv", "dlc"),
        ("DLTdv8 xypts CSV — pt1_cam1_X… rows per frame, top-left origin, first pixel = 1 (*.csv)",
         ".csv", "dltdv"),
        ("DLTdv xypts CSV, bottom-left origin — older DLTdv / Argus Clicker (*.csv)", ".csv", "dltdv_bl"),
        ("Sparse TSV — tracked cells only (*.tsv)", ".tsv", "sparse"),
        ("MATLAB — tracks, confidence, segment silhouette, skeleton (*.mat)", ".mat", "mat"),
        ("ALL CAMERAS — DLTdv8 xypts for 3D reconstruction, offsets applied, top-left, first pixel = 1 (*.csv)",
         ".csv", "multi"),
        ("3D landmarks — xyz per reference frame + residual sidecar (*.csv)", ".csv", "xyz"),
        ("3D kinematics — smoothed positions, velocity, acceleration + report (*.csv)", ".csv", "kin"),
        ("Everything — all of the above with one base name (*.csv)", ".csv", "all"),
    ]

    def _export_one(self, key: str, path: str) -> list[str]:
        """Write one format; returns the files written (sidecars included)."""
        s = self.session
        written = [path]
        if key == "wide":
            s.export_csv(path)
        elif key == "dlc":
            s.export_dlc_csv(path)
        elif key in ("dltdv", "dltdv_bl"):
            s.export_dltdv_csv(path, flip_y=(key == "dltdv_bl"))
            written.append(str(Path(path).with_name(Path(path).stem + "_pointnames.csv")))
        elif key == "sparse":
            s.export_tsv_sparse(path)
        elif key == "mat":
            s.export_mat(path)
        elif key == "multi":
            # every camera in one file, landmarks matched BY NAME and each view
            # sampled through its own offset — the input a 3D solver wants
            return self.project.export_multi_dltdv(path)
        elif key == "xyz":
            r = self.project.reconstruction if self.project else None
            if r is None:
                # a 3D-only format on a project without 3D writes nothing: the
                # "Everything" path skips it, a direct choice is told why
                self.toast.show_message("No 3D reconstruction yet: run 3D -> Reconstruct 3D "
                                        "Landmarks (Ctrl+3) first.", "warn", 7000)
                return []
            r.export_csv(path)
            return [path, str(Path(path).with_name(Path(path).stem + "_xyzres.csv"))]
        elif key == "kin":
            from cotracker_app.kinematics import export_kinematics
            p = self.project
            r = p.reconstruction if p else None
            if r is None:
                self.toast.show_message("No 3D reconstruction yet: run 3D -> Reconstruct 3D "
                                        "Landmarks (Ctrl+3) first.", "warn", 7000)
                return []
            cutoff = self._ask_smoothing() if not getattr(self, "_export_all_running", False) else "auto"
            if cutoff is False:
                return []
            fps = float(p.sessions[0].fps) if p.sessions else float(self.info.fps)
            unit = (p.calibration.unit if p.calibration is not None and p.calibration.unit else "") or r.unit or ""
            return export_kinematics(path, r, fps, unit, cutoff)
        stem = str(Path(path).with_suffix(""))
        if key != "mat":
            if s.events or s.notes:            # frame notes live in this file too (I125)
                side = stem + "_events.csv"
                s.export_events_csv(side)
                written.append(side)
            if s.masks is not None and s.masks.n_masked() > 0:
                side = stem + "_segment.csv"
                s.export_animal_csv(side)
                written.append(side)
        return written

    def _ask_smoothing(self):
        """How to smooth before differentiating: "auto" (residual analysis), a
        cutoff in Hz, None (raw), or False (cancelled). Explained in place."""
        items = ["Automatic (recommended): the program picks the cutoff by residual analysis and says why",
                 "Choose the cutoff frequency myself (Hz)",
                 "None: raw differences (noisy - accelerations will be dominated by tracking jitter)"]
        choice, ok = QInputDialog.getItem(
            self, "Smoothing before velocity and acceleration",
            "Differentiating tracked positions amplifies their jitter, so positions are low-pass filtered "
            "first. The report states the cutoff used and how to check the result by hand.",
            items, 0, False)
        if not ok:
            return False
        if choice == items[2]:
            return None
        if choice == items[1]:
            fps = float(self.project.sessions[0].fps) if self.project and self.project.sessions else 30.0
            val, ok = QInputDialog.getDouble(self, "Cutoff frequency",
                                             f"Cutoff (Hz): movement at half this rate keeps 94 % or more of its "
                                             f"size, at the cutoff itself half. The camera runs at {fps:.0f} fps; "
                                             f"{fps / 10:.0f}-{fps / 4:.0f} Hz is typical for animal movement.",
                                             fps / 10.0, 0.1, fps / 2.0, 1)
            return float(val) if ok else False
        return "auto"

    def _export_dialog(self):
        if self.session is None:
            return
        base = Path(self.project_path or self.info.path)
        cam = ""
        if self.project is not None and self.project.n_views > 1:
            # every camera used to get the same default name, and one camera's
            # files overwrote another's (I108)
            cam = "_" + "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in self.project.name(self.project.active))
        start = str(base.with_suffix("")) + cam + "_tracks.csv"
        filters = ";;".join(f[0] for f in self.EXPORT_FORMATS)
        path, chosen = QFileDialog.getSaveFileName(self, "Export tracks", start, filters)
        if not path:
            return
        label, suffix, key = next((f for f in self.EXPORT_FORMATS if f[0] == chosen),
                                  self.EXPORT_FORMATS[0])
        QApplication.setOverrideCursor(Qt.WaitCursor)
        written: list[str] = []
        try:
            if key == "all":
                stem = str(Path(path).with_suffix(""))
                for lab, suf, k in self.EXPORT_FORMATS:
                    if k == "all" or (k == "multi" and self.project.n_views < 2):
                        continue    # the all-cameras file is meaningless for one camera
                    if k in ("xyz", "kin") and self.project.reconstruction is None:
                        continue    # no 3D yet: nothing to write
                    if k == "dltdv_bl":
                        continue    # "Everything" writes the DLTdv8 convention once, not both
                    tag = {"wide": "", "dlc": "_dlc", "dltdv": "_dltdv", "sparse": "",
                           "mat": "", "multi": "_allcams", "xyz": "_xyz", "kin": "_kinematics"}[k]
                    self._export_all_running = True       # kinematics: automatic smoothing, no question
                    try:
                        written += self._export_one(k, stem + tag + suf)
                    finally:
                        self._export_all_running = False
            else:
                if not path.lower().endswith(suffix):
                    path += suffix
                written = self._export_one(key, path)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            self.toast.show_message(f"Export failed: {e}", "error", 8000)
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QApplication.restoreOverrideCursor()
        n = int(self.session.tracked.any(axis=1).sum())
        names = ", ".join(dict.fromkeys(Path(w).name for w in written))
        which = (f" (2D files: camera {self.project.name(self.project.active)})"
                 if self.project is not None and self.project.n_views > 1 else "")
        self.statusBar().showMessage(f"Exported {n} tracked frames ✓  {names}{which}", 8000)
        self.toast.show_message(f"Exported: {names}{which}", "success", 7000)
        self._autosave()

    def _export_overlay(self):
        """File → Export Overlay Video: render the tracked result into an MP4
        in a background thread (its own VideoCapture), with a cancellable
        progress dialog. The app stays usable meanwhile."""
        from cotracker_app.render import OverlayDialog, OverlayRenderer
        s = self.session
        if s is None or self.state != READY or self._overlay is not None:
            return
        base = Path(self.project_path or self.info.path)
        default = str(base.with_suffix("")) + "_overlay.mp4"
        dlg = OverlayDialog(self, s, default, self.current, self.timeline.sel_range, n_frames=self.n_frames)
        if dlg.exec() != QDialog.Accepted or dlg.result_options is None:
            return
        opts, out = dlg.result_options, dlg.result_path
        bones = s.bones() if self.act_show_bones.isChecked() else []
        r = OverlayRenderer(s, self.info.path, out, opts, bones, self._mask_opacity)
        total = opts.end - opts.start + 1
        prog = QProgressDialog(f"Rendering {total} frames to {Path(out).name}…", "Cancel", 0, total, self)
        prog.setWindowTitle("Overlay video")
        prog.setWindowModality(Qt.NonModal)
        prog.setMinimumDuration(0)
        prog.setAutoClose(False)
        prog.setAutoReset(False)
        prog.canceled.connect(r.request_cancel)
        r.progress.connect(lambda d, t: prog.setValue(d))

        def _done(path, codec):
            prog.close()
            self._overlay = None
            self._apply_state()
            note = getattr(r, "note", "")
            written = getattr(r, "frames_written", None) or total
            if note:                      # it stopped early: say why, not the requested count (I41)
                self.toast.show_message(f"Overlay video written: {Path(path).name} ({codec}). {note}",
                                        "warn", 12000)
            else:
                self.toast.show_message(f"Overlay video written: {Path(path).name} ({codec}, {written} frames)",
                                        "success", 8000)
            self.statusBar().showMessage(f"Overlay video: {path}", 10000)

        def _fail(msg):
            prog.close()
            self._overlay = None
            self._apply_state()
            if msg == "cancelled":
                self.statusBar().showMessage("Overlay render cancelled — no file written", 5000)
            else:
                self.toast.show_message(f"Overlay render failed: {msg}", "error", 8000)

        r.finished_ok.connect(_done)
        r.error.connect(_fail)
        self._overlay = r
        self._apply_state()
        prog.show()
        r.start()

    def _refresh_seg_menu(self):
        """Relabel the Segment ▾ model list. Rebuilt every time it opens because
        a status can change under it — saving a token flips SAM 3 from 'gated'
        to 'download', and the first run caches the weights."""
        for key, act in self._seg_acts.items():
            label = BACKENDS[key][2]
            status, note = backend_status(key)
            badge = {"ready": "ready", "download": f"downloads {BACKENDS[key][3]}",
                     "gated": "needs a token"}.get(status, status)
            act.setText(f"Model: {label}   [{badge}]")
            act.setToolTip(note)
            act.setChecked(key == self._seg_backend)

    def _set_seg_backend(self, key: str, announce: bool = True):
        """The ONE place the segmentation model changes — the Segment ▾ menu and
        the Settings dialog both come through here, so they cannot disagree."""
        if key not in BACKENDS:
            return
        changed = key != self._seg_backend
        self._seg_backend = key
        self._refresh_seg_menu()
        if not (changed and announce):    # a project restore is not a user choice
            return
        status, note = backend_status(key)
        self.toast.show_message(
            f"Segmentation model: {BACKENDS[key][2]}. {note} "
            "It applies to the next silhouette / tracking run.",
            "warn" if status == "gated" else "info", 9000)
        if self.session is not None:
            self.session.dirty = True

    def _show_settings(self):
        statuses = {}
        for key, (repo, family, label, size) in BACKENDS.items():
            status, note = backend_status(key)
            statuses[key] = (label, status, note)
        dlg = SettingsDialog(self, self._seg_backend, self._mask_opacity, has_token(), statuses,
                             self.session.annotator if self.session is not None else self._default_annotator())
        dlg.btn_token.clicked.connect(lambda: self._save_token(dlg))
        if dlg.exec() != QDialog.Accepted:
            return
        self._mask_opacity = dlg.chosen_opacity()
        self._apply_annotator(dlg.chosen_annotator())
        self._set_seg_backend(dlg.chosen_backend())
        if self.session is not None:
            self.session.dirty = True
        self._refresh_overlay()

    def _save_token(self, dlg):
        tok = dlg.token.text().strip()
        if not tok:
            self.toast.show_message("Paste a Hugging Face read token first (hf_…).", "warn")
            return
        try:
            save_token(tok)
        except OSError as e:
            self.toast.show_message(f"Could not store the token: {e}", "error")
            return
        dlg.token.clear()
        dlg.token.setPlaceholderText("token stored")
        self.toast.show_message("Token stored in models/hf/token. Gated models can now be "
                                "downloaded.", "success")

    # ----------------------------------------------------------------- close

    def closeEvent(self, ev):
        if self._overlay is not None and self._overlay.isRunning():
            self._overlay.request_cancel()
            self._overlay.wait(10000)
            self._overlay = None
        for th in (self._body_worker, self._body_video):
            if th is not None and th.isRunning():
                th.request_cancel()
                th.wait(10000)
        self._body_worker = self._body_video = None
        if self.state == TRACKING and self.worker is not None:
            self.worker.request_pause()
            # let its last rows land (queued chunk_ready / finished) before the save
            t_end = time.time() + 20.0
            while self.worker is not None and not self.worker.wait(100) and time.time() < t_end:
                QApplication.processEvents()
            QApplication.processEvents()
        if self._retrack is not None:
            # a half-done automatic re-track has no Keep / Undo on exit: put it back (I107)
            st = self._retrack
            self._retrack = None
            try:
                self._retrack_restore(st)
            except Exception:       # noqa: BLE001
                pass
        if self._preview is not None:
            self._preview.wait(15000)
        # Always the PROJECT, like the 30 s autosave: writing `self.session`
        # here saved a multi-camera project as one camera (v3) on exit, and the
        # resume prompt then offered a file missing the other cameras and the
        # calibration.
        if self.project is not None and self.project.dirty:
            self._sync_ui_state()
            path = self._autosave_path()
            if path is not None:
                try:
                    self.project.save_npz(path)
                except OSError as e:
                    # never lose work silently on the way out (I104)
                    side = Path(self.info.path + AUTOSAVE_SUFFIX) if self.info is not None else None
                    saved = False
                    if side is not None and side != path:
                        try:
                            self.project.save_npz(side)
                            saved = True
                        except OSError:
                            pass
                    if not saved and QMessageBox.question(
                            self, "Your work could not be saved",
                            f"{path}\n\n{e}\n\nClose anyway and lose the changes since the last save?",
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                        ev.ignore()
                        return
        # Every thread this window owns must be stopped before the interpreter
        # tears Qt down, or Qt aborts ("QThread: Destroyed while thread is
        # still running" = exit 0xC0000409 on Windows). Only the working
        # camera's decoder used to be stopped here.
        for timer in (getattr(self, "_render_timer", None), getattr(self, "_play_timer", None)):
            if timer is not None:
                timer.stop()
        for rt in list(getattr(self, "_views", [])):
            try:
                rt.stop()
            except Exception:       # noqa: BLE001
                pass
        probe = getattr(self, "_dev_probe", None)
        if probe is not None and probe.isRunning():
            probe.wait(30000)
        # the video probe (a slow frame-count check on a 4K file or a share) and
        # any thread that outlived its earlier wait (I109, I112)
        for th in [getattr(self, "_probe", None)] + list(_ORPHANS):
            try:
                if th is not None and th.isRunning():
                    th.wait(30000)
            except RuntimeError:
                pass
        ev.accept()


def main():
    import os
    import sys
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")   # no telemetry (see __main__.py)
    os.environ.setdefault("DO_NOT_TRACK", "1")
    # Under pythonw.exe / GUI-mode launches there is no console and
    # sys.stdout/sys.stderr are None — but torch.hub, tqdm, and warnings all
    # write there. Give them a safe sink so a Track click can't crash.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    apply_theme(app)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        arg = sys.argv[1]
        if arg.endswith(PROJECT_SUFFIX):
            QTimer.singleShot(0, lambda: win._open_project_from_path(arg))
        else:
            QTimer.singleShot(0, lambda: win._open_video(arg))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
