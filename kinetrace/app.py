"""Main window: state machine, tracking loop wiring, projects, exports.

Threading rules (the GUI thread never blocks):
- video decoding: SeekService thread (scrubbing) / TrackingWorker's own source
- model load + inference: TrackingWorker (QThread)
- torch is imported lazily in background threads (app startup stays instant)
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import weakref
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PySide6.QtCore import QEvent, QEventLoop, QObject, QRect, QSettings, QSize, Qt, QTimer, QThread, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractSpinBox, QTextEdit, QAbstractItemView, QApplication, QDialog, QDialogButtonBox,
                               QDockWidget, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QMenu, QMessageBox, QPlainTextEdit, QProgressDialog,
                               QSizePolicy, QSpinBox, QSplitter, QTextBrowser,
                               QToolButton, QVBoxLayout, QWidget)

from kinetrace import APP_NAME, APP_TAGLINE, APP_VERSION, theme
from kinetrace import alltracker_backend
from kinetrace.camerapanel import CameraPanel
from kinetrace.layers import LayersPanel, ROLE_HAS, ROLE_NAME
from kinetrace.view3d import CalibrationDialog, Scene3D, View3D
from kinetrace.canvas import (ZOOM_STEP as CANVAS_ZOOM_STEP,
                                  TRAIL_FRAMES, TRAIL_MAX, DISPLAY_FILTERS, REGION_SHAPES)
from kinetrace.project import MAX_VIEWS, REFERENCE_VIEW, Project
from kinetrace.segmenter import (BACKENDS, DEFAULT_BACKEND, backend_status, has_token,
                                     model_is_cached as seg_is_cached, preferred_backend,
                                     save_token, working_size)
from kinetrace import gpmf, projectfile, recovery, trackio
from kinetrace.lens import lens_label
from kinetrace.errors import plain_error as _plain_error
from kinetrace.session import TrackingSession
from kinetrace.viewgrid import ViewGrid, caption_for
from kinetrace.skeletons import all_templates, save_user_template
from kinetrace.theme import apply_theme
from kinetrace.timeline import TIME_ZOOM_STEP, TimelinePanel
from kinetrace.video_source import DEFAULT_CACHE_BYTES, FrameCache, SeekService, VideoInfo, probe_video
from kinetrace import icons
from kinetrace.widgets import LoadingOverlay, ManualDialog, OnboardingStrip, SettingsDialog, Toast, native_keys

PROJECT_SUFFIX = projectfile.SUFFIX
# the manual heading Help / Track ▾ open at (R11: it was spelled out three times)
MANUAL_WHICH_MODEL = "Which point model should I use?"

_AT_AVAILABLE: dict = {}        # the `available` function asked -> True once it said so (R23)


def _alltracker_available() -> bool:
    """`alltracker_backend.available()` (a file stat) asked once per process, not per selected point
    on every playhead move and twice per row at every list rebuild (R23). Only a True answer is
    kept (the vendored code does not disappear while the app runs; a missing one may be fetched
    meanwhile), and it is keyed by the function, so a test that replaces `available` is asked."""
    fn = alltracker_backend.available
    if _AT_AVAILABLE.get(fn):
        return True
    ok = bool(fn())
    if ok:
        _AT_AVAILABLE[fn] = True
    return ok

IDLE, READY, TRACKING = range(3)
VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.webm *.mpg *.mpeg);;All files (*)"
TRACKS_FILTER = ("Tracks - DeepLabCut, SLEAP, DLTdv / Argus, Kinetrace (*.csv);;All files (*)")
# a companion camera only ever displays ONE frame, so it needs room for a couple
# of 4K frames and nothing more (see MainWindow._rebudget_caches)
COMPANION_CACHE_BYTES = 96 * 1024 ** 2
# the other cameras (G24): playhead moves closer together than STEP_GAP_S are one
# continuous motion (playing, scrubbing, a held key); a lone move is a STEP, and on
# a step Sync all views shows every camera's new picture at once (waiting at most
# LOCKSTEP_MS for a slow decoder). "Active view only" never decodes them (G27).
STEP_GAP_S = 0.15
LOCKSTEP_MS = 300
# the loading card stays up until the working camera's first picture is on screen,
# or this long at most (a file whose first picture never comes is reported elsewhere)
FIRST_FRAME_WAIT_MS = 6000
# videos probed at once when several cameras open (each decodes its first frames
# and its last one: more at a time only thrash the disk)
PROBE_PARALLEL = 4
OPEN_STAGES = 5          # the loading card's bar for one video: file, rate, frames, timeline, picture

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
<tr><td class=k>T</td><td>start tracking / pause (semi-automatic mode: one step). Only what is <b>selected</b> in LAYERS is tracked: the points selected there, and an animal row = its silhouette and all its points — nothing selected, nothing tracked</td></tr>
<tr><td class=k>Ctrl+A</td><td>select everything to track: every animal and every point</td></tr>
<tr><td class=k>several selected</td><td>right-click one of them in LAYERS (or hold right on one of their markers): one menu for all — track these, clear here / in the window / whole tracks, delete, hidden here, show / hide, Tracker, fill gaps, <b>Move to</b> another animal, and for two points of one animal <b>Connect them with a bone</b>; a short right click on one of their markers clears all of them on this frame</td></tr>
<tr><td class=k>AT / CT / MS</td><td>beside a point's name in LAYERS: its tracker — AllTracker, CoTracker3 or Moving spot (right-click → Tracker). One Track press tracks each selected point with its own tracker; AllTracker and CoTracker3 points run one after the other (two passes over the same frames), and a point that stops ends the run for all</td></tr>
<tr><td class=k>Shift+T</td><td>several cameras: this run in <b>every camera</b> that has the point(s) here, all at the same time (Track ▾ → <i>Every camera</i> makes T and F always do that)</td></tr>
<tr><td class=k>X / Space</td><td>pause a running track (during 3D → Re-track Disagreeing Stretches: stops the whole queue and asks whether to keep what was re-tracked; during an every-camera run: stops every camera at once)</td></tr>
<tr><td class=k>Track ▾</td><td>dropdown: Automatic (to the end) or Semi-automatic (F steps); <b>Every camera</b> (several cameras: the selected points tracked in each camera that has them at this instant, ball markers included, all at the same time, each live in its own view; the button says "· 3 cams"; X stops them all; one Ctrl+Z undoes all; the menu stays open while you tick, so mode, Every camera and point model combine); <b>point model</b>: sets the tracker of the selected points and the project's default for the others — AllTracker (default: animals and objects with a visible shape), CoTracker3 (faster, sub-pixel on high-contrast markers) or Moving spot (only a target small enough to be ONE point — a dot up to about 20 px with no visible shape; no model, stops where it loses the spot); <b>Test the point models on my clicks</b> (place a point by hand on 20 frames in a row first) recommends one; the choice is saved with the project</td></tr>
<tr><td class=k>Auto-pause</td><td>stop the run when the model loses a point (a brief occlusion doesn't trigger it); the point's track is <b>cut</b> at the first unreliable frame and the playhead goes there</td></tr>
<tr><td class=k>ROI</td><td>track inside a crop around the points when it clearly helps</td></tr>
<tr><td class=k>Ctrl+Z</td><td>undo the last tracking run, bulk edit or hand edit (a click, a right-click clear, Ctrl+click, deleted point, Shift+X) — one step</td></tr>
</table>
<h3>Points &amp; regions (on the video)</h3><table>
<tr><td class=k>N (or Point)</td><td>arm the crosshair: the next click places a new point — stray clicks never edit. It joins the animal selected in LAYERS, else the animal whose silhouette is under the click, else the animal of the selected point, else Scene (a notice says where, with a click to move it)</td></tr>
<tr><td class=k>click (armed)</td><td>place a new point — or continue the selected point where it has no data</td></tr>
<tr><td class=k>hold left + move</td><td>pan the view (from a marker too: points are never dragged — a click places them)</td></tr>
<tr><td class=k>right-click a marker</td><td>clear that point on <b>this frame only</b> and select it (one Ctrl+Z step); with several points selected, a short right click on one of their markers clears all of them on this frame (points only — the silhouettes are not touched)</td></tr>
<tr><td class=k>hold right on a marker</td><td>(half a second) the point's menu — the same menu as a right click on its name in LAYERS</td></tr>
<tr><td class=k>click (not armed)</td><td><b>annotate by hand</b>: place the point selected in the list here on this frame, replacing what the tracker put there. Nothing selected = nothing is placed (a notice on the video says so). One Ctrl+Z step. A click on or right beside the ◇ places it exactly there. A landmark derived from the silhouette cannot be placed by hand (right-click → Data source → Track by appearance first)</td></tr>
<tr><td class=k>＋ Point</td><td>LAYERS: a named point with no position yet, in the selected animal (else in Scene), selected (and in every camera's list) — then click it on the video</td></tr>
<tr><td class=k>A</td><td>with a calibration: place the selected point at the <b>◇</b> — where two or more other cameras put it — exactly (one Ctrl+Z step)</td></tr>
<tr><td class=k>Alt+click</td><td>with a calibration: <b>look here</b> — where this spot can be in the other cameras (a dashed line there, a "?" ring here). Nothing is edited; Esc clears it</td></tr>
<tr><td class=k>Shift+&lt; / Shift+&gt;</td><td>jump to the selected point's <b>first / last frame with data</b>; with no point selected, the first / last silhouette of the animals selected in LAYERS (else of every animal)</td></tr>
<tr><td class=k>point menu (curve)</td><td><b>Fill its gaps between hand placements</b> / <b>Replace everything between its hand placements with that curve</b>: keyframe digitizing — a smooth curve through the frames you placed by hand fills the frames between (confidence 0.6); frames you marked hidden are left alone; Ctrl+Z undoes</td></tr>
<tr><td class=k>point menu</td><td>also: go to its first / last frame, its first / last hand-placed frame, its first doubtful stretch; clear its position on this frame, in the selected frame window, or its whole track; with a calibration, <b>Snap to the other cameras' rays here</b> and <b>Place it where the other cameras put it (◇)</b></td></tr>
<tr><td class=k>, / .</td><td>previous / next hand-placed frame of the selected point</td></tr>
<tr><td class=k>J / Shift+J</td><td>next / previous low-confidence stretch (the red runs) — of the selected points, or of all</td></tr>
<tr><td class=k>Shift+X</td><td>mark the selected point <b>hidden</b> on this frame (kept, not exported, not used for 3D); again to unmark. On the timeline: Shift+drag a window, right-click → Mark hidden</td></tr>
<tr><td class=k>Shift+N</td><td>note on this frame (▲ on the timeline; right-click it to edit)</td></tr>
<tr><td class=k>Point ▾</td><td>region shape for an armed drag: circle · rectangle · polygon (click corners, Enter closes) · <b>Ball marker</b>: click a ball, SAM outlines it every frame and the fitted circle's centre is the point (wand balls, markers, a dropped ball — add one per ball; they track together in one window, and balls too far apart for one (about 740 px) get a window each — any spacing works)</td></tr>
<tr><td class=k>O</td><td>onion skin: ghost markers of the previous (solid) and next (dashed) frame</td></tr>
<tr><td class=k>L</td><td>loupe: magnifier under the cursor with a crosshair on the exact pixel</td></tr>
<tr><td class=k>View → Trails</td><td>Off / Last 10 frames / Custom… (any length) — fading, optional upcoming path; in every camera while Track ▾ → Every camera is ticked; View → Display filter: contrast / brighten / frame difference (display only)</td></tr>
<tr><td class=k>drag (armed)</td><td>outline a region in the Point ▾ shape (circle or rectangle; a polygon is clicked corner by corner) → tracked as one point (its fitted center)</td></tr>
<tr><td class=k>Ctrl+click</td><td>move the selected point here (one Ctrl+Z step)</td></tr>
<tr><td class=k>right-click (LAYERS) / hold right (marker)</td><td>the point's menu: rename · lock to seed appearance · data source · hidden on this frame · may leave its silhouette · Move to (another animal / Scene) · use it as its animal's head · tracker · delete</td></tr>
<tr><td class=k>Ctrl/Shift+click (LAYERS)</td><td>select several rows (points and animals)</td></tr>
<tr><td class=k>drag in LAYERS</td><td>drop points on another animal (or Scene) to move them there: renamed "&lt;animal&gt; &lt;part&gt;" in every camera, one Ctrl+Z step</td></tr>
<tr><td class=k>Delete</td><td>delete what is selected in LAYERS (an animal asks whether its points go too) — or clear the selected frame window (below)</td></tr>
<tr><td class=k>Esc</td><td>drops a drag or polygon in progress; then, one per press: the segment tool, the armed crosshair, the pan tool, a half-marked event, the frame-window selection, the look-here line (Alt+click); then deselects</td></tr>
</table>
<h3>Animals, silhouettes &amp; skeletons</h3><table>
<tr><td class=k>LAYERS</td><td>the right panel: each <b>animal</b> with its points under it, then <b>Scene</b> (points of no animal: wand ends, reference markers). <b>＋ Animal</b> makes one; double-click renames (its points follow: "&lt;animal&gt; &lt;part&gt;"); its checkbox shows / hides its silhouette; <b>right-click</b> an animal: outline it (S), keep its points on its silhouette, jump to its first / last silhouette, clear its silhouettes, save its points and bones as a skeleton template, forget its bones, remove it (its points go to Scene, or with it)</td></tr>
<tr><td class=k>S (or Segment)</td><td>segment tool, for the animal selected in LAYERS (with no animal yet, it makes one): <b>click the animal</b> — its silhouette appears within a second. Shift+click = "not the animal", drag = box around it. Right-click a click marker to remove it. S or Esc when done. Optional: points track without a segment. The ▾ on the Segment button picks the segmentation model; its last entry is Settings (Ctrl+,)</td></tr>
<tr><td class=k>Track ▶</td><td>with an animal's silhouette in the run (its row selected, or one of its points held on it), a run also segments every frame: the silhouette follows the animal, the crop follows the silhouette, and silhouette landmarks (tail tip, midline, feet) fill in</td></tr>
<tr><td class=k>Skeleton ▾</td><td>give the selected animal a named landmark set (its points become "&lt;animal&gt; &lt;part&gt;"); select a landmark, press N and click it. Landmarks marked <i>from silhouette</i> need no click — they come from the mask. Bones: select two points of one animal, right-click → Connect them with a bone</td></tr>
<tr><td class=k>right-click a point</td><td>Data source: track by appearance, or derive from the silhouette (tail tip, midline %, extremities). Switching asks first, because it erases the point's track (Ctrl+Z brings it back). Feet and wing tips are named as seen from above; filmed from below, left and right swap</td></tr>
<tr><td class=k>Mask</td><td>show / hide every silhouette (View menu: midline, bones; Settings: opacity, model)</td></tr>
<tr><td class=k>keep on the silhouette</td><td>an animal's points tracked by appearance stay on its silhouette: a point a few pixels off is nudged back; a point that <b>leaves</b> it stops the run at that frame and its track ends there (right-click the animal → Keep its points on its silhouette; a point → "May leave its silhouette" exempts it; Scene points are never held)</td></tr>
<tr><td class=k>fix a wrong mask</td><td>pause, go to the frame, press S, click again (Shift+click to exclude), press Track — the correction applies from there on</td></tr>
</table>
<h3>New to tracking?</h3><table>
<tr><td class=k>F1</td><td><b>Help → User Manual</b> — the full manual, written for someone who has never tracked anything before: what the ideas mean, a first session step by step, how to fix mistakes, and a glossary. This page is the quick reference; that one explains <i>why</i></td></tr>
</table>
<h3>Several cameras</h3><table>
<tr><td class=k>＋ Add video</td><td>CAMERAS panel (top of the right dock): put every camera of the same event in one project. Each view shows its own points and silhouette; up to 15</td></tr>
<tr><td class=k>File → Open Folder of Videos</td><td>every video of a folder listed: tick the ones to import, put them in <b>camera order</b> (Move up / Move down; camera 1 = the reference clock; the order of calibrations, 3D and exports), and they open as one project — saved at once as &lt;folder&gt;.kinetrace unless you untick it. <b>＋ Add video</b> asks the order of the cameras added; <b>Camera order…</b> under the CAMERAS list changes it any time</td></tr>
<tr><td class=k>offset  ◂ ▸</td><td><b>the frame this camera shows when camera 1 (the reference) is at its frame 0</b>: a camera switched on 12 frames after the reference has offset −12, one switched on earlier a positive offset. The first camera loaded is the reference: its offset is 0 by definition and locked, so every other number is measured against it and stays put when you switch cameras. Scrub to something every camera saw (a flash, a clap, first contact) and nudge until it lines up — or press <b>Align here</b> to take what you see as the match. The panel then reports the overlap window</td></tr>
<tr><td class=k>click a view</td><td>switch to that camera. Every camera has the same list of points (in the same order); their positions, the silhouette, events and the timeline are always the <i>working</i> camera's; the playhead keeps the same instant and the selected point stays selected. With <b>N (Add) armed</b> the click switches AND places the point in the camera clicked; otherwise it only switches. Switching puts the segment tool down, drops a half-marked event and forgets the undo step; the tool toggles (Auto-pause, ROI, Body, Follow, the point model…) stay as you set them</td></tr>
<tr><td class=k>guides &amp; ◇</td><td>with a calibration: the selected point's <b>dashed line</b> from every other camera that has it, run to the picture's edges (dotted where a lens model is only guessing). Once <b>two cameras</b> have it, every camera's lines turn faint and each camera that has it shows its <b>3D rmse</b> beside it (the reconstruction residual in px — green good, amber usable, red poor; also in the status bar). Once two <i>other</i> cameras have it, a <b>◇</b> shows where they put it (the lines fade; A or a click on it places it); in a camera that already has it, the ◇ shows how many px its placement is from the others'. No ◇ when those cameras disagree or see it along one line — the line's label says why. A camera added after calibrating gets none; the others keep theirs</td></tr>
<tr><td class=k>View → Other cameras</td><td><b>Sync all views</b> (the default: every camera follows the playhead; a single step shows every camera's new picture at once) · <b>Active view only</b> (Ctrl+Shift+2, or untick <i>Sync all</i> in the CAMERAS panel: only the working camera reads its video; the others stay on the picture they last showed, veiled, until you tick Sync all again or click one to work in it — its guides still come from the others' tracks) ·<b>Only the working camera</b> (Ctrl+2: the others are hidden and stop decoding)</td></tr>
<tr><td class=k>eye / title bar</td><td>the <b>eye</b> on a camera's line in CAMERAS shows / hides its view (hidden = not decoded; the others get the room); <b>drag a view's title bar</b> onto another view to move it there (only the screen changes, the cameras keep their numbers); right-click a title bar: hide it, show every camera, views back in camera order. <b>▶</b> on a camera's line opens its controls (Align here, offset, frame rate). With Track ▾ → Every camera, a <b>hidden camera is tracked too</b>, without being drawn: Track says "(n hidden)" and a warning names them</td></tr>
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
<tr><td class=k>H (or Pan)</td><td>pan tool: left-drag pans instead of editing (does nothing until a video is open); H, Esc, or picking Add (N) or Segment (S) ends it — only one of Pan / Add / Segment is on at a time. Middle-drag always pans, any time — even while tracking</td></tr>
<tr><td class=k>Follow</td><td>off by default; when on, keeps the selected point in view while zoomed in — or auto-frames all points when none is selected. R always fits the whole picture</td></tr>
<tr><td class=k>marker px</td><td>marker size on screen (shrink it to see exact placement)</td></tr>
<tr><td class=k>Shift+C</td><td>failsafe: clear the frame cache and re-decode this frame from the file (use if the picture ever looks stale or garbled; tracked data is untouched)</td></tr>
</table>
<h3>Files</h3><table>
<tr><td class=k>Ctrl+O / Ctrl+Shift+O</td><td>open video / project</td></tr>
<tr><td class=k>Ctrl+S / Ctrl+Shift+S</td><td>save project / save as (projects restore the exact working state). Ctrl+S during a run: saved as soon as the run stops</td></tr>
<tr><td class=k>Ctrl+Q</td><td>quit (File → Quit; asks first when there are unsaved changes)</td></tr>
<tr><td class=k>Ctrl+E</td><td>export tracks (CSV / TSV / MATLAB; events included)</td></tr>
<tr><td class=k>Ctrl+,</td><td>Settings: segmentation model, access token, silhouette opacity (also the last entry of the Segment ▾ menu)</td></tr>
</table>
"""


class _Job(QThread):
    """One piece of work off the GUI thread for `MainWindow._in_background`: the
    function gets `report(detail, value, total)` and `cancelled()` when it takes
    them (keyword arguments), and must not touch a widget."""
    step = Signal(str, int, int)

    def __init__(self, fn, kwargs: dict):
        super().__init__()
        self._fn, self._kw = fn, kwargs
        self.result, self.exc, self.cancel_requested = None, None, False

    def run(self):
        try:
            self.result = self._fn(**self._kw)
        except BaseException as e:  # noqa: BLE001 - handed back to the GUI thread
            self.exc = e


class _VideoProbe(QThread):
    done = Signal(object)  # VideoInfo | str(error)
    step = Signal(str, object)  # probe_video's stage + facts, for the loading card

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def run(self):
        try:
            info = probe_video(self._path, progress=lambda st, facts: self.step.emit(st, facts))
            # (G145) THE GoPro flag: GoPro footage carries its own metadata (lens model, settings, sensors);
            # anything else gets None and none of the GoPro workflow. Never fails the open.
            info.gopro = gpmf.read_safe(self._path)
            self.done.emit(info)
        except Exception as e:  # noqa: BLE001 — reported to the user verbatim
            self.done.emit(str(e))


def on_network_drive(path: str) -> bool:
    """A UNC path or a mapped network drive: opening and scrubbing are several
    times slower there (measured: a cold model load 126 s vs 7 s, 4K tracking
    10 vs 16 fps), which the loading card says."""
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or p.startswith("//"):
        return True
    if sys.platform.startswith("win"):
        drive = os.path.splitdrive(p)[0]
        if drive:
            try:
                import ctypes
                return ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == 4      # DRIVE_REMOTE
            except Exception:       # noqa: BLE001
                return False
    return False


def camera_names(paths, taken=()) -> list[str]:
    """One name per camera for `paths`, none equal to another or to `taken` (I223).
    Cameras are named after the file (24 characters of its stem); cameras filmed into
    sub-folders often share a stem (cam1/GX010001.MP4 ... cam8/GX010001.MP4), and
    Import -> Camera Offsets matches BY NAME, so a collision takes the PARENT FOLDER's
    name when that tells them apart, else a " (2)" suffix."""
    paths = [Path(p) for p in paths]
    stems = [p.stem[:24] for p in paths]
    taken_l = {str(t).lower() for t in taken}
    clash = {s for k, s in enumerate(stems) if stems.count(s) > 1 or s.lower() in taken_l}
    if clash:
        folders = [p.parent.name[:24] for p in paths]
        alt = [f if (s in clash and f) else s for s, f in zip(stems, folders)]
        if len(set(a.lower() for a in alt)) == len(alt) and not any(a.lower() in taken_l for a in alt):
            stems = alt
    used = set(taken_l)
    out: list[str] = []
    for s in stems:
        base = s or "camera"
        name, k = base, 2
        while name.lower() in used:
            name, k = f"{base} ({k})", k + 1
        used.add(name.lower())
        out.append(name)
    return out


class _DeviceProbe(QThread):
    """Imports torch in the background (slow) and reports the compute device."""
    got = Signal(str)

    def run(self):
        try:
            from kinetrace.device import probe
            # the whole hardware probe (device, GPU, driver, RAM, what is
            # slower / off on this computer) is computed here, off the GUI
            # thread, and cached: the status bar, Help -> System Check and the
            # Body dialog's gating then read it without touching torch
            self.got.emit(probe()["label"])
        except Exception as e:  # noqa: BLE001
            self.got.emit(f"device unavailable: {e}")


class _MaskPreviewWorker(QThread):
    """Segments ONE frame from the user's prompts so the silhouette shows right
    after a click (a tracking run re-segments every frame). Loads the model on
    first use — which downloads the weights once."""
    loading = Signal(str)
    progress = Signal(str, float, float)   # a first-use download: label, bytes done, bytes in all (G45)
    done = Signal(object)      # mask summary dict (see segmenter.summarize_mask) + "frame"
    error = Signal(str)
    plain = False              # True when `error` carries a sentence (a download), not a traceback
    cancelled = False

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
            from kinetrace.segmenter import (MIDLINE_SAMPLES, Prompt, get_segmenter,
                                                 summarize_mask)
            from kinetrace.silhouette import midline as silhouette_midline, resample
            self.loading.emit(self._backend)
            seg = get_segmenter(self._backend, progress=lambda label, d, t: self.progress.emit(label, float(d), float(t)),
                                cancel=lambda: self.cancelled)
            rgb = self._rgb
            if rgb is None:
                from kinetrace.video_source import VideoSource
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
        except Exception as e:  # noqa: BLE001 — reported to the user with guidance
            from kinetrace.downloads import DownloadError
            if isinstance(e, DownloadError):
                self.plain = True
                self.error.emit(str(e))
            else:
                self.error.emit(traceback.format_exc())


def _quiet_close(dlg) -> None:
    """Close a progress dialog WITHOUT its canceled signal: QProgressDialog emits
    canceled when it is closed, and the download dialogs' Cancel pauses the run --
    closing one at the run's start stopped every run that first loaded SAM."""
    if dlg is None:
        return
    try:
        dlg.canceled.disconnect()
    except (RuntimeError, TypeError):
        pass
    dlg.close()
    dlg.deleteLater()


def _download_status(clock: dict, label: str, done: float, total: float) -> tuple[int, int, str]:
    """A download's progress for a QProgressDialog: (maximum, value, text) --
    how far and about how long, from the rate since this file began (G45)."""
    now = time.monotonic()
    if clock.get("label") != label:
        clock.update(label=label, t0=now, d0=done)
    rate = (done - clock["d0"]) / max(1e-3, now - clock["t0"])
    if total > 0:
        left = (total - done) / rate if rate > 1e3 else None
        eta = "" if left is None else (f", about {left / 60:.0f} min left" if left >= 90 else
                                       f", about {max(1.0, left):.0f} s left")
        line, mx, val = f"{done / 1e6:.0f} of {total / 1e6:.0f} MB{eta}", 1000, int(1000 * min(1.0, done / total))
    else:
        line, mx, val = f"{done / 1e6:.0f} MB so far", 0, 0
    return mx, val, (f"{label} (first use only)\n{line}\n\nThe file is kept inside the Kinetrace folder "
                     "(models/). Cancel stops it; the next try goes on from where it stopped.")


def _crash_text(tb: str, hint: str, fallback: str) -> str:
    """A worker's traceback for the user (G54): the hint (or `fallback`) first, the
    last line in the program's words, the full traceback to the error log --
    Help > Error Report shows it -- instead of 1500 characters of it in a dialog."""
    import logging
    logging.getLogger("kinetrace.errors").warning("handled worker error:\n%s", tb.rstrip())
    last = next((ln.strip() for ln in reversed(tb.strip().splitlines()) if ln.strip()), "")
    return ((hint or fallback) + (f"\n\nIn the program's words: {last[:300]}" if last else "")
            + "\n\nHelp > Error Report shows the full details (nothing is sent anywhere).")


def _final_error_lines(tb: str) -> str:
    """The exception line(s) a traceback ENDS with (G104), lower-cased: the
    non-indented lines after the last stack frame. A traceback's frames are
    indented, so file names, line numbers and echoed source never reach the
    matching below (a line 403 in balls.py used to read as an HTTP 403)."""
    out: list[str] = []
    for ln in reversed(tb.strip().splitlines()):
        if not ln.strip():
            continue
        if ln[0].isspace() or ln.startswith(("Traceback", "During handling", "The above exception")):
            break
        out.append(ln)
        if len(out) >= 8:
            break
    return "\n".join(reversed(out)).lower()


def _model_error_hint(tb: str) -> str:
    """Turn a traceback into one actionable sentence for the user. Only the
    final exception line(s) are matched, with specific markers (G104)."""
    low = _final_error_lines(tb)
    if "out of memory" in low:
        return ("The GPU ran out of memory. Close other GPU applications (or pick a smaller "
                "segmentation model in Settings) and try again.")
    if any(k in low for k in ("gatedrepoerror", "401 client error", "403 client error", "gated repo",
                              "is gated", "gated model")):
        return ("The model weights are gated on Hugging Face. Request access to the model, "
                "then paste a read token in Settings (it is stored inside this folder).")
    if any(k in low for k in ("urlerror", "connectionerror", "connecterror", "maxretryerror", "max retries exceeded",
                              "name or service", "getaddrinfo", "temporary failure in name resolution",
                              "timed out", "could not be downloaded", "offline mode", "connection refused",
                              "connection reset", "connection aborted", "no connection")):
        return ("The model could not be downloaded. Check your internet connection and try "
                "again — the weights are only needed once, then they live in models/.")
    if "cuda error" in low or ("cuda" in low and "device" in low) or ("mps" in low and "not supported" in low):
        return ("The graphics card could not be used. Help → System Check… says what was found; "
                "update the NVIDIA driver, or run on the CPU (slow) by starting Kinetrace with "
                "KINETRACE_DEVICE=cpu set.")
    if "no click, box or mask" in low:
        return "Press S, click the animal on the current frame, then start tracking again."
    return ""


class _ChoiceMenu(QMenu):
    """A menu of settings (Track ▾): ticking an entry leaves the menu open, so the
    run mode, Every camera and the point model -- independent choices -- can all be
    set in one visit (G32). Esc or a click outside closes it."""

    def _toggle(self, act) -> bool:
        if act is None or not act.isEnabled() or not act.isCheckable():
            return False
        act.trigger()
        return True

    def mouseReleaseEvent(self, ev):
        if self._toggle(self.actionAt(ev.position().toPoint())):
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space) and self._toggle(self.activeAction()):
            ev.accept()
            return
        super().keyPressEvent(ev)


class _SideRun(QObject):
    """One camera of a simultaneous every-camera run (I141). It records how the
    camera's run ended; for a camera other than the working one (`write`) it also
    puts the run's rows, silhouettes and ball radii into THAT camera's session
    and keeps its frames for its tile -- the working camera keeps the full live
    pipeline. A QObject in the GUI thread, so the worker's signals (emitted on the
    tracking thread) arrive here queued, in order."""

    def __init__(self, win, view: int, worker, start: int, write: bool):
        super().__init__(win)
        self.win, self.view, self.worker, self.start = win, view, worker, int(start)
        self.pids = list(worker.point_ids)
        self.write = write
        self.frames: deque = deque()
        self.last: int | None = None
        self.paused = False
        self.error: str | None = None
        self.autopause: tuple[int, int] | None = None
        worker.autopaused.connect(self.on_autopaused)
        worker.finished_ok.connect(self.on_finished)
        worker.error.connect(self.on_error)
        if write:
            worker.chunk_ready.connect(self.on_chunk)
            worker.masks_ready.connect(self.on_masks)
            worker.balls_ready.connect(self.on_balls)

    def _session(self):
        p = self.win.project
        return p.sessions[self.view] if p is not None and self.view < p.n_views else None

    def on_chunk(self, w0, tracks, vis, conf, members, new_frames):
        s = self._session()
        if s is not None:
            s.write_segment(w0, tracks, vis, self.pids, conf)
            self.frames.extend(new_frames)

    def on_masks(self, summaries):
        s = self._session()
        if s is not None:
            s.write_mask_summaries(summaries)

    def on_balls(self, rows):
        s = self._session()
        if s is not None:
            s.write_ball_radii(rows)

    def on_autopaused(self, frame: int, pid: int):
        self.autopause = (int(frame), int(pid))

    def on_finished(self, last: int, was_paused: bool):
        self.last, self.paused = int(last), bool(was_paused)

    def on_error(self, text: str):
        self.error = text


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
        self.bad_frames: set = set()         # frames already reported as undecodable (I40)

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


class _SaveWorker(QThread):
    """Writes a frozen project off the GUI thread: the project folder
    (projectfile.write_folder_over: only the files that changed, I145) or a
    single file (projectfile.write: the recovery copies, Export Project as One
    File). Autosave does not wait for it; Save does, while the window keeps
    repainting. `stats` = what the folder save wrote."""
    done = Signal(bool, str)

    def __init__(self, frozen, path, folder: bool = False, **kw):
        super().__init__()
        self._frozen, self._path, self._folder, self._kw = frozen, path, folder, kw
        self.stats: dict | None = None

    def run(self):
        try:
            if self._folder:
                self.stats = projectfile.write_folder_over(self._frozen, self._path, **self._kw)
            else:
                projectfile.write(self._frozen, self._path, **self._kw)
            self.done.emit(True, "")
        except Exception as e:      # noqa: BLE001 - reported to the user, never silent
            self.done.emit(False, _plain_error(e, "The save did not finish"))      # in words (G54)


# ui_state entries that belong to the USER (tools, display), not to one camera:
# carried across a camera switch (I50). Frame, selection and zoom stay per camera.
GLOBAL_UI_KEYS = ("follow", "autopause", "roi", "track_mode", "track_all", "marker_size", "show_mask", "mask_opacity",
                  "show_midline", "show_bones", "seg_backend", "point_backend", "trail_len",
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


@dataclass
class _OpenPlan:
    """What opening a project decided (R9, `MainWindow._decide_unsaved_work`): the project to adopt
    and its state, the id and the save it came from, where Save goes (None = nowhere yet), whether
    it holds unsaved work, and the recovery to set aside once its copy is open (I209)."""
    proj: Project
    state: dict
    meta: dict
    pid: str
    saved_at: str | None
    file_path: Path | None
    unsaved: bool
    decline_after: str | None


class _SegmentSpecs(NamedTuple):
    """The segments' part of a run (R8, `MainWindow._segment_specs`): what the worker gets for them
    (G149: any number of segments, each with its head landmark, its on-body points and the points held
    on it; a stored one = the silhouettes an earlier pass wrote, the second pass of a two-pass run)."""
    animals: list             # [AnimalSpec], [] = no segment in this run
    derived: list             # [DerivedSpec] (each with the position of its segment in `animals`)


# Where a run stopped at a point, by `MainWindow._stop_kind` (the worker's `_autopause_reason`): `short`
# = how the every-camera summary names it, `toast` (+ `ms`) and `status` = what the run's end says
# ({name}, {frame}, {prev} = frame - 1); one table instead of the sentences spelt out twice (R8).
_STOP_TEXTS = {
    "segment": dict(
        short="{name} was lost", ms=15000,
        toast=("Auto-paused: <b>{name}</b> was lost (or left the frame) around frame {frame}. If it is "
               "visible here, press <b>S</b>, click it, then Track. Or turn <b>Auto-pause</b> off (bottom "
               "bar) to run through."),
        status="Auto-paused: the segment was lost around frame {frame}"),
    "switched": dict(     # (I260) SAM took another object after losing the animal: stops whatever Auto-pause says
        short="{name} turned into another object", ms=20000,
        toast=("Stopped at frame {frame}: <b>{name}</b> disappeared here, and SAM then picked up "
               "<b>something else</b> (another animal or a look-alike, far from where yours was heading), so "
               "its silhouette ends at frame {prev}. If your animal is visible on a later frame, go there, "
               "press <b>S</b>, click it, then Track."),
        status="Stopped: the segment was lost at frame {frame} and SAM took another object instead"),
    "ball": dict(
        short="{name} was lost", ms=15000,
        toast=("Stopped at frame {frame}: SAM could not find the ball <b>{name}</b> any more, and it had "
               "not reached the picture edge, so its track ends at frame {prev}. If the ball is visible "
               "here, select it, Point ▾ → Ball marker, click it, then Track. A ball that leaves the picture "
               "never stops the run."),
        status="Stopped: the ball {name} was lost at frame {frame}"),
    "spot": dict(
        short="{name} was not found (Moving spot)", ms=15000,
        toast=("Stopped at frame {frame}: {said}, so its track ends at frame {prev}. Moving spot stops "
               "instead of guessing. If you can see it here, click it (it is selected) and press Track; "
               "clicking it on the next frame too gives it its speed. If it is not a small spot, switch "
               "Track ▾ → Point model."),
        status="Stopped: Moving spot {saw} {name} at frame {frame}",
        said_two=("two spots look equally like <b>{name}</b> here (another spot or a glint right beside "
                  "it, or two crossing)"),
        said_none=("<b>{name}</b> was not found where its speed put it (it faded, was hidden, or turned "
                   "sharply)"),
        saw_two="saw two candidates for", saw_none="lost"),
    "exit": dict(
        short="{name} left the segment", ms=15000,
        toast=("Stopped at frame {frame}: <b>{name}</b> left its animal's silhouette, so its track ends "
               "at frame {prev}. Click it where it really is and press Track. If it is allowed off the "
               "silhouette, right-click it → <i>May leave its silhouette</i>."),
        status="Stopped: {name} left its animal's silhouette at frame {frame}"),
    "unreliable": dict(
        short="{name} was lost", ms=12000,
        toast=("Auto-paused: <b>{name}</b> became unreliable at frame {frame}, so its track is cut there "
               "(its timeline lane is empty from this frame on). It is selected: click where it really is "
               "on the video and press Track — or turn <b>Auto-pause</b> off (bottom bar) to run through."),
        status=("Auto-paused: the model lost {name} at frame {frame} (low confidence; its track is cut "
                "there). Click where it really is and press Track — or turn Auto-pause off to run "
                "through")),
}


def _stop_short(reason: str, name: str) -> str:
    """How the every-camera summary names a stop (`_STOP_TEXTS` `short`): by the worker's reason."""
    return _STOP_TEXTS[reason if reason in ("exit", "spot", "switched") else "unreliable"]["short"].format(name=name)


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
        # who this project is on disk: the id inside the file (recovery is
        # found by it, wherever the file moves) and the save it came from
        self._project_id = projectfile.new_id()
        self._saved_at: str | None = None
        self._save_worker: _SaveWorker | None = None
        self._saving = False
        # (I265) Save pressed while a run is going: saved the moment the run is over (it used to do nothing)
        self._save_after_run = False
        # an older single-file project the user chose to keep as one file (I145)
        self._keep_single_file = False
        self._video_dirs: list[Path] = []          # where a project's videos are looked for by name
        self._exports_worker = None                # refreshes the project's exports/ after a save (G42)
        self._recovery_sig = None
        self._camera_entries: list = []
        self._project_dir: Path | None = None
        self._after_open = None          # (video path, job): runs once that video has opened
        self._undo_snap = None
        self._epi_probe = None           # (view, x, y): a look-here Alt+click (G19, G21)
        # the other cameras (G24): "all" = follow the playhead; "active" = only the
        # working video decodes, the others stay where they were (G27). View →
        # Other cameras also has Hidden (act_solo).
        self._sync_mode = "all"
        self._companions_stale = False   # "active": the others are not at the playhead yet
        self._last_goto_t = 0.0
        self._lock: dict | None = None   # a discrete step: {"want": {view}, "frames": {view: rgb}}
        self._multi: dict | None = None  # tracking in every camera, all at once (G29, I141)
        self._side: dict | None = None   # {view: _SideRun} of the other cameras while they track (I141)
        self._display_queue: deque = deque()
        self._fps_ema = 0.0
        self._last_emit_t = 0.0
        self._probe: _VideoProbe | None = None
        self._model_dialog: QProgressDialog | None = None
        self._model_worker = None           # the worker whose models are being prepared (G45)
        self._model_dl: dict = {}
        self._pending_event: int | None = None      # E pressed once: start frame
        self._autopause_info: tuple[int, int] | None = None  # (frame, pid)
        self._member_frames: dict[int, dict] = {}   # frame -> {pid: (M,2)} overlay
        # _track_mode ("auto" | "semi") is a property over the Track menu's actions (I139)
        self._step_run = False                      # current run is a semi-auto step
        self._track_blocked: str | None = None      # why Track cannot start here (G34)
        self._seg_backend = preferred_backend()     # segmentation model (Settings)
        self._point_backend = self._preferred_point_backend()   # Track dropdown
        self._spot_hints: set = set()          # the one-time Moving spot hints already shown (G58)
        self._passes = None                    # a Track press with AllTracker AND CoTracker3 points (G63)
        # (G158) each session's lane count when LAYERS last rebuilt (weak: a freed session drops out)
        self._tl_rows_seen = weakref.WeakKeyDictionary()
        self._spot_corrections: dict = {}      # (project, camera, point) -> frames corrected by hand
        self._mask_opacity = 0.35
        self._region_shape = "circle"               # Point ▾: circle | rect | polygon
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
        # what the UI builders used to create and the rest of the window reads (R7)
        self._place_kind = "point"          # what the next armed click creates: "point" | "ball"
        self._retrack = None                # an automatic epipolar re-track in progress (see _retrack_start)
        self._stopping_for_close = False    # `_stop_runs_for_close` is running: no window opens meanwhile
        self._retrack_last = None
        self._trail_custom = 30             # the last length typed into View → Trails → Custom…
        # the loading card's levels (G31, see `_busy_push`) and the wait for the first picture
        self._busy_stack: list[dict] = []
        self._busy_next = 0
        self._first_frame_token: int | None = None

        self._build_ui()
        self._apply_state()
        self._fit_to_screen()

        self._device_label.setText("probing GPU…")
        self._device_kind = ""            # "GPU" / "CPU" once probed: shown in the tracking speed readout
        self._dev_probe = _DeviceProbe()
        self._dev_probe.got.connect(self._on_device_probed)
        self._dev_probe.start()

        self._render_timer = QTimer(self, interval=33, timerType=Qt.CoarseTimer)
        self._render_timer.timeout.connect(self._render_tick)
        self._autosave_timer = QTimer(self, interval=30_000)
        # never in the middle of an open (the project is half-built until it ends)
        self._autosave_timer.timeout.connect(lambda: None if self._loading else self._autosave())
        self._autosave_timer.start()
        QTimer.singleShot(2500, self._announce_recovery)
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        # lockstep: a camera that does not deliver in time is not waited for
        self._lock_timer = QTimer(self, singleShot=True, interval=LOCKSTEP_MS)
        self._lock_timer.timeout.connect(self._flush_lockstep)

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
        # (G169, G170) a view hidden / moved from its title bar: decode, guides and the eyes follow
        self.grid.arrangement_changed.connect(self._on_views_arranged)
        self.grid.name_of = lambda i: self.project.name(i) if self.project is not None else ""
        self._wire_canvas(self.grid.canvases[0])
        self.toast = Toast(self.grid)   # important notices float over the video
        self._build_ui_rest()
        # opening videos / a project: a card over the window says what is happening
        # (G31: without it the wait looked like a frozen app)
        self.overlay = LoadingOverlay(self)
        self._first_frame_timer = QTimer(self, singleShot=True, interval=FIRST_FRAME_WAIT_MS)
        self._first_frame_timer.timeout.connect(self._first_frame_arrived)

    def _wire_canvas(self, canvas) -> None:
        """Every view's canvas reports to the same handlers. Only the active one
        is interactive, so an edit can never arrive from the camera the user is
        not working in."""
        canvas.add_requested.connect(self._on_add)
        canvas.annotate_requested.connect(self._on_annotate)
        canvas.group_requested.connect(self._on_add_group)
        canvas.point_selected.connect(self._on_select)
        canvas.reposition_requested.connect(self._on_reposition)
        canvas.clear_frame_requested.connect(self._on_clear_frame)
        canvas.delete_requested.connect(self._on_delete)
        canvas.rename_requested.connect(self._on_rename)
        canvas.anchor_toggled.connect(self._on_anchor_toggled)
        canvas.source_change_requested.connect(self._on_source_change)
        canvas.free_toggled.connect(self._on_free_toggled)
        canvas.region_requested.connect(self._on_add_region)
        canvas.occluded_toggled.connect(self._on_occluded_toggled)
        canvas.menu_extra = self._extend_point_menu
        canvas.menu_extra_action = self._point_menu_extra_action
        canvas.multi_menu = self._maybe_multi_menu
        canvas.animal_click.connect(self._on_animal_click)
        canvas.animal_box.connect(self._on_animal_box)
        canvas.prompt_remove_requested.connect(self._on_prompt_remove)
        canvas.probe_requested.connect(self._on_probe)
        # (a click on a companion switches to it through ViewGrid.view_activated -> _set_active_view)
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

    def _build_ui_rest(self):
        """The window's widgets and menus, section by section in the order they depend on each other:
        the View menu needs the tool buttons, 3D adds to File's Import submenu. (No shortcut strip: the
        full reference lives in Help → Keyboard & Mouse Reference; chrome stays out of the content's way.)
        """
        self._build_transport()
        self._build_tool_buttons()
        self._build_track_menu()
        self._build_control_bar()
        self._build_centre()
        self._build_side_panel()
        self._build_file_menu()
        self._build_edit_menu()
        self._build_view_menu()
        self._build_skeleton_events_menus()
        self._build_3d_menu()
        self._build_body_menu()
        self._build_help_menu()
        # QMenu hides action tooltips unless told otherwise: every explanation
        # written on a menu entry (what a wizard needs, what an export holds)
        # was invisible until this (release sweep G1, 2026-09-22)
        for m in self.findChildren(QMenu):
            m.setToolTipsVisible(True)
        self._build_nav_shortcuts()
        self._set_focus_policies()
        self._build_status_bar()

    def _build_transport(self) -> None:
        """Transport bar and the frame / step / marker boxes: previous / play / next, the timeline's zoom
        buttons -- text glyphs, not QStyle bitmap icons: those ignore the dark palette and vanish
        black-on-black."""
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
        # (G152) the time-axis zoom is on the timeline itself (TimelinePanel.btn_zoom_*): beside play /
        # pause it read as the video's zoom
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

    def _build_tool_buttons(self) -> None:
        """The labelled tool buttons of the control bar (Follow, Auto-pause, ROI, Point ▾, Segment ▾, Mask, Body,
        Pan), each from `_tool_button`."""
        self.btn_follow = self._tool_button(
            "Follow", icons.follow(),
            "Follow (off by default). When on, keeps tracked points in view while zoomed in: pans to\n"
            "follow the selected point (holds still where its track has a gap); with nothing\n"
            "selected, re-frames ALL points automatically. R always fits the whole frame.",
            checked=False)      # OFF by default: the view moves only when asked
        self.btn_follow.toggled.connect(
            lambda on: [cv.set_follow(on) for cv in self.grid.canvases])
        self.canvas.set_follow(self.btn_follow.isChecked())

        self.btn_autopause = self._tool_button(
            "Auto-pause", icons.autopause(),
            "Auto-pause: stop the run when the model has lost a point: its confidence stays low\n"
            "for 16 frames in a row (a point hidden for a moment keeps its confidence, so ordinary\n"
            "occlusion does not trigger this). Its track is cut back to the first unreliable frame,\n"
            "the playhead jumps there and the point is selected: click where it really is, then\n"
            "Track. Also stops when the segment is lost for 16 frames, or a ball marker is lost\n"
            "inside the picture. Uncheck to always run to the end. (A landmark held on its animal's\n"
            "silhouette that leaves it stops the run either way.)",
            checked=True)

        self.btn_roi = self._tool_button(
            "ROI", icons.roi(),
            "ROI zoom: when the tracked points sit in a small part of a high-res frame,\n"
            "track inside a crop around them so small objects keep real detail at the\n"
            "model's internal resolution. Engages only when it clearly helps (≥2× zoom).",
            checked=True)

        # (G152) named by its noun, like Segment: arming the tool is the "add"
        self.btn_add = self._tool_button(
            "Point", icons.add(),
            "Point (N): arms the crosshair — the next click places a new point\n"
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
            "Add one per ball; they are tracked together in one 960-pixel window, and balls too far\n"
            "apart for it (about 740 px) are split into groups with a window each, so any spacing works.\n"
            "Choosing this arms the crosshair like N.")
        self.act_add_ball.triggered.connect(self._arm_ball)
        menu_add.addAction(self.act_add_ball)
        self.btn_add.setMenu(menu_add)

        self.btn_animal = self._tool_button(
            "Segment", icons.segment(),
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

        self.btn_mask = self._tool_button(
            "Mask", icons.mask(),
            "Mask: show / hide the animals' silhouettes (each animal's checkbox in LAYERS hides its own)",
            checked=True)
        # the overlay only: no LAYERS row shows the toggle (each animal's own checkbox is `shown`)
        self.btn_mask.toggled.connect(lambda _on: self._refresh_overlay())
        # (G156) no global Body toggle: holding points on a silhouette is a setting of each animal
        # (its LAYERS menu -> Keep its points on its silhouette), and a point of no animal (Scene)
        # is never held

        self.btn_pan = self._tool_button(
            "Pan", icons.pan(),
            "Pan tool (H): left-drag moves the view instead of editing points.\n"
            "H or Esc ends it, and so does picking Add or Segment (one tool at a time).\n"
            "Middle-drag always pans, in any mode — even while tracking runs.")
        self.btn_pan.toggled.connect(self._on_pan_mode)

    def _build_track_menu(self) -> None:
        """The Track button and its ▾ menu: run mode, Every camera, point model, the test."""
        # Track button with a mode dropdown: automatic (run to end) or
        # semi-automatic (each F tracks exactly one frame, then pauses)
        self.btn_track = QToolButton()
        self.btn_track.setObjectName("primary")  # the one filled accent button
        self.btn_track.setMinimumWidth(150)
        self.btn_track.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_track.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_track.clicked.connect(lambda _=False: self._toggle_tracking())   # clicked(bool) is not all_cameras
        menu_track = _ChoiceMenu(self.btn_track)     # ticking a choice keeps it open (G32)
        self._mode_group = QActionGroup(self)
        self.act_mode_auto = QAction("Automatic — track to the end of the video", self,
                                     checkable=True, checked=True)
        self.act_mode_semi = QAction("Semi-automatic — each F tracks ONE frame, then pauses",
                                     self, checkable=True)
        for act, mode in ((self.act_mode_auto, "auto"), (self.act_mode_semi, "semi")):
            self._mode_group.addAction(act)
            menu_track.addAction(act)
            act.triggered.connect(lambda _=False, m=mode: self._set_track_mode(m))
        # several cameras: the same points tracked in every camera that has them (G29)
        menu_track.addSeparator()
        self.act_track_all = QAction("Every camera — track the point(s) in each camera that has them here",
                                     self, checkable=True)
        self.act_track_all.setToolTip(
            "With several cameras: Track (T, or F in semi-automatic mode) tracks the selected points "
            "in EVERY camera that has them at this instant, ball markers included, all cameras at the same "
            "time, each shown live in its own view. X stops them all at once. One Ctrl+Z undoes it in every "
            "camera. Shift+T does this once without ticking it. Combines with the run mode and the point model "
            "(the menu stays open while you tick).")
        self.act_track_all.toggled.connect(lambda _on: (self._update_track_button(), self._refresh_companions()))
        menu_track.addAction(self.act_track_all)
        menu_track.setToolTipsVisible(True)
        # point-tracking model: AllTracker (robust on animals) or CoTracker3 (fast)
        menu_track.addSeparator()
        self._point_group = QActionGroup(self)
        self.act_pm_alltracker = QAction(
            "Point model: AllTracker — animals and objects with a visible shape (default)", self, checkable=True)
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
        # a third point model for small, fast, featureless targets (I160, G56)
        self.act_pm_spot = QAction(
            "Point model: Moving spot — a target small enough to be one point (a dot, no visible shape)", self,
            checkable=True)
        from kinetrace.spots import WHICH_MODEL
        self.act_pm_spot.setToolTip(
            WHICH_MODEL + " It needs no model and no graphics card: on every frame the dot is searched where its "
            "speed puts it — the brightest or darkest small blob there, or what changes much more than that "
            "background usually does — and it STOPS where it cannot find the dot or sees two alike, instead of "
            "drifting. Click the dot on two frames in a row before you track (that gives its speed). Regions "
            "are left out of a Moving spot run. Choosing it here gives it to the SELECTED points (and makes it "
            "the default for points without their own); right-click a point's row → Tracker does it for one.")
        self._pm_acts = {"alltracker": self.act_pm_alltracker, "cotracker3": self.act_pm_cotracker,
                         "spot": self.act_pm_spot}
        for key, act in self._pm_acts.items():
            self._point_group.addAction(act)
            menu_track.addAction(act)
            act.triggered.connect(lambda _=False, k=key: self._set_point_backend(k))
        self.act_test_models = QAction("Test the point models on my clicks…", self,
                                       triggered=lambda: self._test_point_models())
        self.act_test_models.setToolTip(
            "Which point model suits this footage? Place the selected point by hand on at least 20 frames in a "
            "row (select it, click it, F, click it…), then this starts AllTracker, CoTracker3 and Moving spot "
            "from your first click, counts how often each one has to be put back on your clicks, and "
            "recommends one. Nothing is changed unless you press Use.")
        menu_track.addAction(self.act_test_models)
        self.act_which_model = QAction(MANUAL_WHICH_MODEL, self,
                                       triggered=lambda: self._show_manual(MANUAL_WHICH_MODEL))
        self.act_which_model.setToolTip("The manual's short guide: a visible shape (an animal, an object) -> "
                                        "AllTracker + Segment; a target small enough to be one point -> Moving "
                                        "spot; a round marker -> Ball marker")
        menu_track.addAction(self.act_which_model)
        at_ok = _alltracker_available()
        self.act_pm_alltracker.setEnabled(at_ok)
        if not at_ok:
            # the installer fetches AllTracker's code; say how to get it instead of a silent grey entry
            self.act_pm_alltracker.setToolTip(
                "AllTracker is not installed yet, so CoTracker3 is used. Start Kinetrace with run.bat / "
                "run.sh while connected to the internet: the launcher fetches AllTracker's code (277 KB) "
                "into models/alltracker, and its 63 MB checkpoint downloads on the first Track.")
        self._pm_acts.get(self._point_backend, self.act_pm_cotracker).setChecked(True)
        self.btn_track.setMenu(menu_track)

        # no run progress bar: it only repeated the timeline playhead; the status
        # bar carries fps / ETA (G35)

    def _build_control_bar(self) -> None:
        """The control bar under the timeline: frame box + transport | tools | toggles ... Track, folded to
        icons one button at a time when the window is narrow (G2, G16)."""
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
        tl.addWidget(self.btn_autopause)
        tl.addWidget(self.btn_roi)
        tl.addStretch(1)
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
                              self.btn_mask, self.btn_autopause, self.btn_roi]
        self._compact_order = [self.btn_pan, self.btn_roi, self.btn_mask, self.btn_follow,
                               self.btn_autopause, self.btn_animal, self.btn_add]
        self._compact_icons = [self.btn_prev, self.btn_play, self.btn_next]
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
        # measured with the Track button at its 150 px minimum; its label grows past that
        # ("Track · 3 points + segment ▶ · 3 cams", G61 / G29): `_fit_track_label` (G139)
        self._controls_min_w = tl.minimumSize().width()
        self._track_base_w = self.btn_track.minimumWidth()
        self._track_extra = 0
        self._set_compact_controls(0)
        self._controls_level = 0            # how many labels are folded (0 = all shown)
        self._controls_compact = False
        self._controls_hook = _ResizeHook(self._fit_controls)
        controls.installEventFilter(self._controls_hook)

    def _build_centre(self) -> None:
        """The centre: the video grid above the timeline in a vertical splitter, the onboarding strip over
        the video, the control bar under the timeline."""
        self.timeline = TimelinePanel()
        # (G173) where the other cameras have the working camera's landmarks (a thin line in its lanes)
        self.timeline.elsewhere = self._landmarks_elsewhere
        self.timeline.elsewhere_names = self._landmark_cameras_at
        # (G152) the time zoom buttons are the timeline's own, in the corner left of its ruler
        self.btn_tz_out, self.btn_tz_in, self.btn_tz_fit = (
            self.timeline.btn_zoom_out, self.timeline.btn_zoom_in, self.timeline.btn_zoom_fit)
        self.timeline.seek_requested.connect(self._goto)
        self.timeline.point_selected.connect(self._on_select)
        self.timeline.events_changed.connect(self._on_events_changed)
        self.timeline.clear_requested.connect(self._clear_tracked_window)
        self.timeline.clear_masks_requested.connect(
            lambda f0, f1: self._clear_masks_window(f0, f1, self.timeline.sel_segs))     # (G149) the lanes covered
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
        bl.addWidget(self._controls)
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

    def _build_side_panel(self) -> None:
        """The right dock: cameras, then LAYERS -- every animal with its points, then Scene (G154)."""
        self.layers = LayersPanel()
        self.layers.setToolTip(
            "LAYERS: each animal with its points under it, then Scene (points of no animal).\n"
            "Click a row to select it (Ctrl / Shift+click for several, Ctrl+A for everything): Track tracks\n"
            "exactly what is selected -- an animal row = its silhouette and all its points.\n"
            "With a point selected, a plain click on the video places it on this frame.\n"
            "Drag points onto another animal (or Scene) to move them there. Double-click renames;\n"
            "the checkbox shows / hides; right-click for everything else. AT / CT / MS beside a point =\n"
            "its tracker.")
        self.layers.itemChanged.connect(self._on_layer_item_changed)
        self.layers.currentItemChanged.connect(self._on_layer_current)
        # the run scope follows the selection: keep the Track button's wording current
        self.layers.itemSelectionChanged.connect(self._on_point_selection_changed)     # (G80)
        self.layers.setContextMenuPolicy(Qt.CustomContextMenu)
        self.layers.customContextMenuRequested.connect(self._layers_menu)
        self.layers.move_requested.connect(self._on_layers_move)

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
        self.cameras.fps_requested.connect(self._set_camera_fps)
        self.cameras.sync_toggled.connect(lambda on: self._set_sync_mode("all" if on else "active"))
        self.cameras.shown_toggled.connect(self._set_view_shown)            # (G169) a row's eye
        self.cameras.show_all_requested.connect(self._show_all_views)
        self.cameras.order_requested.connect(self._camera_order_dialog)     # (G172)
        pl.addWidget(self.cameras)
        sep0 = QWidget()
        sep0.setFixedHeight(1)
        sep0.setStyleSheet(f"background: {theme.HAIRLINE};")
        pl.addWidget(sep0)
        head = QLabel("LAYERS")
        head.setStyleSheet(f"color: {theme.TEXT_DIM}; font-weight: 600; letter-spacing: 1px;")
        pl.addWidget(head)
        # two short rows (one long row widened the panel and folded the control bar's labels)
        self.btn_new_segment = QToolButton()
        self.btn_new_segment.setText("＋ Animal")
        self.btn_new_segment.setFocusPolicy(Qt.NoFocus)
        self.btn_new_segment.setToolTip(
            "Add an animal (a layer): give it points by selecting it and clicking them on the video (N), or by "
            "dragging points onto it; press S and click it on the video to outline its silhouette (optional).")
        self.btn_new_segment.clicked.connect(self._new_segment)
        self.btn_new_point = QToolButton()
        self.btn_new_point.setText("＋ Point")
        self.btn_new_point.setFocusPolicy(Qt.NoFocus)
        self.btn_new_point.setToolTip(
            "Make a named point with no position yet, in the selected animal (else in Scene), and select it; "
            "then click it on the video. With several cameras it is in every camera's list: click it in one "
            "camera, click the next camera, click it there (on the dashed line, or on the ◇ once two cameras "
            "have it). Double-click the name to rename it. (N + click makes and places a point in one go.)")
        self.btn_new_point.clicked.connect(self._new_point)
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        row1.addWidget(self.btn_new_segment)
        row1.addWidget(self.btn_new_point)
        row1.addStretch(1)
        pl.addLayout(row1)
        self.btn_skeleton = QToolButton()
        self.btn_skeleton.setText("Skeleton ▾")
        self.btn_skeleton.setPopupMode(QToolButton.InstantPopup)
        self.btn_skeleton.setToolTip("Give the selected animal a named set of body points (a template), or save "
                                     "its points and bones as one")
        self.m_skeleton_btn = QMenu(self.btn_skeleton)
        self.btn_skeleton.setMenu(self.m_skeleton_btn)
        self.btn_clear_animal = QToolButton()
        self.btn_clear_animal.setText("Delete")
        self.btn_clear_animal.setFocusPolicy(Qt.NoFocus)
        self.btn_clear_animal.setToolTip("Delete what is selected in LAYERS: points (with their tracks) and / or "
                                         "animals (you choose whether their points go too). Ctrl+Z undoes points.")
        self.btn_clear_animal.clicked.connect(lambda _=False: self._delete_layers())
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        row2.addWidget(self.btn_skeleton)
        row2.addWidget(self.btn_clear_animal)
        row2.addStretch(1)
        pl.addLayout(row2)
        pl.addWidget(self.layers, 1)
        self.animal_label = QLabel("")
        self.animal_label.setWordWrap(True)
        self.animal_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.animal_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        pl.addWidget(self.animal_label)
        dock = QDockWidget("Layers", self)
        dock.setWidget(panel)
        dock.setMinimumWidth(230)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.dock = dock
    def _build_file_menu(self) -> None:
        """File menu (and its Import submenu)."""
        # menu / toolbar actions
        m_file = self.menuBar().addMenu("&File")
        self.act_open = QAction("Open &Video…", self, shortcut=QKeySequence("Ctrl+O"),
                                triggered=self._open_video_dialog)
        self.act_open_proj = QAction("Open &Project…", self, shortcut=QKeySequence("Ctrl+Shift+O"),
                                     triggered=self._open_project_dialog)
        self.act_open_folder = QAction("Open &Folder of Videos…", self, triggered=self._open_folder_dialog)
        self.act_open_folder.setToolTip("Several cameras of one event in one folder: pick the folder, tick the "
                                        "videos to import and choose the base (reference) camera — they open "
                                        "as one project, saved at once if you like")
        self.act_save = QAction("&Save Project", self, shortcut=QKeySequence("Ctrl+S"),
                                triggered=self._save_project)
        self.act_save_as = QAction("Save Project &As…", self, shortcut=QKeySequence("Ctrl+Shift+S"),
                                   triggered=self._save_project_as)
        self.act_export_one = QAction("Export Project as &One File…", self, triggered=self._export_single_file)
        self.act_export_one.setToolTip("The whole project in one .kinetrace file, to e-mail or archive "
                                       "(File → Open Project opens it); the project folder stays as it is")
        self.act_exports = QAction("&Keep Exports Up to Date…", self, triggered=self._exports_dialog)
        self.act_exports.setToolTip("Choose files for DeepLabCut, DLTdv or MATLAB that are written into the "
                                    "project folder's exports/ at every save, whenever their data changed")
        self.act_recover = QAction("&Recover Unsaved Work…", self, triggered=self._recover_dialog)
        self.act_recover.setToolTip("Work that was never saved (the program closed, or you chose not to "
                                    "save) is kept in Kinetrace's recovery folder: open it from here")
        self.act_import_tracks = QAction("&Import Tracks…", self, triggered=lambda: self._import_tracks_dialog())
        self.act_import_tracks.setToolTip("Tracks made in DeepLabCut, SLEAP, DLTdv / Argus or another Kinetrace "
                                          "project, into the camera on screen (points matched by name)")
        self.act_export = QAction("&Export Tracks…", self, shortcut=QKeySequence("Ctrl+E"),
                                  triggered=self._export_dialog)
        self.act_overlay = QAction("Export Overlay &Video…", self, triggered=self._export_overlay)
        self.act_overlay.setToolTip("An MP4 with markers, names, skeleton, silhouette, trails, frame "
                                    "counter, events and notes drawn on it — for talks and for checking "
                                    "a result without the app")
        self.act_import_xyz = QAction("3D &Points…", self, triggered=self._import_points3d)
        self.act_import_xyz.setToolTip("3D landmarks from Anipose, DLTdv (xyzpts) or another Kinetrace project, "
                                       "for the 3D view and the kinematics export")
        self.act_import_offsets = QAction("Camera &Offsets…", self, triggered=self._import_offsets)
        self.act_import_offsets.setToolTip("Each camera's frame offset and frame rate from a CSV "
                                           "(camera, name, offset, rate), matched by camera name")
        self.act_import_masks = QAction("&Silhouettes (mask images)…", self, triggered=self._import_masks)
        self.act_import_masks.setToolTip("A folder of black / white mask images named with their frame number "
                                         "(mask_000012.png), from another segmentation tool")
        for a in (self.act_open, self.act_open_folder, self.act_open_proj, self.act_recover, None, self.act_save,
                  self.act_save_as, None):
            m_file.addSeparator() if a is None else m_file.addAction(a)
        m_import = m_file.addMenu("&Import")
        self.act_import_tracks.setText("&Tracks…")
        # act_calib ("Import Calibration…" in the 3D menu) is created below; added in _add_import_calib
        self._m_import = m_import
        for a in (self.act_import_tracks, self.act_import_xyz, self.act_import_offsets, self.act_import_masks):
            m_import.addAction(a)
        # File → Quit (G43): the same close as the window's ×, so unsaved
        # changes are asked about first; Ctrl+Q (Cmd+Q on a Mac, where Qt moves it to the app menu)
        self.act_quit = QAction("&Quit", self, shortcut=QKeySequence("Ctrl+Q"), triggered=self.close)
        self.act_quit.setMenuRole(QAction.QuitRole)
        self.act_quit.setToolTip("Close Kinetrace (asks first when there are unsaved changes)")
        for a in (self.act_export, self.act_exports, self.act_overlay, None, self.act_export_one, None,
                  self.act_quit):
            m_file.addSeparator() if a is None else m_file.addAction(a)

    def _build_edit_menu(self) -> None:
        """Edit menu."""
        m_edit = self.m_edit = self.menuBar().addMenu("&Edit")
        self.act_undo = QAction("&Undo Last Run / Edit", self,
                                shortcut=QKeySequence("Ctrl+Z"),
                                triggered=self._undo_run, enabled=False)
        m_edit.addAction(self.act_undo)
        self.act_select_all = QAction("Select &Everything to Track", self, shortcut=QKeySequence("Ctrl+A"),
                                      triggered=self._select_all_tracked)
        self.act_select_all.setToolTip("Selects every animal and every point in LAYERS: Track then tracks all of "
                                       "them (only what is selected is tracked). Esc clears the selection.")
        m_edit.addAction(self.act_select_all)
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
        m_edit.addSeparator()
        # (G147) fixing which point is which
        self.act_point_tools = QAction("&Point Tools… (swap, move, fill, split)", self,
                                       triggered=lambda: self._point_tools())
        self.act_point_tools.setToolTip("Swap two points the tracker mixed up, give a stretch of one point's data "
                                        "to another, fill one point's empty frames from another, or split a point "
                                        "in two at a frame. Each is one Ctrl+Z step.")
        m_edit.addAction(self.act_point_tools)

    def _build_view_menu(self) -> None:
        """View menu: panel, other cameras, overlays, trails, display filters."""
        m_view = self.menuBar().addMenu("&View")
        act_panel = self.dock.toggleViewAction()
        act_panel.setText("Layers panel")
        act_panel.setShortcut(QKeySequence("Ctrl+1"))
        act_panel.setIcon(icons.panel())
        m_view.addAction(act_panel)
        self.act_onboarding = QAction("Getting started strip", self, checkable=True, checked=True)
        self.act_onboarding.toggled.connect(lambda on: self.onboarding.setVisible(on))
        m_view.addAction(self.act_onboarding)
        # the other cameras (G24): one choice of three
        m_others = m_view.addMenu("Other &cameras")
        m_others.setToolTipsVisible(True)
        self.m_others = m_others
        self.act_sync_all = QAction("&Sync all views (follow the playhead)", self, checkable=True, checked=True)
        self.act_sync_all.setToolTip("Every camera shows the instant the playhead is on, with its points and "
                                     "the epipolar guides; a single step shows every camera's new picture at once")
        self.act_sync_active = QAction("&Active view only (the others stay where they are)", self,
                                       checkable=True, shortcut=QKeySequence("Ctrl+Shift+2"))
        self.act_sync_active.setToolTip(
            "Only the working camera reads its video (fastest with many 4K cameras); the others keep the "
            "picture they last showed, veiled, until you choose Sync all views again or click one to work in "
            "it. The guides and ◇ in the working camera need only the other cameras' tracks, not their "
            "pictures, so they keep working")
        self.act_solo = QAction("Only the &working camera (hide the others)", self, checkable=True, checked=False,
                                shortcut=QKeySequence("Ctrl+2"))
        self.act_solo.setToolTip("Hide the other cameras — they also stop decoding, "
                                 "which gives scrubbing the whole disk back")
        # ExclusiveOptional: Ctrl+2 / Ctrl+Shift+2 pressed again untick their choice
        # (a strictly exclusive group ignores that); the handlers then fall back
        # (Hidden -> how the cameras were followed, Active view only -> Sync all)
        self._others_group = QActionGroup(self)
        self._others_group.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)
        for a in (self.act_sync_all, self.act_sync_active, self.act_solo):
            self._others_group.addAction(a)
            m_others.addAction(a)
        self.act_sync_all.toggled.connect(lambda on: self._on_sync_action("all", on))
        self.act_sync_active.toggled.connect(lambda on: self._on_sync_action("active", on))
        self.act_solo.toggled.connect(self._on_solo_toggled)
        # (G169, G170) which views and where: also the eyes in CAMERAS and each view's title bar
        m_others.addSeparator()
        self.act_show_all_views = QAction("Show &every camera", self, triggered=lambda _=False: self._show_all_views())
        self.act_show_all_views.setToolTip("Every camera's view on screen again (the eye in CAMERAS hides one)")
        self.act_views_in_order = QAction("&Arrange the views in camera order", self,
                                          triggered=lambda _=False: self._views_in_camera_order())
        self.act_views_in_order.setToolTip("Undo the views' arrangement (dragged by their title bars): camera 1 "
                                           "first, then 2, 3, ...")
        for a in (self.act_show_all_views, self.act_views_in_order):
            m_others.addAction(a)
        m_others.setEnabled(False)
        self.act_solo.setEnabled(False)
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
        # and Edit ends with it, as the Preferences entry: on a Mac Qt moves it to the application
        # menu (Kinetrace -> Settings..., Cmd+,), where every Mac app has it (Mac report 2026-10-08:
        # it was in no menu-bar menu, so a Mac user could not find it)
        self.act_settings.setMenuRole(QAction.PreferencesRole)
        self.m_edit.addSeparator()
        self.m_edit.addAction(self.act_settings)
        self._refresh_seg_menu()
        m_view.addAction(self.act_show_midline)
        m_view.addAction(self.act_show_bones)
        m_view.addSeparator()
        # trajectory trails: length + upcoming path
        m_trails = m_view.addMenu("&Trails")
        self._trail_group = QActionGroup(self)
        self._trail_acts: dict[int, QAction] = {}
        # three choices (G33): off, the preset, or a length you type
        for n in (0, TRAIL_FRAMES):
            act = QAction("Off" if n == 0 else f"Last {n} frames", self, checkable=True)
            act.triggered.connect(lambda _=False, k=n: self._set_trail_len(k))
            self._trail_group.addAction(act)
            m_trails.addAction(act)
            self._trail_acts[n] = act
        self.act_trail_custom = QAction("Custom…", self, checkable=True)
        self.act_trail_custom.setToolTip(f"Type how many frames of trail to draw behind each point "
                                         f"(1–{TRAIL_MAX})")
        self.act_trail_custom.triggered.connect(self._ask_trail_len)
        self._trail_group.addAction(self.act_trail_custom)
        m_trails.addAction(self.act_trail_custom)
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
        self.act_epipolar.setToolTip("With a calibration (switched on when one is loaded): dashed lines in EVERY "
                                     "camera showing where the SELECTED landmark, as the other cameras see it at "
                                     "this instant, can lie. Place it on the line (or right-click it → Snap to "
                                     "the other cameras' rays). Once two other cameras have it, a ◇ shows where it "
                                     "is (A places it there). Alt+click on the video shows where any spot can be in "
                                     "the other cameras (Esc clears it).")
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

    def _build_skeleton_events_menus(self) -> None:
        """Skeleton and Events menus."""
        # skeleton templates (also under the panel's Skeleton ▾ button)
        self.m_skeleton = self.menuBar().addMenu("&Skeleton")
        self._refresh_skeleton_menu()

        # E is handled in keyPressEvent (a menu shortcut would steal the letter
        # from rename editors); the menu wording documents it
        self.m_events = self.menuBar().addMenu("E&vents")
        self.act_mark_event = QAction("Mark Event Start / End  (E)", self,
                                      triggered=self._mark_event)
        self._refresh_events_ui()

    def _build_3d_menu(self) -> None:
        """3D menu: calibration -> sub-frame sync -> triangulation -> volume hull."""
        # 3D: calibration -> sub-frame sync -> triangulation -> volume hull
        m_3d = self.menuBar().addMenu("&3D")
        self.act_lens = QAction("Calibrate a &Lens (checkerboard)…", self, triggered=self._lens_wizard)
        self.act_lens.setToolTip("Measure how a lens bends the picture from a video of a printed checkerboard. "
                                 "Needed for wide-angle / action cameras before a wand calibration; explains "
                                 "when you need it and when you do not.")
        # (G148) a camera's lens profile out to a file, and a file onto a camera, at any time
        self.act_export_lens = QAction("E&xport Lens Profile…", self, triggered=self._export_lens_profile)
        self.act_export_lens.setToolTip("Save a camera's lens profile (from the checkerboard, GoPro's lens model or a\n"
                                        "file) to reuse it later: for this camera in other projects, or for other\n"
                                        "cameras of the same model, lens, zoom and recording mode")
        self.act_load_lens = QAction("Load a Lens Profile for This Camera…", self, triggered=self._load_lens_profile)
        self.act_load_lens.setToolTip("Attach a saved lens profile (.klens.json, OpenCV .yml / .json, Argus .txt) to\n"
                                      "the working camera; the picture size must match")
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
        self.act_set_axes = QAction("Set World &Axes…", self, triggered=self._set_world_axes)      # (I170)
        self.act_set_axes.setToolTip("Choose the 3D world yourself: three landmarks give the origin, the +X and the\n"
                                     "+Y direction (+Z follows the right-hand rule). The calibration and the 3D\n"
                                     "result are re-expressed in it; the pictures are not touched. Needs a 3D result\n"
                                     "(Reconstruct, Ctrl+3).")
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
        self.act_gopro = QAction("&GoPro Cameras…", self, triggered=self._gopro_dialog)       # (G145)
        self.act_gopro.setToolTip("For GoPro footage only: each camera's recording settings, its tilt from the\n"
                                  "gravity sensor, dropped frames, when it moved, and GoPro's own lens model\n"
                                  "(for the cameras that have no lens profile yet)")
        # the 3D menu's entry, also under File -> Import (its own action: "Import" is the submenu's word)
        self.act_import_calib = QAction("&Calibration…", self, triggered=self._import_calibration)
        self.act_import_calib.setToolTip(self.act_calib.toolTip())
        self._m_import.insertAction(self.act_import_xyz, self.act_import_calib)
        for a in (self.act_sync, self.act_gopro, None, self.act_lens, self.act_load_lens, self.act_export_lens,
                  self.act_wand, self.act_calib, self.act_export_cal,
                  self.act_offsets3d, None, self.act_recon, self.act_set_axes, self.act_retrack, self.act_hull, None,
                  self.act_view3d,
                  self.act_export_mesh):
            m_3d.addSeparator() if a is None else m_3d.addAction(a)

    def _build_body_menu(self) -> None:
        """Body menu: human joints and joint angles from the footage itself."""
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

    def _build_help_menu(self) -> None:
        """Help menu."""
        m_help = self.menuBar().addMenu("&Help")
        self.act_manual = QAction("&User Manual…", self, shortcut=QKeySequence("F1"),
                                  triggered=self._show_manual)
        self.act_manual.setToolTip("The full manual, written for someone new to tracking")
        m_help.addAction(self.act_manual)
        self.act_help_models = QAction("Which &Point Model Should I Use?", self,
                                       triggered=lambda: self._show_manual(MANUAL_WHICH_MODEL))
        self.act_help_models.setToolTip("A visible shape (an animal, an object) -> AllTracker + Segment, no extra "
                                        "clicks; a target small enough to be one point (a dot up to ~20 px) -> "
                                        "Moving spot; a round marker -> Ball marker -- and the test that decides "
                                        "on your own footage")
        m_help.addAction(self.act_help_models)
        self.act_syscheck = QAction("System &Check… (GPU, memory, what runs here)", self,
                                    triggered=self._show_system_check)
        self.act_syscheck.setToolTip("What this computer has (graphics card, memory, PyTorch build) and "
                                     "which features run on it, run slower, or are switched off")
        m_help.addAction(self.act_syscheck)
        self.act_error_report = QAction("&Error Report… (what went wrong, to copy into a bug report)", self,
                                        triggered=self._show_error_report)
        self.act_error_report.setToolTip(
            "The errors Kinetrace recorded while running (kinetrace.log in the Kinetrace folder's logs "
            "folder) and the System Check, ready to copy into a bug report. Nothing is sent anywhere.")
        m_help.addAction(self.act_error_report)
        self.act_folders = QAction("Kinetrace's &Folders… (where everything is kept)", self,
                                   triggered=self._show_folders)
        self.act_folders.setToolTip(
            "Every folder Kinetrace reads or writes, with its size, and a button to open it -- what "
            "deleting the Kinetrace folder removes, and what it leaves (your projects and exports)")
        m_help.addAction(self.act_folders)
        m_help.addAction(QAction("&Keyboard && Mouse Reference…", self,
                                 triggered=self._show_hotkeys))
        m_help.addSeparator()
        self.act_updates = QAction("Check for &Updates…", self, triggered=self._check_updates)
        self.act_updates.setToolTip(
            "Asks GitHub whether a newer Kinetrace has been published and, if so, installs it and restarts. "
            "Your projects, models and settings are kept. Nothing is checked unless you ask.")
        m_help.addAction(self.act_updates)
        self.act_about = QAction(f"&About {APP_NAME}…", self, triggered=self._show_about)
        self.act_about.setToolTip("Version, licence, where Kinetrace comes from, and the models it builds on")
        m_help.addAction(self.act_about)

    def _build_nav_shortcuts(self) -> None:
        """The Left / Right / Home / End shortcuts (single-letter keys go through `_hotkey`)."""
        # combo shortcuts only — single-letter keys (F/B/N/T/X/E/R/H/Space/
        # Delete/Esc/±) are handled in keyPressEvent so they never steal
        # keystrokes from the point-rename editor or other text fields
        # kept, so the loading card can switch them off with the menus (G71)
        self._nav_shortcuts = [
            QShortcut(QKeySequence("Right"), self, lambda: self._goto(self.current + 1)),
            QShortcut(QKeySequence("Left"), self, lambda: self._goto(self.current - 1)),
            QShortcut(QKeySequence("Shift+Right"), self,
                      lambda: self._goto(self.current + self.step_spin.value())),
            QShortcut(QKeySequence("Shift+Left"), self,
                      lambda: self._goto(self.current - self.step_spin.value())),
            QShortcut(QKeySequence("Home"), self, lambda: self._goto(0)),
            QShortcut(QKeySequence("End"), self, lambda: self._goto(self.n_frames - 1))]

    def _set_focus_policies(self) -> None:
        """Keyboard focus discipline: tool buttons never take the focus, spin boxes only by click."""
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

    def _build_status_bar(self) -> None:
        """Status bar: frame, device and run labels."""
        # status bar
        # eliding labels: a plain QLabel's minimum width is its whole text, so a
        # longer frame / fps / ETA text widened the WINDOW (it can be as narrow
        # as a laptop screen since G2) and refitted the video (G8)
        from kinetrace.widgets import ElidedLabel
        self._frame_label = ElidedLabel("no video", pad=16)
        self._device_label = ElidedLabel("", pad=16)
        self._track_label = ElidedLabel("", pad=16)
        for w in (self._frame_label, self._device_label, self._track_label):
            self.statusBar().addPermanentWidget(w)
            w.setStyleSheet("padding: 0 8px;")

    @staticmethod
    def _tool_button(text: str, icon, tip: str, checked: bool | None = None) -> QToolButton:
        """A labelled, checkable tool button of the control bar: icon + text, ticked or not from the
        start (`checked` None = leave it), with its tooltip (R7). The caller connects its signal
        afterwards, so the initial state never fires a handler."""
        b = QToolButton()
        b.setText(text)
        b.setIcon(icon)
        b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        b.setCheckable(True)
        if checked is not None:
            b.setChecked(checked)
        b.setToolTip(tip)
        return b

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
        # (I258) `map_frame`'s tie rule (a .5 rounds DOWN on the way back to an earlier
        # view), not floor(x + 0.5): the two disagreed at every half-frame offset
        return int(p.reference_index(p.active, self.current))

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
        need = self._controls_full_w + self._track_extra      # the Track label's live width (G139)
        level = 0
        while need > avail and level < len(self._compact_order):
            need -= self._label_saving.get(self._compact_order[level], 0)
            level += 1
        if level == self._controls_level:
            return
        self._set_compact_controls(level)

    def _fit_track_label(self) -> None:
        """The Track button's label changes with what is selected ("Track · 3 points +
        segment ▶ · 3 cams", up to ~285 px); the bar was measured with an empty one
        (150 px), so a long label clipped Track ▾ (G139). The button now keeps the
        width its label needs, the bar's minimum follows, and the tool buttons fold
        against the live width."""
        if getattr(self, "_controls_min_w", None) is None:
            return
        want = max(self._track_base_w, self.btn_track.sizeHint().width())
        extra = want - self._track_base_w
        if extra == self._track_extra:
            return
        self._track_extra = extra
        self.btn_track.setMinimumWidth(want)
        self._controls.setMinimumWidth(self._controls_min_w + extra)
        self._fit_controls()

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
        self.layers.setEnabled(has_video and not tracking)
        # a run belongs to ONE camera: switching views or retiming mid-run would
        # pull the session out from under the worker
        self.cameras.setEnabled(has_video and not tracking)
        self._refresh_cameras()
        s = self.session
        exportable = s is not None and (s.n_points > 0 or any(m.n_masked() > 0 for m in s.seg_masks))  # (G151m)
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
        self.act_import_calib.setEnabled(live3d)
        # the two wizards stay clickable: a first-time user must be able to open
        # them and be TOLD what to prepare, not meet a greyed-out entry
        self.act_wand.setEnabled(not tracking)
        self.act_lens.setEnabled(not tracking)
        self.act_export_cal.setEnabled(live3d)
        self.act_export_lens.setEnabled(live3d)      # (G148) clickable; it says when no camera has a profile
        self.act_load_lens.setEnabled(live3d)
        self.act_offsets3d.setEnabled(live3d)
        self.act_sync.setEnabled(live3d)
        self.act_recon.setEnabled(live3d)
        self.act_set_axes.setEnabled(live3d)        # clickable like the others; it explains what it needs
        self.act_gopro.setEnabled(live3d and bool(self._gopro_infos()))          # (G145) GoPro footage only
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
        # (I265) Save stays available during a run: it is done when the run is over (`_save_project`);
        # greyed out, Ctrl+S did nothing at all and said nothing
        self.act_save.setEnabled(has_video)
        self.act_save_as.setEnabled(has_video and not tracking)
        self.act_export_one.setEnabled(has_video and not tracking)
        self.act_exports.setEnabled(has_video and not tracking)
        self.act_open.setEnabled(not tracking)
        self.act_open_folder.setEnabled(not tracking)
        self.act_open_proj.setEnabled(not tracking)
        # recovering replaces the project on screen: never in the middle of a run
        for a in (self.act_import_tracks, self.act_import_xyz, self.act_import_offsets, self.act_import_masks,
                  self.act_recover):
            a.setEnabled(not tracking)
        self.act_undo.setEnabled(has_video and not tracking and self._undo_snap is not None)
        self.act_mark_event.setEnabled(has_video and not tracking)
        self.timeline.setEnabled(has_video and not tracking)  # still paints progress live
        # ONLY the working camera takes edits — a click on a companion view
        # switches to it instead, so an edit can never land in the wrong session
        active_i = self.project.active if self.project is not None else 0
        for k, cv in enumerate(self.grid.canvases):
            cv.set_interactive(has_video and not tracking and k == active_i)
            cv.set_switchable(has_video and not tracking and k != active_i)
        self.btn_add.setEnabled(has_video and not tracking)
        if tracking and self.btn_add.isChecked():
            self.btn_add.setChecked(False)
        self.btn_animal.setEnabled(has_video and not tracking)
        if tracking and self.btn_animal.isChecked():
            self.btn_animal.setChecked(False)
        self.btn_mask.setEnabled(has_video)
        self.btn_skeleton.setEnabled(has_video and not tracking)
        self.btn_new_point.setEnabled(has_video and not tracking)
        self.btn_new_segment.setEnabled(has_video and not tracking)
        self.act_point_tools.setEnabled(has_video and not tracking)      # (G147) the dialog says when there are no points
        self.m_skeleton.setEnabled(has_video and not tracking)
        # (G164) LAYERS -> Delete deletes points as well as animals (G154): usable whenever there is either
        self.btn_clear_animal.setEnabled(bool(has_video and not tracking and s is not None
                                              and (s.n_segments or s.n_points)))
        self.btn_pan.setEnabled(has_video)  # panning is view-only: fine mid-run
        for act in (self.act_mode_auto, self.act_mode_semi, self.act_track_all):
            act.setEnabled(not tracking)
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
        seg = any(s.has_silhouette(k) for k in range(s.n_segments))   # any (G151m)
        # the skeleton step ticks as soon as a skeleton is applied or any landmark exists
        landmarks = any(a.skeleton for a in s.segments) or s.n_points > 0
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
        """Which points a run covers: EXACTLY what is selected in LAYERS (owner, 2026-10-02, G61 --
        nothing selected used to mean everything): the selected points and every point of a
        selected animal row (G154: an animal row stands for its silhouette and all its points).
        Returns (set of pids, n_selected)."""
        s = self.session
        if s is None:
            return set(), 0
        sel = {p for p in self._selected_pids() if p < s.n_points}
        for k in self._selected_segments():
            sel |= set(s.points_of(k))
        return sel, len(sel)

    def _segment_selected(self) -> bool:
        """An animal row is selected: its silhouette is part of the next run (G61, G154)."""
        return bool(self._selected_segments())

    def _selected_segments(self) -> list[int]:
        """The animals whose rows are selected in LAYERS (G154)."""
        return self.layers.selected_animals()

    def _select_segments(self, idx, on: bool = True) -> None:
        self.layers.set_selected(animals=list(idx), on=on)

    def _selection_animals(self) -> list[int]:
        """The animals the LAYERS selection names (G154): the selected animal rows and the animals of
        the selected points."""
        s = self.session
        if s is None:
            return []
        ks = set(self._selected_segments())
        for q in self._selected_pids():
            k = s.segment_of(q)
            if k is not None:
                ks.add(k)
        return sorted(ks)

    def _s_target(self) -> int | None:
        """The ONE animal the selection names -- what the S tool outlines and the clicks drawn on the
        video belong to (G154); None when the selection names none or several. It replaces the
        hidden 'active segment' (G150, G151a): the active index only ever follows this."""
        ks = self._selection_animals()
        if len(ks) == 1:
            return ks[0]
        s = self.session
        # the only animal there is, when the selection names none (fix a mask: S, click -- as ever)
        return 0 if (not ks and s is not None and s.n_segments == 1) else None

    def _sync_s_target(self) -> None:
        """Point the session's active animal at the selection's one animal, so the S tool and the
        drawn clicks are the selected animal's."""
        s = self.session
        k = self._s_target()
        if s is not None and k is not None and k != s.active_seg:
            s.active_seg = k
        if s is not None:
            self.animal_label.setText(self._animal_status_text())

    def _pick_segment(self, what: str) -> int | None:
        """The ONE animal an action works on (G151, G154): the only animal; else the one the LAYERS
        selection names; else the user picks it. None = no animal or Cancel. Never a hidden
        'active' animal on its own: that is how G150 removed the wrong one."""
        s = self.session
        if s is None or not s.segments:
            return None
        if s.n_segments == 1:
            return 0
        k = self._s_target()
        if k is not None:
            return k
        names = s.segment_names()
        name, ok = QInputDialog.getItem(
            self, "Which animal?",
            f"{what} -- which animal? (Select that animal in LAYERS to skip this question.)",
            names, s.active_seg if 0 <= s.active_seg < len(names) else 0, False)
        return names.index(name) if ok and name in names else None

    def _segments_for_view(self) -> list[int]:
        """The animals a look-at action without a lane of its own uses (G151): the ones the selection
        names, else every animal."""
        s = self.session
        if s is None:
            return []
        return self._selection_animals() or list(range(s.n_segments))

    def _held(self, s, q: int) -> bool:
        """Point q is held on its animal's silhouette (G156): it belongs to an animal that holds its
        points and has a silhouette, is tracked by appearance and is not marked 'may leave'."""
        if not (0 <= q < s.n_points):
            return False
        k = s.segment_of(q)
        m = s.points[q]
        return (k is not None and s.segments[k].hold and s.has_silhouette(k)
                and not m.derived and not m.is_ball and not m.free)

    def _run_segments(self, scope, s=None) -> list[int]:
        """The animals whose silhouettes a run with `scope` covers (G149, G154): the selected animal
        rows, the animal a selected landmark is derived from (it cannot fill in without the
        silhouette), and the animal a selected point is held on (I185, G156)."""
        s = s if s is not None else self.session
        if s is None or not s.segments:
            return []
        out = set(self._selected_segments()) if s is self.session else set()
        for q in scope:
            if 0 <= q < s.n_points:
                k = s.segment_of(q)
                if k is not None and (s.points[q].derived or self._held(s, q)):
                    out.add(k)
        return sorted(k for k in out if k < s.n_segments and s.has_silhouette(k))

    def _seg_list(self, scope, segment, s=None) -> list[int]:
        """The animals of a run: `segment` None = the selection rule (`_run_segments`), False = none,
        True = those or the animals of the scope's points, a list = these animal NAMES (an
        every-camera run names the working camera's animals) plus the ones the scope's points ride
        on (G149)."""
        s = s if s is not None else self.session
        if s is None or not s.segments or segment is False:
            return []
        if isinstance(segment, (list, tuple, set)):
            named = [k for k, n in enumerate(s.segment_names()) if n in set(segment)]
            return sorted(set(named) | set(self._run_segments(scope, s)))
        segs = self._run_segments(scope, s)
        if segment is True and not segs:
            # the animals the run's own points belong to (G151n; was the invisible active row)
            segs = sorted({k for q in scope if 0 <= q < s.n_points for k in [s.segment_of(q)]
                           if k is not None and s.has_silhouette(k)})
        return segs

    def _run_segment(self, scope) -> bool:
        """A silhouette runs (G149: any of them) -- see `_run_segments`."""
        return bool(self._run_segments(scope))

    def _tracker_of(self, pid: int, s=None) -> str:
        """A point's tracker (G62): its own, else the project's default point model."""
        from kinetrace.session import TRACKERS
        s = s if s is not None else self.session
        t = s.points[pid].tracker if s is not None and 0 <= pid < s.n_points else ""
        t = t if t in TRACKERS else self._point_backend
        if t == "alltracker" and not _alltracker_available():
            t = "cotracker3"
        return t

    def _tracker_passes(self, pids) -> list[list[int]]:
        """The passes one Track press needs (G63): AllTracker points (or CoTracker3
        ones) + Moving spot points + ball markers + derived landmarks in the first;
        CoTracker3 points in a second when AllTracker points are selected too
        (owner: one after the other)."""
        s = self.session
        nn: dict[str, list[int]] = {}
        rest = []
        for q in sorted(pids):
            if q >= s.n_points:
                continue
            m = s.points[q]
            t = "" if (m.derived or m.is_ball) else self._tracker_of(q)
            if t in ("alltracker", "cotracker3"):
                nn.setdefault(t, []).append(q)
            else:
                rest.append(q)
        order = [k for k in ("alltracker", "cotracker3") if k in nn]
        if not order:
            return [rest]
        # every segment's head (G151c: `head_pid()` alone is the FIRST segment's)
        heads = {s.head_pid(k) for k in range(s.n_segments)} - {None}
        if len(order) == 2 and heads & set(nn[order[1]]) and not heads & set(nn[order[0]]):
            # the pass holding the head landmark runs FIRST and carries the segment: the other
            # pass's points are then anchored to a silhouette that exists, and are kept on the
            # silhouettes the first pass wrote (the worker's `stored_masks`, I185)
            order.reverse()
        return [nn[order[0]] + rest] + [nn[k] for k in order[1:]]

    def _pass_startable(self, pids, carries_seg: bool, step: bool, every: bool, stops=None) -> bool:
        """Can a pass with these points (and the segment, when it `carries_seg`) start on this
        frame (I203)? A point needs a position here; a pass that carries the segment also starts
        from a silhouette. With Every camera, in at least one camera; `stops` {camera: last frame}
        = a later pass runs only in the cameras the first one ran in (I204)."""
        s = self.session
        p = self.project
        if every and p is not None and p.n_views > 1:
            names = {s.points[q].name for q in pids}
            return bool(self._multi_jobs(step, names=names, segment=carries_seg, stops=stops))
        if stops is not None and stops.get(p.active if p is not None else 0) is None:
            return False
        return bool(any(s.tracked[self.current, q] and not s.points[q].derived for q in pids)
                    or (carries_seg and any(s.animal_seedable_at(self.current, k) for k in range(s.n_segments))))

    def _plan_passes(self, scope, step: bool, every: bool) -> list[list[int]]:
        """The passes one Track press runs, from what can START on this frame (I203, G63):
        `_tracker_passes` without the passes that cannot (points with no position here and no
        segment to carry) -- before, pass 1 with nothing to start made the whole press do nothing,
        and the button counted passes that could never run. The segment, and the silhouette-derived
        landmarks that need it, ride with the first pass that runs."""
        s = self.session
        groups = [list(g) for g in self._tracker_passes(scope)] if (s is not None and scope) else [[]]
        if len(groups) < 2:
            return groups
        seg = self._run_segment(scope)
        derived = [q for g in groups for q in g if s.points[q].derived]
        out, carried = [], False
        for g in ([q for q in g if not s.points[q].derived] for g in groups):
            carries = seg and not carried
            if self._pass_startable(g, carries, step, every):
                out.append(g + (derived if carries else []))
                carried = carried or carries
        return out or [groups[0]]

    def _update_track_button(self):
        semi = self._track_mode == "semi"
        if self.state == TRACKING:
            self.btn_track.setText("Pause ■  (X)")
            self.btn_track.setEnabled(True)
            self._set_track_blocked(None)
            self.btn_track.setToolTip("Stop tracking (X or Space). Corrections are made while paused.")
            self._fit_track_label()
            return
        s = self.session
        scope, n_sel = self._run_scope()
        segs = self._run_segments(scope)    # once per refresh: it runs on every playhead move (simplify 2026-10-04)
        seg = bool(segs)
        # the button itself says what a run covers: exactly what is selected (G61)
        what_sel = []
        if n_sel:
            what_sel.append(f"{n_sel} point{'s' if n_sel != 1 else ''}")
        if seg:
            what_sel.append("silhouette" if len(segs) == 1 else f"{len(segs)} silhouettes")     # (G154) the animals'
        # the same rule the press uses: only passes that can start on this frame count (I203)
        n_pass = len(self._plan_passes(scope, semi, self.act_track_all.isChecked())) \
            if (s is not None and scope and self.state == READY) else 1
        # a held point brings its animal's silhouette although the animal's row is not selected (I185, G160)
        rides = bool(seg and s is not None and not self._segment_selected()
                     and not any(0 <= q < s.n_points and s.points[q].derived for q in scope))
        label = "Step" if semi else "Track"
        self.btn_track.setText(f"{label} · {' + '.join(what_sel)}" + (f" ({n_pass} passes)" if n_pass > 1 else "")
                               + (" ▶  (F)" if semi else " ▶") if what_sel else (f"{label} ▶  (F)" if semi
                                                                                 else f"{label} ▶"))
        jobs = self._multi_jobs(semi) if self.act_track_all.isChecked() and self.state == READY else []
        n_cams = len(jobs)
        # (G174, owner 2026-10-08) cameras whose view is hidden are tracked too, without being drawn: the
        # button says so before the press
        shown_v = set(self.grid.visible_indices())
        hidden_cams = [j["view"] for j in jobs if j["view"] not in shown_v] if n_cams > 1 else []
        if n_cams > 1:
            self.btn_track.setText(self.btn_track.text() + f" · {n_cams} cams"   # Track ▾ → Every camera (G29)
                                   + (f" ({len(hidden_cams)} hidden)" if hidden_cams else ""))
        pids_here, animal_ok = self._startable_here(scope, seg, segs)
        n = len(pids_here)
        # Nothing to start from: the button stays ENABLED, only drawn quiet, because Qt
        # disables a disabled button's menu too and Track ▾'s choices must stay reachable;
        # T / a click then says why instead of starting (G34)
        self.btn_track.setEnabled(True)
        if self.state == IDLE:
            blocked = "Open a video first (Ctrl+O) or a project (Ctrl+Shift+O)"
        elif s is not None and not scope and not seg and (s.n_points or s.animal is not None):
            blocked = ("Select what to track in LAYERS: click a point or an animal (an animal = its silhouette "
                       "and all its points; Ctrl+click for several, Ctrl+A for all) — only what is selected is "
                       "tracked")
        elif s is None or (n == 0 and not animal_ok):
            if scope and all(0 <= q < s.n_points and s.points[q].derived for q in scope):
                # silhouette-derived landmarks are never placed by hand: they fill in from the
                # segment (G120)
                blocked = (f"The selected landmark(s) fill in from the segment's silhouette, which has no click on "
                           f"frame {self.current}: press S and click the animal here (or go to a frame with its "
                           "silhouette) and select the segment's row in SEGMENT too"
                           if s.animal is not None else
                           "The selected landmark(s) fill in from the segment's silhouette: press S and click "
                           "the animal first, then select the segment's row in SEGMENT and Track")
            elif scope:
                blocked = (f"The selected point(s) have no position on frame {self.current}. Select a "
                           "point that exists here, or place it here first (click it on the video)")
            elif s is not None and s.animal is not None:
                blocked = ("Nothing to track from this frame: place a point (N), press S and click the "
                           "segment here, or move to a frame where the points/segment exist")
            else:
                blocked = ("No point has a position at this frame — press N and "
                           "click the animal to add one (a few pixels is enough); "
                           "S outlines a larger animal, optionally")
        else:
            blocked = None
        self._set_track_blocked(blocked)
        if blocked is not None:
            self.btn_track.setToolTip(blocked + ". Track ▾ still sets the mode and the point model.")
        else:
            what = []
            if n:
                what.append(f"the {n} selected point(s)")
            if animal_ok:
                what.append("the segment (silhouette)")
            what = " and ".join(what)
            if rides and animal_ok:
                what += (" (its animal keeps its points on its silhouette, so the silhouette rides along — "
                         "right-click the animal → untick 'Keep its points on its silhouette', or mark a point "
                         "'may leave its silhouette', to track without it)")
            if n_pass > 1:
                what += (" — in two passes, one after the other: the AllTracker points, then the CoTracker3 points "
                         "over the same frames (a point that stops ends the run for all"
                         + ("; the pass holding the head landmark goes first, with the silhouette, and the "
                            "second pass is kept on the silhouettes the first one makes — no second "
                            "segmentation" if seg else "") + ")")   # (I185)
            if semi:
                self.btn_track.setToolTip(
                    f"Semi-automatic: track {what} ONE frame forward from frame "
                    f"{self.current} and pause (F or T). Fix anything, press F again — "
                    "each step re-seeds from what you see.")
            else:
                self.btn_track.setToolTip(f"Track {what} forward from frame {self.current} (T)")
            if n_cams > 1:
                self.btn_track.setToolTip(
                    self.btn_track.toolTip() + f" — in each of the {n_cams} cameras that have them here, all at "
                    "the same time (Track ▾ → Every camera; X stops them all; one Ctrl+Z undoes all)"
                    + (f".\nHIDDEN: {self._hidden_cams_text(hidden_cams)} — tracked too, but not drawn while "
                       "hidden, so you cannot watch it there. Show them with their eye in CAMERAS (or Show all) "
                       "to watch every camera." if hidden_cams else ""))
        self._fit_track_label()

    def _set_track_blocked(self, reason: str | None):
        """None = Track can start here; else the sentence T / a click shows. The
        button is drawn quiet while blocked (the theme's [idle="true"] rule)."""
        self._track_blocked = reason
        idle = reason is not None
        if self.btn_track.property("idle") != idle:
            self.btn_track.setProperty("idle", idle)
            self.btn_track.style().unpolish(self.btn_track)   # a dynamic property needs a re-polish
            self.btn_track.style().polish(self.btn_track)

    @staticmethod
    def _preferred_point_backend() -> str:
        return "alltracker" if _alltracker_available() else "cotracker3"

    def _set_point_backend(self, key: str):
        """Track ▾ -> Point model: applies to the next run, at any time, and is
        the PROJECT's point model -- saved with it and restored when it is opened
        (owner, 2026-10-01; G56). The SELECTED points take it too (`_set_tracker`,
        G62); the test's Use goes through `_set_tracker` for its one point."""
        if key not in self._pm_acts:
            return
        changed = key != self._point_backend
        self._point_backend = key
        self._pm_acts[key].setChecked(True)
        if self.session is not None:
            self.session.ui_state["point_backend"] = key
            if changed:
                self.session.dirty = True        # a project setting: saved with the project
            sel = [q for q in self._selected_pids() if q < self.session.n_points
                   and not self.session.points[q].derived and not self.session.points[q].is_ball]
            if sel:                              # the selected points take it too (G62)
                self._set_tracker(sel, key)
        self.statusBar().showMessage(
            {"alltracker": "Point model: AllTracker — applies to the next Track run",
             "cotracker3": "Point model: CoTracker3 — applies to the next Track run",
             "spot": "Point model: Moving spot, for single-point targets — applies to the next Track run (click "
                     "the dot on two frames in a row first; it stops where it loses it)"}[key], 7000)
        self._update_track_button()

    @property
    def _track_mode(self) -> str:
        """"auto" | "semi" (F = one tracked step), READ from Track ▾'s checked
        mode action: the mode and the menu can never disagree (it used to be a
        separate variable mirroring the QActionGroup; I139)."""
        act = getattr(self, "act_mode_semi", None)
        return "semi" if act is not None and act.isChecked() else "auto"

    @_track_mode.setter
    def _track_mode(self, mode: str) -> None:
        # ticks the matching action (anything but "semi" is automatic); before
        # the menu exists there is nothing to tick -- Automatic starts ticked
        act = getattr(self, "act_mode_semi" if mode == "semi" else "act_mode_auto", None)
        if act is not None and not act.isChecked():
            act.setChecked(True)

    def _set_track_mode(self, mode: str):
        self._track_mode = mode
        self._update_track_button()
        if mode == "semi":
            self.statusBar().showMessage(
                "Semi-automatic mode: F tracks one frame forward and pauses; correct "
                "anything, then F again — B still just steps back", 8000)

    def _on_device_probed(self, label: str):
        """The background probe finished: the status bar shows a coloured badge
        - green "GPU" or amber "CPU" - with the device, and its tooltip says
        how the two compared and what is slower or switched off here."""
        self._device_label.setText(label)
        try:
            from kinetrace.device import cached_device, probe, short_status
            if cached_device() is None:
                return
            p = probe()
            gpu = p["kind"] == "gpu"
            self._device_kind = "GPU" if gpu else "CPU"
            name = p["gpu"] or ("Apple GPU" if p["mps"] and gpu else "")
            if gpu:
                text = f"● GPU  {name}"
            else:
                text = "● CPU" + (f"  ({name} available)" if name else "")
            self._device_label.setText(text)
            colour = theme.GREEN if gpu else theme.AMBER
            self._device_label.setStyleSheet(
                f"padding: 0 8px; color: {colour}; font-weight: 600;")
            b = p.get("bench") or {}
            measured = ""
            if b.get("gpu_ms") is not None and b.get("cpu_ms") is not None:
                measured = (f"\nMeasured: GPU {b['gpu_ms']:.1f} ms vs CPU {b['cpu_ms']:.1f} ms per pass "
                            f"({float(b.get('speedup') or 0):.1f}x) - the faster one is used.")
            self._device_label.setToolTip(
                ("The models run on the GPU: " if gpu else "The models run on the CPU: ") + label
                + measured + "\n" + short_status(p) + "\nHelp → System Check… has the details.")
        except Exception:       # noqa: BLE001 - a badge is never worth an error
            pass
        if label.startswith("cpu") or label.startswith("device unavailable"):
            self.statusBar().showMessage(
                label + " — Help → System Check… says what that means for each feature.", 12000)

    def _show_system_check(self):
        """Help → System Check: the hardware probe in plain words (the same
        text `python -m kinetrace --check` prints), with a Copy button for
        support requests. Runs the probe on the worker if it has not finished."""
        import sys
        from PySide6.QtGui import QFont
        from kinetrace.device import cached_device, describe
        probe = getattr(self, "_dev_probe", None)
        if cached_device() is None and probe is not None and probe.isRunning():
            # PyTorch is still loading in the probe thread: wait for it with the loading card up
            # and the window live (up to 20 s; Cancel shows the dialog without the result, G105)
            def wait_for_probe(cancelled):
                t0 = time.monotonic()
                while probe.isRunning() and time.monotonic() - t0 < 20 and not cancelled():
                    probe.wait(100)

            self._in_background("Checking this computer", wait_for_probe, cancellable=True,
                                detail="Loading PyTorch to look at the graphics card…",
                                hint="This happens once per start of Kinetrace.")
        text = describe() if cached_device() is not None else \
            "The hardware check has not finished yet (PyTorch is still loading). Try again in a moment."
        dlg = QDialog(self)
        dlg.setWindowTitle("System check")
        dlg.setMinimumSize(640, 420)
        lay = QVBoxLayout(dlg)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setFont(QFont("Consolas" if sys.platform.startswith("win") else "Menlo", 10)
                     if sys.platform != "linux" else QFont("Monospace", 10))
        lay.addWidget(view)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        copy = btns.addButton("Copy", QDialogButtonBox.ActionRole)
        copy.clicked.connect(lambda: QApplication.clipboard().setText(text))
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec()
        dlg.deleteLater()

    def _error_report_text(self) -> str:
        """Help → Error Report: the recorded errors (crashlog.report_text) and the
        System Check, one block of text for a bug report."""
        from kinetrace import crashlog
        from kinetrace.device import cached_device, describe
        sysc = describe() if cached_device() is not None else \
            "(the hardware check has not finished yet: PyTorch is still loading)"
        return crashlog.report_text() + "\n\n---- System check ----\n" + sysc

    def _show_error_report(self):
        """Help → Error Report…: what went wrong, with Copy and Open the log
        folder (the files can be attached instead). Nothing is sent anywhere (I140)."""
        import sys
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices, QFont
        from kinetrace import crashlog
        text = self._error_report_text()
        dlg = QDialog(self)
        dlg.setWindowTitle("Error report")
        dlg.setMinimumSize(760, 480)
        lay = QVBoxLayout(dlg)
        note = QLabel("Nothing is sent anywhere. The report contains file and folder names from this computer "
                      "(which can include your user name and your video names): read it before you post it "
                      "somewhere public.")        # (I158)
        note.setWordWrap(True)
        lay.addWidget(note)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setLineWrapMode(QPlainTextEdit.NoWrap)
        view.setFont(QFont("Consolas" if sys.platform.startswith("win") else "Menlo", 10)
                     if sys.platform != "linux" else QFont("Monospace", 10))
        lay.addWidget(view)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        copy = btns.addButton("Copy", QDialogButtonBox.ActionRole)
        copy.clicked.connect(lambda: QApplication.clipboard().setText(text))
        show = btns.addButton("Open the log folder", QDialogButtonBox.ActionRole)

        def _open():
            d = crashlog.log_path().parent
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

        show.clicked.connect(_open)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec()
        dlg.deleteLater()

    def _show_folders(self):
        """Help → Kinetrace's Folders…: `paths.places()` with sizes (measured in the background:
        .venv holds tens of thousands of files) and Show for the selected one (Mac install audit P1)."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QListWidget, QListWidgetItem
        from kinetrace import paths
        places = paths.places()
        sizes = self._in_background("Measuring Kinetrace's folders",
                                    lambda: [paths.size_bytes(p.path) if "registry" not in p.note else None
                                             for p in places],
                                    detail="Adding up the environment and the models…")
        dlg = QDialog(self)
        dlg.setWindowTitle("Kinetrace's folders")
        dlg.setMinimumSize(760, 420)
        lay = QVBoxLayout(dlg)
        note = QLabel("Deleting the Kinetrace folder removes everything listed inside it. Your projects, "
                      "exports and calibration files are saved where you chose them and are never deleted "
                      "by an update or by uninstalling.")
        note.setWordWrap(True)
        lay.addWidget(note)
        lst = QListWidget()
        for inside in (True, False):
            group = [(p, n) for p, n in zip(places, sizes) if p.inside == inside]
            head = QListWidgetItem("Inside the Kinetrace folder" if inside else
                                   "Outside the Kinetrace folder" + ("" if group else ": nothing"))
            head.setFlags(Qt.NoItemFlags)
            lst.addItem(head)
            for p, n in group:
                size = "" if p.what.startswith("Kinetrace folder") or "registry" in p.note \
                    else f"   [{paths.human(n)}]"
                it = QListWidgetItem(f"   {p.what}: {p.path}{size}" + (f"   ({p.note})" if p.note else ""))
                it.setData(Qt.UserRole, str(p.path) if "registry" not in p.note else "")
                lst.addItem(it)
        lay.addWidget(lst)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        show = btns.addButton("Show in " + ("Finder" if sys.platform == "darwin" else "the file manager"),
                              QDialogButtonBox.ActionRole)
        copy = btns.addButton("Copy", QDialogButtonBox.ActionRole)

        def _show():
            it = lst.currentItem()
            target = Path(it.data(Qt.UserRole)) if it is not None and it.data(Qt.UserRole) else None
            if target is None:
                self.statusBar().showMessage("Select a folder in the list first.", 5000)
                return
            while not target.exists() and target != target.parent:     # a file / folder not made yet
                target = target.parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target if target.is_dir() else target.parent)))

        show.clicked.connect(_show)
        lst.itemDoubleClicked.connect(lambda _it: _show())
        copy.clicked.connect(lambda: QApplication.clipboard().setText(
            "\n".join(lst.item(i).text().strip() for i in range(lst.count()))))
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        dlg.exec()
        dlg.deleteLater()

    def _show_about(self):
        """Help → About Kinetrace (G36)."""
        from kinetrace import updatedialog
        updatedialog.show_about(self, on_check=self._check_updates)

    def _update_busy(self) -> str | None:
        """Why an update must wait, or None (G37)."""
        if self.state == TRACKING:
            return "Stop tracking first (X), then press Update now."
        if self._loading:
            return "Wait until the videos have finished opening, then press Update now."
        return None

    def _check_updates(self):
        """Help → Check for Updates… (G37): look, install, restart."""
        from kinetrace import updatedialog
        dlg = updatedialog.UpdateDialog(self, busy=self._update_busy, restart=self._restart_after_update)
        dlg.exec()
        dlg.deleteLater()

    def _restart_after_update(self):
        """Close the usual way (Save / Discard / Cancel for unsaved work), then
        start Kinetrace again through its launcher; Cancel keeps it running."""
        from kinetrace import update
        if self.close():
            update.relaunch()
        else:
            self.statusBar().showMessage("The new version starts the next time you open Kinetrace", 8000)

    def _error_context(self) -> str:
        """One line of where the program was, for an error-log entry. It can be
        called on a worker thread, so plain attributes only -- no Qt calls."""
        p = self.project
        parts = [{IDLE: "no video open", READY: "ready", TRACKING: "tracking"}.get(self.state, str(self.state)),
                 f"frame {self.current}"]
        if p is not None:
            parts.append(f"{p.n_views} camera(s), working in camera {p.active + 1}")
        parts.append(f"point model {self._point_backend}")
        return ", ".join(parts)

    def _on_error_logged(self, text: str) -> None:
        """The error log's on-screen notice (at most one every few seconds;
        the log keeps every error)."""
        self.toast.show_message(text, "error", 12000)

    def _show_manual(self, section: str | None = None):
        """Help → User Manual (F1). Non-modal and reused, so it can stay open
        beside the video while you follow it. `section` = a heading to open at."""
        dlg = getattr(self, "_manual_dlg", None)
        if dlg is None:
            dlg = ManualDialog(self)
            self._manual_dlg = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        if section:
            QTimer.singleShot(0, lambda: dlg.go_to_heading(section))

    def _show_hotkeys(self):
        dlg = getattr(self, "_hotkeys_dlg", None)
        if dlg is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Keyboard & mouse reference")
            lay = QVBoxLayout(dlg)
            tb = QTextBrowser()
            tb.setHtml(native_keys(HOTKEYS_HTML))         # ⌘ / ⌥ on a Mac
            tb.setOpenExternalLinks(False)
            lay.addWidget(tb)
            dlg.resize(620, 680)
            self._hotkeys_dlg = dlg
        dlg.show()          # non-modal: keep it open next to the video
        dlg.raise_()
        dlg.activateWindow()

    # ------------------------------------------------------- the loading card

    @property
    def _loading(self) -> bool:
        """An open is under way that input must not disturb (the project is being
        built). The last phase -- waiting for the first picture -- is passive: the
        card still says so, but clicks and keys go through."""
        return any(not b.get("passive") for b in (getattr(self, "_busy_stack", None) or []))

    def _busy_push(self, title: str | None = None, detail: str | None = None, total: int | None = None,
                   hint: str | None = None, on_cancel=None, immediate: bool = False) -> int:
        """One level of 'please wait': the loading card shows while any level is
        open (nested waits -- a project, then its cameras, then the first picture --
        share one card). Returns a token for `_busy_pop`. The menus are off and the
        hotkeys swallowed meanwhile; Cancel calls the innermost `on_cancel`."""
        self._busy_next += 1
        tok = self._busy_next
        first = not self._busy_stack
        self._busy_stack.append({"tok": tok, "cancel": on_cancel})
        if first:
            self._set_input_blocked(True)
            self.overlay.start(title or "Please wait", detail or "", total or 0, hint or "",
                               on_cancel, immediate)
        else:
            if title is not None:
                self.overlay.title.setText(title)
            self.overlay.step(detail, 0 if total is not None else None, total)
            if hint is not None:
                self.overlay.set_hint(hint)
            self._busy_refresh_cancel()
            self._busy_refresh_mode()        # a new blocking level over a passive one blocks again
        return tok

    def _busy_step(self, detail: str | None = None, value: int | None = None, total: int | None = None) -> None:
        if self._loading:
            self.overlay.step(detail, value, total)

    def _busy_pop(self, tok: int | None) -> None:
        """Close one level (any order); the card goes when none is left."""
        if tok is None:
            return
        self._busy_stack = [b for b in self._busy_stack if b["tok"] != tok]
        if not self._busy_stack:
            self.overlay.finish()
            self._set_input_blocked(False)
        else:
            self._busy_refresh_cancel()
            self._busy_refresh_mode()

    def _busy_set_passive(self, tok: int) -> None:
        """Level `tok` no longer holds input back (the project is built)."""
        for b in self._busy_stack:
            if b["tok"] == tok:
                b["passive"] = True
        self._busy_refresh_mode()

    def _busy_refresh_mode(self) -> None:
        blocking = self._loading
        self._set_input_blocked(blocking)
        self.overlay.set_passive(bool(self._busy_stack) and not blocking)

    def _set_input_blocked(self, blocking: bool) -> None:
        """The menus and the window's own shortcuts (arrows, Home / End, Ctrl+,) are off
        while the loading card holds input back (G71); the keys and clicks in the app's
        other windows are held back by `eventFilter`."""
        self.menuBar().setEnabled(not blocking)
        for sc in getattr(self, "_nav_shortcuts", ()):
            sc.setEnabled(not blocking)
        self.act_settings.setEnabled(not blocking)

    def _busy_cancel_cb(self):
        return next((b["cancel"] for b in reversed(self._busy_stack) if b["cancel"] is not None), None)

    def _busy_refresh_cancel(self) -> None:
        self.overlay.set_cancel(self._busy_cancel_cb())

    def _busy_set_cancel(self, tok: int, on_cancel) -> None:
        for b in self._busy_stack:
            if b["tok"] == tok:
                b["cancel"] = on_cancel
        self._busy_refresh_cancel()

    def _first_frame_arrived(self) -> None:
        """The working camera's first picture is on screen (or it took too long):
        the open is over."""
        tok, self._first_frame_token = self._first_frame_token, None
        self._first_frame_timer.stop()
        self._busy_pop(tok)

    def _open_hint(self, path: str, what: str = "video") -> str:
        """Something useful to read while waiting."""
        if on_network_drive(path):
            return ("This file is on a network drive: opening and scrubbing are several times faster from a copy "
                    "on this computer's own disk.")
        if what == "project":
            return "Every camera, point and track, and where you were, come back exactly as you left them."
        if what == "cameras":
            return ("Next: line the cameras up in time with 3D → Sync Cameras (Sound / Motion); every camera "
                    "shares one list of points.")
        return ("Next: press N and click what you want to track (or S to outline the animal). F1 opens the "
                "manual, written for first-time users.")

    @staticmethod
    def _probe_message(stage: str, facts: dict, prefix: str = "") -> str:
        """probe_video's stage in plain words."""
        if stage == "open":
            return prefix + "Opening the file…"
        w, h, n = facts.get("width", 0), facts.get("height", 0), facts.get("frames", 0)
        fps = facts.get("fps", 0.0) or 0.0
        big = max(w, h) >= 3000
        if stage == "rate":
            return (prefix + f"{w} × {h}" + (f", {fps:.4g} fps" if fps else "") + f", {n:,} frames — "
                    "reading the first frames to check the frame rate…")
        if stage == "frames":
            return (prefix + f"Checking that all {n:,} frames can be read"
                    + (" (a 4K file takes a few seconds)" if big else "") + "…")
        return prefix + "Reading…"

    def _in_background(self, title: str, fn, *, detail: str = "", hint: str = "", total: int = 0,
                       cancellable: bool = False, progress: bool = False):
        """Run `fn` on a worker thread with the loading card up and the window
        repainting (a local event loop; input held back by the card) -> its result,
        or its exception raised here. G46-G52: saving, opening, exports, 3D and
        imports used to run on the GUI thread, a frozen window with a wait cursor.
        `progress` passes report(detail, value, total); `cancellable` adds Cancel to
        the card and passes cancelled() -- the work stops at its next check."""
        job = None

        def cancel():
            if job is not None:
                job.cancel_requested = True
                self._busy_step("Stopping after the current step…")

        kw = {}
        if progress:
            kw["report"] = lambda d="", v=0, n=0: job.step.emit(str(d), int(v), int(n))
        if cancellable:
            kw["cancelled"] = lambda: job.cancel_requested
        job = _Job(fn, kw)
        loop = QEventLoop()
        job.finished.connect(loop.quit)
        job.step.connect(lambda d, v, n: self._busy_step(d or None, v if n else None, n or None))
        tok = self._busy_push(title, detail, total or None, hint, on_cancel=cancel if cancellable else None)
        try:
            job.start()
            if not job.isFinished():
                loop.exec()
            job.wait()
        finally:
            self._busy_pop(tok)
        if job.exc is not None:
            raise job.exc
        return job.result

    def _probe_many(self, paths: list[str], title: str, hint: str | None = None) -> dict:
        """Probe several videos off the GUI thread, `PROBE_PARALLEL` at a time,
        with the loading card counting them; the window keeps repainting (a
        local event loop, input blocked by the card). Returns {path: VideoInfo |
        error text | "cancelled"}. Before, each probe ran ON the GUI thread: the
        window froze ~2.3 s per 4K camera (measured on a real clip)."""
        paths = list(dict.fromkeys(str(x) for x in paths))
        out: dict = {}
        if not paths:
            return out
        loop = QEventLoop()
        queue = list(paths)
        running: dict = {}
        state = {"cancelled": False}

        def say() -> None:
            waiting = ", ".join(Path(x).name for x in running)
            self._busy_step(f"{len(out)} of {len(paths)} ready" + (f" — reading {waiting}" if waiting else ""),
                            len(out), len(paths))

        def launch() -> None:
            while queue and len(running) < PROBE_PARALLEL and not state["cancelled"]:
                path = queue.pop(0)
                th = _VideoProbe(path)
                th.done.connect(lambda r, p=path: finished(p, r))
                running[path] = th
                th.start()
            say()

        def finished(path: str, result) -> None:
            th = running.pop(path, None)
            if th is not None:
                th.wait(2000)
            if state["cancelled"]:
                return
            out[path] = result
            if len(out) == len(paths):
                loop.quit()
            else:
                launch()

        def cancel() -> None:
            state["cancelled"] = True
            loop.quit()

        tok = self._busy_push(title, "", len(paths), hint, on_cancel=cancel)
        launch()
        if len(out) < len(paths):
            loop.exec()
        for th in running.values():
            _retire(th)                  # still reading: kept alive until it ends (never destroyed running)
        for path in paths:
            out.setdefault(path, "cancelled")
        self._busy_pop(tok)
        return out

    # ------------------------------------------------------------ open video

    def _start_folder(self, fallback="") -> str:
        """Where a file dialog opens that belongs to the open project: its folder (the place its
        videos and its files are found from), else `fallback` (R12). The other dialogs keep their
        own rule: the project's name for a save, the video's folder for masks, the camera's name for
        an export (`_default_output`)."""
        return str(self._project_dir or fallback)

    def _open_video_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", VIDEO_FILTER)
        if not path:
            return
        settled = self._settle_unsaved(f"opening {Path(path).name}")     # (G143)
        if settled is not None:
            self._open_video(path, discard=settled == "discard")

    def _settle_unsaved(self, doing: str) -> str | None:
        """(G143) Before another video / project REPLACES this one: the close question (Save /
        Discard / Cancel) when there is unsaved work. Returns None = Cancel (or a Save that did not
        happen): open nothing; "discard" = the opener drops the unsaved work once the new one is
        really opening (`_leave_project(discard=True)`); "keep" = nothing unsaved, saved, or any
        other answer (the unsaved work goes to the recovery folder, as before)."""
        if self.project is None or not self.project.dirty:
            return "keep"
        answer = self._ask_save_before_close(doing)
        if answer == QMessageBox.Cancel:
            return None
        if answer == QMessageBox.Save:
            return "keep" if self._save_project() else None
        return "discard" if answer == QMessageBox.Discard else "keep"

    def _open_folder_dialog(self):
        """File → Open Folder of Videos… (G30): the folder's videos listed, the user
        ticks which to import and picks the base (reference) camera."""
        from kinetrace.folderimport import VideoFolderDialog
        if self.state == TRACKING:
            return
        start = self._start_folder(Path(self.info.path).parent if self.info else Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Open a folder of videos", start)
        if not folder:
            return
        dlg = VideoFolderDialog(self, folder)
        accepted = dlg.exec() == QDialog.Accepted
        dlg.stop_probe()              # however the dialog ended: never delete a running QThread
        paths, save_to = list(dlg.result_paths), dlg.result_project
        dlg.deleteLater()
        if accepted and paths:
            settled = self._settle_unsaved(f"opening the videos of {Path(folder).name}")     # (G143)
            if settled is not None:
                self._import_folder(paths, save_to, discard=settled == "discard")

    def _import_folder(self, paths: list[str], save_to: str | None = None, discard: bool = False) -> None:
        """Open `paths[0]` (the base camera) as a new project, add the others, name
        every camera after its file, and save to `save_to` when given."""
        if not paths or self.state == TRACKING:
            return
        base = paths[0]
        folder = Path(base).parent.name
        n = len(paths)
        if self._first_frame_token is not None:      # a previous open still waiting for its picture (G109)
            self._first_frame_arrived()
        # every camera is read at once (PROBE_PARALLEL at a time), the base included,
        # with the loading card counting them -- then the project is built
        tok = self._busy_push(f"Opening {n} camera{'s' if n != 1 else ''} from {folder}", "", n,
                              self._open_hint(base, "cameras"))
        infos = self._probe_many(paths, f"Opening {n} camera{'s' if n != 1 else ''} from {folder}")
        binfo = infos.get(str(base))
        if isinstance(binfo, str):
            self._busy_pop(tok)
            if binfo != "cancelled":
                QMessageBox.critical(self, "Could not open the base camera", f"{Path(base).name}:\n\n{binfo}")
            else:
                self.statusBar().showMessage("Opening the folder cancelled — nothing was changed", 5000)
            return
        self._busy_set_cancel(tok, None)
        self._busy_step("Building the project: the timeline, every camera's view and the first pictures…")
        try:
            self._attach_video(binfo, discard=discard)
            p = self.project
            names = camera_names(paths)                 # unique, even for cam1/GX01.MP4 ... cam8/GX01.MP4 (I223)
            p.names[0] = names[0] or p.names[0]
            rest = paths[1:]
            added = [x for k, x in enumerate(rest, 1) if self._add_view(x, infos.get(str(x)), name=names[k])]
            self._refresh_cameras()
            self._refresh_companions()
            saved = ""
            if save_to:
                # (I166) never straight over a project that is already there, never inside one:
                # the same question as Save As; No = the cameras stay open, unsaved
                target = self._project_target(save_to)
                if target is None:
                    saved = " Not saved yet: use File → Save Project As… to choose where."
                else:
                    self._busy_step(f"Saving the project as {target.name}…")
                    self.project_path = target
                    self._project_dir = self.project_path.resolve()     # a project folder (I145)
                    if self._save_project():
                        saved = f" Saved as {self.project_path.name}."
        except Exception:
            self._busy_pop(tok)
            raise
        if self.canvas._raw_rgb is None:
            self._first_frame_token = tok
            self._busy_step("Showing the first picture…")
            self._busy_set_passive(tok)
            self._first_frame_timer.start()
        else:
            self._busy_pop(tok)
        missed = len(rest) - len(added)
        self.toast.show_message(
            f"Opened {1 + len(added)} camera(s) from {folder}; the base (reference) camera is "
            f"<b>{p.name(0)}</b>." + (f" {missed} could not be added." if missed else "") + saved
            + " Next: line them up in time with <b>3D → Sync Cameras (Sound / Motion)</b>.", "info", 12000)

    def _open_video(self, path: str, then=None, title: str | None = None, hint: str | None = None,
                    discard: bool = False):
        """Probe in a worker thread, then wire everything up. `then(info)` runs
        after success (used by project loading). The loading card says each step,
        and stays until the first picture is on screen; Cancel (while the file is
        still being read) leaves whatever is open untouched. `discard`: the user
        chose Discard for the open project's unsaved work (G143) -- dropped only
        once the new video really replaces it."""
        name = Path(path).name
        try:
            size = f" ({Path(path).stat().st_size / 1024 ** 2:,.0f} MB)"
        except OSError:
            size = ""
        self.statusBar().showMessage(f"Opening {name}…")
        probe = _VideoProbe(path)
        prev = self._probe
        self._probe = probe  # latest request wins; older probes are ignored below
        if prev is not None:
            _retire(prev)
        if self._first_frame_token is not None:      # a previous open still waiting for its picture
            self._first_frame_arrived()

        def cancel():
            if self._probe is probe:
                self._probe = None           # its result is dropped when it comes
                _retire(probe)               # and the thread is kept alive until it ends
            self._after_open = None          # a follow-up waiting for this video (an Import Tracks) must not run later
            self._busy_pop(tok)
            self.statusBar().showMessage(f"Opening {name} cancelled — nothing was changed", 5000)

        tok = self._busy_push(title or f"Opening {name}", f"Opening the file{size}…", OPEN_STAGES,
                              hint if hint is not None else self._open_hint(path), on_cancel=cancel)
        # the bar counts the stages: open, frame rate, frame count, the timeline, the first picture
        probe.step.connect(lambda st, facts: self._probe is probe and self._busy_step(
            self._probe_message(st, facts, prefix=f"{name}: " if title else ""),
            {"open": 1, "rate": 2, "frames": 3}.get(st)))

        def done(result):
            probe.wait(1000)
            if self._probe is not probe:
                return  # a newer open superseded this one (or it was cancelled)
            self.statusBar().clearMessage()
            if isinstance(result, str):
                self._busy_pop(tok)
                QMessageBox.critical(self, "Could not open video", result)
                return
            self._attach_with_card(result, then, tok, discard)

        probe.done.connect(done)
        probe.start()

    def _attach_with_card(self, info, then, tok: int, discard: bool = False) -> None:
        """Build the project on `info` with the loading card's level `tok` up: past
        this point the open cannot be taken back (no Cancel); the level goes when
        the working camera's first picture is on screen."""
        self._busy_set_cancel(tok, None)
        self._busy_step("Preparing the timeline and the first picture…", OPEN_STAGES - 1, OPEN_STAGES)
        try:
            self._attach_video(info, then, discard=discard)
        except Exception:
            self._busy_pop(tok)
            raise
        if self.state == READY and self.canvas._raw_rgb is None:
            self._first_frame_token = tok            # popped when the picture is on screen
            self._busy_step("Showing the first picture…", OPEN_STAGES, OPEN_STAGES)
            self._busy_set_passive(tok)              # built: clicks and keys go through meanwhile
            self._first_frame_timer.start()
        else:
            self._busy_pop(tok)

    def _attach_video(self, info: VideoInfo, then=None, discard: bool = False):
        """Open `info` as a brand-new single-camera project (extra cameras are
        added afterwards with `_add_video_dialog`). `discard`: drop the replaced
        project's unsaved work instead of keeping it in recovery (G143)."""
        # Track ▾ is usable with no video open (G34): what was chosen there carries
        # into this first video instead of snapping back to the defaults
        chosen = None if self.project is not None else {
            "track_mode": self._track_mode, "track_all": self.act_track_all.isChecked(),
            "point_backend": self._point_backend}
        self._leave_project(discard)   # unsaved work -> recovery (or dropped, G143), where the user was -> view sidecar
        self._teardown_video()
        self._gopro_said = set()                 # (G145) a new project: its GoPro notes are new
        self._gopro_soon()
        self._epi_probe = None
        self._wand_result = (None, None)      # a new project: no wand run belongs to it (I33)
        self.project = Project([TrackingSession(info.path, info.n_frames, info.fps,
                                                info.width, info.height)])
        if chosen is not None:
            self.session.ui_state.update(chosen)   # a recovered / opened copy replaces it with its own
        self._views = [_ViewRuntime(info, DEFAULT_CACHE_BYTES)]
        self.grid.set_count(1)
        self.grid.set_active(0)
        for cv in self.grid.canvases:
            cv.set_stale(False)            # an Active-view-only veil must not carry into this video (G134)
        self._start_seek_service()

        self.canvas.set_video_size(info.width, info.height)
        self.spin.setRange(0, info.n_frames - 1)
        self.spin.setSuffix(f" / {info.n_frames - 1}")
        self._set_play_interval(info.fps)

        if then is None:
            self.project_path = None
            self._keep_single_file = False
            self._project_id, self._saved_at, self._recovery_sig = projectfile.new_id(), None, None
            self._camera_entries, self._project_dir, self._video_dirs = [], None, []
            earlier = recovery.for_video(info.path)
            if earlier is not None:
                self._offer_video_recovery(earlier)
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
        job, self._after_open = self._after_open, None
        if job is not None and Path(job[0]).resolve() == Path(info.path).resolve():
            QTimer.singleShot(0, job[1])
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
        rt.seek.decode_failed.connect(lambda f, msg, r=rt: self._on_decode_failed(self._view_index(r), f, msg))
        rt.seek.start()

    def _set_play_interval(self, fps: float) -> None:
        """Preview playback runs at the video's rate (at most 100 frames a second)."""
        self._play_timer.setInterval(max(10, round(1000 / max(float(fps), 1e-6))))

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
            rt.want_frame = None              # the other cameras ask for their picture again (G133)
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
        if not paths:
            return
        # (G172) the cameras' order is the user's, not the file dialog's (whose order is not even the
        # order the files were picked in): the project's cameras and the new ones, the new ones last
        from kinetrace.cameraorder import CameraOrderDialog
        from kinetrace.folderimport import _natural_key
        p = self.project
        paths = sorted(paths, key=lambda x: _natural_key(Path(x)))
        room = MAX_VIEWS - p.n_views
        if len(paths) > room:
            QMessageBox.information(self, "Too many cameras",
                                    f"A project holds up to {MAX_VIEWS} cameras: the first {room} of these are "
                                    "added.")
            paths = paths[:room]
        entries = ([(("cam", i), p.name(i), "") for i in range(p.n_views)]
                   + [(("new", x), Path(x).name, "new") for x in paths])

        def problem(keys):
            # a calibration of the first cameras must keep them first (Project.order_problem)
            cal = p.calibration
            k = len(cal) if cal is not None else 0
            if 0 < k <= p.n_views and sorted(key[1] for key in keys[:k] if key[0] == "cam") != list(range(k)):
                return (f"the calibration covers cameras 1-{k} only, so they must stay the first {k} cameras; "
                        "put the new videos after them")
            return None
        dlg = CameraOrderDialog(self, entries, f"Add {len(paths)} camera{'s' if len(paths) != 1 else ''}: their order",
                                consequence="The new cameras are marked “new”. Each camera keeps its points and offset; "
                                            "if another camera becomes camera 1, the offsets are re-measured from it.",
                                problem=problem)
        ok = dlg.exec() == QDialog.Accepted
        keys = list(dlg.order)
        dlg.deleteLater()
        if not ok:
            return
        new_paths = [key[1] for key in keys if key[0] == "new"]
        infos = self._probe_many(new_paths, f"Adding {len(new_paths)} camera{'s' if len(new_paths) != 1 else ''}",
                                 self._open_hint(new_paths[0], "cameras"))
        n0 = p.n_views
        added = [x for x in new_paths if self._add_view(x, infos.get(str(x)))]
        if not added:
            return
        index_of = {("cam", i): i for i in range(n0)}
        index_of.update({("new", x): n0 + k for k, x in enumerate(added)})
        final = [index_of[key] for key in keys if key in index_of]      # a file that could not be added is left out
        if final != list(range(p.n_views)):
            self._reorder_cameras(final)
        self._refresh_cameras()
        self._refresh_companions()
        self.toast.show_message(
            f"Added {len(added)} camera(s) — the order: "
            + ", ".join(f"{k + 1} {p.name(k)}" for k in range(p.n_views))
            + ". Line them up in time: <b>3D → Sync Cameras (Sound / "
            "Motion)</b> does it from the sound tracks or the pictures, or scrub to a shared event and nudge each "
            "camera's <b>offset</b> (or press <b>Align here</b>).",
            "info", 12000)

    def _add_view(self, path: str, probed=None, name: str | None = None) -> bool:
        """Probe `path` and append it as a view. `probed`: its VideoInfo (or error
        text / "cancelled") from an earlier `_probe_many` over several files.
        `name`: the camera's name (a folder import has worked the set out together);
        otherwise the file's name, made different from every other camera's (I223).
        Returns once the camera is in (the probe runs off the GUI thread with the
        loading card up; it used to freeze the window ~2.3 s per 4K file)."""
        if self.project is None or self.project.n_views >= MAX_VIEWS:
            return False
        if any(rt.info.path == path for rt in self._views):
            self.statusBar().showMessage(f"{Path(path).name} is already in this project", 5000)
            return False
        info = probed if probed is not None else \
            self._probe_many([path], f"Adding {Path(path).name}", self._open_hint(path, "cameras"))[str(path)]
        if isinstance(info, str):
            if info != "cancelled":
                QMessageBox.critical(self, "Could not open video", info)
            return False
        s = TrackingSession(info.path, info.n_frames, info.fps, info.width, info.height)
        # (G153) the animals (with their skeletons) and the landmarks of the other cameras arrive by
        # name in `sync_landmarks` below: a deleted landmark is not brought back (I234)
        if name is None or name.lower() in {str(n).lower() for n in self.project.names}:
            name = camera_names([path], taken=self.project.names)[0]      # (I223)
        had_3d, had_hull = self.project.reconstruction is not None, bool(self._hull_cache)
        i = self.project.add_view(s, name or None)
        added = self.project.sync_landmarks()      # every point already made is waiting to be placed here (G19)
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
        if added:
            self._refresh_point_list()             # the working camera may have gained landmarks too (I234)
        self._drop_reconstruction("a camera was added", had=had_3d, had_hull=had_hull)    # (I206) the camera set changed
        if getattr(info, "fps_note", ""):
            self.toast.show_message(f"{self.project.name(i)}: {info.fps_note}", "warn", 12000)
        self._gopro_soon()                       # (G145)
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
        self._epi_probe = None
        if self._undo_extra:
            # the undo step spans cameras by their index: after a removal the
            # indices shift, so that step can no longer be taken back safely (G19)
            self._undo_snap = None
            self.act_undo.setEnabled(False)
        had_3d, had_hull = p.reconstruction is not None, bool(self._hull_cache)
        add_on = self._leave_camera_tools()     # the same tool housekeeping as a view switch (G92)
        target = None
        if i == p.active:                 # the next camera takes over at the SAME instant (I111)
            nxt = i + 1 if i + 1 < p.n_views else i - 1
            target = p.map_frame(i, nxt, self.current)
        self._views[i].stop()
        del self._views[i]
        p.remove_view(i)
        self._rebudget_caches()
        self.grid.drop_view(i)             # (G170) the cameras after it move down one, on screen too
        self.grid.set_count(p.n_views)
        for k, rt in enumerate(self._views):          # tiles after i now show another camera
            cv = self.grid.canvas(k)
            if cv is not None:
                cv.set_video_size(rt.info.width, rt.info.height)
            rt.want_frame = None
        self.grid.set_active(p.active)
        self._apply_active_view(target if target is not None else p.session.current_frame)
        self._enter_camera_tools(add_on)
        self._drop_reconstruction("a camera was removed", had=had_3d, had_hull=had_hull)       # (I206)

    # ------------------------------------------------------- the cameras' order (G172)

    def _order_consequence(self) -> str:
        """What reordering an open project does, in words (the order dialog's footnote)."""
        p = self.project
        bits = ["Each camera keeps its points, tracks, offset, frame rate and lens"]
        if p is not None and p.calibration is not None and len(p.calibration):
            bits[0] += " and its calibration (no need to calibrate again)"
        text = bits[0] + ". If another camera becomes camera 1, the offsets are re-measured from it (the cameras stay " \
                         "in sync)."
        if p is not None and p.reconstruction is not None:
            text += " The 3D result is dropped: Reconstruct again."
        return text

    def _camera_order_dialog(self) -> None:
        """CAMERAS -> Camera order...: the open project's cameras in another order (G172)."""
        p = self.project
        if p is None or p.n_views < 2 or self.state != READY:
            return
        from kinetrace.cameraorder import CameraOrderDialog
        entries = [(i, p.name(i), Path(p.sessions[i].video_path).name) for i in range(p.n_views)]
        dlg = CameraOrderDialog(self, entries, "Camera order", consequence=self._order_consequence(),
                                problem=p.order_problem)
        ok = dlg.exec() == QDialog.Accepted
        order = list(dlg.order)
        dlg.deleteLater()
        if ok:
            self._reorder_cameras(order)

    def _reorder_cameras(self, order) -> bool:
        """Put the open project's cameras in `order` (`order[k]` = the camera that becomes camera k + 1),
        G172. Everything indexed by camera follows: the project's lists (`Project.reorder_views`), the
        decode runtimes, the tiles (each shows its camera's video at the new index) and the views'
        arrangement, so every camera stays where it was on screen. The working camera stays the
        working camera at the same frame; the 3D result is dropped (the project does it)."""
        p = self.project
        if p is None or self.state != READY:
            return False
        order = [int(k) for k in order]
        why = p.order_problem(order)
        if why:
            QMessageBox.warning(self, "Camera order", f"This order cannot be used: {why}.")
            return False
        if order == list(range(p.n_views)):
            return False
        had_3d, had_hull = p.reconstruction is not None, bool(self._hull_cache)
        add_on = self._leave_camera_tools()      # the same tool housekeeping as a view switch (G92)
        before = [p.name(i) for i in range(p.n_views)]
        p.reorder_views(order)
        self._views = [self._views[k] for k in order]
        self.grid.remap({old: new for new, old in enumerate(order)})
        self._epi_probe = None
        self._lock = None                        # a step's collected pictures are keyed by the old numbers
        self._wand_result = (None, None)         # the last wand run's report names the old numbers (I33)
        self._undo_snap = None                   # an undo step spans cameras by their number (G19)
        self.act_undo.setEnabled(False)
        for k, rt in enumerate(self._views):     # each tile now shows the camera at its index
            cv = self.grid.canvas(k)
            if cv is not None:
                cv.set_video_size(rt.info.width, rt.info.height)
            rt.want_frame = None
        self.grid.set_active(p.active)
        self.grid.set_arrangement()              # lay the tiles out again with the remapped order
        self._apply_active_view(self.current)    # the same camera, the same frame
        self._enter_camera_tools(add_on)
        self._refresh_companions()
        self._refresh_cameras()
        self._drop_reconstruction("the camera order changed", had=had_3d, had_hull=had_hull)
        now = [p.name(i) for i in range(p.n_views)]
        self.statusBar().showMessage(
            "Camera order: " + ", ".join(f"{k + 1} {n}" for k, n in enumerate(now))
            + (f" — {now[0]} is now the reference (camera 1)" if now[0] != before[0] else ""), 12000)
        return True

    def _set_active_view(self, i: int):
        """Switch the camera being worked on. The playhead follows through the
        offsets, so the picture stays on the same instant."""
        p = self.project
        if p is None or self.state != READY or not (0 <= i < p.n_views) or i == p.active:
            return
        add_on = self._leave_camera_tools()
        # the camera being left keeps its exact state; the tool / display toggles
        # travel with the user: re-applying the other camera's stored copy silently
        # turned auto-pause and ROI back on and swapped the point model (I50)
        self._sync_ui_state()
        carried = {k: self.session.ui_state[k] for k in GLOBAL_UI_KEYS if k in self.session.ui_state}
        # the selected LANDMARK comes along (the cameras share one list, G19): pick
        # a point in one camera, click the next camera, and place it there
        s0 = self.session
        keep = (s0.points[self.selected].name
                if self.selected is not None and self.selected < s0.n_points else None)
        # the WHOLE selection comes along too: it is what Track tracks (G61)
        keep_all = {s0.points[q].name for q in self._selected_pids() if q < s0.n_points}
        keep_seg = [s0.segments[k].name for k in self._selected_segments() if k < s0.n_segments]   # (G149) by name
        prev = p.name(p.active)
        # the camera being left shows ITS frame at this instant; the one taken up is
        # driven by the playhead from now on (Active view only keeps the first where
        # it is and knows it is current, G27)
        self._views[p.active].want_frame = self.current
        self._views[i].want_frame = None
        target = p.set_active(i)
        p.session.ui_state.update(carried)
        self.grid.set_active(i)
        self._apply_active_view(target if target is not None else p.session.current_frame)
        self._update_disagreement()            # the band is per camera: recompute for this one
        j = p.session.pid_by_name(keep) if keep is not None else None
        # selects it (and shows its epipolar guides) and the whole selection, by name; ClearAndSelect
        # (G66)
        sv = p.session
        self.layers.select_only([q for q in (sv.pid_by_name(nm) for nm in keep_all | ({keep} if keep else set()))
                                 if q is not None],
                                [k for k, n in enumerate(sv.segment_names()) if n in keep_seg], current=j)
        if j is not None:
            self.selected = j
        self._sync_s_target()
        self._update_track_button()
        self._enter_camera_tools(add_on)
        msg = f"Working in {p.name(i)} — points, silhouette and timeline are this camera's"
        s = p.session
        if j is not None and not s.tracked[self.current, j] and not s.points[j].derived:
            on_line = (" (the ◇ is where the other cameras put it)" if self.canvas.prediction_count() else
                       f" on the dashed line from {prev}" if self.canvas.guide_count() else "")
            msg = (f"Working in {p.name(i)}: {keep} is not placed in this camera on this frame — "
                   f"click it on the video{on_line}, then Track")
        self.statusBar().showMessage(msg, 9000)

    def _leave_camera_tools(self) -> bool:
        """The tool housekeeping of leaving the working camera, shared by a view
        switch and a camera's removal (G92: removing skipped it, so Add stayed armed on
        another canvas and the next click overwrote the selected point, and a
        half-marked event was finished with the removed camera's frame). Returns
        whether Add was armed, for `_enter_camera_tools`."""
        self.canvas.cancel_gesture()
        # The tools arm ONE canvas; left armed across a switch, the new view took
        # plain clicks as hand placements while S / N still looked on (I47), and a
        # half-marked event was finished with the other camera's frame number (I65).
        # Add is carried OVER instead (G20): armed in one camera, a click in another
        # places the point there -- disarming it swallowed that click, and the next,
        # unarmed click drew the look-here cross instead of a point. The segment
        # tool still goes down: its clicks prompt one camera's SAM session.
        add_on = self.btn_add.isChecked()
        if add_on:
            self.canvas.set_place_mode(False)       # the camera being left
            self.canvas.set_click_only(False)
        if self.btn_animal.isChecked():
            self.btn_animal.setChecked(False)
        if self._pending_event is not None:
            self._pending_event = None
            self.timeline.set_pending_event(None)
        return add_on

    def _enter_camera_tools(self, add_on: bool) -> None:
        """Arm Add on the camera taken up, when it was armed on the one left (G20)."""
        if add_on and self.btn_add.isChecked():
            self.canvas.set_place_mode(True)
            self.canvas.set_click_only(self._place_kind == "ball")

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
        self._set_play_interval(info.fps)
        for k, cv in enumerate(self.grid.canvases):
            cv.set_interactive(k == p.active and self.state == READY)
            cv.set_switchable(k != p.active and self.state == READY)
            cv.set_stale(False)
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
        if not on and not (self.act_sync_all.isChecked() or self.act_sync_active.isChecked()):
            # unticked directly (Ctrl+2 again): back to how the cameras were followed
            self._set_sync_mode(self._sync_mode)
        self._refresh_companions()
        self._refresh_guides()
        self._refresh_cameras()                  # the eyes rest while only the working camera is shown (G169)

    def _on_view_offset(self, i: int, offset: float):
        if self.project is None:
            return
        had_3d, before = self.project.reconstruction is not None, list(self.project.offsets)
        self.project.set_offset(i, float(offset))
        if list(self.project.offsets) != before:        # an unchanged value drops nothing (I206)
            self._after_retime(had_3d)
        self._refresh_companions()
        self._refresh_cameras()

    def _after_retime(self, had_3d: bool, had_hull: bool | None = None) -> None:
        """A changed offset / rate makes the triangulation stale (I23): the
        project drops it; here the volumes, the band and the menus follow."""
        if had_hull is None:
            had_hull = bool(self._hull_cache)
        # (I206) the carved volumes are keyed by reference instant and made with the old
        # timing: dropped on EVERY retime, with or without a 3D result
        self._drop_reconstruction("camera timing changed",
                                  had=had_3d and self.project is not None and self.project.reconstruction is None,
                                  had_hull=had_hull)

    def _align_view_here(self, i: int):
        """Take what camera `i` is showing right now as the match for the active
        camera's current frame (the flash / clap alignment step)."""
        p = self.project
        if p is None or not (0 <= i < p.n_views) or i == p.active:
            return
        # on the reference's row the WORKING camera is the one retimed (G13)
        ref = i == REFERENCE_VIEW
        shown = self._views[i].want_frame
        if shown is None:
            self.statusBar().showMessage(
                f"{p.name(i)} has no frame at this instant — nudge "
                f"{p.name(p.active) + chr(39) + 's' if ref else 'its'} offset first", 5000)
            return
        had_3d, before = p.reconstruction is not None, list(p.offsets)
        off = p.align_to(i, shown, self.current)
        if list(p.offsets) != before:
            self._after_retime(had_3d)
        self._refresh_companions()
        self._refresh_cameras()
        self.statusBar().showMessage(
            f"{p.name(p.active)} aligned to the reference {p.name(i)}: offset {off:+g} frames"
            if ref else
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
        self.m_others.setEnabled(p.n_views > 1)      # View → Other cameras (G24)
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
        hidden = self.grid.hidden()
        self.cameras.update_rows(list(p.names), list(p.offsets), p.active, statuses, note,
                                 rates=list(p.rates), fps=[s.fps for s in p.sessions],
                                 file_fps=[getattr(s, "file_fps", s.fps) for s in p.sessions],
                                 shown=[i not in hidden for i in range(p.n_views)],
                                 solo=self.act_solo.isChecked())
        self.act_show_all_views.setEnabled(bool(hidden - {p.active}))
        self.act_views_in_order.setEnabled(self.grid.display_order() != list(range(p.n_views)))

    # ------------------------------------------- which views are on screen, where (G169, G170)

    def _set_view_shown(self, i: int, shown: bool) -> None:
        """A camera's eye in CAMERAS: its view on screen or not. Display state only (never unsaved);
        a hidden view is not decoded and the others get its room."""
        p = self.project
        if p is None or not (0 <= i < p.n_views):
            return
        if not shown and i == p.active:
            self._refresh_cameras()            # the working camera is always shown: the eye goes back
            return
        self.grid.set_hidden(i, not shown)     # -> _on_views_arranged

    def _landmarks_elsewhere(self):
        """(G173) Where the OTHER cameras have each of the working camera's landmarks tracked at the same
        instant: (key, (T, N) bool) for the timeline -- its lanes are the working camera's, and a point
        tracked only in another view showed an empty lane (owner 2026-10-07: "the point P1 is tracked in
        the top right frame but it is not showing up in the timeline of P1"). Frames are matched through
        the offsets and rates (`Project.local_index`). Cached until a camera's data, its timing or the
        working camera changes; (None, None) with one camera."""
        p, s = self.project, self.session
        if p is None or s is None or p.n_views < 2 or s.n_points == 0:
            return None, None
        key = (id(p), p.active, tuple(x.data_version for x in p.sessions), tuple(p.offsets), tuple(p.rates),
               tuple(m.name for m in s.points))
        cache = getattr(self, "_else_cache", None)
        if cache is not None and cache[0] == key:
            return key, cache[1]
        out = np.zeros((s.n_frames, s.n_points), bool)
        t = (np.arange(s.n_frames, dtype=np.float64) - p.offsets[p.active]) / p.rates[p.active]
        for v in p.others():
            sv = p.sessions[v]
            pairs = [(j, sv.pid_by_name(m.name)) for j, m in enumerate(s.points)]
            pairs = [(j, q) for j, q in pairs if q is not None]
            if not pairs or sv.n_frames == 0:
                continue
            fv = p.local_index(v, t)
            inside = (fv >= 0) & (fv < sv.n_frames)
            idx = np.clip(fv, 0, sv.n_frames - 1)
            js, qs = [j for j, _ in pairs], [q for _, q in pairs]
            out[:, js] |= sv.tracked[idx][:, qs] & inside[:, None]
        self._else_cache = (key, out)
        return key, out

    def _landmark_cameras_at(self, pid: int, frame: int) -> list[str]:
        """(G173) The other cameras that have the working camera's landmark `pid` tracked at the instant of
        `frame` (the timeline's tooltip on its thin line)."""
        p, s = self.project, self.session
        if p is None or s is None or p.n_views < 2 or not (0 <= pid < s.n_points):
            return []
        name = s.points[pid].name
        t = p.reference_time(p.active, frame)
        out = []
        for v in p.others():
            sv = p.sessions[v]
            q = sv.pid_by_name(name)
            if q is None:
                continue
            fv = p.local_index(v, t)
            if 0 <= fv < sv.n_frames and sv.tracked[fv, q]:
                out.append(p.name(v))
        return out

    def _show_all_views(self) -> None:
        if self.project is None or not self.grid.hidden():
            return
        self.grid.set_arrangement(hidden=set())
        self._on_views_arranged()

    def _views_in_camera_order(self) -> None:
        if self.project is None:
            return
        self.grid.set_arrangement(order=list(range(self.project.n_views)))
        self._on_views_arranged()

    def _on_views_arranged(self) -> None:
        """The views on screen changed (an eye, a drag, a menu): the shown ones decode and draw again,
        the hidden ones give their decoder back (`_refresh_companions`), the guides and the eyes follow."""
        if self.project is None:
            return
        self._refresh_companions()
        self._refresh_guides()
        self._refresh_cameras()

    def _set_camera_fps(self, i: int) -> None:
        """A camera's fps button (G38): the rate it REALLY recorded at, when its
        file header gives another (slow-motion files). Camera rates, times and
        speeds follow; offsets are kept."""
        p = self.project
        if p is None or not (0 <= i < p.n_views):
            return
        if self.state == TRACKING:
            self.toast.show_message("Stop tracking first (X), then change the frame rate.", "warn", 5000)
            return
        s = p.sessions[i]
        file_fps = float(getattr(s, "file_fps", s.fps) or s.fps)
        from kinetrace.video_source import MAX_HEADER_FPS
        v, ok = QInputDialog.getDouble(
            self, "Frame rate",
            f"What frame rate did {p.name(i)} really record at (frames per second)?\n\n"
            f"Its video file says {file_fps:g}. High-speed cameras often save their footage for\n"
            "slow-motion playback, so the file says 30 while the camera filmed at 240 or 1000:\n"
            "use the rate set on the camera.\n\n"
            "Times, speeds and the matching of cameras all use this number; offsets are kept.\n"
            f"To go back to the file's rate, enter {file_fps:g}.",
            float(s.fps), 0.1, MAX_HEADER_FPS, 3)
        if not ok:
            return
        had_3d = p.reconstruction is not None
        if not p.set_fps(i, float(v)):
            return
        self._after_retime(had_3d)
        self._refresh_companions()
        self._refresh_cameras()
        self._apply_state()                       # menu / button gates (the project is now unsaved: set_fps dirties it)
        same = abs(float(v) - file_fps) <= 1e-6
        self.toast.show_message(
            f"{p.name(i)}: {float(v):g} fps" + ("  (the file's own rate)" if same else
                                                f"  (its file says {file_fps:g})")
            + ". Camera rates, times and speeds now use it.", "info", 7000)

    def _offer_video_recovery(self, info: dict) -> None:
        """Unsaved work from an earlier session on this video (never saved as
        a project): offer it. Declined work is kept aside, never deleted."""
        pid = info["project_id"]
        try:
            proj, _state, _meta = projectfile.read(recovery.paths(pid)[0])
        except projectfile.ProjectFileError as e:
            moved = recovery.quarantine(pid)
            QMessageBox.warning(self, "Earlier work could not be read",
                                f"Unsaved work for this video was found but could not be read ({e}).\n\n"
                                f"It was kept as {moved.name if moved else 'is'} in the recovery folder.")
            return
        # (I167) the opened file is ONE camera of the recovered project, not necessarily its
        # active one: match it by path, compare THAT camera's frame count, make it the
        # active camera (the open video is its picture), and take the whole project
        want = os.path.normcase(os.path.abspath(self.info.path))
        mine = next((k for k, s in enumerate(proj.sessions)
                     if os.path.normcase(os.path.abspath(s.video_path)) == want), proj.active)
        n_saved = proj.sessions[mine].n_frames
        if n_saved != self.info.n_frames and not (getattr(self.info, "header_frames", 0) == n_saved
                                                  and self.info.n_frames < n_saved):
            moved = recovery.quarantine(pid)
            QMessageBox.warning(self, "Earlier work not restored",
                                f"Unsaved work for this video was found, but it was made on a video of "
                                f"{n_saved} frames and this one has {self.info.n_frames}, so its tracks would "
                                f"sit on the wrong frames. It was kept as {moved.name if moved else 'is'} in "
                                "the recovery folder.")
            return
        restored = proj.sessions[mine]
        tracked_n = int(restored.tracked.any(axis=1).sum())
        cams = (f", {proj.n_views} cameras" if proj.n_views > 1 else "")
        if QMessageBox.question(
                self, "Restore unsaved work?",
                f"Unsaved work on this video was found, from {info.get('written_at', 'earlier')} "
                f"({restored.n_points} points, {tracked_n} tracked frames{cams}).\n\nRestore it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
            recovery.decline(pid)
            self.statusBar().showMessage(f"Earlier unsaved work not restored; it is kept in "
                                         f"{recovery.folder()[0] / 'declined'}", 10000)
            return
        restored.video_path = self.info.path
        proj.active = mine                        # the opened video is this camera's picture
        self._project_id = pid                    # keep writing to the same recovery
        self._adopt_project(proj, self.info)      # the whole project, one camera too: its name, lens, exports
        self.project.dirty = True

    def _teardown_video(self):
        for th in (self._body_worker, self._body_video):
            if th is not None and th.isRunning():
                th.request_cancel()
                th.wait(10000)
            _retire(th)
        self._body_worker = self._body_video = None
        # (I197) a pause ends the run "normally": nothing of the video that is going may start a
        # second pass, the next camera or the next re-track stretch inside the new one (as the close does)
        self._user_paused = True
        self._passes = None
        self._multi = None
        st = self._retrack
        if st is not None:
            st["jobs"] = []
            self._retrack = None             # its Keep / Undo question is about the project that is going
        if self.worker is not None:
            self.worker.request_pause()
            self.worker.wait(5000)
            _retire(self.worker)
            self.worker = None
        else:
            self._user_paused = False        # nothing ran: no pause is pending
        if self._preview is not None:
            self._preview.cancelled = True   # (I197) between chunks: it ends at once, not after the 15 s wait
            self._preview.wait(15000)
            _retire(self._preview)
            self._preview = None
        self._preview_again = False         # a rerun wanted by the video that is going (I188)
        _quiet_close(self._loading_dialog)
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
        self._lock_timer.stop()
        self._lock = None
        self._companions_stale = False
        for cv in self.grid.canvases:
            cv.set_stale(False)             # (G134)
        self._multi = None
        self._side = None                   # (I141)
        self.btn_play.setChecked(False)

    def _on_seek_slow(self, view: int, idx: int):
        if self.project is not None and view == self.project.active:
            self.statusBar().showMessage(f"Seeking frame {idx}…", 3000)

    def _on_decode_failed(self, view: int, idx: int, msg: str):
        """A frame inside the video could not be decoded (twice): say so once
        per frame, and never shorten the video over it (I40)."""
        if not (0 <= view < len(self._views)):
            return
        seen = self._views[view].bad_frames
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

    _INPUT_EVENTS = (QEvent.KeyPress, QEvent.MouseButtonPress, QEvent.MouseButtonRelease,
                     QEvent.MouseButtonDblClick, QEvent.Wheel)

    def _other_app_windows(self) -> list:
        """The app's windows other than the main one: the floated Segment & Points panel,
        the 3D view, the Body window."""
        return [w for w in (self.view3d, getattr(self, "dock", None), getattr(self, "body_win", None))
                if w is not None]

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in self._INPUT_EVENTS and self._loading and QApplication.activeModalWidget() is None:
            # opening: the window is half-built -- keys do nothing, Esc = Cancel where allowed.
            # In EVERY window of the app (G71): T in the floated panel / the 3D view / the Body
            # window used to start a second run while the first Track was still loading its model
            if t == QEvent.KeyPress:
                if QApplication.activeWindow() in (self, *self._other_app_windows()):
                    if ev.key() == Qt.Key_Escape and self._busy_cancel_cb() is not None:
                        self.overlay._on_cancel()
                    elif (ev.key() == Qt.Key_S and ev.modifiers() == Qt.ControlModifier
                          and self.project is not None and self._run_in_progress()):
                        self._save_project()            # (I265) kept for the run's end, said so
                    return True
            elif isinstance(obj, QWidget) and obj.window() in self._other_app_windows():
                return True             # the card only covers the main window: those clicks are held back here
        if t == QEvent.KeyPress and self._hotkeys_apply(ev):
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

    def _fresh_track_blocked(self):
        """Why Track cannot start HERE, judged now (G66): the button's verdict is refreshed first, so a
        semi-automatic F never trusts a stale one."""
        self._update_track_button()
        return self._track_blocked

    def _hotkey(self, ev) -> bool:
        """The hotkey table. Returns True when the key was one of ours."""
        if self._loading:
            return False                # the project is being built: no key acts (G71)
        key, mods = ev.key(), ev.modifiers()
        shift_only = mods == Qt.ShiftModifier
        plain = mods in (Qt.NoModifier, Qt.KeypadModifier)
        if key == Qt.Key_F and plain:
            # semi-automatic mode: F tracks one frame forward instead of just
            # moving (falls back to plain navigation when nothing is seedable)
            if (self._track_mode == "semi" and self.state == READY
                    and self.session is not None
                    and self.current < self.n_frames - 1
                    and self._fresh_track_blocked() is None):
                self._toggle_tracking()             # one step (in every camera with Track ▾ → Every camera)
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
        elif key == Qt.Key_T and shift_only:
            self._toggle_tracking(all_cameras=True)  # Shift+T: this run in every camera (G29)
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
            elif self.btn_pan.isChecked():
                self.btn_pan.setChecked(False)
                self.statusBar().showMessage("Pan tool off", 3000)
            elif self._pending_event is not None:
                self._pending_event = None
                self.timeline.set_pending_event(None)
                self.statusBar().showMessage("Event mark cancelled", 3000)
            elif self.timeline.sel_range is not None:
                self.timeline.clear_selection()
                self.statusBar().showMessage("Frame-window selection cleared", 3000)
            elif self._epi_probe is not None:
                self._epi_probe = None
                self._refresh_guides()
                self.statusBar().showMessage("Look-here line cleared", 3000)
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
        elif key == Qt.Key_A and plain:
            self._accept_prediction()               # A : place the selected point at the ◇ (G23)
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
        # (G24) playing, scrubbing or a held key = one continuous motion; a lone move
        # is a step. Sync all views: a step shows every camera's new picture at once
        # (lockstep). Active view only: ONLY the working video decodes, whatever the
        # move -- the others stay where they were, veiled, until Sync all is back or
        # one of them is clicked (G27; they used to catch up when the playhead
        # stopped, which read as "it still syncs every camera")
        now = time.monotonic()
        continuous = self.btn_play.isChecked() or (now - self._last_goto_t) < STEP_GAP_S
        self._last_goto_t = now
        p = self.project
        multi = p is not None and p.n_views > 1 and len(self.grid.visible_indices()) > 1
        self._companions_stale = bool(multi and self._sync_mode == "active")
        self._lock_timer.stop()
        self._lock = ({"want": {p.active}, "frames": {}}
                      if multi and self._sync_mode == "all" and not continuous
                      and self.seek is not None and self.state != TRACKING else None)
        if self._lock is not None:
            self._lock_timer.start()
        elif self.cache is not None and self.cache.get(idx) is None:
            # instant preview from cache while the real frame decodes
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
        lock = self._lock
        if view == self.project.active:
            if idx == self.current and self.state != TRACKING:
                if lock is not None and view in lock["want"]:
                    lock["frames"][view] = rgb        # goes up with the other cameras (G24)
                    self._try_lockstep()
                    return
                self.canvas.set_frame(rgb)
                if self._first_frame_token is not None:
                    self._first_frame_arrived()          # the open is over: the loading card goes
                # the side-by-side window shows the same instant, and the
                # decoded frame only exists here -- a seek is what fills it
                self._refresh_body_view()
            return
        # a companion view: only the frame we last asked it for (a scrub leaves
        # older replies in flight, and they would show the wrong instant)
        if idx == self._views[view].want_frame:
            if lock is not None and view in lock["want"]:
                lock["frames"][view] = rgb
                self._try_lockstep()
                return
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
            rt = self._views[i] if i < len(self._views) else None
            cv = self.grid.canvas(i)
            if cv is None or rt is None:     # a camera still being set up (a restored setting redraws, G33)
                continue
            if i not in visible:
                # hidden (solo): give the decoder thread and its VideoCapture
                # back — at 15 cameras that is 14 open 4K captures for nothing
                rt.want_frame = None
                rt.stop()
                continue
            f = p.map_frame(p.active, i, self.current)
            s = p.sessions[i]
            if self._tile_stale(i):
                # Active view only (G27): this camera is not decoded; it keeps the
                # picture it last showed, veiled, with the markers of THAT frame
                cv.set_stale(True)
                f = rt.want_frame           # never None here: _tile_stale needs a frame already shown
                self.grid.set_caption(
                    i, f"{p.name(i)}  ·  frame {f} · not following (Active view only) — click to work in it")
            else:
                cv.set_stale(False)
                if f is not None:           # a camera that ends early shows its last frame, and says so
                    f = min(f, rt.n_frames - 1)
                self.grid.set_caption(i, caption_for(p.name(i), f, p.offsets[i], rt.n_frames))
            if f is None:                    # this camera was not recording yet
                rt.want_frame = None
                cv.set_points(np.zeros((0, 2), np.float32), np.zeros(0, bool), [], None)
                cv.set_mask(None)
                cv.set_midline(None)
                continue
            f = min(f, rt.n_frames - 1)
            if rt.want_frame != f and not cv.is_stale():
                rt.want_frame = f
                cached = rt.cache.get(f)
                if self._lock is not None:
                    self._lock["want"].add(i)         # shown with the others (lockstep, G24)
                if cached is not None:
                    if self._lock is not None:
                        self._lock["frames"][i] = cached
                    else:
                        cv.set_frame(cached)
                else:
                    # decoder threads are started on FIRST NEED, not up front: at
                    # 15 cameras that is 15 idle threads each holding a 4K
                    # VideoCapture open for views the user may never look at
                    if rt.seek is None:
                        self._start_seek_service(i)
                    rt.seek.request(f)
            cv.set_masks(self._mask_layers(s, f), self._mask_opacity)       # every segment (G149)
            cv.set_bones(s.bones() if self.act_show_bones.isChecked() else [])
            sel_name = (self.session.points[self.selected].name if self.selected is not None
                        and self.session is not None and self.selected < self.session.n_points else None)
            # the ball markers' fitted circles too: a ball lost its circle as soon as
            # its camera was not the working one (G22)
            trails, future = self._companion_trails(s, f)      # (G33)
            cv.set_points(s.positions_at(f), s.visibility[f], s.points,
                          s.pid_by_name(sel_name) if sel_name is not None else None,
                          trails, future, occluded=s.occluded[f], radii=s.radius[f])
        rt_a = self._rt
        if rt_a is not None:
            self.grid.set_caption(p.active, caption_for(p.name(p.active), self.current,
                                                        p.offsets[p.active], rt_a.n_frames))
        self._try_lockstep()

    def _try_lockstep(self) -> None:
        """A step's pictures go up together once every camera has delivered (G24)."""
        lock = self._lock
        if lock is not None and lock["want"] <= set(lock["frames"]):
            self._flush_lockstep()

    def _flush_lockstep(self) -> None:
        """Show every picture a step has collected (all of them, or -- after
        LOCKSTEP_MS -- the ones that arrived; a late one then shows on arrival)."""
        lock, self._lock = self._lock, None
        self._lock_timer.stop()
        p = self.project
        if not lock or p is None:
            return
        for v, rgb in lock["frames"].items():
            cv = self.canvas if v == p.active else self.grid.canvas(v)
            if cv is not None and 0 <= v < len(self._views):
                cv.set_frame(rgb)
        if p.active in lock["frames"]:
            self._refresh_body_view()
            if self._first_frame_token is not None:
                self._first_frame_arrived()

    def _set_sync_mode(self, mode: str) -> None:
        """View → Other cameras / the CAMERAS panel's Sync all: "all" | "active" |
        "hidden" (G24). One entry point, so the menu and the panel agree."""
        act = {"all": self.act_sync_all, "active": self.act_sync_active, "hidden": self.act_solo}.get(mode)
        if act is not None and not act.isChecked():
            act.setChecked(True)

    def _on_sync_action(self, mode: str, on: bool) -> None:
        if not on:
            # its own shortcut again (nothing ticked now): Active view only goes back
            # to Sync all views, and Sync all views cannot be ticked off by itself. While
            # the group SWITCHES, checkedAction() is None too, but the new choice is
            # already ticked -- so ask the actions, not the group
            if not any(a.isChecked() for a in self._others_group.actions()):
                self._sync_mode = "all"
                self.act_sync_all.setChecked(True)
            return
        self._sync_mode = mode
        # Sync all: the others come back to the playhead now. Active view only: they
        # are at this instant already, and stay where they are from the next move on
        self._companions_stale = False
        self.cameras.set_sync(mode == "all")
        self._refresh_companions()
        self._refresh_guides()
        if mode == "active":
            self.statusBar().showMessage(
                "Active view only: only the working camera reads its video; the others stay where they are "
                "(veiled once the playhead moves) until Sync all views, or until you click one to work in it", 8000)

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

    def _put_down_other_tools(self, keep):
        """Pan, Add and Segment are ONE canvas tool at a time: arming one puts the
        others down, so a click always does what the last-picked tool says (a Pan
        left on used to swallow every Segment click until it was turned off by hand)."""
        for btn in (self.btn_pan, self.btn_add, self.btn_animal):
            if btn is not keep and btn.isChecked():
                btn.setChecked(False)

    def _on_pan_mode(self, on: bool):
        if on:
            self._put_down_other_tools(self.btn_pan)
            self.statusBar().showMessage(
                "Pan tool: drag to move the view. H or Esc when done "
                "(picking Add or Segment also ends it)", 6000)
        for cv in self.grid.canvases:
            cv.set_pan_mode(on)

    def _on_add_mode(self, on: bool):
        if on:
            self._put_down_other_tools(self.btn_add)
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
                "point needs Point ▾ → Region shape: circle first. Esc cancels", 9000)
        else:
            self.statusBar().showMessage(
                "Place a point: click where it should track — or drag to outline a "
                "region (circle or rectangle, chosen under Point ▾). Esc cancels (N toggles)", 8000)

    def _arm_ball(self):
        """Point ▾ → Ball marker: the next click segments a ball with SAM."""
        if self.session is None or self.state != READY:
            return
        self._place_kind = "ball"
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
                self._begin_edit()              # (I123)
                s.add_ball_prompt(self.selected, self.current, x, y)
                self._refresh_overlay()
                self._update_track_button()
                self.statusBar().showMessage(
                    f"{s.points[self.selected].name}: SAM will be prompted here on frame {self.current} — "
                    "press Track to continue it", 7000)
                return
            self._begin_edit()                  # adding is one undo step too (I123)
            pid = s.add_ball(self.current, x, y)
            where = self._assign_new_point(pid, x, y)      # its animal, or Scene (G155)
            self._share_landmarks()
            self.selected = pid
            self._refresh_point_list()
            self._refresh_overlay()
            self._apply_state()
            self.statusBar().showMessage(
                f"Added ball marker {s.points[pid].name}{where} at ({x:.0f}, {y:.0f}) on frame {self.current}. "
                "Add the other balls the same way (Point ▾ → Ball marker), then press Track: SAM outlines "
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
            self._on_place(self.selected, *self._snap_to_prediction(self.selected, x, y))
            rmse = self._residual_sentence(s.points[self.selected].name)      # (G28)
            self.statusBar().showMessage(
                (rmse + " — " if rmse else "")
                + f"{s.points[self.selected].name} placed here (same point, no new point created) — "
                "press Track to continue it, or Esc first if you wanted a new point", 8000)
            return
        # (I123) without a snapshot here, Ctrl+Z right after adding restored an OLDER
        # snapshot: the new point vanished AND the previous run or edit was undone
        self._begin_edit()
        derived_sel = (self.selected is not None and self.selected < s.n_points
                       and s.points[self.selected].derived)
        pid = s.add_point(self.current, x, y)
        where = self._assign_new_point(pid, x, y)          # its animal, or Scene (G155)
        self._share_landmarks()
        self.selected = pid
        self._refresh_point_list()
        self._refresh_overlay()
        self._apply_state()
        self._hint_small_spot(pid, x, y)
        self.statusBar().showMessage(
            f"Added {s.points[pid].name}{where} at ({x:.0f}, {y:.0f}) on frame {self.current}"
            + (" — the landmark that was selected is derived from the silhouette and cannot be placed by "
               "hand, so a NEW point was added" if derived_sel else ""), 6000 if derived_sel else 4000)

    def _owner_for_new_point(self, x: float | None = None, y: float | None = None) -> tuple[int | None, str]:
        """(G155) The animal a new point joins, and why: (1) the one animal row selected in LAYERS;
        (2) the one animal whose silhouette holds the click on this frame; (3) the animal of the
        selected point(s) (adding point after point to one animal); else Scene (None). Two
        silhouettes over the click and no selection to decide = Scene, said."""
        s = self.session
        if s is None or not s.segments:
            return None, ""
        rows = self._selected_segments()
        if len(rows) == 1:
            return rows[0], "selected"
        if x is not None:
            inside = [k for k in range(s.n_segments) if self._silhouette_holds(s, k, self.current, x, y)]
            if len(inside) == 1:
                return inside[0], "silhouette"
            if len(inside) > 1:
                return None, "overlap"
        of = {s.segment_of(q) for q in self._selected_pids()} - {None}
        if len(of) == 1:
            return next(iter(of)), "selected"
        return None, ""

    @staticmethod
    def _silhouette_holds(s, k: int, f: int, x: float, y: float) -> bool:
        """Animal k's silhouette on frame f contains (x, y) (its outline polygons)."""
        import cv2
        polys = s.seg_masks[k].contours.get(f) if 0 <= k < s.n_segments else None
        inside = False
        for poly in polys or []:
            pts = np.asarray(poly, np.float32).reshape(-1, 1, 2)
            if len(pts) >= 3 and cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0:
                inside = not inside                # a hole inside an outline flips it back
        return inside

    def _assign_new_point(self, pid: int, x: float | None = None, y: float | None = None) -> str:
        """Give a point just made its animal (G155, owner: assigned automatically and shown, not asked):
        `_owner_for_new_point`. A notice says where it went when a silhouette decided it or a
        silhouette under the click belongs to another animal, with a click that moves it. Returns
        ' (in <animal>)' / ' (in Scene)' for the status line ('' with no animals)."""
        s = self.session
        if s is None or not s.segments:
            return ""
        k, why = self._owner_for_new_point(x, y)
        if k is not None:
            s.move_points([pid], k)
        name = s.points[pid].name
        where = s.segments[k].name if k is not None else "Scene"
        under = ([j for j in range(s.n_segments) if j != k and self._silhouette_holds(s, j, self.current, x, y)]
                 if x is not None else [])
        if why == "silhouette" or why == "overlap" or under:
            text = (f"<b>{name}</b> was put in <b>{where}</b>"
                    + (" (its silhouette is under the click)" if why == "silhouette" else "")
                    + (" — the click is on the silhouettes of more than one animal" if why == "overlap" else "")
                    + (f" — it is on <b>{s.segments[under[0]].name}</b>'s silhouette" if under else "")
                    + ". Click here to move it, or drag it in LAYERS.")
            self.toast.show_message(text, "info", 9000,
                                    on_click=lambda n=name: self._move_menu_for_name(n))
        return f" (in {where})"

    def _move_menu_for_name(self, name: str) -> None:
        """A small menu at the pointer: move point `name` to an animal or Scene (G155's notice)."""
        s = self.session
        pid = s.pid_by_name(name) if s is not None else None
        if pid is None or self.state != READY:
            return
        menu = QMenu(self)
        acts = {}
        for k, a in enumerate(s.segments):
            if s.segment_of(pid) != k:
                acts[menu.addAction(f"Move {name} to {a.name}")] = k
        if s.segment_of(pid) is not None and not s.points[pid].derived:
            acts[menu.addAction(f"Move {name} to Scene")] = None
        from PySide6.QtGui import QCursor
        chosen = menu.exec(QCursor.pos())
        menu.deleteLater()
        if chosen in acts:
            self._move_points([pid], acts[chosen])

    def _new_point(self) -> None:
        """POINTS → ＋ New point: a point with no position yet, in every camera's
        list and selected -- a plain click on the video then places it (G26)."""
        s = self.session
        if s is None or self.state != READY:
            return
        if self.btn_add.isChecked():
            self.btn_add.setChecked(False)      # the next click places THIS point
        self._begin_edit()                      # one undo step, in every camera (I123, G19)
        pid = s.add_empty_point()
        self._assign_new_point(pid)             # the selected animal, else Scene (G155)
        self._share_landmarks()
        self.selected = pid
        self._epi_probe = None
        self._refresh_point_list()
        self._refresh_overlay()
        self._refresh_companions()
        self._apply_state()
        name = s.points[pid].name
        multi = self.project is not None and self.project.n_views > 1
        self.statusBar().showMessage(
            f"{name} made" + (" in every camera's list" if multi else "") + " with no position yet: click it on "
            "the video to place it" + (", then click the next camera and click it there (the dashed line shows "
                                        "where it can be)" if multi else "") + ". Double-click its name to rename "
            "it; Ctrl+Z removes it", 10000)

    def _on_add_group(self, cx: float, cy: float, radius: float):
        """Circle drag: track a region as one point (robust center of a member
        constellation). Future region shapes (rectangle, polygon) plug in here."""
        if self.session is None:
            return
        self.btn_add.setChecked(False)  # one placement per arm
        s = self.session
        self._begin_edit()                      # (I123)
        pid = s.add_point(self.current, cx, cy, kind="group", radius=radius)
        self._assign_new_point(pid, cx, cy)     # its animal, or Scene (G155)
        self._share_landmarks()
        self.selected = pid
        self._refresh_point_list()
        self._refresh_overlay()
        self._apply_state()
        self.statusBar().showMessage(
            f"Added region {s.points[pid].name} (r={radius:.0f} px) at frame "
            f"{self.current} — it tracks as one point: the region's fitted center", 6000)

    # ---------------------------------------------------------------- animal

    def _on_animal_mode(self, on: bool):
        if on:
            self._put_down_other_tools(self.btn_animal)
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
        """The line under LAYERS: the selected animal's numbers (G154), or what to do."""
        s = self.session
        if s is None or not s.segments:
            return ("No animal yet: ＋ Animal makes one (a layer for its points), or press <b>S</b> and click "
                    "the animal on the video to outline it (optional) — tail tip / midline / feet landmarks can "
                    "then be derived from its silhouette.")
        k = self._s_target()
        if k is None:
            return (f"{s.n_segments} animal(s). Select one in LAYERS to see its numbers; with one animal "
                    "selected, S + a click on the video outlines it.")
        a, m = s.segments[k], s.seg_masks[k]
        n_masked = m.n_masked()
        pf = a.prompt_frames()
        here = " (this frame)" if self.current in pf else ""
        n_pts = len(s.points_of(k))
        txt = (f"<b>{a.name}</b>: {n_pts} point(s); {a.n_prompts()} click(s)/box(es) on {len(pf)} frame(s)"
               f"{here}; silhouette on {n_masked:,} of {s.n_frames:,} frames.")
        n_der = sum(1 for q in s.points_of(k) if s.points[q].derived)
        if n_der:
            txt += f" {n_der} landmark(s) derive from it."
        if n_masked == 0 and pf:
            txt += " Press Track to segment the video."
        elif not pf and not n_masked:
            txt += " S + a click on the video outlines it (optional)."
        return txt

    def _refresh_animal_panel(self):
        """The animals changed (a click, a silhouette, a rename): the line under LAYERS and the tree."""
        self._refresh_point_list()

    def _new_segment(self) -> None:
        """LAYERS -> ＋ Animal (G149, G154): another animal, in every camera's list (by name), selected.
        It is a layer: points join it by being made while it is selected or by being dragged onto it;
        its silhouette is optional (S + a click on the video while it is selected)."""
        s = self.session
        if s is None or self.state != READY:
            return
        k = s.add_segment()
        self._share_landmarks(undoable=False)
        self.timeline.set_session(s)             # one more lane
        self._refresh_point_list(keep=(set(), {s.segments[k].name}))
        self._apply_state()
        self.statusBar().showMessage(
            f"{s.segments[k].name} made and selected: press N and click its points on the video (they join it), "
            "or drag points onto it; S + a click on the video outlines its silhouette (optional). "
            "Double-click its name to rename it", 12000)

    def _on_layer_item_changed(self, item, col: int = 0):
        """A LAYERS row was edited (G154): a point's name (the part: the animal's prefix is kept) or
        checkbox, an animal's name or checkbox (its silhouette shown), Scene's checkbox (its points)."""
        from kinetrace.session import qualified
        s = self.session
        kind = self.layers.kind_of(item)
        if s is None or not kind:
            return
        if kind[0] == "point":
            pid = kind[1]
            if pid >= s.n_points:
                return
            meta = s.points[pid]
            k = s.segment_of(pid)
            text = item.text(0).strip()
            part = s.part_name(pid)
            if not text:
                # an emptied name keeps the old one: the row used to stay blank (G126)
                self.layers.blockSignals(True)
                item.setText(0, part)
                self.layers.blockSignals(False)
            elif text != part:
                applied = self._apply_rename(pid, qualified(s.segments[k].name, text) if k is not None else text)
                self.layers.blockSignals(True)
                item.setText(0, s.part_name(pid))
                item.setData(0, ROLE_NAME, applied)
                self.layers.blockSignals(False)
                self._refresh_overlay()
            want = item.checkState(0) == Qt.Checked
            if want != meta.display:
                self._begin_edit()                    # the checkbox is one Ctrl+Z step too (G68)
                meta.display = want
                s.dirty = True
                self._refresh_overlay()
        elif kind[0] == "animal":
            k = kind[1]
            if not (0 <= k < s.n_segments):
                return
            a = s.segments[k]
            want = item.checkState(0) == Qt.Checked
            if want != a.shown:
                a.shown = want                      # display state: view.json, not an edit (G154)
                self._refresh_overlay()
            desired = item.text(0).strip()
            if desired and desired != a.name:
                self._rename_segment(k, desired)
            elif not desired:
                self.layers.blockSignals(True)
                item.setText(0, a.name)
                self.layers.blockSignals(False)
        elif kind[0] == "scene":
            want = item.checkState(0) == Qt.Checked
            mine = [q for q in s.points_of(None) if s.points[q].display != want]
            if mine:
                self._begin_edit()
                for q in mine:
                    s.points[q].display = want
                s.dirty = True
                self._refresh_point_list()
                self._refresh_overlay()

    # ---- an animal's context menu -----------------------------------------

    def _build_animal_menu(self, k: int | None = None):
        """Menu + action map, built separately so tests can drive the entries
        (the canvas point menu uses the same pattern). `k` = the animal the menu is for: the row
        right-clicked (G151b; None = the active one); it rides in acts["_seg"]."""
        s = self.session
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        if s is not None and not (k is not None and 0 <= k < s.n_segments):
            k = s.active_seg
        acts = {"_seg": k}
        ok = s is not None and 0 <= k < s.n_segments
        a = s.segments[k] if ok else None
        m = s.seg_masks[k] if ok else None
        n_masked = m.n_masked() if m is not None else 0
        acts["rename"] = menu.addAction("Rename the animal…")
        acts["outline"] = menu.addAction("Outline it on the video (S)")
        acts["outline"].setToolTip("Selects this animal and switches the S tool on: click the animal on the video "
                                   "(Shift+click = not it, a drag = a box)")
        a_mask = menu.addAction("Show its silhouette")
        a_mask.setCheckable(True)
        a_mask.setChecked(bool(a is not None and a.shown))
        acts["show_mask"] = a_mask
        a_hold = menu.addAction("Keep its points on its silhouette")
        a_hold.setCheckable(True)
        a_hold.setChecked(bool(a is not None and a.hold))
        a_hold.setToolTip("Its points tracked by appearance are nudged back when they slip just past the "
                          "silhouette's edge, and a run stops where one clearly leaves it (a point marked 'May "
                          "leave its silhouette' is exempt). Needs a silhouette.")
        acts["hold"] = a_hold
        a_mid = menu.addAction("Show the midlines")
        a_mid.setCheckable(True)
        a_mid.setChecked(self.act_show_midline.isChecked())
        acts["show_midline"] = a_mid
        menu.addSeparator()
        acts["first"] = menu.addAction("Jump to its first silhouette")
        acts["last"] = menu.addAction("Jump to its last silhouette")
        for key in ("first", "last"):
            acts[key].setEnabled(bool(n_masked > 0))
        a_here = menu.addAction(f"Clear its silhouette on frame {self.current}")
        a_here.setEnabled(bool(m is not None and m.has(self.current)))
        acts["clear_here"] = a_here
        sel = self.timeline.sel_range
        a_win = menu.addAction(
            f"Clear its silhouettes in frames {sel[0]}–{sel[1]}" if sel else
            "Clear its silhouettes in the selected frame window")
        a_win.setEnabled(bool(sel is not None and n_masked > 0))
        a_win.setToolTip("Shift+drag across the timeline to select a frame window first")
        acts["clear_window"] = a_win
        a_all = menu.addAction(f"Clear ALL {n_masked:,} of its silhouettes (keep the clicks)")
        a_all.setEnabled(bool(n_masked > 0))
        a_all.setToolTip("The clicks stay, so pressing Track segments the video again")
        acts["clear_all"] = a_all
        menu.addSeparator()
        acts["save_tpl"] = menu.addAction("Save its points and bones as a skeleton template…")
        acts["save_tpl"].setEnabled(bool(ok and s.points_of(k)))
        acts["save_tpl"].setToolTip("Another animal (this video or the next) then gets the same points from "
                                    "Skeleton ▾")
        acts["forget_sk"] = menu.addAction("Forget its bones and head (keep the points)")
        acts["forget_sk"].setEnabled(bool(a is not None and a.skeleton))
        menu.addSeparator()
        acts["remove"] = menu.addAction("Remove the animal (its points go to Scene)")
        acts["remove_all"] = menu.addAction("Remove the animal and its points")
        acts["remove_all"].setEnabled(bool(ok and s.points_of(k)))
        return menu, acts

    def _rename_segment(self, i: int, desired: str) -> str:
        """Animal i renamed in every camera (one name free in all, G149), its points with it (G153)."""
        s, p = self.session, self.project
        old = s.segments[i].name
        names = [s.points[q].name for q in s.points_of(i)]
        # its points' names and its own name come back with Ctrl+Z in every camera -- one where it has
        # no points too (G153)
        self._begin_edit(names, animals=[old])
        new = p.rename_segment(old, desired)     # every camera; one camera is the one-element case
        self._refresh_companions()
        self.timeline.refresh()
        self._refresh_point_list()
        self._refresh_overlay()
        self.statusBar().showMessage(f"Animal renamed to {new} (its points with it)", 5000)
        return new

    def _animal_menu_action(self, chosen, acts):
        """Apply one entry of the animal menu (the exec result, or an action picked by a test) to the
        menu's animal, acts["_seg"] (G151b)."""
        s = self.session
        if chosen is None or s is None or not s.segments:
            return
        k = acts.get("_seg")
        if k is None or not (0 <= k < s.n_segments):
            k = s.active_seg
        a = s.segments[k]
        if chosen is acts["rename"]:
            name, ok = QInputDialog.getText(self, "Rename the animal", "Name:", text=a.name)
            if ok and name.strip():
                self._rename_segment(k, name)
        elif chosen is acts["outline"]:
            self._refresh_point_list(keep=(set(), {a.name}))
            self._on_point_selection_changed()
            if not self.btn_animal.isChecked():
                self.btn_animal.setChecked(True)
        elif chosen is acts["show_mask"]:
            a.shown = acts["show_mask"].isChecked()     # display state: view.json, not an edit (G154)
            if a.shown and not self.btn_mask.isChecked():
                self.btn_mask.setChecked(True)
            self._refresh_point_list()
            self._refresh_overlay()
        elif chosen is acts["hold"]:
            self._begin_edit(animals=[a.name])      # every camera share_animal changes
            a.hold = acts["hold"].isChecked()
            if self.project is not None:
                self.project.share_animal(a.name)
            s.dirty = True
            self._refresh_point_list()
            self.statusBar().showMessage(
                f"{a.name}: its points are kept on its silhouette (next run)" if a.hold else
                f"{a.name}: its points may drift off its silhouette", 6000)
        elif chosen is acts["show_midline"]:
            self.act_show_midline.setChecked(acts["show_midline"].isChecked())
            self._refresh_overlay()
        elif chosen is acts["first"] or chosen is acts["last"]:
            fr = s.mask_frames(k)
            if len(fr):
                self._goto(int(fr[-1] if chosen is acts["last"] else fr[0]))
        elif chosen is acts["clear_here"]:
            self._clear_masks_window(self.current, self.current, [k])
        elif chosen is acts["clear_window"]:
            sel = self.timeline.sel_range
            if sel is not None:
                self._clear_masks_window(sel[0], sel[1], [k])
        elif chosen is acts["clear_all"]:
            n = s.seg_masks[k].n_masked()
            if QMessageBox.question(
                    self, "Clear silhouettes",
                    f"Remove {a.name}'s silhouettes on {n:,} frames?\n\nIts clicks stay, so "
                    "pressing Track segments the video again. Ctrl+Z undoes this.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
                self._clear_masks_window(0, s.n_frames - 1, [k])
        elif chosen is acts["save_tpl"]:
            self._save_animal_template(k)
        elif chosen is acts["forget_sk"]:
            self._clear_skeleton_of(k)
        elif chosen is acts["remove"]:
            self._clear_animal([k])                 # the row right-clicked (G150)
        elif chosen is acts["remove_all"]:
            self._clear_animal([k], with_points=True)

    def _save_animal_template(self, k: int) -> None:
        """An animal's points, bones, head and derived rules as a template in skeletons/ (G157), so
        the next animal gets them from Skeleton ▾."""
        from kinetrace.skeletons import save_user_template, user_template_path
        s = self.session
        name, ok = QInputDialog.getText(self, "Save as a skeleton template", "Template name:",
                                        text=(s.segments[k].skeleton or {}).get("name") or s.segments[k].name)
        if not ok or not name.strip():
            return
        t = s.skeleton_template(k, name.strip())
        try:
            try:
                path = save_user_template(t)
            except FileExistsError:
                if QMessageBox.question(
                        self, "Replace the saved skeleton?",
                        f"{user_template_path(t).name} already exists in skeletons/. Replace it?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                    return
                path = save_user_template(t, overwrite=True)
        except OSError as e:
            QMessageBox.warning(self, "Template not saved", _plain_error(e, "The template could not be saved"))
            return
        self._refresh_skeleton_menu()
        self.toast.show_message(
            f"Template <b>{t['name']}</b> saved ({len(t['landmarks'])} points, {len(t['bones'])} bones): select "
            f"another animal and choose it under Skeleton ▾. ({Path(str(path)).name})", "success", 9000)

    def _layers_menu(self, pos):
        """Right-click in LAYERS (G154): a point = the point menu (several selected = the menu for all of
        them, G64); an animal = its menu; Scene = a short one."""
        item = self.layers.itemAt(pos)
        s = self.session
        if item is None or s is None or self.state != READY:
            return
        kind = self.layers.kind_of(item)
        gpos = self.layers.viewport().mapToGlobal(pos)
        if kind and kind[0] == "point":
            self._point_list_menu_for(kind[1], gpos)
        elif kind and kind[0] == "animal":
            menu, acts = self._build_animal_menu(kind[1])
            chosen = menu.exec(gpos)
            menu.deleteLater()          # release the shown menu (see canvas._context_menu)
            self._animal_menu_action(chosen, acts)
        elif kind and kind[0] == "scene":
            menu = QMenu(self)
            a_new = menu.addAction("＋ Point in Scene")
            chosen = menu.exec(gpos)
            menu.deleteLater()
            if chosen is a_new:
                self.layers.select_only()
                self._on_point_selection_changed()
                self._new_point()

    def _on_layers_move(self, pids, target: int) -> None:
        """Points dragged onto an animal (`target`) or Scene (-1) in LAYERS (G154, owner: "allow the user
        to drag and drop points from one animal parent to another")."""
        self._move_points(list(pids), None if target < 0 else int(target))

    def _move_points(self, pids, k: int | None) -> None:
        """Points `pids` into animal k (None = Scene) in every camera, renamed "<animal> <part>", as one
        Ctrl+Z step (G153). A landmark derived from the silhouette needs an animal: it is not moved to
        Scene."""
        s = self.session
        if s is None or self.state != READY:
            return
        derived = [q for q in pids if 0 <= q < s.n_points and s.points[q].derived]
        if k is None and derived:
            self.toast.show_message(
                "A landmark derived from the silhouette (" + ", ".join(s.points[q].name for q in derived)
                + ") belongs to an animal: it was not moved to Scene. Drag it onto another animal, or switch it "
                "to tracking by appearance first (right-click → Data source).", "warn", 9000)
            pids = [q for q in pids if q not in derived]
        pids = [q for q in pids if 0 <= q < s.n_points and s.segment_of(q) != k]
        if not pids:
            return
        names = [s.points[q].name for q in pids]
        target = s.segments[k].name if k is not None else ""
        self._begin_edit(names)
        changes = self.project.move_landmarks(names, target)    # every camera (one = the 1-element case)
        self._refresh_companions()
        new_names = {n for _o, n in changes}
        self._refresh_point_list(keep=(new_names, set()))
        self._on_point_selection_changed()
        self._refresh_overlay()
        self.timeline.refresh()
        where = target or "Scene"
        self.statusBar().showMessage(
            f"{len(changes)} point(s) moved to {where}" + (f" (now {', '.join(sorted(new_names))})"
                                                           if len(new_names) <= 4 else "")
            + " — Ctrl+Z puts them back", 8000)
    def _current_rgb(self):
        return self.cache.get(self.current) if self.session is not None else None

    def _segment_tool_target(self) -> tuple[int | None, bool]:
        """The animal an S click / box outlines (G154): the one the LAYERS selection names; with no
        animal yet, a new one (True = made now). (None, False) = the selection names none or several
        of the animals there are: nothing is clicked, a notice says to select one."""
        s = self.session
        k = self._s_target()
        if k is not None:
            s.active_seg = k
            return k, False
        if not s.segments:
            return s.add_segment(), True
        self.toast.show_message(
            "Select the animal to outline in LAYERS (one animal), or ＋ Animal for a new one: the S tool "
            "outlines the selected animal.", "warn", 8000)
        return None, False

    def _on_animal_click(self, x: float, y: float, positive: bool):
        s = self.session
        if s is None or self.state != READY:
            return
        k, new = self._segment_tool_target()
        if k is None:
            return
        dropped = self._undo_not_for_prompts()
        s.segments[k].add_click(self.current, x, y, positive)
        s._touch()
        self._after_prompt_change("excluded" if not positive else "marked", new, dropped)

    def _on_animal_box(self, x0: float, y0: float, x1: float, y1: float):
        s = self.session
        if s is None or self.state != READY:
            return
        k, new = self._segment_tool_target()
        if k is None:
            return
        dropped = self._undo_not_for_prompts()
        s.segments[k].set_box(self.current, (x0, y0, x1, y1))
        s._touch()
        self._after_prompt_change("boxed", new, dropped)

    def _on_prompt_remove(self, index: int):
        s = self.session
        if s is None or s.animal is None:
            return
        clicks = s.animal.prompts.get(self.current, [])
        if 0 <= index < len(clicks):
            dropped = self._undo_not_for_prompts()
            del clicks[index]
            if not clicks:
                s.animal.prompts.pop(self.current, None)
            s._touch()
            if s.animal.has_prompt(self.current):
                self._after_prompt_change("updated", dropped_undo=dropped)
            else:
                s.clear_masks(self.current, self.current)
                self._refresh_overlay()
                self._refresh_animal_panel()
                self._apply_state()
                self.statusBar().showMessage("Click removed — no prompt left on this frame"
                                             + (self._UNDO_DROPPED if dropped else ""), 6000 if dropped else 4000)

    _UNDO_DROPPED = " (the earlier Ctrl+Z step is gone: segment clicks cannot be undone — right-click a click to remove it)"

    def _undo_not_for_prompts(self) -> bool:
        """The segment's clicks and boxes are not part of an undo snapshot, so no
        snapshot can take one back (G70): left alone, the next Ctrl+Z undid the PREVIOUS
        tracking run and kept the click. The undo point is cleared instead, as for
        removing the segment (I124). True when there was one to lose."""
        had = self._undo_snap is not None
        self._undo_snap = None
        self.act_undo.setEnabled(False)
        return had

    def _after_prompt_change(self, verb: str, new_segment: bool = False, dropped_undo: bool = False):
        if new_segment:
            self._share_landmarks(undoable=False)     # the first segment in every camera's list (G149)
            self.timeline.set_session(self.session)
        self._refresh_overlay()
        self._refresh_animal_panel()
        if new_segment:
            # the animal a click has just made is part of the next run, like a new point (G120), and
            # the S tool's target (G154): its row alone is selected, so "Press Track now" is true
            s = self.session
            self.layers.select_only(animals=[s.active_seg])
            self._on_point_selection_changed()
        self._apply_state()
        a = self.session.animal
        self.statusBar().showMessage(f"{a.name if a is not None else 'Animal'} {verb} on frame {self.current} — "
                                     "computing its silhouette…" + (self._UNDO_DROPPED if dropped_undo else ""),
                                     8000 if dropped_undo else 5000)
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
        hp = s.head_pid(s.active_seg)          # the head of the segment clicked for (G151d)
        if hp is not None and s.tracked[self.current, hp]:
            head = s.tracks[self.current, hp]
        clicks = list(s.animal.prompts.get(self.current, []))
        box = s.animal.boxes.get(self.current)
        w = _MaskPreviewWorker(self.info.path, self.current, self._current_rgb(),
                               (self.info.width, self.info.height), clicks, box,
                               self._seg_backend, head)
        w.target_session = s            # the camera may change while SAM works (I66)
        # (I262) ... and the animal it was clicked for: the OBJECT, so a rename while SAM works still finds it (G149)
        w.target_animal = s.animal
        w.loading.connect(self._on_seg_loading)
        w.progress.connect(self._on_seg_progress)
        # bound to THIS worker: a result that arrives after the video was replaced must
        # not be written into the next video's session (I188)
        w.done.connect(lambda summ, w=w: self._on_preview_done(summ, w))
        w.error.connect(lambda tb, w=w: self._on_preview_error(tb, w))
        self._preview = w
        self._preview_again = False
        QApplication.setOverrideCursor(Qt.BusyCursor)
        w.start()

    def _on_seg_loading(self, backend: str):
        from kinetrace.segmenter import loaded_backends
        if backend in loaded_backends() or self._loading_dialog is not None:
            return
        label = BACKENDS.get(backend, BACKENDS[DEFAULT_BACKEND])[2]
        if seg_is_cached(backend):
            text = f"Loading the segmentation model ({label})…"
        else:
            text = (f"Downloading the segmentation model ({label}) — first use only.\n"
                    "It is stored inside the tool folder (models/hf).")
        dlg = QProgressDialog(text, "Cancel", 0, 0, self)
        dlg.setWindowTitle("Preparing the model")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumDuration(0 if not seg_is_cached(backend) else 300)
        dlg.canceled.connect(self._cancel_seg_download)
        dlg.setValue(0)
        self._loading_dialog = dlg
        self._seg_clock = {}

    def _on_seg_progress(self, label: str, done: float, total: float):
        if self._loading_dialog is None:
            return
        mx, val, text = _download_status(self._seg_clock, label, done, total)
        self._loading_dialog.setRange(0, mx)
        self._loading_dialog.setValue(val)
        self._loading_dialog.setLabelText(text)

    def _cancel_seg_download(self):
        if self._preview is not None:
            self._preview.cancelled = True          # between chunks: the worker ends with a sentence

    def _close_loading_dialog(self):
        _quiet_close(self._loading_dialog)
        self._loading_dialog = None

    def _end_preview(self, w) -> bool:
        """What every preview result starts with (I188): put the cursor back, close
        the download card, let the worker finish (`_retire`: still cleaning up, it is
        kept until it ends, never destroyed running, I133). True when `w` is the
        CURRENT preview -- its result is wanted; a worker the video's replacement
        already dropped is only cleaned up."""
        QApplication.restoreOverrideCursor()
        current = w is not None and w is self._preview
        if current:
            self._close_loading_dialog()
        if w is not None:
            w.wait(2000)
            _retire(w)
        if current:
            self._preview = None
        return current

    def _on_preview_done(self, summ: dict, w=None):
        w = w if w is not None else self._preview
        target = getattr(w, "target_session", None)
        if not self._end_preview(w):
            return                      # its video / project is gone (I188)
        s = target if target is not None else self.session
        if s is None or s.animal is None or self.project is None or not any(s is x for x in self.project.sessions):
            return                      # the camera it was clicked in was removed (I188)
        f = int(summ["frame"])
        # the animal it was clicked for (G149), found by identity: renamed meanwhile is the same animal;
        # removed meanwhile = the silhouette belongs to no one and is dropped (never the active animal's)
        want = getattr(w, "target_animal", None) if w is not None else None
        k = next((i for i, a in enumerate(s.segments) if a is want), None) if want is not None else s.active_seg
        if k is None:
            return
        summ["seg"] = k
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

    def _on_preview_error(self, tb: str, w=None):
        w = w if w is not None else self._preview
        if not self._end_preview(w):
            return                      # a worker of a video that is gone: nothing to say about it (I188)
        if w is not None and w.plain:                   # a download: its sentence, no traceback (G45)
            if w.cancelled:
                self.statusBar().showMessage(tb + " Click the segment again to go on with it.", 9000)
            else:
                self.toast.show_message(tb, "error", 12000)
                QMessageBox.warning(self, "The segmentation model could not be downloaded", tb)
            return
        hint = _model_error_hint(tb)
        self.toast.show_message("Segmentation failed. " + (hint or "Try the click again, or drag a box."),
                                "error", 10000)
        QMessageBox.critical(self, "Segmentation failed", _crash_text(
            tb, hint, "The segment could not be found on this frame because of an unexpected error."))

    def _delete_layers(self) -> None:
        """LAYERS -> Delete (and the Delete key when no frame window is selected): exactly the rows
        selected in LAYERS (G150, G154) -- points with their tracks, animals (asked whether their
        points go too). Nothing selected = nothing deleted, and said."""
        s = self.session
        if s is None or self.state != READY:
            return
        ks, pids = self._selected_segments(), self._selected_pids()
        if ks:
            self._clear_animal(ks, extra_pids=[q for q in pids if s.segment_of(q) not in ks])
        elif len(pids) > 1:
            self._delete_points(pids)
        elif pids:
            self._on_delete(pids[0])
        else:
            self.toast.show_message("Select what to delete in LAYERS (points and / or animals), or right-click a "
                                    "row.", "info", 6000)

    def _clear_animal(self, idx=None, with_points: bool | None = None, extra_pids=()) -> None:
        """Remove animals `idx` (indices; None = the active one) in this camera and, by name, in every
        other camera. ONE question names them all and, when they have points, whether those go too
        (`with_points` None = ask; kept points move to Scene with their tracks). `extra_pids` = other
        selected points deleted in the same step. An animal cannot be brought back by Ctrl+Z (its
        clicks are in no snapshot, G70 / I124): the question says so."""
        s = self.session
        if s is None or not s.segments or self.state != READY:
            return
        idx = [s.active_seg] if idx is None else sorted({i for i in idx if 0 <= i < s.n_segments})
        if not idx:
            return
        names = [s.segments[i].name for i in idx]
        mine = sorted({q for k in idx for q in s.points_of(k)})
        extra = [q for q in extra_pids if 0 <= q < s.n_points and q not in mine]
        n = sum(s.seg_masks[i].n_masked() for i in idx)
        clicks = sum(s.segments[i].n_prompts() for i in idx)
        p = self.project
        others = [p.name(v) for v in p.others()
                  if any(p.sessions[v].segment_index(nm) is not None for nm in names)] \
            if p is not None and p.n_views > 1 else []
        what = (f"the animal {names[0]}: its" if len(names) == 1
                else f"{len(names)} animals ({', '.join(names)}): their")
        text = (f"Remove {what} {clicks} click(s)/box(es) and silhouettes on {n:,} frames?"
                + (f" {'It goes' if len(names) == 1 else 'They go'} from the other cameras too "
                   f"({', '.join(others)})." if others else "")
                + (f"\n\nAlso delete the {len(extra)} other selected point(s) "
                   f"({', '.join(s.points[q].name for q in extra[:4])}{', …' if len(extra) > 4 else ''})."
                   if extra else ""))
        keep = True
        if mine and with_points is None:
            ans = QMessageBox.question(
                self, "Remove animal" if len(names) == 1 else "Remove animals",
                text + f"\n\nKeep {'its' if len(names) == 1 else 'their'} {len(mine)} point(s)?\n"
                "Yes = they move to Scene with their tracks.  No = they are deleted too.\n\n"
                "This cannot be undone with Ctrl+Z.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if ans not in (QMessageBox.Yes, QMessageBox.No):
                return
            keep = ans == QMessageBox.Yes
        else:
            if with_points is not None:
                keep = not with_points
            if QMessageBox.question(
                    self, "Remove animal" if len(names) == 1 else "Remove animals",
                    text + ("" if not mine else (f"\n\nIts {len(mine)} point(s) " + (
                        "move to Scene with their tracks." if keep else "are deleted too.")))
                    + "\n\nThis cannot be undone with Ctrl+Z.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
        self.canvas.cancel_gesture()
        gone = [s.points[q].name for q in sorted(set(extra) | (set() if keep else set(mine)))]
        for q in sorted((s.pid_by_name(nm) for nm in gone), reverse=True):
            if q is not None:
                s.remove_point(q)
        self._remove_landmarks_elsewhere(gone)
        for name in names:                         # by NAME: the indices shift as each one goes
            p.remove_segment(name, keep_points=True)  # one list of animals in every camera (G149)
        # an older snapshot would restore something else and not the animal (I124)
        self._undo_snap = None
        self.act_undo.setEnabled(False)
        self.btn_animal.setChecked(False)
        self.selected = None
        self.timeline.set_session(s)   # lane layout changes
        self._refresh_point_list(keep=(set(), set()))
        self._on_point_selection_changed()
        self._refresh_overlay()
        self._apply_state()
        self.statusBar().showMessage(
            f"{'Animal' if len(names) == 1 else 'Animals'} {', '.join(names)} removed"
            + (f"; {len(mine)} point(s) moved to Scene" if keep and mine else "")
            + (f"; {len(gone)} point(s) deleted" if gone else ""), 6000)

    def _clear_masks_window(self, f0: int, f1: int, segs=None):
        """Drop silhouettes inside [f0, f1] -- of `segs` (the timeline lanes the marquee covered, or
        the menu's row), else `_resolve_window_segs` (G151h) -- leaving every tracked point alone (G149)."""
        self._clear_window(f0, f1, [], do_points=False, do_masks=True, segs=segs)
        self._refresh_animal_panel()
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

    def _set_trail_len(self, n: int, announce: bool = True):
        self._trail_len = n = int(max(0, min(TRAIL_MAX, n)))
        act = self._trail_acts.get(n)
        if act is None:                             # any other length is a custom one (G33)
            self._trail_custom = n
            act = self.act_trail_custom
        act.setChecked(True)
        self.act_trail_custom.setText(f"Custom… ({self._trail_custom} frames)"
                                      if act is self.act_trail_custom else "Custom…")
        for cv in self.grid.canvases:
            cv.set_trails(self._trail_len, self._trail_future)
        self._refresh_overlay()
        self._refresh_companions()

    def _ask_trail_len(self):
        """View → Trails → Custom…: how many frames of trail to draw (G33)."""
        n, ok = QInputDialog.getInt(
            self, "Trail length",
            f"How many frames of trail to draw behind each point (1–{TRAIL_MAX}; the line fades towards "
            "its oldest frame):", self._trail_custom, 1, TRAIL_MAX, 1)
        # cancelled: the entry chosen before is ticked again
        self._set_trail_len(int(n) if ok else self._trail_len)

    def _trails_for(self, s, f: int):
        """View → Trails for session `s` at frame `f`: (the past path, the upcoming
        path) per point, or None where that part is off. One place, so the working
        camera and the others draw the same trails (G33)."""
        n = self._trail_len
        if n <= 0 or not (0 <= f < s.n_frames):
            return None, None
        lo = max(0, f - n)
        past = [s.tracks[lo:f + 1, i] for i in range(s.n_points)]
        future = ([s.tracks[f:min(s.n_frames, f + n + 1), i] for i in range(s.n_points)]
                  if self._trail_future else None)
        return past, future

    def _companion_trails(self, s, f: int):
        """The other cameras draw their trails only while Track ▾ → Every camera is
        ticked -- the cameras being tracked together (G33)."""
        return self._trails_for(s, f) if self.act_track_all.isChecked() else (None, None)

    def _on_trail_future(self, on: bool):
        self._trail_future = bool(on)
        for cv in self.grid.canvases:
            cv.set_trails(self._trail_len, self._trail_future)
        self._refresh_overlay()
        self._refresh_companions()

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
        self._begin_edit()                      # (I123)
        pid = s.add_point(self.current, float(c[0]), float(c[1]), kind="group", radius=radius,
                          shape=shape, outline=pts.tolist())
        self._assign_new_point(pid, float(c[0]), float(c[1]))     # its animal, or Scene (G155)
        self._share_landmarks()
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
        self._begin_edit()                      # Ctrl+Z takes the mark back, nothing older (I71)
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
        self._begin_edit()
        n = s.set_occluded_window(use, f0, f1, on)
        self.timeline.clear_selection()
        self._refresh_overlay()
        names = ", ".join(s.points[p].name for p in use[:4]) + (" …" if len(use) > 4 else "")
        self.statusBar().showMessage(
            f"{'Marked' if on else 'Unmarked'} {n} cells hidden: {names}, frames {f0}–{f1} "
            "(Ctrl+Z undoes)", 6000)

    # ------------------------------------------------ notes + annotator

    @staticmethod
    def _settings() -> QSettings:
        """settings.ini in the Kinetrace folder (`recovery.settings_path`), never the registry
        or a plist: deleting the folder removes it (Mac install audit P1)."""
        from kinetrace import recovery
        return QSettings(str(recovery.settings_path()), QSettings.IniFormat)

    def _default_annotator(self) -> str:
        try:
            return str(self._settings().value("annotator", "") or "")
        except Exception:      # noqa: BLE001
            return ""

    def _apply_annotator(self, name: str):
        name = (name or "").strip()
        try:
            st = self._settings()
            st.setValue("annotator", name)
            st.sync()
        except Exception:      # noqa: BLE001
            pass
        # every camera's events and notes are marked by the same person (I233)
        for s in (self.project.sessions if self.project is not None else
                  [self.session] if self.session is not None else []):
            s.annotator = name
            s.dirty = True

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
        self._begin_edit()                      # its own undo step: Ctrl+Z restored an older one (G68)
        s.points[pid].free = bool(free)
        s.dirty = True
        name = s.points[pid].name
        self.statusBar().showMessage(
            f"{name}: may leave its silhouette (not held on it)" if free
            else f"{name}: held on its animal's silhouette when the animal keeps its points on it (right-click "
                 "the animal in LAYERS; if the point clearly leaves it, the run stops there)", 6000)

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
        k = None
        if spec and s.n_segments > 1:
            # (G151e) from the segment the user names (one highlighted row, or asked), never the
            # invisible active row
            k = self._pick_segment(f"Derive {meta.name} from the silhouette of")
            if k is None:
                self._refresh_point_list()
                return
        # an undo point whether or not there was data to lose: the source change itself
        # is an edit, and Ctrl+Z would otherwise undo an older one (G70); every camera the move to its
        # animal renames (G153)
        self._begin_edit([meta.name])
        if spec:
            if s.animal is None:
                self.toast.show_message(
                    f"<b>{meta.name}</b> will derive from the animal's silhouette — press "
                    "<b>S</b> and click the animal first, then Track.", "info", 8000)
            else:
                # (G153) the landmark becomes the animal's: "<animal> <part>", in every camera
                if k is None:
                    k = 0 if s.segment_of(pid) is None else s.segment_of(pid)
                if s.segment_of(pid) != k:
                    self.project.move_landmarks([meta.name], s.segments[k].name)
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
            from kinetrace.skeletons import user_template_problems
            probs = user_template_problems()
            if probs:           # a broken file in skeletons/ is said, not silently skipped (I62)
                menu.addAction(f"{len(probs)} problem(s) in the skeletons/ folder…",
                               lambda pr=tuple(probs): QMessageBox.warning(
                                   self, "Skeleton files", "\n\n".join(pr)))
            if self.session is not None and any(a.skeleton for a in self.session.segments):
                menu.addSeparator()
                menu.addAction("Forget the selected animal's bones and head (keep the points)", self._clear_skeleton)
            menu.setToolTipsVisible(True)

    def _apply_skeleton_template(self, t: dict):
        s = self.session
        if s is None or self.state != READY:
            return
        # (G151f, G153) the animal the skeleton is for: the one the LAYERS selection names, else asked;
        # with no animal yet, a new one ("animal") is made for it -- a skeleton belongs to an animal
        if s.segments:
            k = self._pick_segment(f"Put the skeleton {t.get('name', '')} on")
            if k is None:
                return
            made = False
        else:
            k, made = s.add_segment(), True
        seg_name = s.segments[k].name
        # one undo step, in every camera the template gives landmarks to (G70, G19)
        self._undo_snap = s.snapshot()
        self.act_undo.setEnabled(True)
        if self.project is not None and self.project.n_views > 1:
            for v in self.project.others():
                self._undo_extra[v] = self.project.sessions[v].snapshot()
        new = s.apply_skeleton(t, k)
        if self.project is not None and self.project.n_views > 1:
            # the same animal in every camera: its skeleton (bones, head) and its landmarks (G19)
            self._share_landmarks(undoable=False)
            self.project.share_animal(seg_name)
        self._refresh_point_list(keep=(set(), {seg_name}))
        self._on_point_selection_changed()
        if made:
            self.timeline.set_session(s)
        self._refresh_point_list()
        self._refresh_overlay()
        self._refresh_skeleton_menu()
        self._refresh_animal_panel()
        self._apply_state()
        QTimer.singleShot(0, self._fit_timeline_height)
        derived = [s.points[p].name for p in new if s.points[p].derived]
        placed = [s.points[p].name for p in new if not s.points[p].derived]
        msg = f"Skeleton <b>{t['name']}</b>" + (f" on {seg_name}" if seg_name else "") + \
            f": {len(new)} landmark(s) added."
        if placed:
            head = t["head"] if t.get("head") else s.part_name(s.pid_by_name(placed[0]))
            msg += (f" Select one in LAYERS, press <b>N</b> and click it on the video "
                    f"(start with <b>{head}</b>).")
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
        pairs, dropped = [], []
        for line in bones.toPlainText().splitlines():
            if not line.strip():
                continue
            # "a - b" first: landmark names may hold a hyphen ("left-hip - left-knee" was cut at
            # the first one and the bone silently lost, G132); a line with no " - " is split at
            # a hyphen only where both halves are landmarks
            cuts = ([line.split(" - ", 1)] if " - " in line
                    else [[line[:k], line[k + 1:]] for k, ch in enumerate(line) if ch == "-"])
            pair = next(([a.strip(), b.strip()] for a, b in cuts
                         if a.strip() in landmarks and b.strip() in landmarks), None)
            if pair is not None:
                pairs.append(pair)
            else:
                dropped.append(line.strip())
        t = {"name": name.text().strip() or "custom", "head": head.text().strip() or landmarks[0],
             "landmarks": landmarks, "bones": pairs, "derived": derived, "note": ""}
        from kinetrace.skeletons import validate_template
        t, problems = validate_template(t)
        left_out = [f"bone line “{ln}” — both ends must be landmarks of this skeleton, written “a - b”"
                    for ln in dropped]                                                  # (G132)
        if problems or left_out:
            # a typo such as midline:50 used to become a landmark that never fills (I62)
            QMessageBox.warning(self, "Skeleton rules corrected",
                                "Some of the rules could not be used and were left out:\n\n• "
                                + "\n• ".join(list(problems) + left_out)
                                + ("\n\nThose landmarks are tracked by appearance instead (select, N, click)."
                                   if problems else ""))
        try:
            try:
                save_user_template(t)
            except FileExistsError:
                # a template of that name is already in skeletons/: replaced only when asked (G122)
                from kinetrace.skeletons import user_template_path
                fname = user_template_path(t["name"]).name
                if QMessageBox.question(
                        self, "Replace the saved skeleton?",
                        f"skeletons/{fname} already exists.\n\nReplace it with this skeleton? (No: use the skeleton "
                        "now without saving it over the old one.)",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
                    save_user_template(t, overwrite=True)
        except OSError as e:
            self.statusBar().showMessage(f"Template not saved to skeletons/: {e}", 6000)
        self._apply_skeleton_template(t)

    def _clear_skeleton(self):
        """Skeleton ▾ -> Forget: the bones and head of the animal the selection names (asked otherwise)."""
        s = self.session
        if s is None or not s.segments:
            return
        k = self._pick_segment("Forget the bones and head of")
        if k is not None:
            self._clear_skeleton_of(k)

    def _clear_skeleton_of(self, k: int) -> None:
        """Animal k's bones and head forgotten in every camera, its points kept: one Ctrl+Z step
        (Skeleton ▾ -> Forget and the animal menu)."""
        s = self.session
        # the skeleton is in the snapshot (G70), of every camera share_animal changes
        self._begin_edit(animals=[s.segments[k].name])
        s.clear_skeleton(k)
        if self.project is not None:
            self.project.share_animal(s.segments[k].name)
        self._refresh_skeleton_menu()
        self._refresh_overlay()
        self.statusBar().showMessage(f"{s.segments[k].name}'s bones and head forgotten — its points remain "
                                     "(Ctrl+Z)", 6000)

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
        from kinetrace.session import starts_formula
        if starts_formula(name):              # (M7) a spreadsheet would run it as a formula
            self.toast.show_message(f"“{name.strip()}”: a name may not start with = + - or @ (a spreadsheet "
                                    "opening the events export would run it as a formula), so those "
                                    "characters were left out.", "warn", 8000)
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
        # clear() leaves the submenus (one per event type, one for Notes) alive as children of
        # the menu: they are released with it, or every rebuild leaked them (I259). Found with
        # findChildren, not through the actions' wrappers (a wrapper GC'd by Python deletes
        # the C++ menu; see the audit_sweep harness notes)
        old = self.m_events.findChildren(QMenu, "", Qt.FindDirectChildrenOnly)
        self.m_events.clear()
        for sub in old:
            sub.deleteLater()
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
        """Make `pid` THE selection (G66): ClearAndSelect -- a bare `setCurrentRow` with the
        signals blocked only replaced the LAST selection operation (P2 + P3 + P4 selected, P1
        clicked -> P1 + P2 + P3) -- and the Track button follows, since T tracks exactly what is
        selected (it kept the stale verdict, and a semi-automatic F trusted it)."""
        self.selected = pid
        self.layers.select_only([pid], current=pid)
        self._sync_s_target()
        self._refresh_overlay()
        self._refresh_companions()          # the same landmark is ringed in the other cameras (G22)
        self._update_track_button()

    def _on_point_selection_changed(self):
        """The LAYERS selection changed (a click, Ctrl+click, Ctrl+A, ...): `self.selected` -- the
        point a plain click on the video places and Delete acts on -- stays one of the SELECTED
        point rows, or None when none is selected (G80: Ctrl+click-deselecting the current row left
        it on a point that was not highlighted, and Ctrl+A left it None, so a click on the video said
        that no point is selected); the S tool's animal follows the selection (G154); then the Track
        button, which follows the selection (G61)."""
        s = self.session
        if s is not None:
            sel = self._selected_pids()
            want = self.selected
            cur = self.layers.current_pid()
            if not sel:
                want = None
            elif cur in sel:
                # the row just clicked (Ctrl+click adds it): Qt moves the current row BEFORE the selection,
                # so `_on_layer_current` saw it unselected and left `selected` behind (G80)
                want = cur
            elif want is None or want not in sel:
                want = sel[0]
            target_was = s.active_seg
            self._sync_s_target()
            if want != self.selected or s.active_seg != target_was:
                self.selected = want
                self._epi_probe = None
                self._refresh_overlay()
                self._refresh_companions()
        self._update_track_button()

    def _on_layer_current(self, cur, _prev=None):
        """The current LAYERS row moved to a point: it is the one a click on the video places."""
        kind = self.layers.kind_of(cur)
        if not kind or kind[0] != "point":
            return
        row = kind[1]
        if not cur.isSelected():
            return                               # a Ctrl+click that deselected it (G80)
        self.selected = row
        self._epi_probe = None
        self._refresh_overlay()
        self._refresh_companions()          # the same landmark is ringed in the other cameras
        s = self.session
        if s is not None and row < s.n_points and not s.tracked[self.current, row]:
            self.statusBar().showMessage(
                (f"{s.points[row].name} is derived from the silhouette: it fills in when you track "
                 "its animal (it cannot be placed by hand)" if s.points[row].derived else
                 f"{s.points[row].name} has no position on this frame — click on the video "
                 "to place it here and continue the same point"), 6000)
    def _set_undo_point(self, snap, extra: dict | None = None) -> None:
        """Make `snap` the undo point and enable Ctrl+Z (R10): for an edit that took its snapshot
        BEFORE it knew whether it would change anything (a fill that found nothing to fill takes no
        undo step), or whose snapshot is not the working camera's as it stands -- `_begin_edit` is
        the one for an edit that always happens. `extra`: the other cameras' snapshots
        {view: Snapshot} that belong to the same step."""
        self._undo_snap = snap                   # (setting it forgets the extras)
        if extra:
            self._undo_extra.update(extra)
        self.act_undo.setEnabled(True)

    def _set_run_undo_point(self, snaps: dict) -> None:
        """One Ctrl+Z for a run that went through several cameras: `snaps` = {view: the snapshot
        taken before the run}; the working camera's is the undo point (a fresh one when it ran
        nothing), the others' ride along (R10; `_multi_finish`, `_passes_finish`, the re-track's
        Keep)."""
        p = self.project
        self._set_undo_point(snaps.get(p.active, self.session.snapshot()),
                             {v: sn for v, sn in snaps.items() if v != p.active})

    def _begin_edit(self, names=None, animals=None) -> None:
        """The undo point BEFORE an edit (I123, G68): the working camera's snapshot, plus -- for an
        edit made by landmark name, which changes every camera that has it (G19), or to an ANIMAL
        (`animals`, by name: its skeleton, hold or name, which `Project.share_animal` /
        `rename_segment` change in every camera, G153, G162) -- each of those cameras' (`_undo_extra`).
        Not while a run is live: its pre-run snapshot is the undo point."""
        s = self.session
        if s is None or self.state != READY:
            return
        extra = {}
        p = self.project
        if (names or animals) and p is not None and p.n_views > 1:
            for v in p.others():
                sv = p.sessions[v]
                if (any(sv.pid_by_name(n) is not None for n in names or ())
                        or any(sv.segment_index(a) is not None for a in animals or ())):
                    extra[v] = sv.snapshot()
        self._undo_snap = s.snapshot()           # (setting it forgets the extras)
        self._undo_extra.update(extra)
        self.act_undo.setEnabled(True)

    def _on_place(self, pid: int, x: float, y: float):
        """Ctrl+click / N-continue: a manual correction at the current frame (a marker is no longer
        dragged since G59)."""
        if self.session is None or pid >= self.session.n_points or self.state != READY:
            return
        if self.session.points[pid].derived:        # the same refusal as a plain click (I69)
            self._refresh_overlay()
            self.statusBar().showMessage(
                f"{self.session.points[pid].name} is derived from the silhouette — to place it by hand, "
                "right-click it → Data source → Track by appearance", 6000)
            return
        # a Ctrl+click / N-continue is ONE undo step, like a plain click:
        # without it Ctrl+Z threw away the whole previous tracking run (I64)
        self._begin_edit()
        corrected = bool(self.session.tracked[self.current, pid] and not self.session.manual[self.current, pid])
        self.session.set_position(self.current, pid, x, y)
        if corrected:
            self._hint_corrections(pid)
        self._refresh_overlay()
        self._update_track_button()
        name = self.session.points[pid].name
        rmse = self._residual_sentence(name)                   # (G28)
        self.statusBar().showMessage(
            (rmse + " — " if rmse else "")
            + f"{name} set to ({x:.1f}, {y:.1f}) at frame {self.current} — press Track to "
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
            # (G21) the notice goes ON the video: in the status bar alone it was missed,
            # and the look-here cross this click used to draw was taken for a point
            # that "changed into a crosshair" and never reached the POINTS list
            multi = self._guides_ready()
            self.toast.show_message(
                "Nothing was placed: no point is selected. Press <b>N</b> (Point) and click to add a point"
                + (" (it appears in every camera's list)" if multi else "")
                + ", or pick one in LAYERS and click where it is."
                + (" <b>Alt+click</b> shows where a spot can be in the other cameras." if multi else ""),
                "info", 7000)
            return
        if s.points[pid].derived:
            self.statusBar().showMessage(
                f"{s.points[pid].name} is derived from the silhouette — to annotate it by hand, "
                "right-click it → Data source → Track by appearance", 6000)
            return
        x, y = self._snap_to_prediction(pid, x, y)
        self._begin_edit()               # Ctrl+Z takes the click back
        corrected = bool(s.tracked[self.current, pid] and not s.manual[self.current, pid])
        s.set_position(self.current, pid, x, y)
        if corrected:
            self._hint_corrections(pid)
        self._refresh_overlay()
        self._update_track_button()
        n = int(s.manual[:, pid].sum())
        plural = "s" if n != 1 else ""
        rmse = self._residual_sentence(s.points[pid].name)      # (G28)
        self.statusBar().showMessage(
            (rmse + " — " if rmse else "")
            + f"{s.points[pid].name} annotated at ({x:.1f}, {y:.1f}) on frame {self.current} "
            f"— {n} hand-placed frame{plural} (Shift+< / Shift+> jump to first / last; "
            "Ctrl+Z undoes)", 8000 if rmse else 6000)

    def _goto_clicked_frame(self, last: bool):
        """Shift+> / Shift+<: jump to the LAST / FIRST frame the selected point
        has data on — tracked or hand-placed, whichever the point actually has
        (on an ordinary tracked point the hand-placed-only rule looked like a
        dead key). With no point selected it walks the
        silhouettes of the highlighted segments (else every segment's, G151g) instead. `,` / `.` still step between hand-placed
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
        # (G151g) the highlighted segments' silhouettes, else every segment's (not the active one's)
        ks = self._segments_for_view()
        frames = (np.unique(np.concatenate([s.mask_frames(k) for k in ks])) if ks
                  else np.zeros(0, np.int64))
        if len(frames):
            name = s.segments[ks[0]].name if len(ks) == 1 else "The segments"
            self._goto(int(frames[-1] if last else frames[0]))
            self.statusBar().showMessage(
                f"{name}: {which} frame with a silhouette ({len(frames):,} frames). "
                "Select a point in the list to walk that point instead", 5000)
            return
        self.statusBar().showMessage(
            "Nothing to jump to: select a point in LAYERS (Shift+< / Shift+> go to its "
            "first / last frame), or outline an animal with S and Track", 6000)

    def _on_delete(self, pid: int):
        if self.session is None or pid >= self.session.n_points:
            return
        self.canvas.cancel_gesture()  # a gesture in progress would hold a stale pid
        name = self.session.points[pid].name
        n_tracked = int(self.session.tracked[:, pid].sum())
        n_other, cams = self._landmark_data_elsewhere([name])
        also = (f"\n\nIt is also removed from {', '.join(cams)} (every camera shares one list of points), "
                f"with {n_other} tracked frame(s) there." if cams else "")
        if (n_tracked > 1 or n_other) and QMessageBox.question(
                self, "Delete point",
                f"Delete {name} and its {n_tracked} tracked frames?{also}\n\nCtrl+Z restores it.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        # one undo step: without it Ctrl+Z restored an OLDER snapshot and took
        # back the last tracking run instead (I71); the other cameras that hold the name join it
        self._begin_edit([name])
        self.session.remove_point(pid)
        self._remove_landmarks_elsewhere([name])
        self.timeline.clear_selection()   # its lane rows would now point elsewhere
        if self.selected is not None:
            if self.selected == pid:
                self.selected = None
            elif self.selected > pid:
                self.selected -= 1
        self._refresh_point_list()
        self._refresh_overlay()
        self._apply_state()

    def _landmark_data_elsewhere(self, names: list[str]) -> tuple[int, list[str]]:
        """(tracked frames, camera names) the landmarks `names` have in the OTHER
        cameras -- what deleting them there too removes (G19)."""
        p = self.project
        if p is None or p.n_views < 2:
            return 0, []
        n, cams = 0, []
        for v in p.others():
            sv = p.sessions[v]
            js = [j for j in (sv.pid_by_name(nm) for nm in names) if j is not None]
            if js:
                cams.append(p.name(v))
                n += sum(int(sv.tracked[:, j].sum()) for j in js)
        return n, cams

    def _remove_landmarks_elsewhere(self, names: list[str]) -> None:
        """A deleted landmark leaves every camera (one shared list, G19); the other
        cameras' copies join the undo step the caller has just set."""
        p = self.project
        if p is None or p.n_views < 2:
            return
        for v in p.others():
            sv = p.sessions[v]
            js = sorted((j for j in (sv.pid_by_name(nm) for nm in names) if j is not None), reverse=True)
            if js:
                self._undo_extra.setdefault(v, sv.snapshot())
                for j in js:
                    sv.remove_point(j)
        self._refresh_companions()

    def _animal_item(self, k: int):
        """Animal k's LAYERS row (None when it has none)."""
        return self.layers.animal_item(k)

    def _selected_pids(self) -> list[int]:
        """The point rows selected in LAYERS (Ctrl/Shift+click; not the points of a selected animal
        row -- `_run_scope` adds those for a run)."""
        return self.layers.selected_pids()

    def _delete_selected(self):
        """Delete key: clear the timeline's selected frame window — the lanes the
        marquee covered decide whether that is the segmentation, points, or both
        — else delete the selected point(s)."""
        if self.state != READY or self.session is None:
            return
        if self.timeline.request_delete_selection():
            return
        pids = self._selected_pids()
        if self._selected_segments():
            self._delete_layers()                # animals selected in LAYERS (G154)
        elif len(pids) > 1:
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
        gone = [s.points[p].name for p in pids]
        n_other, cams = self._landmark_data_elsewhere(gone)
        also = (f"\n\nThey are also removed from {', '.join(cams)} (every camera shares one list of "
                f"points), with {n_other} tracked frame(s) there." if cams else "")
        if QMessageBox.question(
                self, "Delete points",
                f"Delete {len(pids)} points ({names}) and their {n_cells} tracked "
                f"frames?{also}\n\nCtrl+Z restores them.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._begin_edit(gone)
        self.canvas.cancel_gesture()
        for pid in sorted(pids, reverse=True):  # descending keeps indices valid
            s.remove_point(pid)
        self._remove_landmarks_elsewhere(gone)
        self.timeline.clear_selection()   # its lane rows would now point elsewhere
        self.selected = None
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

    def _resolve_window_segs(self, f0: int, f1: int) -> list[int] | None:
        """Whose silhouettes a window clear takes when the drag covered no segment lane (G151h, the
        rule `_resolve_window_pids` has for points): the highlighted rows, else the only segment, else
        -- after a question -- every segment. None = the user backed out."""
        s = self.session
        sel = self._selected_segments()
        if sel or s.n_segments <= 1:
            return sel or list(range(s.n_segments))
        if QMessageBox.question(
                self, "Clear every animal's silhouettes?",
                f"The drag named no silhouette lane and no animal is selected in LAYERS — clear the silhouettes "
                f"of ALL {s.n_segments} animals ({', '.join(s.segment_names())}) in frames {f0}–{f1}?\n\n"
                "Ctrl+Z restores them.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return None
        return list(range(s.n_segments))

    def _clear_window(self, f0: int, f1: int, pids, do_points: bool, do_masks: bool, segs=None):
        """Core of the timeline-window deletes: tracked points and / or silhouettes inside [f0, f1],
        as ONE undoable step. The points themselves survive — only that stretch of their tracks is
        removed. `segs` = the segments whose silhouettes go (G149; None / empty = see
        `_resolve_window_segs`, G151h)."""
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
        segs = [k for k in (segs or []) if 0 <= k < s.n_segments]
        if do_masks and not segs:
            segs = self._resolve_window_segs(f0, f1)
            if segs is None:
                return                      # confirmation declined
        if not use and not do_masks:
            self.statusBar().showMessage(
                "Nothing to clear in that window — Shift+drag across the lanes you "
                "want to delete (the segment lane clears silhouettes)", 6000)
            return
        # (G82) count first: a clear that removes nothing takes no snapshot -- it used to replace
        # the undo point, so the last tracking run could no longer be undone -- and says so
        lo, hi = max(0, min(f0, f1)), min(s.n_frames - 1, max(f0, f1))
        would_pts = int(s.tracked[lo:hi + 1][:, use].sum()) if use else 0
        would_msk = sum(int((s.seg_masks[k].area[lo:hi + 1] > 0).sum()) for k in segs) if do_masks else 0
        if not would_pts and not would_msk:
            self.statusBar().showMessage(
                f"Nothing to clear in frames {f0}–{f1}: "
                + ("the selected points have no position there" if use else "there is no silhouette there")
                + (" and no silhouette" if use and do_masks else "") + " — nothing was changed", 6000)
            return None
        self._begin_edit()
        n_pts = s.clear_window(use, f0, f1) if use else 0
        n_msk = sum(s.clear_masks(f0, f1, k) for k in segs) if do_masks else 0
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
        return n_pts, n_msk

    def _clear_tracked_window(self, f0: int, f1: int, pids=None):
        """Blank the tracked data inside [f0, f1] (timeline clear_requested)."""
        self._clear_window(f0, f1, pids, do_points=True, do_masks=False)

    def _on_clear_frame(self, pid: int) -> None:
        """Right click on a marker (G59): that point -- now the selected one -- loses
        its position on THIS frame only; one Ctrl+Z step."""
        s = self.session
        if s is None or self.state != READY or not (0 <= pid < s.n_points):
            return
        sel = self._selected_pids()
        if pid in sel and len(sel) > 1:
            # one of several selected points: all of them on this frame, the selection kept (G64).
            # The POINTS only, never the silhouette, even with the segment's row selected (G84,
            # owner): the menu entries keep that option
            self._multi_clear(sel, self.current, self.current, masks=False)
            return
        self._on_select(pid)
        name = s.points[pid].name
        if not s.tracked[self.current, pid]:
            self.statusBar().showMessage(f"{name} has no position on frame {self.current}", 4000)
            return
        self._clear_tracked_window(self.current, self.current, [pid])
        self.statusBar().showMessage(
            f"{name} cleared on frame {self.current} (this frame only; Ctrl+Z brings it back). "
            "A long right press opens its menu", 6000)

    def _clear_window_both(self, f0: int, f1: int, pids=None):
        """Points AND silhouettes (of the segment lanes the marquee covered, G149) inside [f0, f1],
        as one undo step."""
        self._clear_window(f0, f1, pids, do_points=True, do_masks=True, segs=self.timeline.sel_segs)

    def _select_all_tracked(self) -> None:
        """Ctrl+A: every point and every animal -- Track covers only what is selected (G61)."""
        s = self.session
        if s is None:
            return
        self.layers.select_only(range(s.n_points), range(s.n_segments))
        self._on_point_selection_changed()      # a current row too (G80) + the Track button
        self.statusBar().showMessage("Everything selected: Track tracks every point"
                                     + (" and every silhouette" if s.segments else ""), 5000)

    def _deselect(self):
        if self._segment_selected() and not self._selected_pids():
            self.layers.select_only()
            self._sync_s_target()
            self._update_track_button()
        if self.selected is not None or self._selected_pids():
            self.selected = None
            self.layers.select_only()
            self.layers.setCurrentItem(None)
            self._sync_s_target()
            self._refresh_overlay()
            self._refresh_companions()      # the other cameras drop the selection ring too (G22)
            self._update_track_button()     # nothing selected = nothing to track (G61)
            self.statusBar().showMessage("Selection cleared — a plain click on the video now edits nothing; "
                                         "press N, then click, to add a new point", 5000)

    def _apply_rename(self, pid: int, desired: str) -> str:
        """Rename a point with collision protection; explains the auto-suffix
        in the status bar. Returns the name actually applied. Single collision
        UX for both the dialog and the in-list editor. With several cameras the
        landmark is renamed in all of them, to a name free in every camera: they
        are joined by name for 3D (G19)."""
        from kinetrace.session import starts_formula
        if starts_formula(desired):
            # a spreadsheet would run it as a formula when an export is opened (M7, owner decision)
            self.toast.show_message(f"“{desired}” was not used: a name may not start with = + - or @, "
                                    "because a spreadsheet opening an export would run it as a formula.",
                                    "warn", 8000)
            return self.session.points[pid].name
        p = self.project
        old = self.session.points[pid].name
        if desired == old:
            return old
        # (G68) its own undo step, in every camera it renames: without one the next Ctrl+Z restored
        # a snapshot from before an EARLIER edit and put the old name back in the working camera
        # only, which split the landmark across the cameras (3D joins by name)
        self._begin_edit([old])
        if p is not None and p.n_views > 1:
            applied = p.rename_landmark(old, desired)
            self._refresh_companions()
        else:
            applied = self.session.rename_point(pid, desired)
        it = self.layers.point_item(pid)
        if it is not None and applied != old:
            self.layers.blockSignals(True)
            it.setData(0, ROLE_NAME, applied)     # LAYERS carries its selection by name (G67)
            self.layers.blockSignals(False)
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

    def _point_list_menu_for(self, pid: int, gpos):
        """Same menu as right-clicking the marker — findable even when markers
        overlap or the point currently has no position on screen."""
        if self.session is None or self.state != READY or not (0 <= pid < self.session.n_points):
            return
        if self._maybe_multi_menu(pid, gpos):
            return                              # several selected: the menu for all of them (G64)
        menu, acts = self.canvas._build_context_menu(pid)
        chosen = menu.exec(gpos)
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

    def _maybe_multi_menu(self, pid: int, global_pos) -> bool:
        """Right click (POINTS) / long right press (video) on one of SEVERAL selected
        points: the entries that act on all of them (G64). False = one point's menu."""
        sel = self._selected_pids()
        if self.session is None or self.state != READY or pid not in sel or len(sel) < 2:
            return False
        menu, acts = self._build_multi_menu(sel)
        chosen = menu.exec(global_pos)
        menu.deleteLater()
        self._multi_menu_action(chosen, acts, sel)
        return True

    def _build_multi_menu(self, sel: list[int]):
        """The menu for several selected points (and the segment when its row is
        selected too): every entry acts on all of them, each as one Ctrl+Z step."""
        s = self.session
        seg = self._segment_selected() and s.animal is not None
        n = len(sel)
        who = f"the {n} selected points" + (" and the segment" if seg else "")
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        acts: dict = {}
        head = menu.addAction(f"{n} points selected" + (" + the segment" if seg else ""))
        head.setEnabled(False)
        menu.addSeparator()
        acts["track"] = menu.addAction(f"Track {who} from frame {self.current}")
        acts["track"].setToolTip("Only these: each point with its own tracker (T does the same)")
        menu.addSeparator()
        acts["clear_here"] = menu.addAction(f"Clear {who} on frame {self.current}")
        acts["clear_here"].setToolTip("This frame only; the points stay. One Ctrl+Z step "
                                      "(a right click on one of their markers does the same, for the points)")
        # nothing selected has data here: nothing to clear (G82)
        acts["clear_here"].setEnabled(any(bool(s.tracked[self.current, q]) for q in sel)
                                      or bool(seg and s.masks is not None and s.masks.has(self.current)))
        rng = self.timeline.sel_range
        acts["clear_window"] = menu.addAction(f"Clear {who} in frames {rng[0]}\u2013{rng[1]}" if rng
                                              else "Clear them in the selected frame window")
        acts["clear_window"].setEnabled(rng is not None)
        acts["clear_window"].setToolTip("Shift+drag across the timeline to select a frame window first")
        acts["clear_all"] = menu.addAction(f"Clear the whole tracks of the {n} points (keep the points)")
        acts["clear_all"].setToolTip("Every frame of their tracks; the points stay in the list, ready to be "
                                     "placed again. One Ctrl+Z step")
        acts["delete"] = menu.addAction(f"Delete the {n} points")
        acts["delete"].setToolTip("Remove them from the project (asks first; Ctrl+Z brings them back). The "
                                  "segment has its own row menu for removing it")
        menu.addSeparator()
        all_hidden = all(s.occluded[self.current, q] for q in sel)
        acts["hidden"] = menu.addAction(("Unmark them hidden" if all_hidden else "Mark them hidden")
                                        + f" on frame {self.current}")
        acts["hidden"].setToolTip("Hidden = kept, but left out of exports and 3D (Shift+X for one point)")
        all_shown = all(s.points[q].display for q in sel)
        acts["display"] = menu.addAction("Hide them on the video" if all_shown else "Show them on the video")
        plain = [q for q in sel if not s.points[q].derived and not s.points[q].is_ball]
        acts["tracker"] = self._tracker_submenu(
            menu, plain, "Give every selected point this tracker (ball markers and silhouette "
                         "landmarks keep theirs; Moving spot follows single points, not regions)")
        acts["fill"] = menu.addAction("Fill their gaps between hand placements")
        acts["fill"].setToolTip("For each point with two or more hand-placed frames: a smooth curve through "
                                "them fills the empty frames between (one Ctrl+Z step)")
        # (G153, G157) their animal; two points of one animal: a bone between them
        menu.addSeparator()
        acts["move_to"] = self._move_submenu(menu, list(sel))
        two = len(sel) == 2
        why = s.bone_problem(sel[0], sel[1]) if two else "Select exactly two points of one animal."
        has = two and why is None and s.has_bone(sel[0], sel[1])
        acts["bone"] = menu.addAction("Remove the bone between them" if has else "Connect them with a bone")
        acts["bone"].setEnabled(two and why is None)
        acts["bone"].setToolTip("Bones are drawn on the video and in the 3D view, and saved with the animal's "
                                "skeleton" if why is None else why)
        # (G147) two points selected: the identity tools
        menu.addSeparator()
        acts["swap_two"] = menu.addAction("Swap these two points…" if two else "Swap two points… (select two)")
        acts["swap_two"].setEnabled(two and s.point_tool_problem(sel[0], sel[1]) is None)
        acts["swap_two"].setToolTip("They exchange their data over the frames you choose: where the tracker "
                                    "swapped them (left and right foot). Edit → Point Tools… has the other tools.")
        return menu, acts

    TRACKER_HINTS = {"alltracker": " — a visible shape (an animal, an object)",
                     "cotracker3": " — faster; high-contrast markers",
                     "spot": " — a target small enough to be one point"}

    def _tracker_submenu(self, menu, pids: list[int], tip: str) -> dict:
        """The Tracker submenu of both point menus (R13): one entry per tracker, ticked when every
        point in `pids` (the plain, appearance-tracked ones) has it, {action: key}. Moving spot is
        offered for single points only -- a run leaves a region out (G124)."""
        s = self.session
        sub = menu.addMenu("Tracker")
        out: dict = {}
        point_kind = any(s.points[q].kind == "point" for q in pids)
        for key, name in self.TRACKER_NAMES.items():
            a = sub.addAction(name + self.TRACKER_HINTS[key])
            a.setCheckable(True)
            a.setChecked(bool(pids) and all(self._tracker_of(q) == key for q in pids))
            a.setEnabled(bool(pids) and (key != "alltracker" or _alltracker_available())
                         and (key != "spot" or point_kind))
            out[a] = key
        sub.menuAction().setEnabled(bool(pids))
        sub.menuAction().setToolTip(tip)
        return out

    def _multi_menu_action(self, chosen, acts: dict, sel: list[int]) -> None:
        s = self.session
        if chosen is None or s is None:
            return
        if chosen is acts["track"]:
            self._toggle_tracking()
        elif chosen is acts["clear_here"]:
            self._multi_clear(sel, self.current, self.current)
        elif chosen is acts["clear_window"] and self.timeline.sel_range is not None:
            self._multi_clear(sel, *self.timeline.sel_range)
        elif chosen is acts["clear_all"]:
            self._clear_whole_tracks(sel, "Clear tracks",
                                     f"Clear every frame of the tracks of {len(sel)} points? The points stay in the "
                                     "list. Ctrl+Z brings the tracks back.")
        elif chosen is acts["delete"]:
            self._delete_points(list(sel))
        elif chosen is acts["hidden"]:
            on = not all(s.occluded[self.current, q] for q in sel)
            self._occlude_window(self.current, self.current, list(sel), on)
        elif chosen is acts["display"]:
            show = not all(s.points[q].display for q in sel)
            self._begin_edit()                    # one Ctrl+Z step (G68)
            for q in sel:
                s.points[q].display = show
            s.dirty = True
            self._refresh_point_list()
            self._refresh_overlay()
        elif chosen in acts["tracker"]:
            self._set_tracker(sel, acts["tracker"][chosen])
        elif chosen is acts.get("swap_two") and len(sel) == 2:
            self._point_tools(sel[0], sel[1], "swap")
        elif chosen in acts.get("move_to", {}):
            self._move_points(list(sel), acts["move_to"][chosen])
        elif chosen is acts.get("bone") and len(sel) == 2:
            self._toggle_bone(sel[0], sel[1])
        elif chosen is acts["fill"]:
            snap, done = self._interpolate_pids(
                [q for q in sel if not s.points[q].derived and len(s.manual_frames(q)) >= 2], replace=False)
            n = sum(k for k, _span in done)
            if n:
                self._set_undo_point(snap)
                self._refresh_overlay()
                self.timeline.refresh()
            self.statusBar().showMessage(f"{n} frame(s) filled from the curves through the hand placements"
                                         + (" — Ctrl+Z undoes" if n else " (nothing to fill)"), 7000)

    def _move_submenu(self, menu, pids: list[int]) -> dict:
        """'Move to' with every animal and Scene (G153): {action: animal index or None}."""
        s = self.session
        sub = menu.addMenu("Move to")
        out: dict = {}
        owners = {s.segment_of(q) for q in pids}
        for k, a in enumerate(s.segments):
            act = sub.addAction(a.name)
            act.setEnabled(owners != {k})
            out[act] = k
        act = sub.addAction("Scene (no animal)")
        act.setEnabled(owners != {None} and not any(s.points[q].derived for q in pids))
        act.setToolTip("A landmark derived from a silhouette needs its animal")
        out[act] = None
        sub.menuAction().setEnabled(bool(s.segments))
        sub.menuAction().setToolTip("Give the point(s) to another animal or to Scene — renamed "
                                    "\u201c<animal> <part>\u201d in every camera; drag in LAYERS does the same"
                                    if s.segments else "Make an animal first (＋ Animal in LAYERS)")
        return out

    def _toggle_bone(self, a: int, b: int) -> None:
        """Connect two points of one animal with a bone, or remove the one there is (G157): one Ctrl+Z
        step, in every camera (the skeleton belongs to the animal)."""
        s = self.session
        why = s.bone_problem(a, b)
        if why:
            self.statusBar().showMessage(why, 6000)
            return
        k = s.segment_of(a)
        self._begin_edit(animals=[s.segments[k].name])    # every camera share_animal changes
        had = s.has_bone(a, b)
        (s.remove_bone if had else s.connect_bone)(a, b)
        if self.project is not None:
            self.project.share_animal(s.segments[k].name)
        if not self.act_show_bones.isChecked() and not had:
            self.act_show_bones.setChecked(True)
        self._refresh_overlay()
        self._refresh_companions()
        self.statusBar().showMessage(
            f"Bone {'removed' if had else 'drawn'} between {s.part_name(a)} and {s.part_name(b)} "
            f"({s.segments[k].name}) — Ctrl+Z undoes it", 6000)

    def _set_head(self, pid: int) -> None:
        """A point becomes its animal's head (G157): the silhouette's midline starts there."""
        s = self.session
        k = s.segment_of(pid)
        if k is None or s.points[pid].derived or s.points[pid].is_ball:      # `set_head`'s refusal: no undo step
            self.statusBar().showMessage("Only a point of an animal, tracked by appearance, can be its head", 6000)
            return
        self._begin_edit(animals=[s.segments[k].name])    # every camera share_animal changes
        s.set_head(pid)
        if self.project is not None:
            self.project.share_animal(s.segments[k].name)
        self._refresh_point_list()
        self.statusBar().showMessage(f"{s.part_name(pid)} is now {s.segments[k].name}'s head: its midline (tail tip, "
                                     "midline points) is measured from it — Ctrl+Z undoes it", 7000)

    def _multi_clear(self, sel, f0: int, f1: int, masks: bool | None = None) -> None:
        """Several selected points (and the segment, when its row is selected and `masks` is not
        False) cleared on frames f0..f1, as one undo step (G64). The message counts what really
        had data (G82); nothing to clear = no undo point, said by `_clear_window`."""
        s = self.session
        seg = (self._segment_selected() if masks is None else bool(masks)) and s.animal is not None
        lo, hi = max(0, min(f0, f1)), min(s.n_frames - 1, max(f0, f1))
        n_pts = sum(1 for q in sel if q < s.n_points and s.tracked[lo:hi + 1, q].any())
        n_seg = bool(seg and any((s.seg_masks[k].area[lo:hi + 1] > 0).any() for k in self._selected_segments()
                                 if k < s.n_segments))
        if self._clear_window(f0, f1, list(sel), do_points=True, do_masks=seg, segs=self._selected_segments()) is None:
            return
        self.statusBar().showMessage(
            f"{n_pts} of the {len(sel)} points" + (" and the segment" if n_seg else "")
            + (f" cleared on frame {f0}" if f0 == f1 else f" cleared in frames {f0}-{f1}")
            + " — Ctrl+Z brings them back", 6000)

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
        # (G153, G157) its animal: move it, make it the head
        menu.addSeparator()
        acts["move_to"] = self._move_submenu(menu, [pid])
        k = s.segment_of(pid)
        a_head = menu.addAction(f"Use it as {s.segments[k].name}'s head (the midline's anchor)" if k is not None
                                else "Use it as its animal's head")
        a_head.setCheckable(True)
        a_head.setChecked(k is not None and s.head_pid(k) == pid)
        a_head.setEnabled(k is not None and not s.points[pid].derived and not s.points[pid].is_ball)
        a_head.setToolTip("The silhouette's midline (tail tip, midline points) is measured from this point"
                          if k is not None else "Only a point of an animal can be its head: drag it onto the animal")
        acts["set_head"] = a_head
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
                          "while it is selected)")
        acts["interp_fill"] = a_fill
        a_repl = menu.addAction("Replace everything between its hand placements with that curve")
        a_repl.setEnabled(len(mf) >= 2 and not s.points[pid].derived)
        a_repl.setToolTip("The same curve, but it also overwrites tracked frames between the placements - for a "
                          "stretch where the tracker drifted and you have re-placed the part by hand at both "
                          "ends (and anywhere in between). Ctrl+Z undoes it.")
        acts["interp_replace"] = a_repl
        # (G147) which point is which: swap, move, fill, split
        menu.addSeparator()
        why = s.point_tool_problem(pid)
        a_tools = menu.addAction("Swap / move / fill with another point…")
        a_tools.setEnabled(why is None and s.n_points > 1)
        a_tools.setToolTip("Point Tools with this point as A: swap it with another point, give its data to another, "
                           "or fill another point's empty frames from it" if why is None else f"Not for this point: {why}")
        acts["point_tools"] = a_tools
        a_split = menu.addAction(f"Split it into a new point from frame {self.current}")
        a_split.setEnabled(why is None and bool(s.tracked[self.current:, pid].any()))
        a_split.setToolTip(f"From frame {self.current} on, {name}'s data becomes a new point named "
                           f"\u201c{name} (2)\u201d (rename it afterwards); {name} ends on the frame before. "
                           "One Ctrl+Z step." if why is None else f"Not for this point: {why}")
        acts["split_here"] = a_split
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
        pr = self._prediction(self.project.active, s.points[pid].name, self.current) \
            if self._guides_ready() else None
        a_pred = menu.addAction("Place it where the other cameras put it (◇)   A")
        a_pred.setEnabled(pr is not None and pr["why"] is None and not s.points[pid].derived)
        a_pred.setToolTip(
            "Two or more other cameras have this landmark at this instant: their rays meet in 3D, and the ◇ is "
            "that point seen from this camera. This places the landmark exactly there (hand-placed, one "
            "Ctrl+Z step); Track re-seeds from it." if pr is not None and pr["why"] is None else
            f"No ◇ here: {pr['why']}" if pr is not None else
            "Needs this landmark placed in at least two other calibrated cameras at this instant")
        acts["accept_prediction"] = a_pred
        # which point model suits this point (G57)
        menu.addSeparator()
        q = s.points[pid]
        a_test = menu.addAction(f"Test the point models on its clicks…  ({len(manual)} hand-placed frame"
                                f"{'' if len(manual) == 1 else 's'})")
        a_test.setEnabled(q.kind == "point" and not q.derived and not q.is_ball)
        a_test.setToolTip("Compares AllTracker, CoTracker3 and Moving spot on the frames where you placed this "
                          "point by hand (at least 20 in a row) and recommends one")
        acts["test_models"] = a_test
        # this point's own tracker (G62): applies to the selected points when it is one of them
        acts["tracker"] = self._tracker_submenu(
            menu, [] if (q.derived or q.is_ball) else [pid],
            "Which tracker follows this point (shown as AT / CT / MS on its row). With several points "
            "selected, all of them change.")

    def _point_menu_extra_action(self, chosen, acts: dict, pid: int) -> bool:
        """Apply one of the frame-aware point entries. Returns True when it
        handled the choice, so the canvas leaves its own dispatch alone."""
        s = self.session
        if chosen is None or s is None or not (0 <= pid < s.n_points):
            return False
        name = s.points[pid].name
        if chosen in acts.get("move_to", {}):
            self._move_points([pid], acts["move_to"][chosen])
            return True
        if chosen is acts.get("set_head"):
            self._set_head(pid)
            return True
        if chosen is acts.get("test_models"):
            self._test_point_models(pid)
            return True
        if chosen is acts.get("point_tools"):
            self._point_tools(pid)
            return True
        if chosen is acts.get("split_here"):
            self._apply_point_tool(dict(op="split", a=pid, b=None, f0=self.current, f1=s.n_frames - 1, name=None))
            return True
        if chosen in acts.get("tracker", {}):
            # this menu is only shown for ONE point (several selected = the multi menu), so [pid]
            self._set_tracker([pid], acts["tracker"][chosen])
            return True
        if chosen is acts.get("snap_epipolar"):
            self._snap_to_epipolar(pid)
            return True
        if chosen is acts.get("accept_prediction"):
            self._accept_prediction(pid)
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
            self._clear_whole_tracks(
                [pid], "Clear track",
                f"Clear {name}'s positions on all {n:,} frames?\n\nThe point itself stays in "
                "the list, so you can place it again. Ctrl+Z undoes this.")
            return True
        return False

    def _point_tools(self, a: int | None = None, b: int | None = None, op: str = "swap") -> None:
        """Edit → Point Tools… (G147): the dialog, started on point `a` (the selected point when None)
        and `b`, then `_apply_point_tool` with what was chosen."""
        from kinetrace.pointedit import PointToolsDialog
        s = self.session
        if s is None or self.state != READY:
            return
        if s.n_points == 0:
            self.statusBar().showMessage("Point tools work on tracked points: add some first (N, then click)", 7000)
            return
        if a is None:
            a = self.selected if self.selected is not None and 0 <= self.selected < s.n_points else 0
        cam = self.project.name(self.project.active) if self.project is not None and self.project.n_views > 1 else ""
        dlg = PointToolsDialog(self, s, a, b, op, self.current, self.timeline.sel_range, cam)
        ok = dlg.exec() == QDialog.Accepted
        res = dlg.result
        dlg.deleteLater()
        if ok and res:
            self._apply_point_tool(res)

    def _apply_point_tool(self, r: dict) -> None:
        """Apply one point tool (`pointedit.OPS`) to the working camera as ONE undo step; a split's
        new point joins every camera's list (G19) in the same step. Nothing changed = no undo point."""
        s = self.session
        if s is None or self.state != READY:
            return
        op, a, b, f0, f1 = r["op"], int(r["a"]), r.get("b"), int(r["f0"]), int(r["f1"])
        why = s.point_tool_problem(a, None if b is None else int(b))
        if why is not None:
            self.statusBar().showMessage(f"Point tools: {why}", 7000)
            return
        an = s.points[a].name
        bn = s.points[int(b)].name if b is not None else ""
        snap = s.snapshot()
        new = None
        if op == "swap":
            n = s.swap_points(a, int(b), f0, f1)
            said = f"{an} and {bn} swapped on {n:,} frame(s) in {f0}\u2013{f1}"
        elif op == "move":
            n = s.move_point_data(a, int(b), f0, f1)
            said = f"{n:,} frame(s) of {an}'s data in {f0}\u2013{f1} moved to {bn}"
        elif op == "fill":
            n = s.fill_point_gaps(a, int(b), f0, f1)
            said = f"{bn}'s empty frames filled from {an} on {n:,} frame(s) in {f0}\u2013{f1}"
        else:
            new, n = s.split_point(a, f0, r.get("name"))
            said = (f"{an} split at frame {f0}: its {n:,} frame(s) from there on are now "
                    f"\u201c{s.points[new].name}\u201d (double-click the name to rename it)" if n else "")
        if not n:
            self.statusBar().showMessage("Point tools: nothing to change in those frames", 6000)
            return
        self._set_undo_point(snap)
        if new is not None:
            self._share_landmarks()              # the new name in every camera, in the same Ctrl+Z step
            self.selected = new
        self._refresh_point_list()
        self._refresh_overlay()
        self._refresh_companions()
        self.timeline.refresh()
        self._update_track_button()
        self._apply_state()
        self.statusBar().showMessage(said + " \u2014 Ctrl+Z undoes it", 9000)

    def _clear_whole_tracks(self, pids, title: str, text: str) -> None:
        """'Clear the whole track(s)?' of the single and the multi-point menu (R13): ask (No is the
        default), then clear every frame of the points' tracks, one undo step. Each menu words its
        own question."""
        s = self.session
        if QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) == QMessageBox.Yes:
            self._clear_window(0, s.n_frames - 1, list(pids), do_points=True, do_masks=False)

    def _interpolate_pids(self, pids, replace: bool):
        """The keyframe fill of the single and the multi-point menu (R13): a curve through each
        point's hand-placed frames fills the frames between (`replace`: also overwrites them),
        inside the timeline's selected window if there is one. Returns (the snapshot taken BEFORE,
        [(frames filled, span) per point]); the caller makes it the undo point only when it filled
        something."""
        s = self.session
        snap = s.snapshot()
        window = self.timeline.sel_range
        return snap, [s.interpolate_keyframes(q, replace=replace, window=window) for q in pids]

    def _interpolate_keyframes(self, pid: int, replace: bool) -> None:
        """Point menu: a curve through the hand-placed frames fills the gaps
        (or replaces everything) between them; one undo step."""
        s = self.session
        if s is None or self.state != READY or not (0 <= pid < s.n_points):
            return
        name = s.points[pid].name
        snap, done = self._interpolate_pids([pid], replace)
        n, span = done[0]
        if not n:
            self.statusBar().showMessage(
                f"{name}: nothing to fill — it needs at least two hand-placed frames"
                + (" with empty frames between them" if not replace else "")
                + (" inside the selected window" if self.timeline.sel_range else ""), 7000)
            return
        self._set_undo_point(snap)
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
        if self.session.points[pid].anchor != bool(on):
            self._begin_edit()                    # its own Ctrl+Z step (G68)
        self.session.points[pid].anchor = bool(on)
        self.session.dirty = True
        name = self.session.points[pid].name
        msg = (f"{name}: appearance lock ON — future runs snap back to how it "
               f"looked at its seed frame" if on
               else f"{name}: appearance lock off — plain tracking")
        self.statusBar().showMessage(msg, 6000)


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

    def _refresh_point_list(self, keep=None):
        """Rebuild LAYERS from the session (G154): the animals with their points, then Scene. A
        rebuild keeps the selection BY NAME (G61, G67: when a point is deleted the rows after it move
        up, and row numbers used to select the neighbours: P2 + P3 selected, P1 deleted -> P3 + P4,
        and T overwrote P4's track). `keep` = (point names, animal names) to select instead."""
        if getattr(self, "timeline", None) is not None and self.project is not None:
            self._update_disagreement()        # columns follow the point list
        s = self.session
        if keep is None:
            pk, ak = self.layers.selected_names()
            if getattr(self, "_point_list_project", None) is not self.project:
                pk, ak = set(), set()          # another video / project: nothing carries over by name
            cur_name = s.points[self.selected].name if (s is not None and self.selected is not None
                                                        and self.selected < s.n_points) else None
            if cur_name is not None and cur_name not in pk:
                pk, ak = {cur_name}, set()     # the current point is a NEW one: select just it
            # (G81: a Ctrl+A selection with no current point is kept)
            keep = (pk, ak)
        self._point_list_project = self.project
        # (G163) an explicit `keep` is the WHOLE selection (a new animal alone, the points just moved): the current
        # point is not added to it -- it named a second animal, and S then had no one animal to outline
        if s is not None and self.selected is not None and (self.selected >= s.n_points
                                                            or s.points[self.selected].name not in keep[0]):
            self.selected = None
        has = s.tracked.any(axis=0) if s is not None else None     # once per rebuild (simplify 2026-10-04)
        self.layers.rebuild(
            s, tag_of=(lambda q: "" if (s.points[q].derived or s.points[q].is_ball)
                       else self.TRACKER_TAGS.get(self._tracker_of(q, s), "")) if s is not None else None,
            style_point=(lambda it, q: self._style_point_item(it, q, bool(has[q])))
            if s is not None else None,
            keep=keep, current=self.selected)
        if s is not None:
            sel = self._selected_pids()
            if self.selected is None and sel:
                self.selected = sel[0]           # a selection always has a current point (G80)
                self.layers.select_only(sel, self._selected_segments(), current=self.selected)
            if self.selected is not None and self.selected not in sel:
                self.selected = sel[0] if sel else None
            self._sync_s_target()
            # (G158) the timeline gained lanes (an animal, a point): give it room for them once, as after
            # a skeleton template -- a timeline the user shrank is not grown again until more lanes come
            # A session seen for the first time (a project opened, another camera) only records its count:
            # its layout was just restored and must stand (verify_recovery's exact working state)
            # (counted from the SESSION: while a project opens the timeline still shows the previous one)
            n_rows = s.n_points + s.n_segments
            seen = self._tl_rows_seen
            if s in seen and n_rows > seen[s]:
                QTimer.singleShot(0, self._fit_timeline_height)
            seen[s] = n_rows
        else:
            self.animal_label.setText(self._animal_status_text())

    def _style_point_item(self, item, pid: int, has_any: bool) -> None:
        """A LAYERS point row's tooltip and dimming: dim while the point has no position in this camera
        (a landmark another camera made, or a cleared track)."""
        s = self.session
        meta = s.points[pid]
        item.setData(0, ROLE_HAS, has_any)
        item.setData(0, Qt.ForegroundRole, None if (meta.derived or has_any) else QColor(theme.TEXT_DIM))
        k = s.segment_of(pid)
        owner = f" ({s.segments[k].name})" if k is not None else " (Scene)"
        if meta.derived:
            item.setToolTip(0, f"{meta.name}: derived from its animal's silhouette ({meta.spec}) "
                               "— fills in when you track the animal with its silhouette (S)")
        elif not has_any:
            item.setToolTip(0, f"{meta.name}{owner}: not placed in this camera yet — select it and click it "
                               "on the video (with a calibration, on the dashed line from the other "
                               "cameras), then Track")
        elif meta.is_ball:
            item.setToolTip(0, f"{meta.name}{owner}: a ball marker — SAM outlines it each frame and the "
                               "fitted circle's centre is the point (its circle is drawn on the video)")
        else:
            t = self._tracker_of(pid, s)
            item.setToolTip(0, f"{meta.name}{owner}: tracked by {self.TRACKER_NAMES.get(t, t)}"
                               + ("" if meta.tracker else " (the project's default)")
                               + (", kept on its animal's silhouette" if self._held(s, pid) else "")
                               + " — right-click → Tracker to change it; drag it onto another animal to move it")

    def _restyle_point_rows(self) -> None:
        """Rows go stale after edits that do not rebuild the list (G126): a placeholder's first
        placement undims it, a cleared whole track dims it, a name that changed under the row (an
        undo) is put back. Cheap: one `any` over the bool array; not while a run is live."""
        s = self.session
        if s is None or self.state == TRACKING or self.layers.n_point_rows() != s.n_points:
            return
        has = s.tracked.any(axis=0)
        blocked = self.layers.blockSignals(True)
        try:
            for pid in range(s.n_points):
                item = self.layers.point_item(pid)
                if item is None:
                    continue
                if item.data(0, ROLE_NAME) != s.points[pid].name:
                    item.setText(0, s.part_name(pid))
                    item.setData(0, ROLE_NAME, s.points[pid].name)
                if item.data(0, ROLE_HAS) != bool(has[pid]):
                    self._style_point_item(item, pid, bool(has[pid]))
        finally:
            self.layers.blockSignals(blocked)
    # ------------------------------------------------ epipolar guides (3D)

    GUIDE_COLORS = [(255, 120, 200), (120, 220, 255), (255, 200, 90), (170, 255, 140),
                    (255, 140, 120), (200, 170, 255)]

    def _guide_depth(self, f: int):
        """A 3D point in front of the cameras for the guides' forward sign: this
        instant's reconstruction centroid, else calib.working_probe."""
        from kinetrace.calib import working_probe
        p = self.project
        r = p.reconstruction
        # the reference frame of this instant with `map_frame`'s tie rule, not floor(t + .5) (I258)
        k = p.reference_index(p.active, f) - r.t0 if r is not None else -1
        if r is not None and r.n_frames and 0 <= k < r.n_frames:
            # the row is the REFERENCE instant: comparing the working camera's own
            # frame number with the reference range raised IndexError (or wrapped
            # to another row) whenever the working camera is not camera 1 (I67)
            row = r.xyz[k]
            row = row[np.isfinite(row).all(axis=1)]
            if len(row):
                return row.mean(axis=0)
        return working_probe(p.calibration)

    def _cal_cam(self, v: int):
        """Camera `v`'s calibration, or None when the calibration does not cover
        it -- a camera added after calibrating (G25). Calibration cameras are in
        view order, and a camera added later is appended after them."""
        p = self.project
        cal = p.calibration if p is not None else None
        if cal is None or not (0 <= v < min(len(cal.cameras), p.n_views)):
            return None
        return cal.cameras[v]

    def _guides_ready(self) -> bool:
        """Two or more calibrated cameras: the guides and the ◇ predictions work
        among them, even when a camera added later is not calibrated (G25) -- the
        3D entries still need every camera (`_need_calibration`)."""
        p = self.project
        return (p is not None and p.n_views > 1 and p.calibration is not None
                and min(len(p.calibration.cameras), p.n_views) >= 2)

    def _epipolar_guides(self, pid: int) -> list:
        """[(polyline (M, 2) native px of the working camera, colour, camera name), ...]: where
        landmark `pid` of the working camera, as each OTHER camera sees it at this instant, can lie
        in its picture. Empty without a calibration, without the landmark, or when no other camera
        has it here."""
        s = self.session
        if not self._guides_ready() or s is None or not (0 <= pid < s.n_points):
            return []
        return self._guides_into(self.project.active, s.points[pid].name, self.current)

    def _observations(self, name: str, f: int, exclude: int) -> list:
        """[(view, calibration, raw (x, y)), ...]: the calibrated cameras other
        than `exclude` that have landmark `name` at the instant of the working
        camera's frame `f` (hidden cells left out). Read at the FRACTIONAL frame the way Reconstruct
        does (I253): between the two neighbouring frames when both have a position, else at the
        nearest frame -- with a sub-frame offset or a 2x camera the nearest frame is up to half a
        frame of motion off, and A stores the prediction as a hand placement."""
        p = self.project
        out = []
        for c in range(p.n_views):
            cal = self._cal_cam(c)
            if c == exclude or cal is None:
                continue
            sc = p.sessions[c]
            j = sc.pid_by_name(name)
            if j is None:
                continue
            uv = None
            fx = p.map_frame_exact(p.active, c, f)
            lo = int(np.floor(fx))
            a = fx - lo
            if 1e-9 < a < 1.0 - 1e-9:
                u0, u1 = self._cell_xy(sc, lo, j), self._cell_xy(sc, lo + 1, j)
                if u0 is not None and u1 is not None:
                    uv = (1.0 - a) * u0 + a * u1
            if uv is None:
                fc = p.map_frame(p.active, c, f)
                uv = self._cell_xy(sc, fc, j) if fc is not None else None
            if uv is not None:
                out.append((c, cal, uv))
        return out

    @staticmethod
    def _cell_xy(sc, fr: int, j: int):
        """Landmark `j`'s position at frame `fr` of session `sc`, or None (outside the video, no
        position, hidden, not finite)."""
        if not (0 <= fr < sc.n_frames) or not sc.exportable_at(fr, j):
            return None
        uv = sc.tracks[fr, j]
        return np.asarray(uv, np.float64) if np.isfinite(uv).all() else None

    def _guides_into(self, target: int, name: str, f: int, trusted: list | None = None,
                     obs: list | None = None, depth=None) -> list:
        """The lines in camera `target`, cut exactly at its picture's edges
        (I132); `trusted`, if given, receives each line's (M,) mask -- False
        where the lens model is guessing. `obs` / `depth`: what `_observations(..., exclude=target)`
        and `_guide_depth(f)` give, when the caller (`_refresh_guides`) has them already (R23)."""
        from kinetrace.calib import epipolar_polyline
        p = self.project
        dst = self._cal_cam(target)
        if dst is None:
            return []
        if depth is None:
            depth = self._guide_depth(f)
        if obs is None:
            obs = self._observations(name, f, exclude=target)
        out = []
        for c, cal, uv in obs:
            info: dict = {}
            try:
                pts = epipolar_polyline(cal, dst, uv, depth, margin=0.0, info=info)
            except Exception:       # noqa: BLE001 -- a degenerate calibration must not break the overlay
                continue
            if len(pts) >= 2:
                out.append((pts, self.GUIDE_COLORS[c % len(self.GUIDE_COLORS)], p.name(c)))
                if trusted is not None:
                    tr = info.get("trusted")
                    trusted.append(tr if tr is not None and len(tr) == len(pts) else np.ones(len(pts), bool))
        return out

    def _prediction(self, target: int, name: str, f: int, obs: list | None = None, depth=None) -> dict | None:
        """Where landmark `name` is in camera `target` according to the OTHER
        calibrated cameras (G23): their rays triangulated (DLT, DLTdv's residual)
        and projected into `target`. None with fewer than two of them; otherwise
        {"xy", "residual", "angle_deg", "views", "why"} -- `why` a reason in
        words when the point must NOT be offered: the rays nearly parallel (the
        crossing is noise), the cameras disagreeing (one is misplaced), or the
        point behind / outside this camera's picture. `obs` / `depth`: as for `_guides_into` (R23)."""
        from kinetrace.calib import PARALLEL_LINES_DEG, front_sign
        p = self.project
        dst = self._cal_cam(target)
        if dst is None:
            return None
        tri = self._triangulate(obs if obs is not None else self._observations(name, f, exclude=target))
        if tri is None:
            return None
        xyz, res, widest, views = tri["xyz"], tri["residual"], tri["angle_deg"], tri["views"]
        try:
            xy = dst.project(xyz[None])[0]
        except Exception:           # noqa: BLE001 -- a degenerate calibration must not break the overlay
            return None
        who = " and ".join(p.name(c) for c in views)
        why = None
        thr = self._residual_bands(views)[1]
        L = np.asarray(dst.coefs, np.float64).reshape(11)
        probe = depth if depth is not None else self._guide_depth(f)
        if widest < PARALLEL_LINES_DEG:
            why = (f"{who} see it along nearly the same line ({widest:.1f}°), so they cannot say where on "
                   "that line it is — place it on the dashed line by eye")
        elif not np.isfinite(res) or res > thr:
            why = f"{who} disagree by {res:.1f} px (more than {thr:.0f}) — one of them is misplaced here"
        elif probe is not None and np.sign(float(L[8:11] @ xyz) + 1.0) != front_sign(L, probe):
            why = f"{who} put it behind {p.name(target)}"
        elif not np.isfinite(xy).all() or not (-0.5 <= xy[0] < dst.width - 0.5 and -0.5 <= xy[1] < dst.height - 0.5):
            why = f"{who} put it outside {p.name(target)}'s picture (out of the picture = no data)"
        return {"xy": xy, "residual": float(res), "angle_deg": widest, "views": views, "why": why}

    def _residual_bands(self, views) -> tuple[float, float]:
        """(good, ok) limits in px for a triangulation residual among cameras `views`: the
        Reconstruct report's bands (calib.residual_bands, scaled by the pictures' long side)."""
        from kinetrace.calib import residual_bands
        sess = [self.project.sessions[c] for c in views]
        return max((residual_bands(sv.width or 1920, sv.height or 0) for sv in sess), default=(1.5, 5.0))

    TWO_CAM_NOTE = "2 cams: errors along the line are invisible"

    def _triangulate(self, obs: list) -> dict | None:
        """The 3D point of observations [(view, calibration, raw xy), ...] from
        two or more cameras: {"xyz", "residual" (DLTdv's rmse, px), "angle_deg" (widest angle
        between the rays), "views", "verdict" good / ok / poor (the Reconstruct report's bands),
        "note" ("" or why two cameras cannot say "good")}; None with fewer. Two rays are blind to
        a mistake along the line, so two cameras cap at "ok" as the Reconstruct report does (G83,
        I101)."""
        from kinetrace.calib import dlt_ray, triangulate
        rows = []
        for c, cal, uv in obs:
            q = cal.to_calib_frame(uv)[0]
            if np.isfinite(q).all():
                rows.append((c, cal, q))
        if len(rows) < 2:
            return None
        try:
            xyz, res, _err = triangulate(np.array([cal.coefs for _c, cal, _q in rows]),
                                         np.array([q for _c, _cal, q in rows]))
            dirs = np.array([dlt_ray(cal.coefs, q[None], xyz)[1][0] for _c, cal, q in rows])
        except Exception:           # noqa: BLE001 -- a degenerate calibration must not break the overlay
            return None
        views = [c for c, _cal, _q in rows]
        good, usable = self._residual_bands(views)
        verdict = "good" if res <= good else "ok" if res <= usable else "poor"
        if len(views) == 2 and verdict == "good":
            verdict = "ok"
        return {"xyz": xyz, "residual": float(res),
                "angle_deg": float(np.degrees(np.arccos(np.abs(np.clip(dirs @ dirs.T, -1.0, 1.0)).min()))),
                "views": views, "verdict": verdict, "note": self.TWO_CAM_NOTE if len(views) == 2 else ""}

    def _residual_sentence(self, name: str) -> str:
        """'P1 in 2 cameras: 3D rmse 0.84 px (ok: ...)' -- or '' below two cameras (G28)."""
        if not self._guides_ready():
            return ""
        tri = self._triangulate(self._observations(name, self.current, exclude=-1))
        if tri is None:
            return ""
        return (f"{name} in {len(tri['views'])} cameras: 3D rmse {tri['residual']:.2f} px ({tri['verdict']}"
                + (f", {tri['note']}" if tri["note"] else "")
                + (", the rays are nearly parallel so its depth is uncertain" if tri["angle_deg"] < 5.0 else "")
                + ")")

    def _probe_guides(self, target: int) -> list:
        """The look-here click (Alt+click, G21): the spot's line in every other
        camera, and a dashed '?' ring where it was clicked -- a ring the size of
        a region, so it cannot be taken for a point marker."""
        from kinetrace.calib import epipolar_polyline
        p = self.project
        pr = self._epi_probe
        if pr is None or not (0 <= pr[0] < p.n_views):
            return []
        v, x, y = pr
        src, dst = self._cal_cam(v), self._cal_cam(target)
        if src is None or dst is None:
            return []
        col = self.GUIDE_COLORS[v % len(self.GUIDE_COLORS)]
        if target == v:
            # at least 16 screen px across whatever the zoom: never marker-sized
            cv = self.grid.canvas(v)
            r = max(0.015 * p.sessions[v].width, 16.0 * (cv.scene_px_per_screen_px() if cv is not None else 1.0))
            a = np.linspace(0.0, 2 * np.pi, 41)
            return [(np.column_stack([x + r * np.cos(a), y + r * np.sin(a)]), col, "?")]
        try:
            pts = epipolar_polyline(src, dst, np.array([x, y], np.float64), self._guide_depth(self.current),
                                    margin=0.0)
        except Exception:           # noqa: BLE001
            return []
        return [(pts, col, f"{p.name(v)} (look-here)")] if len(pts) >= 2 else []

    def _refresh_guides(self) -> None:
        """Epipolar guides on EVERY camera on screen (G19): the selected landmark
        as the other cameras see it -- so a point placed in one camera shows at
        once, in the others, the line it must lie on; with two or more of them,
        the ◇ where they put it (G23) -- plus the look-here click (Alt+click).
        Once two cameras have the point, its 3D rmse is written beside it in each
        of them and the lines turn faint everywhere (G28)."""
        p = self.project
        pid = self.selected
        show = self.act_epipolar.isChecked() and self.state == READY and self._guides_ready()
        canvases = self.grid.canvases
        visible = set(self.grid.visible_indices()) if show else set()
        name = (self.session.points[pid].name
                if pid is not None and self.session is not None and pid < self.session.n_points else None)
        allobs = self._observations(name, self.current, exclude=-1) if (name is not None and visible) else []
        tri = self._triangulate(allobs)
        placed_at = {c: uv for c, _cal, uv in allobs}
        verdict_col = {"good": theme.GREEN, "ok": theme.AMBER, "poor": theme.RED}
        depth, have_depth = None, False     # this instant's, asked once for every camera's lines (R23)
        for t, cv in enumerate(canvases):
            lines, faint, preds, notes = [], [], [], []
            if t in visible and not self._tile_stale(t):
                if tri is not None and t in placed_at:
                    q = QColor(verdict_col[tri["verdict"]])
                    notes.append((float(placed_at[t][0]), float(placed_at[t][1]),
                                  f"3D rmse {tri['residual']:.2f} px · "
                                  + (tri["note"] if tri["note"] else f"{len(tri['views'])} cams")
                                  + (" · rays nearly parallel" if tri["angle_deg"] < 5.0 else ""),
                                  (q.red(), q.green(), q.blue())))
                if name is not None:
                    if not have_depth:
                        depth, have_depth = self._guide_depth(self.current), True
                    obs_t = [o for o in allobs if o[0] != t]      # = _observations(..., exclude=t)
                    masks: list = []
                    for (pts, col, lab), tr in zip(self._guides_into(t, name, self.current, masks,
                                                                     obs=obs_t, depth=depth), masks):
                        # the parts where the lens model is guessing are drawn faint (I132);
                        # each run overlaps the one before by a point, so the line stays whole
                        edges = np.flatnonzero(np.diff(tr.astype(np.int8))) + 1
                        for a, b in zip(np.r_[0, edges], np.r_[edges, len(pts)]):
                            seg = pts[max(a - 1, 0):b]
                            if tr[a]:
                                lines.append((seg, col, lab))
                                lab = ""                    # the camera's name once
                            else:
                                faint.append((seg, col))
                    pr = self._prediction(t, name, self.current, obs=obs_t, depth=depth)
                    if pr is not None and pr["why"] is None:
                        st = p.sessions[t]
                        j = st.pid_by_name(name)
                        ft = p.map_frame(p.active, t, self.current)
                        placed = (st.tracks[ft, j] if j is not None and ft is not None and 0 <= ft < st.n_frames
                                  and st.exportable_at(ft, j) and np.isfinite(st.tracks[ft, j]).all() else None)
                        col = tuple(st.points[j].color) if j is not None else (255, 255, 255)
                        x, y = float(pr["xy"][0]), float(pr["xy"][1])
                        n = len(pr["views"])
                        label = (f"{name}: {np.hypot(placed[0] - x, placed[1] - y):.1f} px from the {n} other cameras"
                                 if placed is not None else f"{name} ◇ from {n} cameras")
                        preds.append((x, y, col, label, placed))
                    elif pr is not None and lines:
                        pts0, col0, lab0 = lines[0]
                        lines[0] = (pts0, col0, f"{lab0} — no ◇: {pr['why']}")
                if self._epi_probe is not None:
                    lines += self._probe_guides(t)
            cv.set_guides(lines, faint, preds, dim=bool(preds) or tri is not None, notes=notes)
        if not canvases:
            self.canvas.set_guides([])

    def _tile_stale(self, v: int) -> bool:
        """Camera `v` is on screen but NOT at the playhead's instant: Active view
        only leaves the other cameras where they were (G27)."""
        p = self.project
        if not self._companions_stale or p is None or v == p.active or not (0 <= v < len(self._views)):
            return False
        f = p.map_frame(p.active, v, self.current)
        rt = self._views[v]
        # a camera that has shown nothing yet (just added, or a project reopened in
        # this mode) is decoded once, at this instant; after that it stays put
        return f is not None and rt.want_frame is not None and rt.want_frame != min(f, rt.n_frames - 1)

    def _snap_to_prediction(self, pid: int, x: float, y: float) -> tuple[float, float]:
        """A click on (or next to) the ◇ of the selected landmark places it EXACTLY
        at the ◇ (G23); anywhere else the click is taken as it is."""
        s = self.session
        if not self._guides_ready() or not self.act_epipolar.isChecked() or s is None or pid >= s.n_points:
            return x, y
        pr = self._prediction(self.project.active, s.points[pid].name, self.current)
        if pr is None or pr["why"] is not None:
            return x, y
        reach = (self.canvas._guides.PRED_R + 4.0) * self.canvas.scene_px_per_screen_px()
        px, py = float(pr["xy"][0]), float(pr["xy"][1])
        return (px, py) if np.hypot(px - x, py - y) <= reach else (x, y)

    def _accept_prediction(self, pid: int | None = None) -> None:
        """A (or the point menu): place the selected landmark where two or more
        other cameras put it (G23). Hand-placed, one undo step."""
        s = self.session
        pid = self.selected if pid is None else pid
        if self.state != READY or s is None:
            return
        if pid is None or pid >= s.n_points:
            self.statusBar().showMessage("A places the selected point at the ◇ — select a point first", 5000)
            return
        name = s.points[pid].name
        if s.points[pid].derived:
            self.statusBar().showMessage(f"{name} is derived from the silhouette — it cannot be placed by hand", 5000)
            return
        pr = self._prediction(self.project.active, name, self.current) if self._guides_ready() else None
        if pr is None:
            n = len(self._observations(name, self.current, self.project.active)) if self._guides_ready() else 0
            self.statusBar().showMessage(
                f"No ◇ for {name}: it needs a calibration and {name} placed in two other calibrated cameras at this "
                f"instant ({n} {'has' if n == 1 else 'have'} it)", 7000)
            return
        if pr["why"] is not None:
            self.statusBar().showMessage(f"No ◇ for {name}: {pr['why']}", 9000)
            return
        x, y = float(pr["xy"][0]), float(pr["xy"][1])
        self._begin_edit()
        s.set_position(self.current, pid, x, y)
        self._refresh_overlay()
        self._update_track_button()
        rmse = self._residual_sentence(name)                   # (G28)
        self.statusBar().showMessage(
            f"{name} placed where {' and '.join(self.project.name(c) for c in pr['views'])} put it on frame "
            f"{self.current}" + (f" — {rmse}" if rmse else "") + " — Track re-seeds from here; Ctrl+Z undoes",
            8000)

    def _on_probe(self, x: float, y: float) -> None:
        """Alt+click: where can this spot be in the other cameras? Nothing is
        edited (G21)."""
        p = self.project
        if self.state != READY or p is None:
            return
        if not self._guides_ready() or self._cal_cam(p.active) is None:
            self.toast.show_message(
                "Alt+click shows where a spot can be in the other cameras: it needs a calibration of this camera "
                "and at least one other (3D → Import Calibration, or calibrate with a wand).", "info", 7000)
            return
        self._epi_probe = (p.active, float(x), float(y))
        self._guides_on()
        others = ", ".join(p.name(v) for v in p.others() if self._cal_cam(v) is not None)
        self.statusBar().showMessage(
            f"Look-here: the dashed line in {others} shows where this spot can be in that camera. Nothing was "
            "placed. Esc clears it", 9000)

    def _snap_to_epipolar(self, pid: int) -> None:
        """Move landmark `pid` at this frame onto the other cameras' epipolar
        line(s): the nearest point with one camera, their crossing with two or
        more. Flagged hand-placed, one undo step; Track re-seeds from it."""
        from kinetrace import retrack
        s = self.session
        if s is None or self.state != READY or not (0 <= pid < s.n_points) or s.points[pid].derived:
            return
        p = self.project
        name = s.points[pid].name
        none_msg = f"{name}: no other camera has it at this instant (or no calibration)"
        if not self._guides_ready():
            self.statusBar().showMessage(none_msg, 6000)
            return
        cur = s.tracks[self.current, pid]
        info: dict = {}
        # (R14) the hand snap is `retrack.ray_target` with its own rules: polylines cut exactly at the
        # picture's edges, the observations read at the fractional frame among the calibrated cameras
        got = retrack.ray_target(p, p.active, name, self.current, margin=0.0, info=info,
                                 observations=self._observations(name, self.current, exclude=p.active))
        if got is None:
            if "outside_px" in info:
                # the rays cross OUTSIDE this picture: clipping that onto the edge stored
                # a hand placement where the part is not (I9)
                self.statusBar().showMessage(
                    f"{name}: the other cameras put it {info['outside_px']:.0f} px outside this camera's "
                    f"picture at frame {self.current} — out of the picture means no data here; clear it on "
                    "this frame instead", 8000)
            elif not info.get("n_lines"):
                self.statusBar().showMessage(none_msg, 6000)
            return
        n_cams = int(got[1])
        x, y = float(got[0][0]), float(got[0][1])
        self._begin_edit()
        s.set_position(self.current, pid, x, y)
        self._refresh_overlay()
        self._update_track_button()
        moved = float(np.linalg.norm(np.array([x, y]) - cur)) if np.isfinite(cur).all() else float("nan")
        self.statusBar().showMessage(
            f"{s.points[pid].name} snapped onto {n_cams} camera{'s' if n_cams != 1 else ''}' rays at "
            f"frame {self.current}" + (f" (moved {moved:.1f} px)" if np.isfinite(moved) else "")
            + (" — the cameras stand in a line, so their rays coincide here: it was moved onto the line "
               "only; check its place ALONG the line by eye" if info.get("mode") == "parallel" else "")
            + " — press Track to re-seed from here; Ctrl+Z undoes", 10000)

    # ------------------------------------------- automatic epipolar re-tracking

    def _disagree_thresholds(self) -> list[float]:
        """The timeline's band threshold per camera: 5 px at 1920 wide, scaled -- the timeline's own
        rule (`disagree_threshold`), so the band drawn and the stretches re-tracked are one
        (R6, G83)."""
        from kinetrace.timeline import disagree_threshold
        return [disagree_threshold(s.width or 1920) for s in self.project.sessions]

    def _retrack_dialog(self):
        """3D → Re-track Disagreeing Stretches: explain, list, ask, run."""
        from kinetrace import retrack
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
        from kinetrace import retrack
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
        from kinetrace import retrack
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
            self._start_tracking(stop_after=job.local1, only_pids=[pid], quiet=True, segment=True)
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
        self._refresh_companions()          # the other cameras drew the discarded positions (G125)
        self.timeline.refresh()
        self._apply_state()

    def _retrack_finish(self):
        from kinetrace import retrack
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
        # (G69) Ctrl+Z takes the re-track back, in every camera it touched: the undo point is the
        # re-track's own snapshots (the quiet runs took none, so it was left from BEFORE the
        # original tracking run -- and the working camera only), as `_multi_finish` does
        snaps = {v: sn for v, sn in st["snaps"].items() if v < p.n_views}
        if snaps:
            self._set_run_undo_point(snaps)
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
        local = p.local_index(p.active, t)              # map_frame's tie rule, not np.round (I258)
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
        trails, future = self._trails_for(s, f)
        onion = self.act_onion.isChecked()
        ghost_prev = s.tracks[f - 1] if (onion and f > 0) else None
        ghost_next = s.tracks[f + 1] if (onion and f + 1 < s.n_frames) else None
        positions = s.positions_at(f)
        # animal overlays first (they sit under the markers)
        self.canvas.set_masks(self._mask_layers(s, f), self._mask_opacity)     # every segment (G149)
        # the clicks of the animal the selection names (G154): S adds to it, a right click removes them
        k = self._s_target() if s is self.session else None
        if k is not None and self.state == READY:
            a = s.segments[k]
            self.canvas.set_prompts(a.prompts.get(f, []), a.boxes.get(f), a.color)
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
        self._restyle_point_rows()          # a placement / clear that did not rebuild the list (G126)

    # -------------------------------------------------------------- tracking

    def _toggle_tracking(self, all_cameras: bool | None = None, step: bool | None = None):
        """T / the Track button. `all_cameras` (Shift+T = True) overrides Track ▾ →
        Every camera for this one run; `step` forces one frame forward whatever the
        mode (`_track_step`)."""
        if self.state == TRACKING:
            self._pause_tracking()
            return
        if self._loading:                            # (G71) a video / project is opening: nothing starts
            return
        self._update_track_button()                  # the reason must be about THIS frame
        blocked = getattr(self, "_track_blocked", None)
        if blocked is not None:
            self.toast.show_message(blocked + ".", "warn", 7000)   # instead of a silent no-op (G34)
        elif self.state == READY:
            every = self.act_track_all.isChecked() if all_cameras is None else all_cameras
            step = self._track_mode == "semi" if step is None else bool(step)
            scope, _n = self._run_scope()
            s = self.session
            passes = self._plan_passes(scope, step, every)      # only what can start here (I203)
            n_full = len(self._tracker_passes(scope)) if scope else 1
            self._passes = None
            if len(passes) > 1:
                self._passes = {"queue": [[s.points[q].name for q in g] for g in passes[1:]],
                                "groups": [[s.points[q].name for q in g] for g in passes],
                                "labels": [self._pass_label(g) for g in passes],
                                "view": self.project.active, "frame": self.current, "step": step,
                                "every": every, "done": [], "snap": None, "msnaps": None}
            elif n_full > 1:
                left = [q for g in self._tracker_passes(scope) for q in g
                        if q not in passes[0] and not s.points[q].derived]
                self.toast.show_message(
                    f"{', '.join(s.points[q].name for q in left)} {'has' if len(left) == 1 else 'have'} no "
                    f"position on frame {self.current}, so this run tracks the others only (their tracks are "
                    "kept).", "info", 7000)
            self._start_pass(None if (len(passes) == 1 and n_full == 1) else passes[0], step, every,
                             segment=None, stops=None, quiet=False)
            if self._passes is not None and self.state != TRACKING and self._multi is None:
                self._passes = None             # pass 1 could not start: no later pass is waiting for it

    def _pass_label(self, pids) -> str:
        """'AllTracker (A, B), ball markers (C)': what a pass tracks, each name under the thing that
        really follows it (G100: Moving spot, ball and silhouette-derived points were listed under
        AllTracker)."""
        s = self.session
        parts: dict[str, list[str]] = {}
        for q in pids:
            m = s.points[q]
            key = ("silhouette landmarks" if m.derived else "ball markers" if m.is_ball
                   else self.TRACKER_NAMES.get(self._tracker_of(q), "?"))
            parts.setdefault(key, []).append(m.name)
        return ", ".join(f"{k} ({', '.join(v)})" for k, v in parts.items())

    def _start_pass(self, pids, step: bool, every: bool, segment, stops, quiet: bool) -> None:
        """One pass of a Track press (G63): `pids` None = the selection as it is; `stops`
        {camera: last frame} keeps a later pass inside the first one, each camera by its OWN frame
        numbers (I204)."""
        p = self.project
        names = None if pids is None else {self.session.points[q].name for q in pids}
        jobs = self._multi_jobs(step, names=names, segment=segment, stops=stops) if every else []
        if every and len(jobs) > 1:
            self._start_multi_tracking(step=step, names=names, segment=segment, stops=stops)
            if self._passes is not None and self._passes["msnaps"] is None and self._multi is not None:
                self._passes["msnaps"] = dict(self._multi["snaps"])
            return
        if every and p is not None and p.n_views > 1:
            # it used to fall back to the working camera without a word (G32)
            self._say_single_camera(jobs, names=names, stops=stops)
        stop = None
        if stops is not None:
            stop = stops.get(p.active)
            if stop is None:
                return              # the first pass did not run in this camera: no bound, no run (I204)
        if step:
            if self.current >= self.n_frames - 1:
                self.statusBar().showMessage("Already at the last frame", 3000)
                return
            stop = self.current + 1 if stop is None else min(stop, self.current + 1)
        if pids is None and segment is None:
            kw = {}
            if stop is not None:
                kw["stop_after"] = stop
            if step:
                kw["step"] = True
            self._start_tracking(**kw)
        else:
            self._start_tracking(stop_after=stop, only_pids=pids, quiet=quiet, segment=segment, step=step)
        if self._passes is not None and self._passes["snap"] is None and self.state == TRACKING:
            self._passes["snap"] = self._undo_snap

    def _passes_record(self, lasts: dict, fail, user_stop: bool) -> None:
        """A pass of a two-pass run ended: note where, then go on (G63)."""
        st = self._passes
        if st is None:
            return
        st["done"].append({"lasts": dict(lasts), "fail": fail, "user_stop": user_stop, "group": len(st["done"])})
        QTimer.singleShot(0, self._passes_next)

    def _passes_skip(self, why: str) -> None:
        """The next pass cannot run: record it with the reason and go on to the one after (I203)."""
        st = self._passes
        st["done"].append({"lasts": {}, "fail": None, "user_stop": False, "skipped": True, "reason": why,
                           "group": len(st["done"])})

    def _passes_next(self) -> None:
        """Start the next pass from the same camera and frame, never past where the previous one
        stopped (a point that stops ends the run for all, owner), in exactly the cameras the
        previous one ran in, each bounded by its own end (I204); a pass that cannot start is
        recorded as skipped and the one after it runs (I203). After the last, every point of the
        run ends on the same frame (`_passes_finish`)."""
        st = self._passes
        if st is None:
            return
        if self.state != READY:
            QTimer.singleShot(100, self._passes_next)
            return
        prev = st["done"][-1]
        if prev["user_stop"] or not st["queue"]:
            self._passes_finish()
            return
        p = self.project
        if p.active != st["view"]:
            self._set_active_view(st["view"])
        self._goto(st["frame"], force=True)
        stops = {}
        for v, last in prev["lasts"].items():
            start_v = p.map_frame(st["view"], v, st["frame"]) if v < p.n_views else None
            if last is not None and start_v is not None and last > start_v:
                stops[v] = int(last)
        while st["queue"]:
            names = st["queue"].pop(0)
            s = self.session
            pids = [q for q in (s.pid_by_name(n) for n in names) if q is not None]
            if not stops:
                self._passes_skip("the first pass did not get past the start frame")
                continue
            if not self._pass_startable(pids, False, st["step"], st["every"], stops):
                self._passes_skip(f"none of its points has a position on frame {st['frame']} "
                                  "in the cameras the first pass ran in")
                continue
            self.statusBar().showMessage(
                f"Second pass: {self._pass_label(pids)} over the same frames"
                + (", kept on the first pass's silhouettes" if st.get("seg_views") else "")
                + " — X stops it", 8000)       # (I185)
            self._start_pass(pids, st["step"], st["every"], segment=False, stops=stops, quiet=True)
            if self.state == TRACKING or self._multi is not None:
                return
            self._passes_skip("it could not start")
        self._passes_finish()

    def _passes_finish(self) -> None:
        """The last pass is done: the points of an earlier pass that went further go back to their
        pre-run data after the run's end (one end frame for all); the playhead goes to where the
        run ended, or to the first stop; ONE Ctrl+Z undoes every pass -- set AFTER the camera
        switch, which forgets the undo point (I202)."""
        st, self._passes = self._passes, None
        p = self.project
        if st is None or p is None:
            return
        ran = [(k, d) for k, d in enumerate(st["done"]) if not d.get("skipped")]
        snaps = st["msnaps"] or ({st["view"]: st["snap"]} if st["snap"] is not None else {})
        trimmed = []
        if len(ran) > 1:
            for v in set().union(*[d["lasts"].keys() for _k, d in ran]):
                lasts = [d["lasts"].get(v) for _k, d in ran]
                if any(x is None for x in lasts) or v >= p.n_views or v not in snaps:
                    continue
                end = min(lasts)
                sv = p.sessions[v]
                for k, d in ran:
                    if d["lasts"][v] > end:
                        pids = [q for q in (sv.pid_by_name(n) for n in st["groups"][d.get("group", k)])
                                if q is not None]
                        sv.restore_cells(snaps[v], pids, end + 1, d["lasts"][v])
                        trimmed.append((p.name(v), end))
        fails = [d["fail"] for _k, d in ran if d.get("fail") is not None]
        if fails:
            v, f, pid = min(fails, key=lambda t: t[1])
            if v < p.n_views and p.active != v:
                self._set_active_view(v)
            self._goto(min(f, self.n_frames - 1), force=True)
            if 0 <= pid < self.session.n_points:
                self._on_select(pid)
        else:
            if st["view"] < p.n_views and p.active != st["view"]:
                self._set_active_view(st["view"])
            ends = [d["lasts"].get(st["view"]) for _k, d in ran if d["lasts"].get(st["view"]) is not None]
            if ends:                      # not back at the start frame, where the second pass began
                self._goto(min(min(ends), self.n_frames - 1), force=True)
        if snaps:
            self._set_run_undo_point(snaps)
        self._refresh_overlay()
        self._refresh_companions()
        self.timeline.refresh()
        self._update_track_button()
        # worded from why each pass ended (G100)
        labels = st.get("labels") or [", ".join(g) for g in st["groups"]]
        s = self.session
        said = []
        for k, label in enumerate(labels):
            d = next((x for kk, x in enumerate(st["done"]) if x.get("group", kk) == k), None)
            last = None if d is None else next((x for x in [d["lasts"].get(st["view"])]
                                                 + list(d["lasts"].values()) if x is not None), None)
            if d is None:
                said.append(f"{label}: not run (you stopped the run first)")
            elif d.get("skipped"):
                said.append(f"{label}: not run ({d.get('reason', 'it could not start')})")
            elif d["user_stop"]:
                said.append(f"{label}: stopped by you" + (f" at frame {last}" if last is not None else ""))
            elif d.get("fail") is not None:
                fv, ff, fp = d["fail"]
                nm = s.points[fp].name if 0 <= fp < s.n_points else "the segment"
                said.append(f"{label}: stopped at frame {ff} ({nm} was lost)")
            else:
                said.append(f"{label}: to frame {last}" if last is not None else f"{label}: nothing tracked")
        msg = ("Two passes" if len(labels) == 2 else f"{len(labels)} passes") + ", one after the other — " \
            + "; ".join(said) + "."
        if trimmed:
            msg += (f" The second pass stopped earlier (frame {trimmed[0][1]}), so the first pass's points end "
                    "there too — every point of the run ends on the same frame.")
        msg += " One Ctrl+Z undoes " + ("both passes." if len(labels) == 2 else "all of them.")
        unusual = bool(fails or trimmed or any(d.get("skipped") or d["user_stop"] for d in st["done"])
                       or len(ran) < len(labels))
        if st["step"] and not unusual:
            # a step is a status line, not a 12 s notice every F press (G100)
            self.statusBar().showMessage(msg, 8000)
        else:
            self.toast.show_message(msg, "warn" if unusual else "info", 12000)

    def _track_step(self):
        """Semi-automatic: track exactly one frame forward, then pause. Each step re-seeds from the
        current (possibly corrected) positions, so F → check → fix → F walks the video with the user
        in the loop. A thin wrapper over the Track press (passes, Every camera and the reason Track
        is blocked all apply)."""
        if self.state != READY or self.session is None:
            return
        self._toggle_tracking(step=True)

    # ------------------------------------------- tracking in every camera (G29)

    def _multi_jobs(self, step: bool = False, names=None, segment=None, stops=None) -> list[dict]:
        """One run per camera that has something to track at this instant: the
        points of the run's scope (the panel selection, by NAME, or all) that
        have a position at that camera's frame -- ball markers included -- or its
        segment. The working camera first. Empty with one camera."""
        p = self.project
        if p is None or p.n_views < 2 or self.session is None:
            return []
        scope, _n = self._run_scope()
        if names is None:
            names = {self.session.points[i].name for i in scope if i < self.session.n_points}
        # (G149) the working camera's run segments, BY NAME: each camera runs its segments of those names
        seg_names = (list(segment) if isinstance(segment, (list, tuple, set)) else
                     [self.session.segments[k].name for k in self._seg_list(scope, segment)])
        jobs = []
        for v in [p.active] + p.others():
            sv = p.sessions[v]
            fv = p.map_frame(p.active, v, self.current)
            rt = self._views[v] if v < len(self._views) else None
            if fv is None or rt is None or not (0 <= fv < rt.n_frames - (1 if step else 0)):
                continue
            pids = [q for q in sv.seedable_at(fv) if sv.points[q].name in names]
            # this camera's animals ride along for the points they hold (I185, G160)
            segs_v = self._seg_list(scope, seg_names, sv) if segment is not False else []
            if segment is None:
                segs_v = sorted(set(segs_v) | set(self._run_segments([q for q in pids], sv)))
            seg_v = [sv.segments[k].name for k in segs_v]
            seg_ok = bool(any(sv.animal_seedable_at(fv, k) for k in segs_v))
            if not pids and not seg_ok:
                continue
            # a run that re-segments fills the silhouette-derived landmarks of the run too (I184):
            # without them the old derived positions stayed beside the new silhouettes
            dpids = [q for q in sv.derived_pids() if sv.points[q].name in names] if seg_ok else []
            stop = int(fv) + 1 if step else None
            if stops is not None:
                if v not in stops or stops[v] is None:
                    continue
                stop = stops[v] if stop is None else min(stop, stops[v])
            jobs.append({"view": v, "frame": int(fv), "pids": pids + dpids, "stop": stop,
                         "segment": seg_v if seg_v else False})
        return jobs

    def _tracking_engine_ready(self) -> bool:
        """The tracking code (and PyTorch) imported -- off the GUI thread with the
        card when it is not yet (G46): the first Track after a start used to wait
        on the device probe's torch import with the window frozen and no word."""
        if "kinetrace.tracker" in sys.modules:
            return True
        import importlib
        try:
            self._in_background("Starting the tracking engine", lambda: importlib.import_module("kinetrace.tracker"),
                                detail="Loading PyTorch; the first start after installing takes the longest…")
        except Exception as e:      # noqa: BLE001
            QMessageBox.critical(self, "Tracking cannot start", _plain_error(e, "The tracking code could not be loaded"))
            return False
        return True

    def _start_multi_tracking(self, step: bool = False, names=None, segment=None, stops=None) -> None:
        """Track in every camera that has the points, ALL AT ONCE (I141): one worker
        per camera, advanced together on one thread (tracker.MultiTrackingWorker),
        so the GPU holds one run's memory (~11 GB for a 4K AllTracker run) and each
        camera's result is bit-identical to tracking it alone; each camera draws
        live in its own view; X stops them all at once; one Ctrl+Z undoes it in every
        camera. (Before I141 the cameras ran one after another, and a pause during
        the first left the others untracked -- "only the active camera tracks".)"""
        if not self._tracking_engine_ready():
            return
        p = self.project
        jobs = self._multi_jobs(step, names=names, segment=segment, stops=stops)
        if self.state != READY or len(jobs) < 2:
            return
        # all the cameras start now (I141); their pre-run snapshots are the one undo step
        self._multi = {"orig": p.active, "orig_frame": self.current, "step": step,
                       "snaps": {j["view"]: p.sessions[j["view"]].snapshot() for j in jobs}, "results": [],
                       "stopped": None}
        if p.active not in self._multi["snaps"]:
            self._multi["snaps"][p.active] = self.session.snapshot()
        # one worker per camera, each seeded at ITS frame of this instant: built by
        # the ordinary start path in each camera in turn, then run together
        from kinetrace.tracker import MultiTrackingWorker
        orig, orig_frame = p.active, self.current
        built = []
        for job in jobs:
            v = job["view"]
            if v >= p.n_views:
                continue
            if p.active != v:
                self._set_active_view(v)
            self._goto(job["frame"], force=True)
            pids = [q for q in job["pids"] if q < self.session.n_points]
            w = self._start_tracking(stop_after=job["stop"], only_pids=pids, quiet=True, build_only=True,
                                     segment=job.get("segment"), step=step)
            if w is None:
                self._multi["results"].append({"view": v, "last": None, "why": "nothing to start from"})
                continue
            built.append((v, w, int(job["frame"])))
        main = next((b for b in built if b[0] == orig), built[0] if built else None)
        if main is None:
            self._multi_finish()
            return
        if p.active != main[0]:
            self._set_active_view(main[0])
        self._goto(main[2] if main[0] != orig else orig_frame, force=True)
        order = [main] + [b for b in built if b is not main]
        runs = [(v, _SideRun(self, v, w, f, write=v != main[0])) for v, w, f in order]
        self._multi["runs"] = runs
        self._side = {v: r for v, r in runs if v != main[0]}
        drv = MultiTrackingWorker([w for _v, w, _f in order])
        drv.finished.connect(self._multi_parallel_done)
        self._launch_run(main[1], driver=drv)
        self.statusBar().showMessage(
            f"Tracking in {len(order)} cameras at the same time: "
            + ", ".join(p.name(v) for v, _w, _f in order) + " — X stops them all", 8000)
        self._warn_hidden_tracked([v for v, _w, _f in order], step)

    def _hidden_cams_text(self, views) -> str:
        p = self.project
        names = [p.name(v) for v in views] if p is not None else []
        return (", ".join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")
                + (" is hidden" if len(names) == 1 else " are hidden"))

    def _warn_hidden_tracked(self, views, step: bool = False) -> None:
        """(G174, owner 2026-10-08: "give a clear warning to the user that tracking will continue even for
        hidden cameras in every-camera run") An every-camera run tracks EVERY camera that has the points --
        a camera whose view is hidden (its eye, or Only the working camera) too, but it is not drawn: the
        user cannot watch it or pause it on a mistake there. Said at the start of each run; a
        semi-automatic step says it again only when the hidden cameras changed (F after F would repeat it)."""
        shown_v = set(self.grid.visible_indices())
        hidden = [v for v in views if v not in shown_v]
        if not hidden:
            self._hidden_warned = None
            return
        key = tuple(sorted(hidden))
        if step and getattr(self, "_hidden_warned", None) == key:
            return
        self._hidden_warned = key
        self.toast.show_message(
            f"Tracking in {len(views)} cameras — {self._hidden_cams_text(hidden)}, and "
            f"{'it is' if len(hidden) == 1 else 'they are'} <b>tracked too</b>, without being drawn: you cannot "
            "watch the tracking there. <b>Click here</b> to show them (or their eye in CAMERAS) and watch every "
            "camera; <b>X</b> stops the run.", "warn", 15000, on_click=self._show_tracked_views)

    def _show_tracked_views(self) -> None:
        """(G174) The hidden-camera notice clicked: every camera on screen (Only the working camera off too),
        so a run in every camera can be watched -- the side runs draw the views that come back."""
        if self.act_solo.isChecked():
            self.act_solo.setChecked(False)
        if self.grid.hidden():
            self.grid.set_arrangement(hidden=set())
        self._refresh_cameras()
        self._update_track_button()

    def _mask_layers(self, s, f: int, midline: bool = True) -> list:
        """Every segment's silhouette on frame f, each in its colour, with its midline when shown
        (G149): what `VideoCanvas.set_masks` draws. Empty with the silhouettes switched off."""
        if s is None or not s.segments or not self.btn_mask.isChecked():
            return []
        mid = midline and self.act_show_midline.isChecked()
        # an animal whose LAYERS checkbox is off is not drawn (G154)
        return [(m.contours.get(f) if a.shown else None, a.color, m.midline.get(f) if (mid and a.shown) else None)
                for a, m in zip(s.segments, s.seg_masks)]

    def _render_side(self) -> None:
        """Draw the newest frame of each other camera of a simultaneous run in its
        own view, with its points (I141); never more than 16 frames behind."""
        p = self.project
        if p is None:
            return
        visible = set(self.grid.visible_indices())
        for v, run in self._side.items():
            if not run.frames:
                continue
            while len(run.frames) > 16:
                run.frames.popleft()
            idx, rgb = run.frames.popleft()
            cv = self.grid.canvas(v)
            if v >= p.n_views or v not in visible or cv is None:
                continue
            s = p.sessions[v]
            if not (0 <= idx < s.n_frames):
                continue
            cv.set_stale(False)
            cv.set_frame(rgb)
            cv.set_masks(self._mask_layers(s, idx, midline=False), self._mask_opacity)    # (G149)
            trails, future = self._companion_trails(s, idx)    # (G33)
            cv.set_points(s.positions_at(idx), s.visibility[idx], s.points, None,
                          trails, future, occluded=s.occluded[idx], radii=s.radius[idx])
            self.grid.set_caption(v, f"{p.name(v)}  ·  frame {idx} · tracking")

    def _multi_parallel_done(self) -> None:
        """Every camera of a simultaneous run has stopped (I141). Their data is in;
        now each camera's end goes through the ordinary end-of-run handling in
        turn (a lost point's cut, messages; `_multi_replay_next`), then the one
        Ctrl+Z and the summary (`_multi_finish`)."""
        st = self._multi
        side, self._side = self._side, None
        if side:
            for run in side.values():           # the last pictures up
                while len(run.frames) > 1:
                    run.frames.popleft()
            self._side = side
            self._render_side()
            self._side = None
        if st is None:
            # closing or torn down: the cameras' rows are written; just end the run
            _retire(self.worker)
            self.worker = None
            self.state = READY
            return
        _retire(self.worker)
        self.worker = None
        st["replay"] = list(st.get("runs", []))
        st["user_paused"] = bool(getattr(self, "_user_paused", False))
        self._user_paused = False
        self.state = READY
        self._multi_next()

    def _multi_replay_next(self) -> None:
        """The next camera of a finished simultaneous run: switch to it and hand
        its end to `_on_track_finished` / `_on_track_error`, exactly as if it had
        just run there (they come back through `_multi_next`)."""
        st = self._multi
        p = self.project
        while st["replay"]:
            v, run = st["replay"].pop(0)
            if v >= p.n_views:
                continue
            if p.active != v:
                self._set_active_view(v)
            st["current"] = v
            self.worker = run.worker            # its _autopause_reason, _ball_ended, decode_failed_at
            self._autopause_info = run.autopause
            self._track_pids = run.pids
            self._run_start = run.start
            self._run_hand = getattr(run.worker, "_kt_hand", None)      # (I189)
            self._step_run = bool(st["step"])
            self._user_paused = bool(st["user_paused"] and run.paused)
            self.state = TRACKING
            if run.error is not None:
                self._on_track_error(run.error)
            else:
                self._on_track_finished(run.last if run.last is not None else run.start - 1, run.paused)
            return
        self._multi_finish()

    def _say_single_camera(self, jobs: list, names=None, stops=None) -> None:
        """Track ▾ → Every camera is ticked but only one camera can start here:
        say which camera lacks what, so it does not look like a broken option (G32). `names` =
        this pass's points; `stops` = a later pass runs only where the first one ran, so a camera
        outside it is not blamed for a missing position (I204)."""
        p = self.project
        s = self.session
        if names is None:
            scope, _n = self._run_scope()
            names = {s.points[i].name for i in scope if i < s.n_points}
        names = sorted(n for n in names if (s.pid_by_name(n) is None or not s.points[s.pid_by_name(n)].derived))
        have = {j["view"] for j in jobs}
        if p.active not in have:
            return                  # nothing starts here at all: the ordinary message says so
        miss = []
        for v in p.others():
            if v in have or (stops is not None and v not in stops):
                continue
            fv = p.map_frame(p.active, v, self.current)
            miss.append(f"{p.name(v)} has no position for {', '.join(names) or 'them'} on its frame {fv}"
                        if fv is not None else f"{p.name(v)} has no frame at this instant")
        if miss:
            self.toast.show_message(
                "Every camera: only this camera can start from here — " + "; ".join(miss)
                + ". Place the point there (select it and click it in that camera), or go to a frame where "
                "every camera has it, then press Track again.", "warn", 12000)

    def _multi_next(self) -> None:
        st = self._multi
        if st is None:
            return
        if self.state != READY:
            QTimer.singleShot(100, self._multi_next)
            return
        self._multi_replay_next()

    def _free_side_runs(self, st: dict) -> None:
        """A finished every-camera run lets go of its workers (I196): each `_SideRun` is a child
        of the window and held its TrackingWorker -- SAM sessions (GPU memory), ball trackers, mask
        caches -- until the window closed, a few more per F press in semi-automatic mode until CUDA
        ran out of memory."""
        for _v, run in st.get("runs", []):
            run.worker = None
            run.frames.clear()
            run.deleteLater()
        st["runs"] = []

    def _multi_finish(self) -> None:
        """Every camera done (or the run stopped): one undo step for all of them,
        a summary, and the working camera back -- unless a camera stopped on a lost
        point, which is then where the playhead goes (as after any auto-pause)."""
        st, self._multi = self._multi, None
        p = self.project
        if st is None or p is None:
            return
        self._free_side_runs(st)
        user_stop = st["stopped"] == "user"
        problem = next((r for r in st["results"] if r.get("fail") is not None), None)
        if user_stop:
            problem = None      # X stopped every camera at once: back to the one watched (I141)
        if problem is not None:
            if p.active != problem["view"]:
                self._set_active_view(problem["view"])
            self._goto(problem["fail"], force=True)
            if problem.get("pid") is not None and problem["pid"] >= 0 and problem["pid"] < self.session.n_points:
                self._on_select(problem["pid"])
        else:
            orig = st["orig"]
            if orig < p.n_views and p.active != orig:
                self._set_active_view(orig)
            own = next((r for r in st["results"] if r["view"] == orig and r.get("last") is not None), None)
            if st["step"]:
                self._goto(min(st["orig_frame"] + 1, self.n_frames - 1), force=True)
            elif own is not None:
                self._goto(own["last"], force=True)
        # one Ctrl+Z for the whole run, in every camera it touched (the switches above
        # cleared the undo point, as any camera switch does)
        snaps = {v: s for v, s in st["snaps"].items() if v < p.n_views}
        self._set_run_undo_point(snaps)
        self._refresh_companions()
        self._refresh_guides()
        self._update_track_button()
        done = [r for r in st["results"] if r.get("last") is not None]
        parts = []
        for r in st["results"]:
            nm = p.name(r["view"]) if r["view"] < p.n_views else "?"
            if r.get("fail") is not None:
                parts.append(f"<b>{nm}</b> stopped at frame {r['fail']} ({r.get('why') or 'a point was lost'})")
            elif r.get("last") is not None:
                parts.append(f"{nm} to frame {r['last']}")
            else:
                parts.append(f"{nm}: {r.get('why') or 'not tracked'}")
        if self._passes is not None:
            # one pass of a two-pass run (G63): where each camera ended, then the next pass
            self._passes_record({r["view"]: r.get("last") for r in st["results"] if r.get("view") is not None},
                                (problem["view"], problem["fail"], problem.get("pid", -1)) if problem else None,
                                user_stop)
        if st["step"] and not user_stop and problem is None:
            self.statusBar().showMessage(
                f"Stepped one frame in {len(done)} cameras — F for the next; Ctrl+Z undoes the step in all", 6000)
            return
        self.toast.show_message(
            ("Stopped: " if user_stop else "Tracked in every camera: ") + "; ".join(parts)
            + ". One Ctrl+Z undoes the run in every camera.", "warn" if (problem or user_stop) else "info", 12000)

    def _pause_tracking(self):
        if self.state == TRACKING and self.worker is not None:
            self._user_paused = True
            self.btn_track.setText("Pausing…")
            self.btn_track.setEnabled(False)
            self.worker.request_pause()

    def _start_tracking(self, stop_after: int | None = None, only_pids=None, quiet: bool = False,
                        build_only: bool = False, segment: bool | None = None, step: bool = False):
        """`only_pids` overrides the panel selection (automatic runs); `quiet`
        skips the overwrite guard and keeps the caller's undo snapshot;
        `build_only` returns the worker, unstarted (None when nothing can start),
        for a simultaneous every-camera run (I141); `step` = one semi-automatic frame
        (no ETA, no overwrite question, "Stepped to frame" at its end): a bound
        (`stop_after`) alone no longer makes a run a step -- the second pass of a two-pass
        run was one (G99).

        In order (R8): what can start (`_run_targets`), the overwrite question, the undo point,
        the points' specs split by tracker, the segment's specs, the worker."""
        if self._loading:
            return None         # a project / video is still opening behind the card (G71)
        if not self._tracking_engine_ready():
            return None
        s = self.session
        scope, pids, run_seg, animal_ok = self._run_targets(only_pids, segment)
        if not pids and not animal_ok:
            self._say_nothing_to_start(scope, run_seg)
            return
        if self._preview is not None and self._preview.isRunning():
            self.toast.show_message("One moment — the silhouette preview is still computing.",
                                    "info", 4000)
            return
        self._step_run = bool(step)
        self._say_what_runs(scope, pids, animal_ok)
        if not self._overwrite_ok(scope, pids, animal_ok, quiet):
            return

        if not quiet:
            self._undo_snap = s.snapshot()
        self.act_undo.setEnabled(False)  # re-enabled when the run ends

        from kinetrace.tracker import TrackingWorker        # lazy: imports torch
        seeds = s.positions_at(self.current)
        pids, ball_pids, spot_specs, backend = self._split_by_tracker(pids, seeds, quiet)
        specs = self._point_specs(pids, seeds)
        balls = self._ball_specs(ball_pids, seeds)
        seg = self._segment_specs(pids, scope, animal_ok, segment, specs)
        if not specs and not balls and not spot_specs and not seg.animals:
            self.toast.show_message(
                f"Nothing to start from on frame {self.current}: place a point with <b>N</b>, a ball "
                "with Point ▾ → Ball marker, or press <b>S</b> and click the segment here.", "warn", 7000)
            self.act_undo.setEnabled(self._undo_snap is not None)       # no run: Ctrl+Z stays what it was
            return
        end = self.n_frames if stop_after is None else min(self.n_frames, stop_after + 1)
        w = TrackingWorker(self.info.path, self.current, None, None,
                           self.cache, end, refine=True,
                           specs=specs,
                           roi=self.btn_roi.isChecked(),
                           autopause=self.btn_autopause.isChecked(),
                           animals=seg.animals, derived=seg.derived,             # (G149) every segment
                           point_backend=backend, balls=balls, spots=spot_specs)
        # the points hand-placed on the start frame: `write_segment` clears the flag of every row it
        # writes, the start row (the user's click, position kept) included -- restored at the end of
        # the run (I189)
        w._kt_hand = (self.current, [q for q in pids + ball_pids + [sp.pid for sp in spot_specs]
                                     if s.manual[self.current, q]])
        if build_only:
            return w                    # one camera of a simultaneous run (I141)
        self._launch_run(w)

    def _run_targets(self, only_pids, segment):
        """What a run would start from on this frame: (scope = the points it covers, the scope's
        points that have a position here, whether the segment runs, whether it can start here). The
        rule behind the Track button's wording is `_update_track_button`; the sentences differ there
        (a tooltip, not a notice), so the two are kept apart (R8)."""
        s = self.session
        scope, _n_sel = self._run_scope()   # exactly the panel selection (G61)
        if only_pids is not None:
            scope = set(int(q) for q in only_pids)
        segs = self._seg_list(scope, segment)                 # (G149) every segment of the run
        run_seg = bool(segs)
        pids, animal_ok = self._startable_here(scope, run_seg, segs)
        return scope, pids, run_seg, animal_ok

    def _startable_here(self, scope, run_seg: bool, segs=None):
        """(the scope's points that have a position on this frame, whether a segment of the run --
        `segs`, else the selection rule -- can start here from a click or a silhouette): what a run
        can start from, one rule for the Track button (`_update_track_button`) and for a Track press
        (R8, G149)."""
        s = self.session
        if s is None:
            return [], False
        if run_seg and segs is None:
            segs = self._run_segments(scope)
        return ([p for p in s.seedable_at(self.current) if p in scope],
                bool(run_seg and any(s.animal_seedable_at(self.current, k) for k in (segs or []))))

    def _say_nothing_to_start(self, scope, run_seg: bool) -> None:
        """Track was pressed with nothing that can start on this frame: why, as a notice (the
        Track button's tooltip says the same in `_update_track_button`)."""
        s = self.session
        if not scope and not run_seg:
            self.toast.show_message(
                "Nothing is selected, so nothing was tracked. Select what to track in <b>LAYERS</b>: points, "
                "or an animal (its silhouette and all its points); Ctrl+click for several, <b>Ctrl+A</b> for all.",
                "warn", 8000)
        elif scope:
            self.toast.show_message(
                f"The selected point(s) have no position on frame {self.current}. Select a "
                "point that exists here, or place it here first (click it on the video).",
                "warn", 7000)
        elif s.animal is not None:
            self.toast.show_message(
                f"Nothing to start from on frame {self.current}: place a point with <b>N</b> "
                "(a few pixels is enough), or press <b>S</b> and click the segment here.",
                "warn", 7000)

    def _say_what_runs(self, scope, pids, animal_ok: bool) -> None:
        """The status-bar line of a starting run: what is tracked, and which selected points have no
        position here and are skipped (their tracks are kept). Not for a one-frame step."""
        if self._step_run:
            return
        s = self.session
        skipped = sum(1 for i in range(s.n_points) if not s.points[i].derived and i in scope) - len(pids)
        self.statusBar().showMessage(
            f"Tracking the {len(pids)} selected point(s)"
            + ((" and the silhouette" if len(self._run_segments(scope)) < 2 else
                f" and {len(self._run_segments(scope))} silhouettes") if animal_ok else "")
            + " — only what is selected in LAYERS is tracked", 6000)
        if skipped:
            self.statusBar().showMessage(
                f"{skipped} point(s) have no position at frame {self.current} and will be "
                "skipped (their existing tracks are kept)", 6000)

    def _overwrite_ok(self, scope, pids, animal_ok: bool, quiet: bool) -> bool:
        """The accidental-overwrite guard: starting far before a large body of work asks first
        (irrelevant for a one-frame step, which would nag on every F, and for a quiet run). It
        counts the frames THIS run would overwrite: its own points' tracked frames after the
        start, plus the segment's silhouette frames when it runs -- every point's, a new point D
        alone read "will overwrite 4900 already-tracked frames" (G101). False = the user said no."""
        s = self.session
        cols = list(pids) + ([q for q in s.derived_pids() if q in scope] if animal_ok else [])
        ahead = (s.tracked[self.current + 1:][:, cols].any(axis=1) if cols
                 else np.zeros(max(0, s.n_frames - self.current - 1), bool))
        if animal_ok:
            has_mask = np.zeros(s.n_frames, bool)
            for k in self._seg_list(scope, True):      # the run's segments (G149, G151n)
                has_mask[s.mask_frames(k)] = True
            ahead = ahead | has_mask[self.current + 1:]
        frames_after = int(ahead.sum())
        last = self.current + 1 + int(np.nonzero(ahead)[0][-1]) if frames_after else self.current
        if not quiet and not self._step_run and frames_after > 2000 and self.current < 0.5 * last:
            if QMessageBox.question(
                    self, "Re-track from here?",
                    f"Tracking from frame {self.current} will progressively overwrite "
                    f"{frames_after} already-tracked frames ahead of it.\n\n"
                    "You can undo with Ctrl+Z after it finishes. Continue?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
                return False
        return True

    def _split_by_tracker(self, pids, seeds, quiet: bool):
        """Sort the run's points by what follows them: (the points the ONE appearance model of this
        worker tracks, the ball markers, the Moving spot specs, that model's key). Point model:
        Moving spot (I160, G56): plain points are searched as spots (spots.py, no model); a region
        needs AllTracker / CoTracker3. Each point by its own tracker (G62): Moving spot points as
        spots, the rest by ONE appearance model (a Track press splits AllTracker and CoTracker3
        points into passes, G63; anything else keeps the first)."""
        from kinetrace.tracker import SpotSpec
        s = self.session
        ball_pids = [pid for pid in pids if s.points[pid].is_ball]
        pids = [pid for pid in pids if not s.points[pid].is_ball]
        spot_specs, left_out, nn = [], [], []
        for pid in pids:                # (`seedable_at` never returns a silhouette-derived point)
            q = s.points[pid]
            t = self._tracker_of(pid)
            if t == "spot":
                if q.kind != "point":
                    left_out.append(q.name)
                    continue
                spot_specs.append(SpotSpec(pid, seeds[pid], self._spot_velocity(pid, self.current), q.spot))
            else:
                nn.append((pid, t))
        kinds = [k for k in ("alltracker", "cotracker3") if any(t == k for _q, t in nn)]
        backend = kinds[0] if kinds else self._point_backend
        not_now = [s.points[q].name for q, t in nn if t != backend]
        if not_now and not quiet:
            self.statusBar().showMessage(f"Not in this run (another tracker): {', '.join(not_now)}", 8000)
        pids = [q for q, t in nn if t == backend]
        if left_out and not quiet:
            self.toast.show_message(
                "<b>Moving spot</b> follows single points only, so "
                + ", ".join(f"<b>{n}</b>" for n in left_out)
                + " (a region) is left out of this run. Give it AllTracker or CoTracker3 (right-click its row "
                "in LAYERS → Tracker).", "info", 9000)
        return pids, ball_pids, spot_specs, backend

    @staticmethod
    def _region_offsets(meta):
        """A region's polygon as vertex offsets from its centre (the worker samples members inside
        it), None for a circle / a point."""
        if meta.kind != "group" or meta.outline is None or len(meta.outline) < 3:
            return None
        o = np.asarray(meta.outline, np.float32).reshape(-1, 2)
        return o - o.mean(axis=0)

    def _point_specs(self, pids, seeds) -> list:
        """The appearance-tracked points' `PointSpec`s, seeded from their positions on this frame."""
        from kinetrace.tracker import PointSpec
        s = self.session
        return [PointSpec(pid, seeds[pid].astype(np.float32).copy(),
                          s.points[pid].kind, s.points[pid].radius,
                          s.points[pid].anchor, self._region_offsets(s.points[pid]))
                for pid in pids]

    def _ball_specs(self, ball_pids, seeds) -> list:
        """Ball markers: SAM segments them, a circle is fitted, the centre is the
        point (balls.py) - prompts from this frame on, seeded from the ball's
        current position, the last known radius as the size hint."""
        from kinetrace.tracker import BallSpec
        s = self.session
        balls = []
        for pid in ball_pids:
            q = s.points[pid]
            prompts = {f: cs for f, cs in (q.ball_prompts or {}).items() if f >= self.current}
            rad = s.radius[:, pid]
            known = rad[np.isfinite(rad)]
            balls.append(BallSpec(pid, prompts, seeds[pid], float(known[-1]) if len(known) else None,
                                  self._seg_backend))
        return balls

    def _segment_roles(self, s, k: int, pids) -> dict:
        """Animal k's landmarks among `pids` (G149, G156): the head (when among them) and the points
        it holds -- its appearance-tracked points not marked 'may leave', when the animal keeps its
        points on its silhouette: they are both demoted when off the body and constrained to it. A
        point of no animal (Scene) is never held."""
        hp = s.head_pid(k)
        held = {q for q in pids if s.segment_of(q) == k and self._held(s, q)}
        return dict(head_pid=hp if hp in pids else None, on_body=set(held), constrain=set(held))

    def _segment_specs(self, pids, scope, animal_ok: bool, segment, specs) -> _SegmentSpecs:
        """The segments' part of a run (G149: every segment of the run, one SAM session): each one's
        prompts (or its stored silhouette on this frame as the seed), the derived landmarks to fill in,
        and its landmarks' on-body rules (the constraint, the stop where a landmark leaves the animal,
        the off-body demotion). A two-pass run's second pass gets the silhouettes the first pass wrote
        as STORED segments (I185): no SAM, the same rules."""
        from kinetrace.tracker import AnimalSpec, DerivedSpec
        s = self.session
        animals: list = []
        derived: list = []
        if animal_ok:
            segs = [k for k in self._seg_list(scope, segment) if s.animal_seedable_at(self.current, k)]
            for j, k in enumerate(segs):
                a, mt = s.segments[k], s.seg_masks[k]
                seed_mask = None
                if not a.has_prompt(self.current) and mt.has(self.current):
                    ww, wh = working_size(self.info.width, self.info.height)
                    seed_mask = mt.rasterize(self.current, wh, ww)
                animals.append(AnimalSpec({f: list(v) for f, v in a.prompts.items()}, dict(a.boxes), seed_mask,
                                          self._seg_backend, index=k, name=a.name, **self._segment_roles(s, k, pids)))
                derived += [DerivedSpec(pid, s.points[pid].spec, seg=j) for pid in s.derived_pids()
                            if pid in scope and s.segment_of(pid) == k]
            if segs and self._passes is not None and not self._passes.get("done") and self.project is not None:
                # pass 1 of a two-pass run carries this camera's segments: their silhouettes are what
                # the later pass is kept on (I185)
                self._passes.setdefault("seg_views", {})[self.project.active] = [s.segments[k].name for k in segs]
        pp = self._passes
        if (segment is False and not animals and pp is not None and pp.get("done") and specs
                and self.project is not None and self.project.active in pp.get("seg_views", {})):
            # (I185) the second pass of a two-pass run: no SAM, but the silhouettes pass 1 just wrote
            # into the session hold its on-body points to the same rules as a run with the segments.
            # {camera: segment names}; a plain set of cameras (no names) = every segment there
            sv = pp["seg_views"]
            names = sv[self.project.active] if isinstance(sv, dict) else s.segment_names()
            for name in names:
                k = s.segment_index(name)
                if k is not None and len(s.mask_frames(k)):
                    animals.append(AnimalSpec(index=k, name=name, stored=s.seg_masks[k],
                                              **self._segment_roles(s, k, pids)))
        return _SegmentSpecs(animals, derived)

    def _spot_velocity(self, pid: int, f: int):
        """A Moving spot run's starting speed (px / frame): from the frame before
        when the point has data there, else from the frame after when it was
        placed there by hand (click two frames in a row, track from the first).
        None = unknown: the first search is wider."""
        s = self.session
        p = s.tracks[f, pid].astype(np.float64)
        if f - 1 >= 0 and s.tracked[f - 1, pid] and np.isfinite(s.tracks[f - 1, pid]).all():
            return p - s.tracks[f - 1, pid].astype(np.float64)
        if f + 1 < s.n_frames and s.manual[f + 1, pid] and np.isfinite(s.tracks[f + 1, pid]).all():
            return s.tracks[f + 1, pid].astype(np.float64) - p
        return None

    # ------------------------------------------------ which point model (G57, G58)

    def _test_point_models(self, pid: int | None = None) -> None:
        """Track ▾ / the point menu -> Test the point models on my clicks: the
        selected point (else the one with the most hand-placed frames)."""
        s = self.session
        if s is None or self.state != READY:
            if self.state == TRACKING:
                self.statusBar().showMessage("Pause the run first (X): the test uses the same video", 6000)
            elif s is None:
                self.toast.show_message(
                    "Open a video first. Then place the point you want to follow by hand on 20 frames in a row "
                    "(<b>N</b>, click it; <b>F</b>, click it…) and run the test again.", "info", 8000)
            return
        from kinetrace import spots
        from kinetrace.pointtest import PointModelTest
        plain = [i for i in range(s.n_points) if s.points[i].kind == "point" and not s.points[i].derived
                 and not s.points[i].is_ball]
        if pid is None:
            pid = self.selected if self.selected in plain else None
        if pid is None and plain:
            pid = max(plain, key=lambda i: len(spots.clicked_stretch(s.manual_frames(i))))
        if pid is None:
            self.toast.show_message(
                "The test needs a point placed by hand: press <b>N</b>, click the spot, then click it on 20 "
                "frames in a row (F steps one frame) and run the test again.", "info", 9000)
            return
        q = s.points[pid]
        if q.kind != "point" or q.derived or q.is_ball:
            self.toast.show_message(f"<b>{q.name}</b> is a {'region' if q.kind != 'point' else 'ball marker' if q.is_ball else 'silhouette landmark'}"
                                    ": the test compares point models on a plain point.", "info", 7000)
            return
        mf = [int(f) for f in s.manual_frames(pid) if np.isfinite(s.tracks[f, pid]).all()]
        clicks = {f: s.tracks[f, pid].astype(np.float64) for f in mf}
        at_ok = _alltracker_available()
        models = [("alltracker", at_ok, "" if at_ok else "not installed yet (see the Track ▾ menu)"),
                  ("cotracker3", True, "")]
        dlg = PointModelTest(self, q.name, self.info.path, self.cache, self.n_frames,
                             (self.info.width, self.info.height), clicks, models, self.btn_roi.isChecked(),
                             gpu=getattr(self, "_device_kind", "") == "GPU",
                             on_use=lambda r, pid=pid: self._apply_test_choice(pid, r))
        self._test_dlg = dlg
        dlg.exec()
        self._test_dlg = None
        dlg.deleteLater()

    def _apply_test_choice(self, pid: int, result) -> None:
        """The test's Use: the project's point model, and for Moving spot the
        settings it found on this point (one undo step)."""
        s = self.session
        if s is None or not (0 <= pid < s.n_points):
            return
        # ONE undo step covering the settings and the tracker in every camera it changes (G68)
        self._begin_edit({s.points[pid].name})
        if result.model == "spot" and result.settings is not None:
            s.points[pid].spot = result.settings.to_dict()
            s.dirty = True                  # a new setting alone changes nothing else (I205)
        self._set_tracker([pid], result.model, undo=False)
        self.toast.show_message(
            f"<b>{s.points[pid].name}</b> is now tracked by <b>{result.label}</b>"
            + (f" ({result.settings.describe()})" if result.model == "spot" else "")
            + ". Its tracker shows on its row in LAYERS; right-click the row → Tracker changes it.", "success", 8000)

    def _hint_corrections(self, pid: int) -> None:
        """G58: once per point, when it has been corrected by hand on 5 of the last
        20 frames while AllTracker / CoTracker3 tracks it -- the squid pattern."""
        s = self.session
        # a ball marker is not tested (the test refuses it): no hint for it (G102)
        if s is None or s.points[pid].kind != "point" or s.points[pid].is_ball or self._tracker_of(pid) == "spot":
            return
        key = ("corr", id(self.project), self.project.active if self.project else 0, s.points[pid].name)
        hist = self._spot_corrections.setdefault(key[1:], [])
        hist.append(self.current)
        recent = [f for f in hist if abs(f - self.current) < 20]
        if len(recent) < 5 or key in self._spot_hints:
            return
        self._spot_hints.add(key)
        name = s.points[pid].name
        self.toast.show_message(
            f"You have corrected <b>{name}</b> by hand on {len(recent)} of the last 20 frames. If it is a target "
            "small enough to be one point (a dot, no visible shape), <b>Point model: Moving spot</b> may follow "
            "it better; if you can see its shape, stay with AllTracker. <b>Click here</b> to test the point models "
            "on your clicks (it needs 20 hand-placed frames in a row).", "info", 15000,
            on_click=lambda nm=name: self._test_named_point(nm))

    def _test_named_point(self, name: str) -> None:
        """The corrections hint's click: the point is looked up BY NAME now -- the index it held
        when the hint was shown shifts when a point is deleted (IndexError, or the wrong point;
        G102)."""
        s = self.session
        pid = s.pid_by_name(name) if s is not None else None
        if pid is not None:
            self._test_point_models(pid)

    def _hint_small_spot(self, pid: int, x: float, y: float) -> None:
        """G58: once per project, when a new point sits on a small isolated spot
        and nothing says it is an animal (no segment in this camera)."""
        s = self.session
        if s is None or s.animal is not None or self._tracker_of(pid) == "spot":
            return
        key = ("tiny", id(self.project))
        if key in self._spot_hints:
            return
        rgb = self.cache.get(self.current) if self.cache is not None else None
        if rgb is None:
            return
        from kinetrace import spots
        look = spots.looks_like_small_spot(rgb, (x, y))
        if look is None:
            return
        self._spot_hints.add(key)
        self.toast.show_message(
            f"<b>{s.points[pid].name}</b> sits on a dot about {look.diameter:.0f} px across. If the whole target "
            "is that dot — small enough to be one point, no visible shape — <b>Track ▾ → Point model: Moving "
            "spot</b> often follows it better than AllTracker. If it is a mark on an animal you can see, stay with "
            "AllTracker. To find out, click it on 20 frames in a row, then <b>Track ▾ → Test the point models on "
            "my clicks</b>. <b>Click here</b> to read when to use which.", "info", 15000,
            on_click=lambda: self._show_manual(MANUAL_WHICH_MODEL))     # (item 3)

    TRACKER_TAGS = {"alltracker": "AT", "cotracker3": "CT", "spot": "MS"}
    TRACKER_NAMES = {"alltracker": "AllTracker", "cotracker3": "CoTracker3", "spot": "Moving spot"}

    def _set_tracker(self, pids, key: str, undo: bool = True) -> None:
        """Give these points their own tracker, in every camera (by name: a landmark
        is one thing in all of them, G19 / G62). One undo step across the cameras it changes
        (G68; `undo=False` when the caller took the point itself). Moving spot follows single
        points: a region keeps its tracker (G124)."""
        s = self.session
        if s is None:
            return
        names = {s.points[q].name for q in pids if q < s.n_points}
        todo, regions = [], set()
        for sv in (self.project.sessions if self.project is not None else [s]):
            for m in sv.points:
                if m.name in names and not m.derived and not m.is_ball:
                    if key == "spot" and m.kind != "point":
                        regions.add(m.name)
                    elif m.tracker != key:
                        todo.append((sv, m))
        if todo and undo:
            self._begin_edit(names)
        for sv, m in todo:
            m.tracker = key
            sv.dirty = True
        self._refresh_point_list()
        self._update_track_button()
        done = sorted(names - regions)
        self.statusBar().showMessage(
            (f"{', '.join(done)}: tracked by {self.TRACKER_NAMES.get(key, key)} from the next run" if done else "")
            + (f"{' — ' if done else ''}{', '.join(sorted(regions))} (a region) keeps its tracker: Moving spot "
               "follows single points only" if regions else ""), 7000)

    def _launch_run(self, w, driver=None) -> None:
        """Wire a built worker to the working camera's live display and start it.
        With `driver` (a MultiTrackingWorker, I141) the driver is started instead,
        and the worker's end goes to the every-camera coordinator, not straight to
        the end-of-run handling."""
        self.worker = driver if driver is not None else w
        self._model_worker = w
        w.balls_ready.connect(self._on_ball_radii)
        w.model_loading.connect(self._on_model_loading)
        w.model_progress.connect(self._on_model_progress)
        w.started_ok.connect(self._on_track_started)
        w.masks_ready.connect(self._on_masks)
        w.chunk_ready.connect(self._on_chunk)
        if driver is None:
            w.autopaused.connect(self._on_autopaused)
            w.finished_ok.connect(self._on_track_finished)
            w.error.connect(self._on_track_error)
        self._track_pids = list(w.point_ids)
        self._run_hand = getattr(w, "_kt_hand", None)       # (I189)
        self._autopause_info = None
        self._member_frames.clear()
        self._run_start = self.current
        self._fps_ema = 0.0
        self._last_emit_t = 0.0
        self.state = TRACKING
        self._apply_state()
        self._track_label.setText("starting…")
        self.worker.start()
        if driver is not None:
            self._render_timer.start()          # the other cameras draw from the start

    def _models_needed(self, w) -> tuple[list[str], bool]:
        """What a run's start will load: (sentences, anything to download). The
        point model the run really uses (AllTracker by default -- the old check
        looked at CoTracker3's file whatever the choice, G45) and SAM for a
        segment OR ball markers."""
        from kinetrace import downloads
        from kinetrace.segmenter import loaded_backends, preferred_backend
        parts, download = [], False
        if getattr(w, "specs", None):
            key = "alltracker" if getattr(w, "point_backend", "") == "alltracker" else "cotracker3"
            f = downloads.FILES[key]
            if not (f.dest.is_file() and downloads.code_present(key)):
                parts.append(f"Downloading {f.label} ({f.size / 1e6:.0f} MB)")
                download = True
        backend = None
        if getattr(w, "animal", None) is not None:
            backend = w.animal.backend
        elif getattr(w, "balls", None):
            backend = w.balls[0].backend or preferred_backend()
        if backend and backend not in loaded_backends():
            label = BACKENDS.get(backend, BACKENDS[DEFAULT_BACKEND])[2]
            if seg_is_cached(backend):
                parts.append(f"Loading the segmentation model ({label})")
            else:
                parts.append(f"Downloading the segmentation model ({label})")
                download = True
        return parts, download

    def _on_model_loading(self):
        parts, download = self._models_needed(self._model_worker)
        if not parts:
            w = self._model_worker
            spot_only = (w is not None and getattr(w, "spots", None) and not w.specs and w.animal is None
                         and not w.balls)
            self.statusBar().showMessage("Starting Moving spot (no model to load)…" if spot_only
                                         else "Starting the tracking models…", 8000)
            return
        text = "\n".join(parts) + ("\n\nThis happens only the first time: the files are kept inside the Kinetrace "
                                   "folder (models/)." if download else "")
        self._open_model_dialog(text, 0 if download else 800)

    def _open_model_dialog(self, text: str, delay_ms: int) -> None:
        if self._model_dialog is None:
            dlg = QProgressDialog(text, "Cancel", 0, 0, self)
            dlg.setWindowTitle("Preparing the models")
            dlg.setWindowModality(Qt.NonModal)              # X / Space still stop the run
            dlg.setMinimumDuration(delay_ms)
            dlg.setAutoClose(False)
            dlg.setAutoReset(False)
            dlg.canceled.connect(self._cancel_model_download)
            dlg.setValue(0)
            self._model_dialog = dlg
            self._model_dl = {}
        else:
            self._model_dialog.setLabelText(text)

    def _on_model_progress(self, label: str, done: float, total: float):
        """A first-use download: what, how far, about how long (G45)."""
        self._open_model_dialog(label, 0)
        mx, val, text = _download_status(self._model_dl, label, done, total)
        self._model_dialog.setRange(0, mx)
        self._model_dialog.setValue(val)
        self._model_dialog.setLabelText(text)

    def _cancel_model_download(self):
        if self.worker is not None and self.state == TRACKING:
            self.worker.request_pause()

    def _on_masks(self, summaries: list):
        if self.session is not None:
            self.session.write_mask_summaries(summaries)

    def _on_ball_radii(self, rows: list):
        if self.session is not None:
            self.session.write_ball_radii(rows)

    def _on_track_started(self):
        _quiet_close(self._model_dialog)
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
        if self._fps_ema and not self._step_run:  # a 1-frame step has no useful ETA
            remaining = (self.n_frames - 1 - head) / max(self._fps_ema, 1e-6)
            mins, secs = divmod(round(remaining), 60)
            eta = f"{mins} min {secs:02d} s" if mins else f"{secs} s"
            mw = self._model_worker
            kind = ("Moving spot · " if mw is not None and getattr(mw, "spots", None) and not mw.specs
                    and mw.animal is None and not mw.balls
                    else f"{self._device_kind} · " if getattr(self, "_device_kind", "") else "")
            self._track_label.setText(f"{kind}tracking {self._fps_ema:.1f} fps · ETA {eta}")

    def _render_tick(self):
        if self._side:
            self._render_side()                 # the other cameras of a simultaneous run (I141)
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

    def _end_run(self) -> bool:
        """What the end of EVERY run shares, finished or error (I241, R8): back to READY, the
        paused / step flags reset (an error never reset `_user_paused`: after X during a first-use
        download the next pause or auto-pause counted as an X), the models dialog closed (a run
        paused while loading left it up), Ctrl+Z back on for the snapshot the run kept (an error
        left it off), per-run caches dropped, and the autosave. Returns whether it was a step."""
        step_run = self._step_run
        self.state = READY
        self._step_run = False
        self._user_paused = False
        _quiet_close(self._model_dialog)
        self._model_dialog = None
        self._end_run_cleanup()
        self.act_undo.setEnabled(self._undo_snap is not None)
        self._autosave()  # before the guidance message — 'Saved ✓' must not clobber it
        return step_run

    def _restore_start_flags(self) -> None:
        """A run clears the hand-placed flag of the frame it starts from (the position stays: it is
        the user's seed); put the flag back for the points that had it (I189) -- otherwise the
        diamond, `,` / `.`, keyframe interpolation, the saved hand_placed column and the test's
        20-click count lose that click after every correction + Track / F."""
        hand, self._run_hand = getattr(self, "_run_hand", None), None
        s = self.session
        if not hand or s is None:
            return
        f, pids = hand
        if not (0 <= f < s.n_frames):
            return
        changed = False
        for q in pids:
            if 0 <= q < s.n_points and s.tracked[f, q] and not s.manual[f, q]:
                s.manual[f, q] = True
                changed = True
        if changed:
            s._touch()

    def _on_track_finished(self, last: int, was_paused: bool):
        user_pause = bool(getattr(self, "_user_paused", False))
        self._user_paused = False
        autopause = self._autopause_info            # the impl consumes it
        reason = getattr(self.worker, "_autopause_reason", "") if self.worker is not None else ""
        pass_view = self.project.active if self.project is not None else 0
        self._on_track_finished_impl(last, was_paused)
        if self._passes is not None and self._multi is None and self._retrack is None:
            # one pass of a two-pass run (G63)
            fail = None if autopause is None else (pass_view, int(autopause[0]), int(autopause[1]))
            self._passes_record({pass_view: int(last) if last >= self._passes["frame"] else None}, fail,
                                bool(user_pause and was_paused))
        if self._multi is not None:
            st = self._multi
            res = {"view": st.get("current"), "last": int(last)}
            if autopause is not None:
                fail, pid = int(autopause[0]), int(autopause[1])
                s = self.session
                nm = (s.points[pid].name if s is not None and 0 <= pid < s.n_points
                      else "the segment" if pid < 0 else f"point {pid}")
                res.update(fail=fail, pid=pid, why=_stop_short(reason, nm))
            st["results"].append(res)
            if user_pause and was_paused:
                # X / Space stops the WHOLE run where it is (the camera and frame on
                # screen), as it does an automatic re-track (I107)
                st["stopped"] = "user"
            # a simultaneous run (I141) stopped every camera at once: each one's
            # end is still handled (cut, messages), then the next camera's
            QTimer.singleShot(0, self._multi_next)
        if self._retrack is not None:
            if user_pause and was_paused:
                # X / Space during an automatic re-track stops the WHOLE queue and
                # goes to the Keep / Undo question; it used to start the next
                # stretch at once (I107)
                self._retrack["jobs"] = []
            QTimer.singleShot(0, self._retrack_next)

    def _on_track_finished_impl(self, last: int, was_paused: bool):
        if self.worker is None:
            # the run was torn down with its video (_teardown_video) and its queued end
            # arrived afterwards: there is nothing left to finish
            return
        self.worker.wait(2000)
        reason = getattr(self.worker, "_autopause_reason", "")
        ball_ended = dict(getattr(self.worker, "_ball_ended", {}) or {})
        spot_ended = dict(getattr(self.worker, "_spot_ended", {}) or {})
        # (G149) segments that ended in this run (lost, or SAM took another object) and the landmarks
        # each held on itself: those end where their segment did
        seg_ended = dict(getattr(self.worker, "_seg_ended", {}) or {})
        seg_holds = {int(a.index): set(a.constrain) for a in (getattr(self.worker, "animals", None) or [])
                     if getattr(a, "stored", None) is None}
        # the result can arrive while the thread still releases its capture (a
        # network share can stall that past the wait): keep it until it ends --
        # dropping the last reference to a running QThread aborts the app (I133)
        _retire(self.worker)
        self.worker = None
        self._restore_start_flags()         # before the autosave (I189)
        step_run = self._end_run()
        self._refresh_animal_panel()
        self._segments_ended(seg_ended, seg_holds, last,
                             say=not (self._autopause_info is not None and self._autopause_info[1] < 0))
        if self._autopause_info is not None:
            fail_frame, pid = self._autopause_info
            self._autopause_info = None
            self._goto(min(fail_frame, self.n_frames - 1), force=True)
            self._apply_state()
            self._track_label.setText("")
            # (G151i) the segment(s) the run ended, not the active one
            ss = self.session
            n_seg = ss.n_segments if ss is not None else 0
            ended = (sorted((k for k in seg_ended if 0 <= k < n_seg), key=lambda k: seg_ended[k][0])
                     or sorted(k for k in seg_holds if 0 <= k < n_seg))      # else the run's segments
            aname = (", ".join(ss.segments[k].name for k in ended) if ended
                     else ss.animal.name if ss is not None and ss.animal is not None else "the segment")
            if pid < 0:  # the animal itself
                self._say_stop("switched" if reason == "switched" else "segment", aname, fail_frame)
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
            kind = self._stop_kind(reason, pid)
            self._say_stop(kind, name, fail_frame,
                           spot_ended.get(pid, (fail_frame, "missing"))[1] if kind == "spot" else "missing")
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
        msg = (f"Paused at frame {last} — select a point and click where it is to correct it, then press "
               "Track to re-track from here"
               if was_paused else f"Tracking complete (through frame {last})")
        self.statusBar().showMessage(msg, 8000)
        s = self.session
        ended = [(pid, f, why) for pid, (f, why) in sorted(ball_ended.items()) if s is not None and pid < s.n_points]
        if ended:
            # auto-pause off: the run went on, but these tracks stopped inside the
            # picture and nothing said so (I127)
            parts = [f"<b>{s.points[pid].name}</b> at frame {f}" for pid, f, why in ended]
            self.toast.show_message(
                "Ball markers lost inside the picture (auto-pause is off, so the run went on): "
                + ", ".join(parts) + ". Their tracks end there. Select one, Point ▾ → Ball marker, click it "
                "where it is and press Track.", "warn", 15000)
        stopped = [(pid, f, why) for pid, (f, why) in sorted(spot_ended.items()) if s is not None and pid < s.n_points]
        if stopped:
            # auto-pause off: Moving spot still stops a spot it cannot find; say which, and where
            parts = [f"<b>{s.points[pid].name}</b> at frame {f}"
                     + (" (two alike)" if why == "ambiguous" else " (not found)") for pid, f, why in stopped]
            self.toast.show_message(
                "Moving spot stopped (auto-pause is off, so the run went on for the rest): " + ", ".join(parts)
                + ". Their tracks end there. Select one, click it where it is and press Track.", "warn", 15000)

    def _segments_ended(self, ended: dict, holds: dict, last: int, say: bool = True) -> None:
        """(G149, owner 2026-10-03: end that one, keep going) Segments that ended during the run: the
        landmarks held on each end with it (from that frame to the run's last; their body is gone, the
        on-body rule cannot hold them), and a notice names each segment, the frame and why."""
        s = self.session
        if not ended or s is None:
            return
        parts, cut = [], 0
        for k, (f, why) in sorted(ended.items(), key=lambda kv: kv[1][0]):
            if not (0 <= k < s.n_segments):
                continue
            pids = [q for q in holds.get(k, ()) if 0 <= q < s.n_points]
            if pids and f <= last:
                cut += s.clear_window(pids, f, last)
            parts.append(f"<b>{s.segments[k].name}</b> at frame {f}"
                         + (" (SAM took another object)" if why == "switched" else " (lost)"))
        if cut:
            self._refresh_overlay()
            self.timeline.refresh()
        if say and parts:
            self.toast.show_message(
                "Segments that ended during the run: " + ", ".join(parts) + ". Their silhouettes end on the frame "
                "before" + (", and so do the landmarks held on them" if cut else "") + "; everything else went on. "
                "Where one is visible again, select its row, press S, click it, and Track.", "warn", 15000)

    def _stop_kind(self, reason: str, pid: int) -> str:
        """Which `_STOP_TEXTS` entry a run's stop at point `pid` takes, from the worker's
        `_autopause_reason`: a ball SAM lost, a Moving spot, a landmark that left the segment, else
        low confidence."""
        s = self.session
        known = bool(s and pid < s.n_points)
        if reason == "lost" and known and s.points[pid].is_ball:
            return "ball"
        if reason == "spot" and known:
            return "spot"
        if reason == "exit":
            return "exit"
        return "unreliable"

    def _say_stop(self, kind: str, name: str, frame: int, why: str = "missing") -> None:
        """The notice and the status-bar line of a run that stopped at `frame` (`_STOP_TEXTS`);
        `why` = what Moving spot reported ("ambiguous" = two spots alike, else not found)."""
        t = _STOP_TEXTS[kind]
        fmt = dict(name=name, frame=frame, prev=frame - 1)
        if kind == "spot":
            two = why == "ambiguous"
            fmt["said"] = t["said_two" if two else "said_none"].format(**fmt)
            fmt["saw"] = t["saw_two" if two else "saw_none"]
        self.toast.show_message(t["toast"].format(**fmt), "warn", t["ms"])
        self.statusBar().showMessage(t["status"].format(**fmt), 12000)

    def _on_track_error(self, tb: str):
        if self._passes is not None and self._multi is None:
            self._passes = None                 # an error ends a two-pass run (G63)
        decode_at = getattr(self.worker, "decode_failed_at", None) if self.worker is not None else None
        transient = bool(getattr(self.worker, "decode_transient", False)) if self.worker is not None else False
        dl_failed = getattr(self._model_worker, "download_failed", None)
        if self._multi is not None:
            # every camera run so far is kept (one Ctrl+Z still undoes them all);
            # the run stops where the error happened
            self._multi["results"].append({"view": self._multi.get("current"), "last": None,
                                           "why": "an error stopped it"})
            # the other cameras of a simultaneous run went on (I141): handle theirs
            QTimer.singleShot(0, self._multi_next)
        if self._retrack is not None:
            st = self._retrack
            self._retrack = None
            self._retrack_restore(st)
            self.statusBar().showMessage("Re-tracking stopped on an error; everything was put back", 8000)
        if self.worker is not None:
            self.worker.wait(2000)
            _retire(self.worker)        # (I133)
        self.worker = None
        self._restore_start_flags()          # (I189)
        self._end_run()          # the ending the finished path shares (I241)
        self._apply_state()
        self._track_label.setText("")
        self._refresh_animal_panel()
        if decode_at is not None:
            # a damaged frame mid-video, not a failure of the model: everything
            # before it was emitted and is kept (I40)
            self._goto(max(0, min(decode_at - 1, self.n_frames - 1)), force=True)
            if transient:
                # (I240) a fresh capture reads it: a hiccup (a slow drive), not a damaged file
                self.toast.show_message(f"Tracking stopped: frame {decode_at} of the video could not be read "
                                        "just now — press Track again. Everything before it is kept.",
                                        "warn", 12000)
                self.statusBar().showMessage(f"Stopped: frame {decode_at} could not be read just now — "
                                             "press Track again", 12000)
                QMessageBox.warning(self, "A frame could not be read just now", tb[-1500:])
                return
            self.toast.show_message(f"Tracking stopped: frame {decode_at} of the video could not be decoded. "
                                    "Everything before it is kept.", "warn", 12000)
            self.statusBar().showMessage(f"Stopped: frame {decode_at} could not be decoded", 12000)
            QMessageBox.warning(self, "Damaged frame in the video", tb[-1500:])
            return
        if dl_failed is not None:
            # a model could not be fetched (G45): the worker's sentence, no traceback
            from kinetrace.downloads import DownloadCancelled
            if isinstance(dl_failed, DownloadCancelled):
                self.toast.show_message(f"{dl_failed} Nothing was tracked; press Track to go on with the "
                                        "download from where it stopped.", "info", 9000)
            else:
                self.toast.show_message("Tracking could not start: " + str(dl_failed), "error", 12000)
                QMessageBox.warning(self, "A model could not be downloaded", str(dl_failed))
            return
        hint = _model_error_hint(tb)
        self.toast.show_message("Tracking stopped. " + (hint or "Everything tracked before it is kept."),
                                "error", 10000)
        QMessageBox.critical(self, "Tracking stopped", _crash_text(
            tb, hint, "Tracking stopped on an unexpected error. Everything tracked before it is kept "
                      "(Ctrl+Z undoes the whole run)."))

    # The undo point is ONE camera's snapshot, plus -- when an edit also changed
    # the other cameras (a landmark added, renamed or deleted in all of them, G19)
    # -- a snapshot of each of those. Setting a new undo point forgets the extra
    # copies, so every existing `self._undo_snap = ...` stays a complete undo step.
    @property
    def _undo_snap(self):
        return self.__dict__.get("_undo_main")

    @_undo_snap.setter
    def _undo_snap(self, snap):
        self.__dict__["_undo_main"] = snap
        self.__dict__["_undo_extra"] = {}

    @property
    def _undo_extra(self) -> dict:
        return self.__dict__.setdefault("_undo_extra", {})

    def _share_landmarks(self, undoable: bool = True) -> int:
        """Give the other cameras the landmarks just made in this one (with no
        data there), so each can be selected in those cameras, clicked on the
        epipolar line and tracked (G19). `undoable`: the edit that made them has
        just set the undo point, and taking it back removes them everywhere."""
        p = self.project
        if p is None or p.n_views < 2:
            return 0
        # every camera the sync changes -- a missing name, or its list in another
        # order (G26); the working camera leads the order, so its ids never move
        changes = p.landmark_changes(p.active)
        if not changes:
            return 0
        if undoable and self._undo_snap is not None:
            for v in changes:
                if v != p.active:
                    self._undo_extra.setdefault(v, p.sessions[v].snapshot())
        n = p.sync_landmarks(p.active)
        self._refresh_companions()
        return n

    def _undo_run(self):
        if self._undo_snap is None or self.session is None or self.state != READY:
            return
        self.session.restore(self._undo_snap)
        p = self.project
        for v, snap in self._undo_extra.items():
            if p is not None and 0 <= v < p.n_views and v != p.active:
                p.sessions[v].restore(snap)
        self._undo_snap = None
        self._refresh_companions()
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
        # the run scope (G61) is part of the working state (G103): the selected points BY NAME and
        # whether the segment's row is selected. The landmark list is shared by every camera, so the
        # names go to all of them -- the selection travels with the user across a camera switch
        names = [self.session.points[q].name for q in self._selected_pids() if q < self.session.n_points]
        seg_sel = bool(self._segment_selected())
        seg_names = [self.session.segments[k].name for k in self._selected_segments() if k < self.session.n_segments]
        for sv in (self.project.sessions if self.project is not None else [self.session]):
            sv.ui_state["selected_names"] = list(names)
            sv.ui_state["segment_selected"] = seg_sel
            sv.ui_state["segments_selected"] = list(seg_names)       # (G149) which segments, by name
        for sv in (self.project.sessions if self.project is not None else [self.session]):
            sv.ui_state["hidden_animals"] = [a.name for a in sv.segments if not a.shown]     # (G154) display
        st["follow"] = self.btn_follow.isChecked()
        st["autopause"] = self.btn_autopause.isChecked()
        st["roi"] = self.btn_roi.isChecked()
        st["track_mode"] = self._track_mode
        st["track_all"] = self.act_track_all.isChecked()
        st["marker_size"] = self.marker_spin.value()
        st["show_mask"] = self.btn_mask.isChecked()
        st["mask_opacity"] = float(self._mask_opacity)
        st["show_midline"] = self.act_show_midline.isChecked()
        st["show_bones"] = self.act_show_bones.isChecked()
        st["seg_backend"] = self._seg_backend
        st["point_backend"] = self._point_backend
        st["trail_len"] = int(self._trail_len)
        st["trail_future"] = bool(self._trail_future)
        st["onion"] = self.act_onion.isChecked()
        st["loupe"] = self.act_loupe.isChecked()
        st["epipolar"] = self.act_epipolar.isChecked()
        st["display_filter"] = self._display_filter_key()
        st["region_shape"] = self._region_shape
        st["timeline"] = [int(v) for v in self.timeline._view]

    def _apply_ui_state(self):
        """Restore the exact working state after a project/autosave load."""
        if self.session is None:
            return
        st = self.session.ui_state
        self.btn_follow.setChecked(bool(st.get("follow", False)))
        self.btn_autopause.setChecked(bool(st.get("autopause", True)))
        self.btn_roi.setChecked(bool(st.get("roi", True)))
        self.btn_mask.setChecked(bool(st.get("show_mask", True)))
        pb = str(st.get("point_backend", "") or "")
        if pb == "alltracker" and not _alltracker_available():
            pb = "cotracker3"
        self._point_backend = pb if pb in self._pm_acts else self._preferred_point_backend()
        self._pm_acts[self._point_backend].setChecked(True)
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
        self.act_track_all.setChecked(bool(st.get("track_all", False)))
        self.marker_spin.setValue(int(np.clip(st.get("marker_size", 3), 2, 24)))
        try:
            tl = int(st.get("trail_len", TRAIL_FRAMES))
        except (TypeError, ValueError):
            tl = TRAIL_FRAMES
        self._set_trail_len(tl, announce=False)         # any saved length: a custom one (G33)
        self.act_trail_future.setChecked(bool(st.get("trail_future", False)))
        self.act_onion.setChecked(bool(st.get("onion", False)))
        self.act_loupe.setChecked(bool(st.get("loupe", False)))
        self.act_epipolar.setChecked(bool(st.get("epipolar", True)))
        self._set_display_filter(str(st.get("display_filter", "none")), announce=False)
        self._set_region_shape(str(st.get("region_shape", "circle")), announce=False)
        if not self.session.annotator:
            self.session.annotator = self._default_annotator()
        sel = int(st.get("selected", -1))
        # the run scope: the whole multi-selection and the segment's row (G103). The saved NAMES
        # are the selection when there are any (a stale per-camera index would add a point the
        # user had not selected); the current point is the saved one when it is among them.
        rows = [q for q in (self.session.pid_by_name(str(nm)) for nm in (st.get("selected_names") or []))
                if q is not None and q < self.session.n_points]
        if rows:
            self._on_select(sel if sel in rows else rows[0])
            self.layers.set_selected(pids=rows)
            self._update_track_button()
        elif 0 <= sel < self.session.n_points:
            self._on_select(sel)
        if isinstance(st.get("segments_selected"), list):       # (G149) by name
            want = set(st.get("segments_selected"))
            self._select_segments([k for k, n in enumerate(self.session.segment_names()) if n in want])
        elif st.get("segment_selected") and self.session.segments:
            self._select_segments([0])
        self._sync_s_target()
        # view restore runs after the pending layout pass, else fit() wins
        zoom = float(st.get("zoom", 0.0))
        cx, cy = float(st.get("center_x", 0.0)), float(st.get("center_y", 0.0))
        uz = bool(st.get("user_zoomed", False))
        QTimer.singleShot(0, lambda: self.canvas.set_view_state(zoom, cx, cy, uz))
        tl = st.get("timeline")
        if isinstance(tl, (list, tuple)) and len(tl) == 2:
            try:
                v0, v1 = int(tl[0]), int(tl[1])
                if 0 <= v0 < v1 < self.session.n_frames:
                    self.timeline._set_view(v0, v1 - v0)
            except (TypeError, ValueError):
                pass

    def _signature(self):
        """Changes since a save: every session's data_version (marking a
        session dirty moves it on)."""
        p = self.project
        return None if p is None else tuple(x.data_version for x in p.sessions)

    def _ui_global_state(self) -> dict:
        """state.json: the window-wide toggles and the layout."""
        s = self.session
        tools = {k: s.ui_state.get(k) for k in GLOBAL_UI_KEYS} if s is not None else {}
        geo = self.normalGeometry() if self.isMaximized() else self.geometry()
        return {"tools": tools, "layout": {
            "window": [geo.x(), geo.y(), geo.width(), geo.height()], "maximized": self.isMaximized(),
            "dock_visible": self.dock.isVisible(), "dock_floating": self.dock.isFloating(),
            "splitter": [int(v) for v in self._split.sizes()], "solo": self.act_solo.isChecked(),
            "sync": self._sync_mode,
            "step": int(self.step_spin.value()), "onboarding": self.act_onboarding.isChecked(),
            "views": self._views_layout()}}

    def _views_layout(self) -> dict:
        """(G169, G170) The views' arrangement and the hidden cameras, BY NAME (so a camera removed or
        renumbered later cannot shift them onto another one): layout state, never unsaved work."""
        p = self.project
        if p is None or p.n_views < 2:
            return {}
        return {"order": [p.name(i) for i in self.grid.display_order()],
                "hidden": [p.name(i) for i in sorted(self.grid.hidden())]}

    def _apply_views_layout(self, views) -> None:
        p = self.project
        if p is None or p.n_views < 2 or not isinstance(views, dict):
            return
        at = {n: i for i, n in enumerate(p.names)}
        order = [at[str(n)] for n in (views.get("order") or []) if str(n) in at]
        hidden = {at[str(n)] for n in (views.get("hidden") or []) if str(n) in at} - {p.active}
        self.grid.set_arrangement(order=order, hidden=hidden)
        self._on_views_arranged()

    def _apply_layout(self, layout: dict | None) -> None:
        """The panels and window as they were saved; a window rectangle that
        no longer fits any screen (another computer, a detached monitor) is
        ignored."""
        if not layout:
            return
        try:
            win = [int(v) for v in layout.get("window") or []]
            if len(win) == 4 and QApplication.platformName() != "offscreen":
                rect = QRect(*win)
                if any(sc.availableGeometry().contains(rect.center()) and
                       rect.width() <= sc.availableGeometry().width() and
                       rect.height() <= sc.availableGeometry().height() for sc in QApplication.screens()):
                    self.setGeometry(rect)
                if layout.get("maximized"):
                    self.showMaximized()
            self.dock.setVisible(bool(layout.get("dock_visible", True)))
            self.dock.setFloating(bool(layout.get("dock_floating", False)))
            sp = layout.get("splitter")
            if isinstance(sp, list) and len(sp) == len(self._split.sizes()) and all(int(v) >= 0 for v in sp):
                self._split.setSizes([int(v) for v in sp])
            if self.act_solo.isEnabled():
                self._set_sync_mode("active" if layout.get("sync") == "active" else "all")    # (G24)
                self.act_solo.setChecked(bool(layout.get("solo", False)))
                self._apply_views_layout(layout.get("views"))                                  # (G169, G170)
            self.step_spin.setValue(int(layout.get("step", self.step_spin.value())))
            self.act_onboarding.setChecked(bool(layout.get("onboarding", True)))
        except (TypeError, ValueError):
            pass            # a hand-edited state.json with odd values: keep the defaults

    def _start_writer(self, frozen, path, on_done=None, wait: bool = False, folder: bool = False,
                      title: str | None = None, detail: str = "", **kw):
        """Run projectfile.write (a single file) or write_folder_over (the
        project folder, `folder`) on a worker thread. wait=True returns (ok,
        error) after it finished, the window repainting meanwhile -- with the
        loading card (`title`) once it takes longer than a blink (G48)."""
        prev = self._save_worker
        if prev is not None and prev.isRunning():
            # (I201) a recovery writer still running (the 30 s autosave started one while a
            # question was open): finish it before the next writer; never drop it running
            prev.wait(60000)
            _retire(prev)
        w = _SaveWorker(frozen, path, folder, **kw)
        result = {}

        def finished(ok, err):
            result.update(ok=ok, err=err)
            if on_done is not None:
                on_done(ok, err)
        w.done.connect(finished)
        self._save_worker = w
        w.start()
        if wait:
            tok = self._busy_push(title, detail) if title else None
            try:
                while w.isRunning() or "ok" not in result:
                    QApplication.processEvents(QEventLoop.AllEvents, 50)
                    w.wait(10)
                QApplication.processEvents()
            finally:
                self._busy_pop(tok)
            return result["ok"], result["err"]
        return None, None

    def _wait_for_exports(self) -> None:
        """A save rewrites the folder the exports/ refresh reads: let it finish
        first -- with the card, the window live (it waited on the GUI thread, G48)."""
        w = self._exports_worker
        if w is not None and w.isRunning():
            self._in_background("Finishing the exports", lambda: w.wait(),
                                detail="Bringing exports/ up to date before the next save…")

    def _autosave(self, wait: bool = False):
        """Unsaved work -> the recovery folder, every 30 s and after runs. The
        project file itself changes only on Save, so the last Save is always
        there to go back to. Binary tables: ~50 ms even for 40k frames."""
        p = self.project
        if p is None or not p.dirty:
            return
        sig = self._signature()
        if sig == self._recovery_sig:
            return
        if self._save_worker is not None and self._save_worker.isRunning():
            if not wait:
                return                         # the next tick catches up
            self._save_worker.wait()
        self._sync_ui_state()
        pid = self._project_id
        zp = recovery.paths(pid)[0]
        info = dict(base_saved_at=self._saved_at,
                    last_path=str(self.project_path) if self.project_path else None,
                    videos=[x.video_path for x in p.sessions], temporary=self._saved_at is None)
        frozen = projectfile.freeze(p, self._ui_global_state(), pid, target=self.project_path, binary_tracks=True,
                                    layout=self._project_layout())

        def done(ok, err):
            if ok:
                recovery.write_info(pid, **info)
                self._recovery_sig = sig
                self.statusBar().showMessage(f"Unsaved work kept safe ✓  {time.strftime('%H:%M:%S')}", 2000)
                if recovery.folder()[1] and not getattr(self, "_recovery_fallback_said", False):
                    self._recovery_fallback_said = True
                    self.toast.show_message(
                        f"Kinetrace's own folder cannot be written, so unsaved work is kept in "
                        f"{recovery.folder()[0]} instead.", "info", 10000)
            else:
                self.statusBar().showMessage(f"Could not keep a recovery copy: {err}", 10000)
        self._start_writer(frozen, zp, done, wait=wait, compresslevel=0, fsync=False, backup=False)

    def _leave_project(self, discard: bool = False) -> None:
        """Before another video or project replaces this one: unsaved work into
        recovery; for a saved project, where the user was (frame, zoom,
        toggles) into a small sidecar applied the next time it opens.
        `discard` (the user chose Discard, G143): the unsaved work is dropped as
        on close, and this state counts as handled, so a later `_leave_project`
        of the same open does not write it back (an edit after it still would)."""
        if self.project is None:
            return
        if discard:
            if self._save_worker is not None and self._save_worker.isRunning():
                self._save_worker.wait()
            recovery.discard(self._project_id)
            self._recovery_sig = self._signature()
            return
        self._autosave(wait=True)
        if self._saved_at is not None and not self.project.dirty:
            self._sync_ui_state()
            recovery.write_view(self._project_id, {
                "state": self._ui_global_state(),
                "cameras": [{"current_frame": int(x.current_frame),
                             "ui": {k: x.ui_state.get(k) for k in projectfile.VIEW_KEYS}}      # (G103)
                            for x in self.project.sessions]}, self._saved_at)

    def _project_layout(self) -> str:
        """"zip" while this project is an older single file the user chose to
        keep as one (I145); "folder" otherwise."""
        p = self.project_path
        return "zip" if p is not None and p.is_file() and self._keep_single_file else "folder"

    def _run_in_progress(self) -> bool:
        """A Track press is not over yet: a run is going, or what follows it is still to come -- the
        second pass of a two-pass run, the next camera of an every-camera run, the next re-track
        stretch (each starts from a timer once the one before has ended)."""
        return (self.state == TRACKING or self._passes is not None or self._multi is not None
                or self._retrack is not None)

    def _save_project(self) -> bool:
        if self.project is not None and self._run_in_progress():
            # (I265) the data is still being written: the save is done the moment the run is over
            # (a Save pressed during a run used to do nothing at all, so the work seemed saved and was not)
            first = not self._save_after_run
            self._save_after_run = True
            self.toast.show_message("Tracking is still running: the project is saved as soon as it stops "
                                    "(<b>X</b> stops it now).", "info", 8000)
            self.statusBar().showMessage("Save: as soon as this run stops", 8000)
            if first:
                QTimer.singleShot(300, self._save_when_run_over)
            return False
        if self.project_path is None:
            return self._save_project_as()
        if self.project is None or self._saving:
            return False
        path = self.project_path
        if path.is_file() and not self._keep_single_file:
            # a project saved before I145 is one zip: offer the folder (once per project)
            ans = QMessageBox.question(
                self, "Save as a project folder?",
                f"{path.name} is a single-file project (the older form). Kinetrace now keeps a project as a "
                "FOLDER of the same name: every table is a CSV you can open directly, and a save writes only "
                f"what changed.\n\nSave it as a folder? The single file is kept beside it as {path.name}.bak.\n"
                "(No keeps saving it as one file.)",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if ans == QMessageBox.Cancel:
                return False
            self._keep_single_file = ans == QMessageBox.No
        # (I201) the question above comes first: a recovery copy the 30 s autosave starts while it
        # is open is waited for here, not orphaned by the save's own writer
        if self._save_worker is not None and self._save_worker.isRunning():
            self._save_worker.wait()
        self._wait_for_exports()                 # it reads the folder this save rewrites (G42)
        folder = self._project_layout() == "folder"
        self._saving = True
        try:
            self._sync_ui_state()
            sig = self._signature()
            saved_at = projectfile.timestamp()
            frozen = projectfile.freeze(self.project, self._ui_global_state(), self._project_id,
                                        target=path, saved_at=saved_at, layout="folder" if folder else "zip")
            self.statusBar().showMessage("Saving…")
            ok, err = self._start_writer(frozen, path, wait=True, folder=folder, title=f"Saving {path.name}",
                                         detail="Writing the files that changed since the last save…")
        finally:
            self._saving = False
        if not ok:
            QMessageBox.critical(
                self, "Could not save the project",
                f"{path}\n\n{err}\n\nNothing was changed there: the previous save is as it was. "
                "Check that the folder exists, that you may write to it and that none of its files is open in "
                "another program (a CSV in Excel, for example), then save again or use File → Save Project "
                "As… to save somewhere else. Your work is still in the program, and a recovery copy is kept.")
            return False
        stats = self._save_worker.stats if folder else None
        self._saved_at = saved_at
        self.project.path = str(path)
        if folder:
            self._project_dir = path.resolve()        # videos are found relative to the folder itself
        if self._signature() == sig:          # nothing changed while it was writing
            self.project.dirty = False
        recovery.discard(self._project_id)
        self._recovery_sig = None
        extra = (f" ({self.project.n_views} cameras)" if self.project.n_views > 1 else "")
        what = ""
        if stats is not None:
            n = max(0, stats["written"] - 1) + stats["removed"]      # kinetrace.json changes every time
            what = f" · {n} file{'s' if n != 1 else ''} updated" if n else " · nothing else changed"
        tidy = (stats or {}).get("tidy_up")
        if tidy:
            # (G110) the save itself counted; only the tidying after it did not all work
            self.statusBar().showMessage(f"Project saved ✓  {path.name}{extra}", 5000)
            self.toast.show_message(
                "Saved. Not everything could be tidied up: " + "; ".join(str(x) for x in tidy[:3])
                + ". Nothing is lost; the next save tries again.", "warn", 10000)
        else:
            self.statusBar().showMessage(f"Project saved ✓  {path.name}{extra}{what}", 5000)
        if folder and self.project.exports:
            self._refresh_exports()
        return True

    def _save_when_run_over(self) -> None:
        """(I265) The save asked for during a run, once the whole Track press is over (every pass,
        camera and re-track stretch) and no question or loading card is up."""
        if not self._save_after_run:
            return
        if self.project is None:
            self._save_after_run = False
            return
        if self._run_in_progress() or self._loading or QApplication.activeModalWidget() is not None:
            QTimer.singleShot(300, self._save_when_run_over)
            return
        self._save_after_run = False
        self._save_project()

    def _changes_text(self, most: int = 8) -> str:
        """For the close question (G44): what differs from
        the last save, in words ("cam1: P1 (positions)") — a stray click that
        moved a point is then seen before it is saved. '' when that cannot be
        told (never saved, a single file)."""
        p, path = self.project, self.project_path
        if p is None or path is None or self._project_layout() != "folder":
            return ""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._sync_ui_state()
            frozen = projectfile.freeze(p, self._ui_global_state(), self._project_id, target=path)
            got = projectfile.changes_since_save(frozen, path)
            if got is None:
                return ""
            folders = projectfile.camera_folders(p.names)
            lm = {f: dict(zip(projectfile.landmark_files([q.name for q in s.points]), [q.name for q in s.points]))
                  for f, s in zip(folders, p.sessions)}
            lines = projectfile.describe_changes(*got, list(zip(folders, p.names)), lm)
        except Exception:        # noqa: BLE001 - the question still works without the list
            return ""
        finally:
            QApplication.restoreOverrideCursor()
        if not lines:
            return ("Nothing differs from your last save any more (what changed was put back), so "
                    "saving or not ends the same.\n\n")
        more = f"\n  … and {len(lines) - most} more" if len(lines) > most else ""
        return "Changed since your last save:\n" + "\n".join(f"  • {x}" for x in lines[:most]) + more + "\n\n"

    def _project_target(self, path: str) -> Path | None:
        """Where Save As may put a project folder (I145), or None after saying
        why not: never inside another project's folder, never over a folder
        that is not a project, and over another project only when asked."""
        p = Path(path)
        if p.name.lower() == projectfile.META:
            p = p.parent
        if not p.name.lower().endswith(PROJECT_SUFFIX):
            p = p.with_name(p.name + PROJECT_SUFFIX)
        inside = next((a for a in p.parents if projectfile.is_folder_project(a)), None)
        if inside is not None:
            # a project is a folder, and the save dialog OPENS a folder whose name is chosen
            # (Windows does) -- so choosing an existing project, e.g. the proposed
            # <video>.kinetrace, came back as a path INSIDE it and was refused: the owner had
            # to invent a new name (G55). Inside a project = that project was meant.
            if self.project_path is not None and inside.resolve() == self.project_path.resolve():
                return inside                                   # this very project: just save it
            if QMessageBox.question(
                    self, "Save as this project?",
                    f"You chose the project folder\n{inside}\n\nSave this work as {inside.name}, replacing "
                    "what that project holds now? (Its last save stays in its .history folder until the next "
                    "save.)\n\nNo: choose another place or name (a project cannot be put inside another "
                    "project).", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return None
            return inside
        if p.is_dir():
            if projectfile.is_folder_project(p):
                mine = self.project_path is not None and p.resolve() == self.project_path.resolve()
                if not mine and QMessageBox.question(
                        self, "Replace project?",
                        f"{p.name} is already a Kinetrace project folder.\n\nReplace that project with this one? "
                        "(Its last save stays in its .history folder until the next save.)",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                    return None
            elif not projectfile.holds_only_own_entries(p):      # (I244) a failed first save leaves only its .cache
                QMessageBox.warning(self, "Save project",
                                    f"{p}\n\nis a folder that is not a Kinetrace project. Choose another name.")
                return None
        return p

    def _save_project_as(self) -> bool:
        if self.session is None:
            return False
        base = self.project_path or (Path(self.info.path) if self.info else None)
        start = str(base.with_suffix(PROJECT_SUFFIX)) if base else ""
        path, _ = QFileDialog.getSaveFileName(self, "Save project (a folder of this name is made)", start,
                                              f"Kinetrace project folder (*{PROJECT_SUFFIX})")
        if not path:
            return False
        target = self._project_target(path)
        if target is None:
            return False
        if self.project_path is not None and target.resolve() == self.project_path.resolve() and target.is_dir():
            return self._save_project()                 # the open project itself: a save, same id (G55)
        before = (self.project_path, self._project_id, self._saved_at, self._keep_single_file, self._project_dir)
        # a new project folder is a new project: its own id, so the two never
        # share unsaved work
        self.project_path, self._project_id = target, projectfile.new_id()
        self._keep_single_file = False
        self._project_dir = target.resolve()
        if not self._save_project():
            (self.project_path, self._project_id, self._saved_at, self._keep_single_file,
             self._project_dir) = before
            return False
        recovery.discard(before[1])
        return True

    def _export_single_file(self) -> None:
        """File → Export Project as One File: the whole project (the folder's
        files, not its .cache / .history / exports) in one .kinetrace zip, to
        e-mail or archive. Kinetrace opens it like a folder (I145)."""
        if self.project is None or self._saving:
            return
        base = self.project_path or Path(self.info.path)
        start = str(base.with_name(base.name.removesuffix(PROJECT_SUFFIX).rsplit(".", 1)[0]
                                   + " (one file)" + PROJECT_SUFFIX))
        path, _ = QFileDialog.getSaveFileName(self, "Export project as one file", start,
                                              f"Kinetrace project, one file (*{PROJECT_SUFFIX})")
        if not path:
            return
        out = Path(path if path.lower().endswith(PROJECT_SUFFIX) else path + PROJECT_SUFFIX)
        if out.is_dir():
            QMessageBox.warning(self, "Export project as one file",
                                f"{out}\n\nis a folder (a project folder, perhaps). Choose another name.")
            return
        if self._save_worker is not None and self._save_worker.isRunning():
            self._save_worker.wait()
        self._sync_ui_state()
        # (I243) the export is a COPY: its own project id, so it and the original never share a recovery slot
        frozen = projectfile.freeze(self.project, self._ui_global_state(), projectfile.new_id(), target=out,
                                    layout="zip")
        self.statusBar().showMessage("Writing the single file…")
        self._saving = True                      # no Save / autosave write meanwhile (G48)
        try:
            ok, err = self._start_writer(frozen, out, wait=True, backup=False, title=f"Writing {out.name}",
                                         detail="The whole project into one file (the larger the project, "
                                                "the longer: about 2 s per million tracked rows)…")
        finally:
            self._saving = False
        if not ok:
            QMessageBox.critical(self, "Could not export the project", f"{out}\n\n{err}")
            return
        self.toast.show_message(f"Project written as one file: <b>{out.name}</b> "
                                f"({out.stat().st_size / 1e6:.1f} MB). File → Open Project opens it; "
                                "your project folder is unchanged.", "success", 9000)

    def _exports_dialog(self) -> None:
        """File → Keep Exports Up to Date… (G42): the formats written into the
        project folder's exports/ at every save."""
        from kinetrace.autoexport import ExportsDialog
        p = self.project
        if p is None:
            return
        dlg = ExportsDialog(self, p.exports, p.n_views, p.reconstruction is not None,
                            self.project_path if self._project_layout() == "folder" else None)
        if dlg.exec() != QDialog.Accepted:
            return
        chosen = dlg.chosen()
        if chosen == p.exports:
            return
        p.exports = chosen
        p.sessions[p.active].dirty = True            # a project setting: saved with the project
        if not chosen:
            self.statusBar().showMessage("Exports on save switched off (files already in exports/ are kept)", 6000)
            return
        self.toast.show_message("These files are written into the project's <b>exports</b> folder at every "
                                "save, only when their data changed. Save the project (Ctrl+S) to write them "
                                "now.", "info", 9000)

    def _refresh_exports(self) -> None:
        """Bring exports/ up to date with the save just made, off the GUI
        thread, from the saved folder itself (so they match it exactly)."""
        from kinetrace.autoexport import ExportsWorker
        self._wait_for_exports()
        scorer = self._dlc_scorer()             # (R12) one rule with Ctrl+E
        w = ExportsWorker(self.project_path, list(self.project.exports), scorer)
        w.done.connect(self._on_exports_done)
        self._exports_worker = w
        self.statusBar().showMessage("Updating exports/…")      # until it is done (it said 3 s, G48)
        w.start()

    def _on_exports_done(self, written: list, errors: list) -> None:
        if errors:
            self.toast.show_message("Some exports could not be written:<br>" + "<br>".join(errors[:4]),
                                    "warn", 12000)
        elif written:
            self.statusBar().showMessage(f"exports/ updated ✓  {', '.join(written[:4])}"
                                         f"{' …' if len(written) > 4 else ''}", 6000)
        else:
            self.statusBar().showMessage("exports/ already up to date ✓", 4000)

    def _import_tracks_dialog(self, path: str | None = None):
        """File → Import Tracks: another program's 2D tracks into the camera on
        screen (trackio.py). With no video open, the video comes first."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Import tracks", "", TRACKS_FILTER)
            if not path:
                return
        try:           # a long DeepLabCut / SLEAP file: off the GUI thread (G52)
            imp = self._in_background("Reading the tracks", lambda: trackio.read(path), detail=Path(path).name)
        except trackio.TrackImportError as e:
            QMessageBox.warning(self, "Cannot import these tracks", str(e))
            return
        except Exception as e:  # noqa: BLE001 - (G142) a failed READ in words, the traceback logged
            QMessageBox.warning(self, "Cannot import these tracks",
                                _plain_error(e, "The tracks could not be read", reading=True))
            return
        if self.session is None:
            QMessageBox.information(self, "Import tracks",
                                    f"{Path(path).name} is a {imp.label}. Now open the video these tracks "
                                    "belong to.")
            vpath, _ = QFileDialog.getOpenFileName(self, "Open the video of these tracks",
                                                   str(Path(path).parent), VIDEO_FILTER)
            if vpath:
                self._after_open = (vpath, lambda: self._apply_imported(imp, path))
                self._open_video(vpath)
            return
        self._apply_imported(imp, path)

    def _import_points3d(self):
        """File -> Import -> 3D Points: another program's 3D landmarks become
        this project's 3D result (3D view, kinematics export)."""
        from kinetrace import calibio
        p = self.project
        if p is None or self.state != READY:
            QMessageBox.information(self, "Import 3D points", "Open the project (or its first video) first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import 3D points", self._start_folder(),
                                              "3D points - Anipose, DLTdv xyzpts, Kinetrace (*.csv);;All files (*)")
        if not path:
            return
        try:
            rec, notes = calibio.read_points3d(path, unit=p.calibration.unit if p.calibration else "")
        except calibio.CalibFormatError as e:
            QMessageBox.warning(self, "Cannot import these 3D points", str(e))
            return
        except Exception as e:  # noqa: BLE001 - (G142)
            QMessageBox.warning(self, "Cannot import these 3D points",
                                _plain_error(e, "The 3D points could not be read", reading=True))
            return
        if p.reconstruction is not None and QMessageBox.question(
                self, "Replace the 3D result?", "This project already has a 3D result. Replace it with the "
                "imported points?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._drop_reconstruction(replacement=rec)           # (I242)
        p.dirty = True
        self._apply_state()
        self.toast.show_message("3D points imported: " + "; ".join(notes) + ". Frames are the reference "
                                f"camera's ({p.name(0)}).", "info", 10000)

    def _drop_reconstruction(self, reason: str = "", replacement=None, had: bool | None = None,
                             had_hull: bool | None = None) -> bool:
        """(I242, I206) The ONE place the 3D layer's results are dropped (or the reconstruction is
        replaced by an imported / re-framed one): the volumes carved from it, the magenta
        disagreement band and its Re-track tooltip, the 3D window, the guides and the menus follow, and
        `reason` says why in the status line when there was something to lose. `had`: the result was
        there before the CALLER's own change, which already dropped it (a retime, a camera added or
        removed: the project does that itself); None = it is dropped here. `had_hull`: volumes were
        carved before that change (None = look at the cache now). True when a result was dropped."""
        p = self.project
        if p is None:
            return False
        if had is None:
            had = p.reconstruction is not None
            p.reconstruction = replacement
        elif replacement is not None:
            p.reconstruction = replacement
        if had_hull is None:
            had_hull = bool(self._hull_cache)
        self._hull_cache.clear()
        if replacement is None and not (had or had_hull):
            return False
        self._update_disagreement()
        self._refresh_view3d(force=True)
        self._refresh_guides()
        self._apply_state()
        dropped = had and replacement is None
        if reason and (dropped or had_hull):
            gone = " and ".join(x for x, on in (("the 3D result", dropped), ("the carved volume", had_hull)) if on)
            self.statusBar().showMessage(
                f"{reason[:1].upper() + reason[1:]}: {gone} made earlier {'were' if ' and ' in gone else 'was'} "
                "cleared — " + " and ".join(x for x, on in (("press Ctrl+3 (3D → Reconstruct)", dropped),
                                                            ("carve with Ctrl+4", had_hull)) if on) + " again", 8000)
        return dropped

    def _import_offsets(self):
        """File -> Import -> Camera Offsets: offsets and rates from a CSV."""
        from kinetrace import calibio
        p = self.project
        if p is None or p.n_views < 2 or self.state != READY:
            QMessageBox.information(self, "Import camera offsets",
                                    "Offsets line up two or more cameras: add the other cameras' videos first "
                                    "(＋ Add video in the CAMERAS panel).")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import camera offsets", self._start_folder(),
                                              "Camera offsets (*.csv);;All files (*)")
        if not path:
            return
        try:
            rows = calibio.read_offsets(path, p)
        except calibio.CalibFormatError as e:
            QMessageBox.warning(self, "Cannot import these offsets", str(e))
            return
        except Exception as e:  # noqa: BLE001 - (G142)
            QMessageBox.warning(self, "Cannot import these offsets",
                                _plain_error(e, "The offsets could not be read", reading=True))
            return
        # (I169) read_offsets already re-based the rows on this project's first camera; a file with no
        # row for it cannot be: its numbers are the file's own, applied only when the user says so
        if getattr(rows, "reference_missing", False) and QMessageBox.question(
                self, "Camera offsets",
                f"The file has no row for {p.name(0)}, the reference camera of this project, so its numbers "
                f"cannot be measured against {p.name(0)}.\n\nApply its numbers as written?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        had_3d = p.reconstruction is not None
        for v, off, rate in rows:
            p.set_rate(v, rate)
        for v, off, rate in rows:
            p.set_offset(v, off)
        p.dirty = True
        self._after_retime(had_3d)
        self._refresh_companions()
        self._refresh_cameras()
        n = sum(1 for v, _o, _r in rows if v != 0)
        rebased = (f" The file's numbers were measured against another camera, so they were re-based on "
                   f"{p.name(0)}." if getattr(rows, "rebased", False) else "")
        self.toast.show_message(f"Offsets imported for {n} camera(s) from {Path(path).name}.{rebased}",
                                "info", 8000)

    def _import_masks(self):
        """File -> Import -> Silhouettes: mask images from another tool become
        this camera's segment."""
        s = self.session
        if s is None or self.state != READY:
            QMessageBox.information(self, "Import silhouettes", "Open the video the masks belong to first.")
            return
        # (G151j) into the segment the user names (one highlighted row, or asked); none yet = a new one
        k = self._pick_segment("Import the silhouettes into") if s.segments else None
        if s.segments and k is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder of mask images", str(Path(self.info.path).parent))
        if not folder:
            return
        # onto an existing segment: one undo step. A NEW segment is not taken back
        # by Ctrl+Z (the undo keeps segments made after it, like removing one, I124)
        had_segment = s.animal is not None
        snap = s.snapshot() if had_segment else None
        try:           # thousands of 4K images: off the GUI thread, counted (G52)
            summ = self._in_background(
                "Importing silhouettes",
                lambda report: trackio.import_masks_png(s, folder, lambda d, n: report(f"{d} of {n} images", d, n),
                                                        i=k),
                detail=Path(folder).name, progress=True)
        except trackio.TrackImportError as e:
            QMessageBox.warning(self, "Cannot import these silhouettes", str(e))
            return
        except Exception as e:  # noqa: BLE001 - (G142)
            QMessageBox.warning(self, "Cannot import these silhouettes",
                                _plain_error(e, "The silhouettes could not be read", reading=True))
            return
        self._undo_snap = snap
        self.act_undo.setEnabled(snap is not None)
        self._refresh_animal_panel()
        self._refresh_overlay()
        self.timeline.update()
        self._apply_state()
        self.toast.show_message(summ["sentence"] + (" Ctrl+Z undoes it." if had_segment else
                                                    " To take it back, right-click the animal in LAYERS → "
                                                    "Remove the animal."), "info", 9000)

    def _apply_imported(self, imp, path: str) -> None:
        p = self.project
        if p is None or self.state != READY:
            return
        if imp.n_cameras == 1:
            targets = [(0, p.active)]
        elif imp.n_cameras == p.n_views:
            targets = [(c, c) for c in range(p.n_views)]       # the file's cameras in the project's order
        else:
            labels = [f"camera {c + 1} of the file" for c in range(imp.n_cameras)]
            choice, ok = QInputDialog.getItem(
                self, "Import tracks",
                f"This file has {imp.n_cameras} cameras and this project {p.n_views}. Which of the file's "
                f"cameras is {p.name(p.active)}?\n\n(To import every camera at once, first add the other "
                "cameras' videos with + Add video, in the file's order.)", labels, 0, False)
            if not ok or choice not in labels:
                return
            targets = [(labels.index(choice), p.active)]
        if len(targets) == 1:
            self._undo_snap = p.sessions[targets[0][1]].snapshot()   # one undo step
            self.act_undo.setEnabled(targets[0][1] == p.active)
        else:
            self._undo_snap = None                                   # the undo holds one camera only
        said = []
        for c, view in targets:
            f_of_row = (None if imp.rows == "frames"
                        else (lambda r, v=view: p.map_frame(0, v, r)))
            summ = trackio.apply(p.sessions[view], imp, camera=c, frame_of_row=f_of_row)
            said.append((f"{p.name(view)}: " if len(targets) > 1 else "") + summ["sentence"])
        self._share_landmarks(undoable=len(targets) == 1)   # new names reach every camera (G19)
        self._refresh_point_list()
        self._refresh_overlay()
        self._refresh_cameras()
        self._update_frame_label()
        self._apply_state()
        extra = [n for n in imp.notes]
        if len(targets) > 1:
            extra.append("Ctrl+Z cannot undo an import into several cameras; your last save is unchanged")
        self.toast.show_message(" ".join(said) + ("<br>" + "<br>".join(extra) if extra else ""), "info", 12000)
        self.statusBar().showMessage(f"Imported {Path(path).name}", 6000)

    def _open_project_dialog(self):
        # a project is a FOLDER (I145): the user opens it and picks its kinetrace.json;
        # a single-file project (*.kinetrace) is picked directly
        path, _ = QFileDialog.getOpenFileName(self, "Open project — in a project folder, choose kinetrace.json",
                                              self._start_folder(),
                                              f"Kinetrace project (kinetrace.json *{PROJECT_SUFFIX})")
        if not path:
            return
        settled = self._settle_unsaved(f"opening {projectfile.project_root(path).name}")     # (G143)
        if settled is not None:
            self._open_project_from_path(path, discard=settled == "discard")

    def _announce_recovery(self) -> None:
        """At start-up: say (without a blocking dialog) that unsaved work is
        waiting, and where to get it back."""
        try:
            recovery.cleanup_stale()
            waiting = [r for r in recovery.scan() if r.get("project_id") != self._project_id]
        except OSError:
            return
        if waiting:
            r = waiting[0]
            what = Path(r.get("last_path") or (r.get("videos") or ["a video"])[0]).name
            more = f" (and {len(waiting) - 1} more)" if len(waiting) > 1 else ""
            self.toast.show_message(
                f"Unsaved work from {r.get('written_at', 'an earlier session')} on <b>{what}</b>{more} can be "
                "restored: <b>click here</b> (or File → Recover Unsaved Work…)", "info", 12000,
                on_click=self._recover_dialog)          # a click used to only close the notice (G39)

    def _recover_dialog(self) -> None:
        if self.state == TRACKING:
            return
        items = [r for r in recovery.scan() if not r.get("damaged")]
        if not items:
            QMessageBox.information(self, "Recover unsaved work", "There is no unsaved work to recover.")
            return
        labels = [f"{r.get('written_at', '?')} — "
                  f"{Path(r.get('last_path') or (r.get('videos') or ['?'])[0]).name}"
                  f"{'' if r.get('last_path') else ' (never saved)'}" for r in items]
        choice, ok = QInputDialog.getItem(self, "Recover unsaved work",
                                          "Unsaved work kept by Kinetrace (newest first):", labels, 0, False)
        if ok and choice in labels:
            item = items[labels.index(choice)]
            # (G143) the open project's unsaved work is asked about first -- unless the copy chosen
            # IS that work (Discard would delete the very file being opened)
            settled = ("keep" if item.get("project_id") == self._project_id
                       else self._settle_unsaved("opening the recovered work"))
            if settled is not None:
                self._open_project_from_path(str(recovery.paths(item["project_id"])[0]), recovered=item,
                                             discard=settled == "discard")

    def _locate_video(self, want: Path, label: str) -> Path | None:
        """Ask the user where a project's video went (projects are portable;
        the footage next to them often is not)."""
        if want.is_file():          # (I208) Path("") is the current folder: it "exists" but is no video
            return want
        QMessageBox.information(
            self, "Locate video",
            f"{label}'s video was not found at:\n{want if str(want) not in ('', '.') else '(no path was stored)'}"
            "\n\nPlease locate it.")
        vpath, _ = QFileDialog.getOpenFileName(self, f"Locate video for {label}", "", VIDEO_FILTER)
        return Path(vpath) if vpath else None

    def _find_video(self, i: int, proj: Project) -> Path | None:
        """Camera i's video on THIS computer: the path relative to the project,
        its absolute path, the same file name in the project folder, its
        videos/ folder, beside it, or beside a camera found earlier (paths
        saved on another OS work) — only then ask the user."""
        entries = self._camera_entries or []
        entry = entries[i] if i < len(entries) else {"video": {"path": proj.sessions[i].video_path}}
        extra = list(self._video_dirs)
        extra += [Path(x.video_path).parent for x in proj.sessions if x.video_path and Path(x.video_path).is_file()]
        got = projectfile.locate_video(entry, self._project_dir, extra)
        return got if got is not None else self._locate_video(Path(proj.sessions[i].video_path), proj.name(i))

    def _open_project_from_path(self, path: str, recovered: dict | None = None, discard: bool = False):
        """Open a project folder (given as the folder or its kinetrace.json) or
        a single-file .kinetrace. `recovered`: the file is a recovery copy
        chosen in Recover Unsaved Work. `discard`: drop the open project's
        unsaved work instead of keeping it in recovery (G143)."""
        path = str(projectfile.project_root(path))   # kinetrace.json -> its folder, the project
        if self.state == TRACKING:
            return
        pname = Path(path).name
        # the card shows at once: reading the file happens right here
        ptok = self._busy_push(f"Opening {pname}", "Reading the project file…", 0,
                               self._open_hint(path, "project"), immediate=True)
        QApplication.processEvents()
        try:
            self._open_project_impl(path, recovered, pname, discard)
        finally:
            self._busy_pop(ptok)             # the camera open, if it started, holds its own level

    def _open_project_impl(self, path: str, recovered: dict | None, pname: str, discard: bool = False):
        """Open a project in five steps (R9): read it, decide about unsaved work, work out where its
        files are, find and read every camera's video, adopt it once the working camera is open."""
        self._leave_project(discard)
        got = self._read_project(path, pname)
        if got is None:
            return
        plan = self._decide_unsaved_work(path, recovered, pname, *got)
        self._set_project_home(path, recovered, plan)
        proj = plan.proj
        located = self._locate_project_videos(proj)
        if located is None:
            return
        # every camera's video is located first (a moved one asks), then ALL are read
        # at once off the GUI thread -- the working one no longer waits in front of the
        # others (2 x 4K cameras: 5.1 -> ~2.6 s; each used to freeze the window 2.4 s)
        video = located[proj.active]
        n = proj.n_views
        infos = self._probe_many([v for v in located.values() if v], f"Opening {pname}",
                                 self._open_hint(str(video), "project"))
        ainfo = infos.get(str(video), "cancelled")
        if isinstance(ainfo, str):
            if ainfo == "cancelled":
                self.statusBar().showMessage(f"Opening {pname} cancelled — nothing was changed", 5000)
            else:
                QMessageBox.critical(self, "Could not open video", f"{proj.name(proj.active)}:\n\n{ainfo}")
            return
        if self._first_frame_token is not None:      # (G109) a previous open still waiting for its picture
            self._first_frame_arrived()
        tok = self._busy_push(f"Opening {pname}", f"{n} camera{'s' if n != 1 else ''} read; building the "
                              "project…")
        self._attach_with_card(ainfo, lambda info: self._adopt_opened_project(info, plan, located, infos), tok)

    def _read_project(self, path: str, pname: str):
        """Step 1 of opening a project: read it off the GUI thread (the card's spinner keeps turning,
        G47). (project, state, meta), or None after saying why it cannot be opened."""
        try:
            return self._in_background(f"Opening {pname}", lambda: projectfile.read(path),
                                       detail="Reading the project file…")
        except projectfile.ProjectFileError as e:
            QMessageBox.critical(self, "Could not open project", f"{path}\n\n{e}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open project",
                                 _plain_error(e, f"{path} could not be read", reading=True))      # (G142)
        return None

    def _decide_unsaved_work(self, path: str, recovered: dict | None, pname: str, proj: Project, state: dict,
                             meta: dict) -> _OpenPlan:
        """Step 2: what the project opens as -- the file as it was saved, a recovery copy the user
        chose (`recovered`), or this file's unsaved changes / another copy's, after asking. Returns
        the plan: the project to adopt (re-read from the recovery copy when it was chosen), its id and
        the save it came from, where Save goes, whether it is unsaved, and a recovery to set aside
        once its copy is open (I209)."""
        pid = str(meta.get("project_id") or projectfile.new_id())
        saved_at = meta.get("saved_at")
        file_path: Path | None = Path(path)
        unsaved = False
        decline_after: str | None = None       # (I209) the recovery set aside only once its copy is open
        if recovered is not None:
            # a recovery copy: Save goes back to the project it came from when
            # that file is still the save the work started from
            last = recovered.get("last_path")
            file_path = Path(last) if last and projectfile.is_project(last) else None
            saved_at = recovered.get("base_saved_at") if file_path is not None else None
            if file_path is not None:
                try:        # which save it is: kinetrace.json alone, not the whole project
                    if projectfile.read_meta(file_path).get("saved_at") != saved_at:
                        file_path, saved_at = None, None      # saved elsewhere since: a separate copy
                except projectfile.ProjectFileError:
                    file_path, saved_at = None, None
            unsaved = True
        else:
            found = recovery.find(pid)
            if found is not None and not found.get("damaged"):
                # (I168) a recovery made by a session that had a camera left out holds fewer cameras
                # than this file: it is not "this project's unsaved changes" (Yes by default)
                n_rec = len(found.get("videos") or [])
                same = found.get("base_saved_at") == saved_at and (not n_rec or n_rec == proj.n_views)
                ask = (f"Unsaved changes to this project from {found.get('written_at', 'earlier')} were found.\n\n"
                       "Restore them? (No opens the last saved version; the unsaved changes are kept in "
                       f"{recovery.folder()[0] / 'declined'}.)" if same else
                       f"Unsaved changes from another copy of this project ({found.get('written_at', 'earlier')}, "
                       f"{found.get('last_path') or 'never saved'}) were found.\n\nOpen them as a separate, "
                       "unsaved copy instead of this file? (No opens this file; the other changes are kept in "
                       f"{recovery.folder()[0] / 'declined'}.)")
                if QMessageBox.question(self, "Unsaved changes found", ask, QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.Yes if same else QMessageBox.No) == QMessageBox.Yes:
                    try:
                        proj, state, meta = self._in_background(
                            f"Opening {pname}", lambda: projectfile.read(recovery.paths(pid)[0]),
                            detail="Reading the unsaved changes…")
                        unsaved = True
                        if not same:
                            # it lives on as a new copy (its own id); the old recovery
                            # is kept aside so this file stops offering it -- but only AFTER the
                            # copy is open: a video not located or a Cancel used to strand the work
                            # in declined/, which File → Recover does not list (I209)
                            decline_after = pid
                            pid, file_path, saved_at = projectfile.new_id(), None, None
                    except projectfile.ProjectFileError as e:
                        recovery.quarantine(pid)
                        QMessageBox.warning(self, "Unsaved changes could not be read",
                                            f"They could not be read ({e}) and were moved aside; the last "
                                            "saved version opens.")
                else:
                    recovery.decline(pid)
            if not unsaved:
                view = recovery.read_view(pid, saved_at)
                if view:                        # where the user was when they last closed it
                    state = view.get("state") or state
                    for x, cv in zip(proj.sessions, view.get("cameras") or []):
                        x.current_frame = int(np.clip(int(cv.get("current_frame", 0)), 0, x.n_frames - 1))
                        x.ui_state.update({k: v for k, v in (cv.get("ui") or {}).items() if v is not None})
                        projectfile.apply_hidden_animals(x)          # (G154) the silhouettes shown / hidden
                    for x in proj.sessions:
                        x.ui_state.update((state.get("tools") or {}))
                        x.dirty = False
        return _OpenPlan(proj, state, meta, pid, saved_at, file_path, unsaved, decline_after)

    def _set_project_home(self, path: str, recovered: dict | None, plan: _OpenPlan) -> None:
        """Step 3: where the project's own files are, so its videos are looked for there
        (`_find_video`), and the notice when its last save was cut short."""
        meta, file_path = plan.meta, plan.file_path
        self._camera_entries = list(meta.get("_cameras") or [])
        # relative video paths start at the project folder itself (format 2) or at
        # the folder holding a single file; a recovery copy uses its project's
        home = file_path if (recovered is not None and file_path is not None) else Path(path)
        try:
            home_meta = projectfile.read_meta(home) if home != Path(path) else meta
        except projectfile.ProjectFileError:
            home, home_meta = Path(path), meta
        self._project_dir = projectfile.video_base(home, home_meta)
        root = projectfile.project_root(home).resolve()
        self._video_dirs = ([root, root / projectfile.VIDEOS_DIR] if root.is_dir() else []) + [root.parent]
        if meta.get("_interrupted") == "undone":
            self.toast.show_message("The last save of this project was cut short (the program or the computer "
                                    "stopped while saving), so the project was put back as it was at the save "
                                    "before. Anything since then is in File → Recover Unsaved Work….",
                                    "warn", 15000)

    def _locate_project_videos(self, proj: Project) -> dict | None:
        """Step 4: every camera's video on this computer, the working camera's first: {view: path, or
        None when it was not found and the user did not locate it}. None = the working camera's video
        is missing: nothing is opened (the project file is unchanged)."""
        video = self._find_video(proj.active, proj)
        if video is None:
            self.statusBar().showMessage(
                f"Project not opened: the video of {proj.name(proj.active)} was not located. Open the project "
                "again and point to the video when asked (the project file is unchanged)", 10000)
            return None
        located = {i: (str(video) if i == proj.active else self._find_video(i, proj)) for i in range(proj.n_views)}
        return {i: (str(v) if v is not None else None) for i, v in located.items()}

    def _adopt_opened_project(self, info: VideoInfo, plan: _OpenPlan, located: dict, infos: dict) -> None:
        """Step 5, run once the working camera's video is open: check its length, take over the
        project's identity (id, the save it came from, where Save goes), build the views
        (`_adopt_project`) and set aside the recovery it replaced."""
        proj = plan.proj
        master = proj.sessions[proj.active]
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
        self.project_path = plan.file_path
        self._keep_single_file = False
        self._project_id, self._saved_at, self._recovery_sig = plan.pid, plan.saved_at, None
        self._adopt_project(proj, info, located=located, infos=infos)
        if plan.unsaved:
            self.project.dirty = True
        if plan.decline_after is not None:
            recovery.decline(plan.decline_after)      # (I209) the copy is open: now the old recovery is set aside
        QTimer.singleShot(0, lambda: self._apply_layout(plan.state.get("layout")))

    def _adopt_project(self, proj: Project, active_info: VideoInfo, located: dict | None = None,
                       infos: dict | None = None):
        """Replace the one-view project `_attach_video` just built with the
        loaded one, opening every other camera's video. `located` {view: path or
        None} and `infos` {path: VideoInfo | error | "cancelled"}: the cameras
        already found and read (Open Project reads them all at once)."""
        active = proj.active
        self.project = proj
        self._wand_result = (None, None)      # (I33)
        runtimes: list[_ViewRuntime | None] = [None] * proj.n_views
        runtimes[active] = self._views[0]
        drop: list[int] = []
        # locate every other camera's video first (a moved one asks), then read them
        # all at once off the GUI thread: this froze the window ~2.4 s per 4K camera
        vids = {}
        for i in range(proj.n_views):
            if i == active:
                continue
            vid = located.get(i) if located is not None else self._find_video(i, proj)
            if vid is None:
                drop.append(i)
            else:
                vids[i] = str(vid)
        infos = dict(infos or {})
        todo = [v for v in vids.values() if v not in infos]
        if todo:
            infos.update(self._probe_many(todo, f"Opening the project's other {len(todo)} camera"
                                          f"{'s' if len(todo) != 1 else ''}", self._open_hint(todo[0], "project")))
        for i in range(proj.n_views):
            if i == active or i not in vids:
                continue
            s = proj.sessions[i]
            info = infos.get(vids[i], "cancelled")
            if isinstance(info, str):
                QMessageBox.warning(self, "Could not open camera",
                                    f"{proj.name(i)}: " + ("opening was cancelled" if info == "cancelled" else info)
                                    + "\n\nIt is left out of this session.")
                drop.append(i)
                continue
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
        # (I207) the WORKING camera too: its video may decode more frames than the project holds, and
        # frames past the arrays raised IndexError on every repaint
        sa = proj.sessions[active]
        if runtimes[active].n_frames > sa.n_frames:
            runtimes[active].n_frames = sa.n_frames
            self.spin.setRange(0, sa.n_frames - 1)
            self.spin.setSuffix(f" / {sa.n_frames - 1}")
        lost = [proj.name(i) for i in drop]
        for i in reversed(drop):       # descending keeps the indices valid
            del runtimes[i]
            proj.remove_view(i)
        if lost:
            # The project FILE still holds these cameras. Autosave used to write
            # the reduced project straight back over it within 30 s, deleting
            # their tracks and calibration for good (I14): Save now asks for a
            # file name (autosave only ever writes the recovery folder).
            kept = self.project_path
            self.project_path = None
            # (I168) a detached project: its own id and no base save, so its recovery copy is never
            # offered later as "unsaved changes to the full project" nor saved over it
            self._project_id, self._saved_at, self._recovery_sig = projectfile.new_id(), None, None
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
        # a project digitized before the cameras shared one landmark list: fill the
        # gaps (points with no data) without calling the file changed (G19)
        was = [s.dirty for s in proj.sessions]
        if proj.landmark_changes():          # a missing name, or another order (G26)
            proj.sync_landmarks()
            for s, d in zip(proj.sessions, was):
                s.dirty = d
            self._refresh_point_list()
        self._rebudget_caches()
        for cv in self.grid.set_count(proj.n_views):
            self._wire_canvas(cv)
        self.grid.reset_arrangement()      # (G170) the views as saved come with the project's layout
        for i, rt in enumerate(self._views):
            cv = self.grid.canvas(i)
            cv.set_video_size(rt.info.width, rt.info.height)
            if i != proj.active:
                cv.set_interactive(False)
        self.grid.set_active(proj.active)
        self.act_solo.setEnabled(proj.n_views > 1)
        self.m_others.setEnabled(proj.n_views > 1)      # View → Other cameras (G24)
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
        self._drop_reconstruction("a new calibration was imported")      # (I242)
        p.dirty = True
        self._apply_state()
        self._guides_on()
        self.toast.show_message(
            f"Calibration loaded for {p.n_views} cameras ({p.calibration.source}). " + self.GUIDES_HINT
            + " Next: <b>3D → Reconstruct 3D Landmarks</b> (Ctrl+3).", "info", 14000)

    GUIDES_HINT = ("Press <b>N</b> (＋ Add) and click a point in one camera: the other cameras show a dashed line "
                   "where that point must be. Click the next camera to work there (the point stays selected) and "
                   "click it on the line — or press N first, and that one click does both. Once two cameras have "
                   "it, a ◇ shows where it is in the rest (<b>A</b> places it there). Alt+click shows where any "
                   "spot can be.")

    def _guides_on(self) -> None:
        """A calibration has just arrived: the epipolar guides are what makes it
        useful while digitizing, so they are switched on (View -> Epipolar
        guides) and drawn at once (G19)."""
        if not self.act_epipolar.isChecked():
            self.act_epipolar.setChecked(True)
        self._refresh_guides()

    def _lens_wizard(self):
        """3D → Calibrate a Lens: intrinsics + distortion of ONE camera from a
        checkerboard video, attached to that camera of this project."""
        from kinetrace.lenswizard import LensWizard
        p = self.project
        if self.state == TRACKING:
            return
        if p is None or self.state != READY:
            # no video open: the wizard still opens (it has its own board-video
            # picker); the result is a lens FILE to attach later. It used to refuse
            # with "Open the checkerboard video first" (G9)
            wiz = LensWizard(self, None, "", None, str(Path.home()))
            accepted = wiz.exec() == QDialog.Accepted
            profile, saved_to = wiz.result_profile, getattr(wiz, "saved_path", "")
            wiz.deleteLater()            # (I249) it holds the scan (~93 MB of thumbnails): not for the session
            if accepted and profile is not None and not saved_to:
                if QMessageBox.question(
                        self, "Save the lens profile?",
                        "No video is open, so the profile is not attached to a camera and would be lost "
                        "when this window closes. Save it as a lens file now?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes:
                    path, _ = QFileDialog.getSaveFileName(self, "Save lens profile", str(Path.home() / "camera.klens.json"),
                                                          "Kinetrace lens (*.klens.json)")
                    if path:
                        saved = profile.save(path)
                        self.toast.show_message(f"Lens profile saved: {Path(saved).name}. Load it for its camera "
                                                "with 3D → Calibrate a Lens → I already have a lens file…",
                                                "success", 9000)
            return
        start = str(Path(self.project_path or self.info.path).parent) if self.info else ""
        wiz = LensWizard(self, p, self.info.path if self.info else "", p.active, start)
        accepted = wiz.exec() == QDialog.Accepted
        profile, rview = wiz.result_profile, wiz.result_view
        views = wiz.result_views() if accepted and profile is not None and rview is not None else []
        wiz.deleteLater()                        # (I249) its scan and corner frames are not kept for the session
        if not accepted or profile is None or rview is None:
            return
        bad = self._lens_misfit(profile, rview)
        if bad:                                  # a profile of another picture size (I31)
            QMessageBox.warning(self, "Lens profile not attached", bad)
            return
        self._set_lenses({k: profile for k in views})   # the identical cameras it was shared with too (G40)
        v = str(profile.report.get("verdict", "loaded")).upper()
        self.toast.show_message(
            f"Lens profile attached to {', '.join(p.name(k) for k in views)} ({v}: {profile.summary()}). "
            "The wand calibration will use it; save the project to keep it.", "success", 9000)

    LENS_SAVE_FILTER = ("Kinetrace lens (*.klens.json);;OpenCV lens (*.yml);;OpenCV lens, JSON (*.json);;"
                        "Argus / DLTdv camera profile (*.txt)")

    def _export_lens_profile(self) -> None:
        """3D -> Export Lens Profile… (G148): a camera's lens profile to a file, whenever, however it was made
        (checkerboard, GoPro's model, a file, the wand wizard's lens rows). The working camera's when it has
        one, else the choice of the cameras that do. Kinetrace's own format keeps the report; OpenCV / Argus
        are for other programs (an Argus line cannot hold a fisheye lens: said, nothing written)."""
        from kinetrace import calibio
        p = self.project
        if p is None or self.state != READY:
            return
        have = [v for v in range(p.n_views) if v < len(p.lenses) and p.lenses[v] is not None]
        if not have:
            QMessageBox.information(
                self, "No lens profile yet",
                "No camera of this project has a lens profile yet.\n\nMake one with 3D → Calibrate a Lens "
                "(checkerboard), load one with 3D → Load a Lens Profile for This Camera…"
                + (", or give the GoPro cameras GoPro's lens model in 3D → GoPro Cameras…" if self._gopro_infos() else "")
                + ". Then export it here.")
            return
        v = p.active if p.active in have else have[0]
        if len(have) > 1 or v != p.active:
            labels = [f"{p.name(k)} — {lens_label(p.lenses[k])}" for k in have]
            choice, ok = QInputDialog.getItem(self, "Export lens profile", "The lens profile of:", labels,
                                              have.index(v), False)
            if not ok or choice not in labels:
                return
            v = have[labels.index(choice)]
        prof = p.lenses[v]
        safe = self._safe_name(p.name(v)).strip("_") or "camera"
        start = Path(self._start_folder(str(Path(self.info.path).parent) if self.info else ""))             / f"{safe}_{prof.width}x{prof.height}.klens.json"
        path, flt = QFileDialog.getSaveFileName(self, f"Export the lens profile of {p.name(v)}", str(start),
                                                self.LENS_SAVE_FILTER)
        if not path:
            return
        try:
            if flt.startswith("Kinetrace") or path.lower().endswith(".klens.json") or not Path(path).suffix:
                path = prof.save(path)
            else:
                calibio.write_lens(prof, path)
        except calibio.CalibFormatError as e:      # a fisheye as an Argus line: a sentence already
            QMessageBox.warning(self, "Lens profile not saved", str(e)[0].upper() + str(e)[1:])
            return
        except Exception as e:      # noqa: BLE001 - a full drive, a read-only folder: said
            QMessageBox.warning(self, "Lens profile not saved", _plain_error(e, "The lens profile could not be saved"))
            return
        self.toast.show_message(
            f"Lens profile of {p.name(v)} saved as {Path(path).name} ({prof.width}×{prof.height}). To use it again: "
            "3D → Load a Lens Profile for This Camera… (or the lens wizard's <i>I already have a lens file…</i>), "
            "for this camera in another project or for cameras of the same model, lens, zoom and recording mode.",
            "success", 12000)

    def _load_lens_profile(self) -> None:
        """3D -> Load a Lens Profile for This Camera… (G148): a saved profile (.klens.json, OpenCV, Argus: the
        lens wizard's reader, `lens.read_lens_for`) attached to the working camera after the picture-size
        check every lens gets (I31). Replacing a profile the camera already has is asked first."""
        from kinetrace import lens as lens_mod
        from kinetrace.lenswizard import LENS_FILTER
        p = self.project
        if p is None or self.state != READY:
            return
        v = p.active
        path, _ = QFileDialog.getOpenFileName(self, f"Lens profile for {p.name(v)}",
                                              self._start_folder(str(Path(self.info.path).parent) if self.info else ""),
                                              LENS_FILTER)
        if not path:
            return
        try:
            prof, which = lens_mod.read_lens_for(path, v, p.name(v))
        except Exception as e:      # noqa: BLE001 - not a lens file, a damaged one: said
            QMessageBox.warning(self, "Lens profile not loaded",
                                _plain_error(e, f"{Path(path).name} could not be read", reading=True))
            return
        if prof is None:
            QMessageBox.warning(self, "Lens profile not loaded", which)
            return
        bad = self._lens_misfit(prof, v)
        if bad:
            QMessageBox.warning(self, "Lens profile not attached", bad)
            return
        old = p.lenses[v] if v < len(p.lenses) else None
        if old is not None and QMessageBox.question(
                self, "Replace the lens profile?",
                f"{p.name(v)} already has a lens profile ({lens_label(old)}). Replace it with "
                f"{Path(path).name}?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._set_lenses({v: prof})
        self.toast.show_message(f"Lens profile {Path(path).name}{which} attached to {p.name(v)} ({prof.summary()}). "
                                "The wand calibration will use it; save the project to keep it.", "success", 10000)

    def _lens_misfit(self, prof, v: int) -> str | None:
        """Why lens profile `prof` does not fit camera v (another picture size, I31), as a sentence;
        None when it fits. The one check of the lens wizard, Load a Lens Profile and GoPro's lens."""
        from kinetrace.calibwizard import lens_size_mismatch
        p = self.project
        sv = p.sessions[v]
        bad = lens_size_mismatch(prof, sv.width, sv.height, p.name(v))
        return bad[0].upper() + bad[1:] if bad else None

    def _set_lenses(self, by_view: dict) -> None:
        """Attach lens profiles {camera: profile} to the project (the list padded to the cameras)."""
        p = self.project
        while len(p.lenses) < p.n_views:
            p.lenses.append(None)
        for v, prof in by_view.items():
            p.lenses[v] = prof
        if by_view:
            p.dirty = True

    def _wand_wizard(self):
        """3D → Calibrate Cameras with a Wand: the native easyWand replacement,
        explained for a first-time user. The result becomes this project's
        calibration (and can be exported for the animal projects)."""
        from kinetrace.calibwizard import WandWizard
        p = self.project
        # one video is enough to OPEN the wizard (the wand may be tracked in
        # one camera for other reasons); the pages say what a calibration
        # still needs. Only "no video at all" is refused here -- with the guidance
        # this entry is enabled to give, BEFORE the state check (G106: with no video
        # open the state is IDLE and it used to return in silence).
        if p is None or p.n_views < 1:
            QMessageBox.information(
                self, "Open a wand video first",
                "Wand calibration works on the wand recording of your cameras, in ONE project:\n\n"
                "1. File → Open Video… — the wand video of the first camera.\n"
                "2. In the CAMERAS panel on the right, ＋ Add video — the wand video of each other "
                "camera; set their offsets so they show the same instant (3D → Sync Cameras).\n"
                "3. In every camera, track the two wand ends (Point ▾ → Ball marker on each ball, or N "
                "on a mark; the SAME names in every camera, e.g. 'wand A' and 'wand B'), then Track ▶.\n"
                "4. Then 3D → Calibrate Cameras with a Wand… again.\n\n"
                "Help → User Manual (F1), section 10, walks through all of it.")
            return
        if self.state != READY:
            return
        start = str(Path(self.project_path or self.info.path).parent) if self.info else ""
        wiz = WandWizard(self, p, start)
        if wiz.exec() != QDialog.Accepted or wiz.result_calibration is None:
            return
        p.calibration = wiz.result_calibration
        self._wand_result = (wiz.result, wiz.gravity)
        self._drop_reconstruction("a new calibration was made")          # (I242)
        p.dirty = True
        self._apply_state()
        v = str(wiz.result.report.get("verdict", "?")).upper() if wiz.result is not None else "?"
        self._guides_on()
        self.toast.show_message(
            f"Wand calibration ({v}) is now this project's calibration. Save the project (Ctrl+S); "
            "<b>3D → Export Calibration</b> writes it for your animal projects. " + self.GUIDES_HINT,
            "success", 14000)

    def _export_calibration(self):
        """3D -> Export Calibration: the cameras for other programs, in the
        formats ticked (calibio.py converts the conventions and checks them)."""
        from kinetrace import calibio
        from kinetrace.calibwizard import dlt_csv_caveat, save_calibration_files
        from kinetrace.view3d import ExportCalibrationDialog
        p = self.project
        if not self._need_calibration("Export Calibration"):
            return
        dlg = ExportCalibrationDialog(self, any(x is not None for x in (p.lenses or [])))
        if dlg.exec() != QDialog.Accepted or not dlg.chosen():
            return
        keys = dlg.chosen()
        start = self._default_output(".kcal.json", camera=False)       # one calibration for every camera (G119)
        path, _ = QFileDialog.getSaveFileName(self, "Export calibration — base name", start,
                                              "Kinetrace calibration (*.kcal.json);;All files (*)")
        if not path:
            return
        stem = path[:-len(".kcal.json")] if path.lower().endswith(".kcal.json") else str(Path(path).with_suffix(""))
        written, notes = [], []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if "kcal" in keys:
                res, grav = self._wand_result
                # written= is filled as each file lands, so an error half-way can say which exist (G113)
                save_calibration_files(res, grav, stem + ".kcal.json", cal=p.calibration, written=written)
                note = dlt_csv_caveat(p.calibration, list(p.names))
                if note:
                    notes.append(note)       # with lens corrections the dltCoefs.csv is for lens-corrected pixels (I32)
            if set(keys) & {"anipose", "opencv_yml", "opencv_json", "matlab", "blender"}:
                rec = p.reconstruction
                probe = (np.nanmedian(rec.xyz.reshape(-1, 3), axis=0)
                         if rec is not None and np.isfinite(rec.xyz).any() else None)
                models = calibio.to_models(p.calibration, list(p.names), probe)
                for key, suffix, fn in (("anipose", "_calibration.toml", calibio.write_anipose),
                                        ("opencv_yml", "_cameras.yml", calibio.write_opencv),
                                        ("opencv_json", "_cameras.json", calibio.write_opencv),
                                        ("matlab", "_cameras.mat", calibio.write_matlab),
                                        ("blender", "_blender_cameras.py", calibio.write_blender)):
                    if key in keys:
                        fn(models, stem + suffix)
                        written.append(stem + suffix)
                notes += models.notes
            if "lenses" in keys:
                for i, prof in enumerate(p.lenses or []):
                    if prof is not None:
                        safe = self._safe_name(p.name(i))
                        calibio.write_lens(prof, f"{stem}_lens_{safe}.yml")
                        written.append(f"{stem}_lens_{safe}.yml")
            if "offsets" in keys:
                calibio.write_offsets(p, stem + "_offsets.csv")
                written.append(stem + "_offsets.csv")
        except Exception as e:      # noqa: BLE001
            QApplication.restoreOverrideCursor()
            done = ("\n\nFiles written before the error: " + ", ".join(Path(w).name for w in written)
                    + ". The others were not written." if written else
                    "\n\nNo file was written.")
            QMessageBox.critical(self, "Export calibration",
                                 _plain_error(e, "The calibration could not be written") + done)
            return
        QApplication.restoreOverrideCursor()
        names = ", ".join(Path(w).name for w in written)
        warn = any(("off by" in n) or ("skew" in n) for n in notes)
        self.toast.show_message(f"Calibration exported: {names}" + ("<br>" + "<br>".join(notes) if notes else ""),
                                "warn" if warn else "success", 14000 if notes else 8000)
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

    # ------------------------------------------------------------- GoPro (G145, G146)
    def _gopro_infos(self) -> dict:
        """{view: gpmf.GoProInfo} of the cameras whose video is GoPro footage (THE flag, read at open)."""
        return {v: rt.info.gopro for v, rt in enumerate(getattr(self, "_views", []))
                if getattr(rt.info, "gopro", None) is not None}

    def _gopro_soon(self) -> None:
        """One GoPro notice after the cameras of an open / add have arrived (coalesced)."""
        if not getattr(self, "_gopro_pending", False):
            self._gopro_pending = True
            QTimer.singleShot(0, self._gopro_report)

    def _gopro_report(self) -> None:
        """Say once per camera (and per rig finding) what in its GoPro metadata can spoil tracking or 3D:
        stabilisation on, dropped frames, a camera that moved, settings that differ between cameras. A
        GoPro camera with nothing to report gets one status line; other footage gets nothing."""
        self._gopro_pending = False
        p = self.project
        infos = self._gopro_infos()
        if p is None or not infos:
            return
        said = getattr(self, "_gopro_said", None)
        if said is None:
            said = self._gopro_said = set()
        lines = []
        for v, info in infos.items():
            key = ("cam", info.path)
            if key in said:
                continue
            said.add(key)
            lines += gpmf.problems(info, p.name(v))
        for s in gpmf.rig_problems({p.name(v): i for v, i in infos.items()}):
            if ("rig", s) not in said:
                said.add(("rig", s))
                lines.append(s)
        self._apply_state()                      # the GoPro entry lights up
        if lines:
            self.toast.show_message("GoPro footage: " + " ".join(lines) + " (Click for 3D → GoPro Cameras…)",
                                    "warn", 25000, on_click=self._gopro_dialog)
        else:
            labels = sorted({i.label for i in infos.values()})
            self.statusBar().showMessage(f"GoPro footage ({'; '.join(labels)}): stabilisation off, no dropped frames, "
                                         "no camera moved. 3D → GoPro Cameras… shows the details and GoPro's lens "
                                         "model.", 9000)

    def _gopro_dialog(self) -> None:
        """3D → GoPro Cameras…: the settings / sensors table and GoPro's lens for the cameras without one."""
        from kinetrace.goprodialog import GoProDialog
        p = self.project
        if p is None or self.state != READY or not self._gopro_infos():
            return
        infos = [getattr(rt.info, "gopro", None) for rt in self._views]
        dlg = GoProDialog(self, [p.name(v) for v in range(p.n_views)], infos,
                          lambda: [p.lenses[v] if v < len(p.lenses) else None for v in range(p.n_views)],
                          on_use_lenses=self._use_gopro_lenses,
                          on_goto=lambda v, f: (dlg.accept(), self._goto_camera_frame(v, f)))
        dlg.exec()
        dlg.deleteLater()

    def _goto_camera_frame(self, view: int, frame: int) -> None:
        if self.project is None or not (0 <= view < self.project.n_views):
            return
        if view != self.project.active:
            self._set_active_view(view)
        self._goto(max(0, min(int(frame), self.n_frames - 1)))

    def _use_gopro_lenses(self, views) -> str:
        """GoPro's lens model (gpmf.lens_profile) for these cameras, which have no lens profile: a profile
        of another picture size is refused, as for any lens (I31). Returns what was done, in words."""
        p = self.project
        done, refused = {}, []
        for v in views:
            info = getattr(self._views[v].info, "gopro", None) if v < len(self._views) else None
            prof = gpmf.lens_profile(info) if info is not None else None
            if prof is None:
                continue
            bad = self._lens_misfit(prof, v)
            if bad:
                refused.append(bad)
                continue
            done[v] = prof
        self._set_lenses(done)
        done = [p.name(v) for v in done]
        said = (f"GoPro's lens model attached to {', '.join(done)}: the wand calibration will undistort with it; "
                "save the project to keep it." if done else "No lens attached.")
        if refused:
            said += " Not attached: " + " ".join(refused)
        self.statusBar().showMessage(said, 10000)
        return said

    def _sync_dialog(self):
        """3D -> Sync Cameras (Sound / Motion): whole-frame offsets from the sound tracks or the pictures."""
        from kinetrace.syncdialog import SyncDialog
        p = self.project
        if p is None or p.n_views < 2:
            QMessageBox.information(
                self, "Add cameras first",
                "Syncing lines up two or more cameras of the same event. Open the first camera's "
                "video, then add the others with ＋ Add video in the CAMERAS panel.")
            return
        if self.state != READY:
            return
        before, had_3d = list(p.offsets), p.reconstruction is not None
        dlg = SyncDialog(self, p, [rt.info.path for rt in self._views], self.current,
                         gopro=[getattr(rt.info, "gopro", None) for rt in self._views])     # (G145) timecode prior
        accepted = dlg.exec() == QDialog.Accepted
        n = getattr(dlg, "applied", 0)
        dlg.deleteLater()                    # (G96) never kept for the session (read `applied` first)
        if not accepted:
            return
        # (I242) the 3D result is dropped only when an offset really moved: ticking rows that hold
        # the offsets already there changes nothing
        moved = any(abs(a - b) > 1e-9 for a, b in zip(before, p.offsets))
        if moved:
            p.dirty = True
            self._drop_reconstruction("the cameras' timing changed", had=had_3d)
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
        from kinetrace.calib import estimate_offsets
        p = self.project
        if not self._need_calibration("Estimate Sub-frame Offsets"):
            return
        t0, t1 = self._t_range_3d()
        if t1 < t0:
            self.toast.show_message("Nothing tracked in the overlap yet — track the cameras first.", "warn", 6000)
            return
        rep: dict = {}

        def work(report, cancelled):
            # (G76) a counted, stoppable search: the card shows the bar and a Cancel button
            last = [0.0]

            def progress(done, total):
                now = time.monotonic()
                if now - last[0] >= 0.1 or done >= total:      # not a signal per cost evaluation
                    last[0] = now
                    report(f"{done} of about {total} offset tests", done, total)
            return estimate_offsets(p.sessions, p.calibration, p.rates, p.offsets, (t0, t1), report=rep,
                                    progress=progress, cancelled=cancelled)
        try:           # a joint search over every camera: off the GUI thread (G50)
            offs, before, after = self._in_background(
                "Estimating sub-frame offsets", work, progress=True, cancellable=True,
                detail=f"Testing offsets of every camera against the tracks of frames {t0}-{t1}…")
        except Exception as e:      # noqa: BLE001
            QMessageBox.critical(self, "Sub-frame offsets", _plain_error(e, "The offsets could not be estimated"))
            return
        if rep.get("verdict") == "cancelled":
            self.toast.show_message("Cancelled: the camera offsets were not changed.", "info", 5000)
            return
        if not np.isfinite(before) or not np.isfinite(after):
            QMessageBox.information(self, "Sub-frame offsets",
                                    "No landmark is seen by two cameras at the same instant, so there is "
                                    "nothing to align. Track the same named landmarks in at least two cameras.")
            return
        per_view = rep.get("per_view", {})
        # (G75) "flat" = the landmarks it shares barely move (nothing to time); "none" = it shares no
        # landmark with another camera in this window (its offset is left as it was)
        words = {"sharp": "well determined", "weak": "weakly determined",
                 "flat": "flat: the landmarks do not move enough to time it -- do not trust",
                 "none": "unchanged: shares no landmark with another camera here"}
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
        self._drop_reconstruction()          # (I242) the reconstruction below makes a new one
        self._refresh_companions()
        self._refresh_cameras()
        self._reconstruct_3d(quiet=True)

    def _reconstruct_3d(self, quiet: bool = False):
        from kinetrace.calib import reconstruct
        p = self.project
        if not self._need_calibration("Reconstruct 3D Landmarks"):
            return
        t0, t1 = self._t_range_3d()
        if t1 < t0:
            self.toast.show_message("Nothing tracked in the overlap yet — track the cameras first.", "warn", 6000)
            return
        try:           # off the GUI thread (G50)
            p.reconstruction = self._in_background(
                "Reconstructing the 3D landmarks",
                lambda: reconstruct(p.sessions, p.calibration, p.rates, p.offsets, (t0, t1)),
                detail=f"Triangulating frames {t0}-{t1} of every camera…")
        except Exception as e:      # noqa: BLE001
            QMessageBox.critical(self, "3D reconstruction", _plain_error(e, "The 3D landmarks could not be made"))
            return
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
        from kinetrace.calib import reconstruction_report
        widths = [s.width for s in p.sessions if s.width] or [1920]
        heights = [s.height for s in p.sessions if s.height] or [0]
        rep = reconstruction_report(r, p.n_views, float(max(widths)), float(max(heights)))     # long side (I250)
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
            from kinetrace import retrack
            n_st = len([q for q in retrack.plan(p, self._disagree_thresholds()) if q.target is not None])
            if n_st and self._retrack is None:
                self.toast.show_message(
                    f"{n_st} stretch{'es' if n_st != 1 else ''} where one camera disagrees with the others "
                    "(magenta band on the timeline). <b>3D → Re-track Disagreeing Stretches</b> puts the landmark "
                    "back on the other cameras' rays and re-tracks it, with a before / after verdict.",
                    "warn", 12000)

    def _set_world_axes(self):
        """3D → Set World Axes… (I170, owner 2026-10-03): the user picks the ORIGIN, the +X and the +Y
        direction as three landmarks of the 3D result; +Z follows the right-hand rule. The calibration
        and the 3D result are re-expressed in that world (calib.world_axes / .reframed): the pictures
        are untouched, every camera projects every point to the same pixel as before, and exported
        cameras and exported 3D points are in one world."""
        from kinetrace import calib
        p = self.project
        if self.state == TRACKING:
            return
        if not self._need_calibration("Set World Axes"):
            return
        r = p.reconstruction
        if r is None or r.n_frames == 0 or not np.isfinite(r.xyz).any():
            QMessageBox.information(
                self, "Set World Axes",
                "The axes are chosen from three landmarks of the 3D result, and there is none yet.\n\n"
                "Reconstruct first (Ctrl+3, 3D → Reconstruct 3D Landmarks), then choose the axes.")
            return
        flat = r.xyz.reshape(-1, 3)
        probe = np.nanmedian(flat[np.isfinite(flat).all(axis=1)], axis=0)
        lh = calib.world_is_left_handed(p.calibration, probe)
        dlg = _WorldAxesDialog(self, r, self._reference_instant(), lh, p.calibration.unit or r.unit)
        ok = dlg.exec() == QDialog.Accepted
        axes = dlg.result_axes
        dlg.deleteLater()
        if not ok or axes is None:
            return
        B, o, note = axes
        try:
            cal2 = p.calibration.reframed(B, o, note)
        except ValueError as e:          # the origin lies on a camera's image plane
            QMessageBox.warning(self, "Set World Axes", str(e))
            return
        rec2 = r.reframed(B, o)
        p.calibration = cal2
        self._drop_reconstruction(replacement=rec2)
        p.dirty = True
        self._apply_state()
        self._refresh_guides()
        self.toast.show_message(
            f"World axes set ({note[len('world: '):]}). +Z follows the right-hand rule"
            + (" (the calibration's own world was mirrored; this one is not)" if lh else "")
            + ". The pictures are unchanged: every camera projects every point to the same pixel as before. "
            "A volume carved earlier was in the old axes and was cleared; carve it again (Ctrl+4). Export "
            "the calibration and the 3D points again to get them in the new world.", "success", 14000)

    def _masks_at_instant(self, t: int, name: str | None = None):
        """(cameras, raw masks) at reference instant t — None where a camera
        has no silhouette there. `name` = the segment (G151k; None = the working camera's active one)."""
        p = self.project
        if name is None:
            name = self.session.animal.name if (self.session is not None and self.session.animal is not None) else None
        masks = []
        for c, s in enumerate(p.sessions):
            f = int(p.local_index(c, t))        # map_frame's tie rule, not Python's half-to-even (I258)
            # (G149) the working camera's active segment, matched by name in every camera
            k = s.segment_index(name) if name is not None else None
            m = s.seg_masks[k] if k is not None else s.masks
            masks.append(m.rasterize(f, s.height, s.width) if m is not None and m.has(f) else None)
        return masks

    def _carve_hull_here(self, quiet: bool = False):
        from kinetrace.hull import bounds_from_points, carve, hull_mesh, mesh_volume
        p = self.project
        if not self._need_calibration("Carve Volume"):
            return
        # (G151k) the segment the user names (one highlighted row, or asked), matched by name in every camera
        s = self.session
        seg_name = None
        if s is not None and s.n_segments > 1:
            k = self._pick_segment("Carve the volume of")
            if k is None:
                return
            seg_name = s.segments[k].name
        t = self._reference_instant()         # the hull cache is keyed by the reference instant (I114)
        masks = self._masks_at_instant(t, seg_name)
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
        def work():
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
            return verts, faces, h, cut_off, voxel

        try:           # coarse carves + a 120^3 one: off the GUI thread (G50)
            verts, faces, h, cut_off, voxel = self._in_background(
                "Carving the volume", work, detail=f"From {n_masks} silhouettes at frame {t}…")
        except Exception as e:      # noqa: BLE001
            QMessageBox.critical(self, "Volume hull", _plain_error(e, "The volume could not be carved"))
            return
        if cut_off:
            QMessageBox.warning(
                self, "Volume hull cut off",
                "The volume still reaches the edge of the space searched around the landmarks after "
                "enlarging it several times, so the reported volume is cut off (too small). Usually a "
                "silhouette includes the background here; check each camera's silhouette at this frame.")
        from kinetrace.hull import THIN_WARN_FRAC
        thin = h.thin_fraction()
        if thin > THIN_WARN_FRAC:                # (I172)
            QMessageBox.warning(
                self, "Volume partly checked by two cameras only",
                f"{100 * thin:.0f} % of the volume was checked by fewer than 3 cameras: a camera cuts the "
                "animal at its picture edge, so it cannot rule out the volume beyond it, and there the "
                "hull rests on two cameras only (a long sliver along the line between them). The volume "
                "is probably too large. Frame the whole animal in every camera, or carve at an instant "
                "when it is fully in view of three.")
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
        t = self._reference_instant()       # (G107) the one rounding rule: the hull cache is keyed by it
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
                           (p.session.bone_names() if s else [])
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
        from kinetrace.bodyview import BodyPoseWorker, BodyRunDialog
        s = self.session
        if s is None or self.state != READY or self._body_worker is not None:
            return
        # (G151l) the silhouette that can drive the run: the segment the user names when several have one
        with_sil = [k for k in range(s.n_segments) if s.seg_masks[k].n_masked() > 0]
        if len(with_sil) > 1:
            k = self._pick_segment("Find the person inside the silhouette of")
            if k is None:
                return
        else:
            k = with_sil[0] if with_sil else None
        seg_masks = s.seg_masks[k] if k is not None else None
        has_masks = seg_masks is not None and seg_masks.n_masked() > 0
        # existing=: the dialog asks before a run that cannot be merged (I82)
        dlg = BodyRunDialog(self, self.n_frames, self.current, self.timeline.sel_range,
                            has_masks, self._body_backend, self._lens_focal_px(),
                            existing=s.body)
        if dlg.exec() != QDialog.Accepted or dlg.result_options is None:
            return
        opts = dlg.result_options
        self._body_backend = opts.backend
        self._begin_edit()                  # Ctrl+Z takes the whole run back
        # target=: the result belongs to THIS camera, whatever is active when it ends (I83)
        w = BodyPoseWorker(self.info.path, self.n_frames, opts, seg_masks, target=s)    # (G78) fps is a no-op
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
        w.stopped.connect(lambda m: self.toast.show_message(m, "info", 6000))       # (G78) Stop = a notice
        w.finished.connect(self._end_body_run)
        dl.show()
        w.start()
        self._apply_state()

    def _on_body_progress(self, done: int, total: int, note: str):
        if self._body_progress is not None:
            self._body_progress.setMaximum(max(1, total))
            self._body_progress.setValue(done)
            loading = note.startswith(("Loading", "Downloading", "the "))
            if loading and total <= 1:
                self._body_progress.setMaximum(0)           # moving: how long is not known
            self._body_progress.setLabelText(note if loading else f"Looking for people…  {note}")

    def _on_body_done(self, track):
        from kinetrace.body import merge_run
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
            # (I183) merge_run's sentence says which frames kept their earlier pose
            head = note if merged.n_posed() and note else (
                "No person was found anywhere in that range." if not merged.n_posed() else
                "No person was found anywhere in that range; the earlier poses were kept.")
            self.toast.show_message(
                head + " Try the segment tool (S) to draw round the person, then run again using "
                "the silhouette.", "warn", 8000)
        else:
            self.toast.show_message(f"{merged.summary()}. {note}".strip(), "info", 8000)
            if self.body_win is None and not self._stopping_for_close:      # not while the window closes
                self.act_body_view.setChecked(True)
                self._toggle_body_view(True)
        self.timeline.update()
        self._refresh_body_view(force=True)

    def _on_body_error(self, msg: str):
        hint = _model_error_hint(msg)                     # no internet / memory / gated weights (G54)
        QMessageBox.warning(self, "Body pose", (hint + "\n\n" + msg[-600:]) if hint and hint not in msg else msg)

    def _end_body_run(self):
        if self._body_progress is not None:
            self._body_progress.close()
            self._body_progress = None
        self._body_worker = None
        self._apply_state()

    def _toggle_body_view(self, on: bool):
        from kinetrace.bodyview import BodySideBySide
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

    @staticmethod
    def _safe_name(name: str) -> str:
        """A camera / landmark name as a piece of a file name (R12: the two copies of this rule)."""
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)

    def _default_output(self, tail: str, camera: bool = True) -> str:
        """The proposed file name of an export (G119, R12): the project (or video) name, the working
        camera's name when the project has several -- every camera used to propose the SAME name and one
        camera's file overwrote another's (I108) -- and `tail`. camera=False for what covers every camera
        (the calibration, a mesh)."""
        base = Path(self.project_path or self.info.path)
        cam = ""
        p = self.project
        if camera and p is not None and p.n_views > 1:
            cam = "_" + self._safe_name(p.name(p.active))
        return str(base.with_suffix("")) + cam + tail

    def _body_default_path(self, tail: str) -> str:
        return self._default_output(tail)

    @staticmethod
    def _export_error(exc: BaseException, what: str) -> str:
        """An export's failure in words: an OS error through plain_error (a file open in another
        program, a full drive), a ValueError's own sentence as it is."""
        return _plain_error(exc, what) if isinstance(exc, OSError) else str(exc)

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
            QMessageBox.warning(self, "Export", self._export_error(exc, "The joint positions could not be written"))
            return
        self.toast.show_message(f"Joint positions written to {Path(path).name}.", "info", 5000)

    def _export_body_angles(self):
        """The angles CSV plus a plain-language report beside it -- the numbers
        are useless without the convention each one uses."""
        from kinetrace.body import angle_report
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
            QMessageBox.warning(self, "Export", self._export_error(exc, "The joint angles could not be written"))
            return
        self.toast.show_message(
            f"Joint angles written to {Path(path).name}, with the conventions and the range "
            f"of motion in {Path(rep).name}.", "info", 7000)

    def _export_body_video(self):
        """Write exactly what the side-by-side window shows to an mp4."""
        from kinetrace.bodyview import PoseDrawOptions, SideBySideRenderer
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
        self._begin_edit()
        s.clear_body()
        self._refresh_body_view(force=True)
        self.timeline.update()
        self._apply_state()

    def _export_mesh(self):
        from kinetrace.hull import save_obj, save_ply
        p = self.project
        t = self._reference_instant()          # the same key as the carve (I114)
        if p is None or t not in self._hull_cache:
            self.toast.show_message("Carve the volume at this frame first (Ctrl+4).", "warn", 5000)
            return
        start = self._default_output(f"_hull_f{t}.obj", camera=False)      # a volume of the whole scene (G119)
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
        except Exception as e:      # noqa: BLE001 - (R12) any failure in words, not only OSError
            QMessageBox.critical(self, "Export failed", _plain_error(e, "The mesh could not be written"))
            return
        mirrored = ""
        try:
            from kinetrace.calib import world_is_left_handed
            if p.calibration is not None and world_is_left_handed(p.calibration):
                mirrored = (" This calibration's world is left-handed (mirrored), so the mesh is mirrored too: "
                            "3D → Set World Axes… first gives it a right-handed world.")
        except Exception:      # noqa: BLE001 - only a hint
            pass
        self.toast.show_message(f"Mesh written: {Path(path).name} ({len(faces)} triangles).{mirrored}",
                                "info", 6000 if not mirrored else 12000)

    # ---------------------------------------------------------------- export

    EXPORT_FORMATS = [
        # (filter label, suffix, key)
        ("Wide CSV — one row per frame (*.csv)", ".csv", "wide"),
        ("DeepLabCut CSV — scorer/bodyparts/coords header, likelihood (*.csv)", ".csv", "dlc"),
        ("DeepLabCut multi-animal CSV — one individual per animal, Scene points as 'single' (*.csv)",
         ".csv", "dlc_ma"),
        ("SLEAP analysis CSV — one track per animal, <part>.x / .y / .score (*.csv)", ".csv", "sleap"),
        ("DLTdv8 xypts CSV — pt1_cam1_X… rows per frame, top-left origin, first pixel = 1 (*.csv)",
         ".csv", "dltdv"),
        ("DLTdv xypts CSV, bottom-left origin — older DLTdv / Argus Clicker (*.csv)", ".csv", "dltdv_bl"),
        ("Sparse TSV — tracked cells only (*.tsv)", ".tsv", "sparse"),
        ("MATLAB — tracks, confidence, segment silhouette, skeleton (*.mat)", ".mat", "mat"),
        ("ALL CAMERAS — DLTdv8 xypts for 3D reconstruction, offsets applied, top-left, first pixel = 1 (*.csv)",
         ".csv", "multi"),
        ("3D landmarks — xyz per reference frame + residual sidecar (*.csv)", ".csv", "xyz"),
        ("3D landmarks — Anipose points_3d CSV (*.csv)", ".csv", "xyz_anipose"),
        ("3D landmarks — DLTdv xyzpts, row = reference frame, NaN = none (*.csv)", ".csv", "xyz_dltdv"),
        ("Silhouette outlines — polygons per frame (*.json)", ".json", "sil_json"),
        ("Silhouette masks — one PNG per frame, into a folder (*.png)", ".png", "sil_png"),
        ("3D kinematics — smoothed positions, velocity, acceleration + report (*.csv)", ".csv", "kin"),
        ("Everything — all of the above with one base name (*.csv)", ".csv", "all"),
    ]

    DLTDV_KEYS = ("dltdv", "dltdv_bl", "multi", "xyz_dltdv")      # (owner 2026-10-03: DLTdv = points only)
    SIDECAR_KEYS = ("wide", "dlc", "sparse")       # the 2D track exports that carry the events / segment files

    def _dlc_scorer(self) -> str:
        """The DeepLabCut 'scorer' name (R12: the one rule): the tracker every plain point was tracked
        with when they all share one, else the project's default point model."""
        names = {"alltracker": "AllTracker", "cotracker3": "CoTracker3", "spot": "MovingSpot"}
        s = self.session
        kinds = ({self._tracker_of(i, s) for i in range(s.n_points)
                  if not (s.points[i].derived or s.points[i].is_ball)} if s is not None else set())
        kind = next(iter(kinds)) if len(kinds) == 1 else self._point_backend
        return "Kinetrace_" + names.get(kind, "Kinetrace")

    def _points_models(self):
        """(I170) The exported cameras' models for the 3D points that go out beside them: 3D points are
        ALWAYS written through `calibio.write_points3d(..., models=)`, so a left-handed calibration's
        points are in the same (mirrored) world as the cameras Export Calibration writes. The probe is
        the median of the finite points (as in Export Calibration). Returns (models or None, note)."""
        from kinetrace import calibio
        p = self.project
        cal, r = p.calibration, p.reconstruction
        if cal is None or not len(cal):
            return None, ""
        flat = r.xyz.reshape(-1, 3)
        fin = flat[np.isfinite(flat).all(axis=1)]
        probe = np.median(fin, axis=0) if len(fin) else None
        try:
            return calibio.to_models(cal, list(p.names), probe), ""
        except calibio.CalibFormatError as e:
            return None, (f"The points were written in Kinetrace's own world, not the exported cameras' "
                          f"({e}).")

    def _export_one(self, key: str, path: str, cutoff="auto", sidecars: bool = True,
                    report=None, cancelled=None) -> tuple[list[str], list[str]]:
        """Write one format -> (the files written, sentences for the user). Runs on a worker thread
        (`_in_background`): it touches NO widget (R12) -- the caller shows the notes, asks the smoothing
        question and owns every dialog. An empty list of files with a note = nothing to write (G108).
        `cutoff`: the kinematics smoothing, asked by the caller. `sidecars`: the events / segment
        files beside a 2D track export (only the formats in SIDECAR_KEYS carry them; DLTdv exports are
        points only, owner 2026-10-03). `report` / `cancelled`: a progress line and a stop test."""
        s = self.session
        p = self.project
        written = [path]
        notes: list[str] = []
        no_3d = "No 3D reconstruction yet: run 3D -> Reconstruct 3D Landmarks (Ctrl+3) first."
        no_sil = "No silhouette to export: segment the animal first (S)."
        any_sil = any(m.n_masked() > 0 for m in s.seg_masks)        # (G149) any segment
        if key == "wide":
            s.export_csv(path)
        elif key == "dlc":
            s.export_dlc_csv(path, scorer=self._dlc_scorer())
        elif key in ("dlc_ma", "sleap"):
            # (G159) individuals = the animals: nothing to write without one
            if not s.segments or not any(s.points_of(k) for k in range(s.n_segments)):
                return [], ["No animal has points: make an animal (＋ Animal in LAYERS) and give it points first; "
                            "the plain DeepLabCut / Wide CSV exports write points of no animal."]
            if key == "dlc_ma":
                s.export_dlc_multi_csv(path, scorer=self._dlc_scorer())
            else:
                s.export_sleap_csv(path)
        elif key in ("dltdv", "dltdv_bl"):
            s.export_dltdv_csv(path, flip_y=(key == "dltdv_bl"))
            written.append(str(Path(path).with_name(Path(path).stem + "_pointnames.csv")))
        elif key == "sparse":
            s.export_tsv_sparse(path)
        elif key == "mat":
            s.export_mat(path)
        elif key == "multi":
            # every camera in one file, landmarks matched BY NAME and each view sampled through its own
            # offset — the input a 3D solver wants
            return p.export_multi_dltdv(path), notes
        elif key in ("xyz", "xyz_anipose", "xyz_dltdv"):
            from kinetrace import calibio
            r = p.reconstruction if p else None
            if r is None:
                return [], [no_3d]
            models, mnote = self._points_models()
            if mnote:
                notes.append(mnote)
            kind = {"xyz": "kinetrace", "xyz_anipose": "anipose", "xyz_dltdv": "dltdv"}[key]
            left_out = calibio.write_points3d(r, path, kind, models=models)         # (I170)
            if key == "xyz":
                written.append(str(Path(path).with_name(Path(path).stem + "_xyzres.csv")))
            if key == "xyz_dltdv":
                written.append(str(Path(path).with_name(Path(path).stem + "_pointnames.csv")))
            if left_out:
                notes.append(f"{left_out} instant(s) before reference frame 0 were left out: a DLTdv xyzpts "
                             "file starts at frame 0.")
            if models is not None and (models.mirrored or np.any(models.shift)):
                notes.append("The 3D points are in the same world as the cameras Export Calibration writes"
                             + (" -- a mirrored (left-handed) one; 3D -> Set World Axes… first gives the "
                                "calibration a right-handed world" if models.mirrored else "") + ".")
        elif key in ("sil_json", "sil_png"):
            # (G151m) every segment with silhouettes, one file / folder each (a "_<segment>" suffix
            # when there are several); it used to be the active segment's alone
            ks = [k for k in range(s.n_segments) if s.seg_masks[k].n_masked() > 0]
            if not ks:
                return [], [no_sil]                  # nothing is written, not even an empty file (G108)
            stem, suf = str(Path(path).with_suffix("")), Path(path).suffix

            def tagged(k):
                return "" if s.n_segments == 1 else "_" + self._safe_name(s.segments[k].name)
            if key == "sil_json":
                files = []
                for k in ks:
                    dest = stem + tagged(k) + suf
                    trackio.export_masks_json(s, dest, k)
                    files.append(dest)
                return files, notes
            total = sum(s.seg_masks[k].n_masked() for k in ks)
            files, done = [], 0
            for k in ks:
                folder = stem + tagged(k) + "_masks"

                def step(d, _t, base=done):
                    if report is not None:
                        report(f"{base + d} of {total} mask images", base + d, total)
                    return not (cancelled is not None and cancelled())
                n = s.seg_masks[k].n_masked()
                wrote = trackio.export_masks_png(s, folder, step, k)
                if wrote:
                    files.append(folder)
                done += wrote
                if wrote < n:
                    return files, [f"Cancelled: {done} of {total} mask images were written."]
            return files, notes                      # the files written: here the folders that hold them
        elif key == "kin":
            from kinetrace.kinematics import export_kinematics
            r = p.reconstruction if p else None
            if r is None:
                return [], [no_3d]
            fps = float(p.sessions[0].fps) if p.sessions else float(self.info.fps)
            unit = (p.calibration.unit if p.calibration is not None and p.calibration.unit else "") or r.unit or ""
            return export_kinematics(path, r, fps, unit, cutoff), notes
        if sidecars and key in self.SIDECAR_KEYS:
            stem = str(Path(path).with_suffix(""))
            if s.events or s.notes:            # frame notes live in this file too (I125)
                side = stem + "_events.csv"
                s.export_events_csv(side)
                written.append(side)
            for k, (a, m) in enumerate(zip(s.segments, s.seg_masks)):    # one file per segment (G149)
                if any_sil and m.n_masked() > 0:
                    side = stem + ("_segment.csv" if s.n_segments == 1 else f"_segment_{self._safe_name(a.name)}.csv")
                    s.export_animal_csv(side, k)
                    written.append(side)
        return written, notes

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
        start = self._default_output("_tracks.csv")          # + the camera's name in a multi-camera project
        filters = ";;".join(f[0] for f in self.EXPORT_FORMATS)
        path, chosen = QFileDialog.getSaveFileName(self, "Export tracks", start, filters)
        if not path:
            return
        label, suffix, key = next((f for f in self.EXPORT_FORMATS if f[0] == chosen),
                                  self.EXPORT_FORMATS[0])
        written: list[str] = []
        notes: list[str] = []
        errors: list[str] = []
        stopped = None
        try:
            if key == "all":
                stem = str(Path(path).with_suffix(""))
                jobs = []
                for lab, suf, k in self.EXPORT_FORMATS:
                    if k == "all" or (k == "multi" and self.project.n_views < 2):
                        continue    # the all-cameras file is meaningless for one camera
                    if k in ("xyz", "kin", "xyz_anipose", "xyz_dltdv") and self.project.reconstruction is None:
                        continue    # no 3D yet: nothing to write
                    if k == "sil_png" or (k == "sil_json" and not any(m.n_masked() for m in self.session.seg_masks)):
                        continue    # a PNG per frame is its own export (it can be many thousand files)
                    if k == "dltdv_bl":
                        continue    # "Everything" writes the DLTdv8 convention once, not both
                    if k in ("dlc_ma", "sleap") and not any(self.session.points_of(j)
                                                            for j in range(self.session.n_segments)):
                        continue    # individuals = animals with points (G159)
                    tag = {"wide": "", "dlc": "_dlc", "dlc_ma": "_dlc_multi", "sleap": "_sleap", "dltdv": "_dltdv",
                           "sparse": "", "mat": "", "multi": "_allcams", "xyz": "_xyz", "kin": "_kinematics",
                           "xyz_anipose": "_points3d_anipose", "xyz_dltdv": "_xyzpts", "sil_json": "_silhouette"}[k]
                    # the events / segment files once, beside the wide CSV; none beside DLTdv files
                    jobs.append((lab.split(" (")[0], k, stem + tag + suf, k == "wide"))

                def write_all(report, cancelled):
                    out = {"written": [], "notes": [], "errors": [], "stopped": None}
                    for i, (lab, k, dest, side) in enumerate(jobs):
                        if cancelled():
                            out["stopped"] = i
                            break
                        report(f"File {i + 1} of {len(jobs)}: {lab}", i, len(jobs))
                        try:       # one format failing (a .mat open in MATLAB) does not stop the others (G108)
                            files, nts = self._export_one(k, dest, cutoff="auto", sidecars=side)
                        except Exception as e:  # noqa: BLE001
                            out["errors"].append(f"{lab}: " + _plain_error(e, "could not be written", short=True))
                            continue
                        out["written"] += files
                        out["notes"] += nts
                    return out
                # every format in turn, off the GUI thread, counted and stoppable (G49)
                res = self._in_background("Exporting everything", write_all, total=len(jobs),
                                          progress=True, cancellable=True)
                written, notes, errors, stopped = res["written"], res["notes"], res["errors"], res["stopped"]
            else:
                if not path.lower().endswith(suffix):
                    path += suffix
                cutoff = "auto"
                if key == "kin" and self.project is not None and self.project.reconstruction is not None:
                    cutoff = self._ask_smoothing()       # asked here, on the GUI thread
                    if cutoff is False:
                        return
                if key == "sil_png":      # thousands of files: counted, with a Cancel on the card
                    written, notes = self._in_background(
                        f"Exporting {Path(path).name}",
                        lambda report, cancelled: self._export_one(key, path, cutoff=cutoff, report=report,
                                                                   cancelled=cancelled),
                        detail=label.split(" (")[0], progress=True, cancellable=True)
                else:
                    written, notes = self._in_background(f"Exporting {Path(path).name}",      # off the GUI thread (G49)
                                                         lambda: self._export_one(key, path, cutoff=cutoff),
                                                         detail=label.split(" (")[0])
        except Exception as e:  # noqa: BLE001
            msg = _plain_error(e, "The export could not be written")
            self.toast.show_message("Export failed: " + msg.split("\n")[0], "error", 8000)
            QMessageBox.critical(self, "Export failed", msg)
            return
        if errors:
            done = (f"Written: {', '.join(dict.fromkeys(Path(w).name for w in written))}." if written
                    else "Nothing was written.")
            self.toast.show_message(f"Export finished with {len(errors)} error(s).", "error", 8000)
            QMessageBox.warning(self, "Some files could not be written",
                                "These formats failed:\n\n" + "\n".join(f"  {x}" for x in errors)
                                + f"\n\n{done}")
            return
        if stopped is not None:
            self.toast.show_message(f"Export stopped: {len(dict.fromkeys(written))} file(s) written before Cancel.",
                                    "warn", 8000)
            return
        if any(n.startswith("Cancelled") for n in notes):
            self.toast.show_message(" ".join(n for n in notes if n.startswith("Cancelled")), "warn", 8000)
            return
        if not written:
            # nothing to write (no 3D, no silhouette): the reason only, never "Exported" (G108)
            self.toast.show_message(" ".join(notes) or "Nothing was exported.", "warn", 7000)
            return
        n = int(self.session.tracked.any(axis=1).sum())
        names = ", ".join(dict.fromkeys(Path(w).name for w in written))
        which = (f" (2D files: camera {self.project.name(self.project.active)})"
                 if self.project is not None and self.project.n_views > 1 else "")
        self.statusBar().showMessage(f"Exported {n} tracked frames ✓  {names}{which}", 8000)
        extra = ("<br>" + "<br>".join(dict.fromkeys(notes))) if notes else ""
        self.toast.show_message(f"Exported: {names}{which}{extra}", "info" if notes else "success",
                                12000 if notes else 7000)
        self._autosave()

    def _export_overlay(self):
        """File → Export Overlay Video: render the tracked result into an MP4
        in a background thread (its own VideoCapture), with a cancellable
        progress dialog. The app stays usable meanwhile."""
        from kinetrace.render import OverlayDialog, OverlayRenderer
        s = self.session
        if s is None or self.state != READY or self._overlay is not None:
            return
        default = self._default_output("_overlay.mp4")           # + the camera's name (G119)
        dlg = OverlayDialog(self, s, default, self.current, self.timeline.sel_range, n_frames=self.n_frames)
        if dlg.exec() != QDialog.Accepted or dlg.result_options is None:
            return
        opts, out = dlg.result_options, dlg.result_path
        bones = s.bones()            # the dialog's "Skeleton bones" tick decides, not the View toggle (I44)
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
            self.toast.show_message(_plain_error(e, "Could not store the token", short=True), "error")
            return
        dlg.token.clear()
        dlg.token.setPlaceholderText("token stored")
        self.toast.show_message("Token stored in models/hf/token. Gated models can now be "
                                "downloaded.", "success")

    # ----------------------------------------------------------------- close

    def _ask_save_before_close(self, doing: str = "closing"):
        """The close question: Save / Discard / Cancel (Esc and the window's x mean Cancel, G140).
        `doing` = what replaces the project ("opening clip.mp4", G143)."""
        name = self.project_path.name if self.project_path else "this project"
        return QMessageBox.question(
            self, "Save changes?",
            f"Save the changes to {name} before {doing}?\n\n{self._changes_text()}If you don't save, "
            "the unsaved work is dropped.",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)

    def _stop_runs_for_close(self) -> None:
        """Everything still running is stopped for the close, in an order that loses nothing: the
        tracking run first (its last rows land), then a Body run (its queued result is merged BEFORE the
        question, I198), the overlay render, an unfinished re-track (put back), the mask preview. Every
        thread waited for goes through _retire: one that outlives its wait is kept until it ends, never
        dropped running (the I112 / I133 abort)."""
        self._stopping_for_close = True        # a Body result landing now is merged, but opens no window
        try:
            self._stop_runs_for_close_steps()
        finally:
            self._stopping_for_close = False

    def _stop_runs_for_close_steps(self) -> None:
        # (I197) nothing may START during the close: the pause below ends the run "normally", and a
        # second pass of a two-pass run or the next re-track stretch used to begin inside closeEvent
        self._user_paused = True
        self._passes = None
        self._multi = None                   # closing: no camera after this one starts (G29)
        st = self._retrack
        self._retrack = None
        if st is not None:
            st["jobs"] = []
        if self.state == TRACKING and self.worker is not None:
            self.worker.request_pause()
            # let its last rows land (queued chunk_ready / finished) before the save
            t_end = time.time() + 20.0
            while self.worker is not None and not self.worker.wait(100) and time.time() < t_end:
                QApplication.processEvents()
            QApplication.processEvents()
        if self._overlay is not None:
            if self._overlay.isRunning():
                self._overlay.request_cancel()
                self._overlay.wait(10000)
            _retire(self._overlay)
            self._overlay = None
        for th in (self._body_worker, self._body_video):
            if th is not None and th.isRunning():
                th.request_cancel()
                th.wait(10000)
            _retire(th)
        # a stopped Body run still ends in finished_ok: its result is merged (the sender is still the
        # current worker) before the question and the save, instead of thrown away (I198)
        QApplication.processEvents()
        self._body_worker = self._body_video = None
        if st is not None:
            # a half-done automatic re-track has no Keep / Undo on exit: put it back (I107)
            try:
                self._retrack_restore(st)
            except Exception:       # noqa: BLE001
                pass
        if self._preview is not None:
            self._preview.cancelled = True          # between chunks: it ends with a sentence, no 15 s wait
            self._preview.wait(15000)
            _retire(self._preview)

    def closeEvent(self, ev):
        if self._first_frame_token is not None:
            self._first_frame_arrived()      # only waiting for a picture: nothing half-built
        if self._loading:
            # in the middle of an open: cancel it where that is safe, and close once it
            # has finished (the project being built must not be torn down half-way)
            cb = self._busy_cancel_cb()
            if cb is not None:
                cb()
            ev.ignore()
            QTimer.singleShot(250, self.close)
            return
        # (I198) with something still running and unsaved work, ASK first: Cancel then leaves
        # the run (a Body run of hours, an overlay render, a re-track) going
        live = bool((self.state == TRACKING and self.worker is not None) or self._retrack is not None
                    or (self._overlay is not None and self._overlay.isRunning())
                    or any(th is not None and th.isRunning() for th in (self._body_worker, self._body_video)))
        answer = None
        if live and self.project is not None and self.project.dirty:
            answer = self._ask_save_before_close()
            if answer == QMessageBox.Cancel:
                ev.ignore()
                return
        self._stop_runs_for_close()
        # Unsaved changes: Save writes the project; Discard drops the unsaved work; Cancel keeps the
        # window open. (Esc / x are Cancel: nothing is dropped silently.)
        if answer is None and self.project is not None and self.project.dirty:
            answer = self._ask_save_before_close()
            if answer == QMessageBox.Cancel:
                ev.ignore()
                return
        if answer == QMessageBox.Save:
            if not self._save_project():
                ev.ignore()
                return
        elif answer == QMessageBox.Discard:
            if self._save_worker is not None and self._save_worker.isRunning():
                self._save_worker.wait()
            recovery.discard(self._project_id)
        else:
            self._leave_project()
        if self._save_worker is not None and self._save_worker.isRunning():
            self._save_worker.wait()
        if self._exports_worker is not None and self._exports_worker.isRunning():
            self._exports_worker.cancel()            # stops between files (G42)
            self._exports_worker.wait()
        # Every thread this window owns must be stopped before the interpreter
        # tears Qt down, or Qt aborts ("QThread: Destroyed while thread is
        # still running" = exit 0xC0000409 on Windows). Only the working
        # camera's decoder used to be stopped here.
        for timer in (getattr(self, "_render_timer", None), getattr(self, "_play_timer", None),
                      getattr(self, "_lock_timer", None)):
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
            _retire(probe)
        # the video probe (a slow frame-count check on a 4K file or a share) and
        # any thread that outlived its earlier wait (I109, I112), also the ones the folder import,
        # the point-model test and the sync dialog keep for the same reason (I198)
        later = []
        for modname in ("kinetrace.pointtest", "kinetrace.syncdialog"):
            mod = sys.modules.get(modname)
            later += list(getattr(mod, "_ORPHANS", []) if mod is not None else [])
        for th in [getattr(self, "_probe", None)] + list(_ORPHANS) + later:
            try:
                if th is not None and th.isRunning():
                    th.wait(30000)
            except RuntimeError:
                pass
        fi = sys.modules.get("kinetrace.folderimport")
        if fi is not None:
            fi.wait_orphans(30000)
        ev.accept()


class _WorldAxesDialog(QDialog):
    """3D → Set World Axes… (I170): the origin, the +X and the +Y direction as three landmarks of the 3D
    result (their positions at the instant on screen, or their median over the whole result for markers
    that do not move). `result_axes` = (B, o, note) for `Calibration.reframed` once accepted."""

    def __init__(self, parent, rec, instant: int, left_handed: bool, unit: str = ""):
        from PySide6.QtWidgets import QComboBox, QRadioButton
        super().__init__(parent)
        self.setWindowTitle("Set world axes")
        self._rec, self._t, self._lh, self._unit = rec, int(instant), bool(left_handed), unit or "units"
        self.result_axes = None
        lay = QVBoxLayout(self)
        intro = QLabel(
            "Choose the 3D world yourself with three landmarks (a corner of a frame, three marks on the "
            "floor, a pole): the <b>origin</b> becomes (0, 0, 0), <b>+X</b> points from it towards the second "
            "landmark, <b>+Y</b> towards the third (made perpendicular to X), and <b>+Z</b> follows the "
            "right-hand rule. The calibration and the 3D result are re-expressed in that world; the "
            "pictures and every pixel a camera sees are not touched.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QFormLayout()
        names = list(rec.names)
        self.cb_origin, self.cb_x, self.cb_y = QComboBox(), QComboBox(), QComboBox()
        for cb, label in ((self.cb_origin, "Origin (0, 0, 0)"), (self.cb_x, "+X points toward"),
                          (self.cb_y, "+Y points toward")):
            cb.addItems(names)
            form.addRow(label, cb)
        lay.addLayout(form)
        self.rb_now = QRadioButton(f"Their positions at the instant on screen (reference frame {self._t})")
        self.rb_median = QRadioButton("Their median position over the whole 3D result (markers that do not move)")
        lay.addWidget(self.rb_now)
        lay.addWidget(self.rb_median)
        # three landmarks that have a position at the instant, else the median mode
        finite = [j for j in range(len(names)) if np.isfinite(self._pos(j, False)).all()]
        if len(finite) >= 3:
            self.rb_now.setChecked(True)
            pick = finite[:3]
        else:
            self.rb_median.setChecked(True)
            pick = [j for j in range(len(names)) if np.isfinite(self._pos(j, True)).all()][:3]
        for cb, j in zip((self.cb_origin, self.cb_x, self.cb_y), pick):
            cb.setCurrentIndex(j)
        self.preview = QLabel("")
        self.preview.setWordWrap(True)
        self.preview.setTextFormat(Qt.RichText)
        lay.addWidget(self.preview)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        for w in (self.cb_origin, self.cb_x, self.cb_y):
            w.currentIndexChanged.connect(self._update)
        self.rb_now.toggled.connect(self._update)
        self._update()

    def _pos(self, j: int, median: bool | None = None) -> np.ndarray:
        """Landmark j's position: at the instant, or its median over the result (NaN if none)."""
        r = self._rec
        median = self.rb_median.isChecked() if median is None else median
        if median:
            a = np.asarray(r.xyz[:, j], np.float64)
            a = a[np.isfinite(a).all(axis=1)]
            return np.median(a, axis=0) if len(a) else np.full(3, np.nan)
        k = self._t - int(r.t0)
        return np.asarray(r.xyz[k, j], np.float64) if 0 <= k < r.n_frames else np.full(3, np.nan)

    def _compute(self):
        """-> (B, o, note, text); B is None when the choice cannot make axes (text says why)."""
        from kinetrace import calib
        js = [self.cb_origin.currentIndex(), self.cb_x.currentIndex(), self.cb_y.currentIndex()]
        names = [self._rec.names[j] for j in js]
        if len(set(js)) < 3:
            return None, None, "", "Choose three different landmarks."
        pts = [self._pos(j) for j in js]
        for nm, pt in zip(names, pts):
            if not np.isfinite(pt).all():
                where = "anywhere in the 3D result" if self.rb_median.isChecked() else f"at reference frame {self._t}"
                return None, None, "", f"{nm} has no 3D position {where}: choose another landmark (or the median option)."
        try:
            B, o = calib.world_axes(pts[0], pts[1], pts[2], self._lh)
        except ValueError as e:
            return None, None, "", str(e)[0].upper() + str(e)[1:] + "."
        when = ("median over the 3D result" if self.rb_median.isChecked() else f"frame {self._t}")
        note = f"world: origin = {names[0]}, +X toward {names[1]}, +Y toward {names[2]} ({when})"
        dx, dy = float(np.linalg.norm(pts[1] - pts[0])), float(np.linalg.norm(pts[2] - pts[0]))
        text = (f"<b>{names[0]}</b> becomes (0, 0, 0); <b>{names[1]}</b> lies {dx:.4g} {self._unit} along +X; "
                f"<b>{names[2]}</b> lies in the XY plane ({dy:.4g} {self._unit} from the origin). ")
        if self._lh:
            text += ("The calibration's present world is mirrored (left-handed); the new one is right-handed, "
                     "so exported cameras and points need no mirroring.")
        return B, o, note, text

    def _update(self, *_):
        B, o, note, text = self._compute()
        ok = B is not None
        self.preview.setText(text if ok else f"<span style='color:{theme.AMBER}'>{text}</span>")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)

    def _accept(self):
        B, o, note, _ = self._compute()
        if B is None:
            return
        self.result_axes = (B, o, note)
        self.accept()


def main():
    import os
    import sys
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")   # no telemetry (see __main__.py)
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple GPU: unsupported ops go to the CPU
    if "--check" in sys.argv[1:]:
        # `python -m kinetrace --check` (run.bat / run.sh --check): the hardware
        # report on the console, no window - what the launchers print at the end
        # of an install and what a support request asks for
        from kinetrace.device import cli
        sys.exit(cli())
    if "--paths" in sys.argv[1:]:
        # `python -m kinetrace --paths` (run.sh / run.bat --paths): every folder Kinetrace reads
        # or writes, with sizes, no window (Mac install audit P2-7)
        from kinetrace.paths import cli as paths_cli
        sys.exit(paths_cli())
    # Under pythonw.exe / GUI-mode launches there is no console and
    # sys.stdout/sys.stderr are None — but torch.hub, tqdm, and warnings all
    # write there. Give them a safe sink so a Track click can't crash.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    # the error log (kinetrace/crashlog.py, I140): with no console an error in a
    # button, a key handler or a thread left no trace at all, and a crash left
    # only the window gone. Started before Qt, so Qt's own messages are kept too
    from kinetrace import crashlog
    crashlog.install(APP_VERSION)
    if sys.platform == "win32":
        # its own taskbar entry and icon: without an app id Windows groups the window
        # under python.exe and shows Python's icon (G41)
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Kinetrace.Kinetrace")
        except Exception:     # noqa: BLE001 - cosmetic only
            pass
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    apply_theme(app)
    from kinetrace import appicon
    app.setWindowIcon(appicon.icon())       # drawn in code, cached in _theme_cache (G41)
    win = MainWindow()
    crashlog.attach(win._error_context, win._on_error_logged)
    win.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        arg = sys.argv[1]
        if arg.endswith(PROJECT_SUFFIX) or projectfile.is_project(arg):
            QTimer.singleShot(0, lambda: win._open_project_from_path(arg))
        elif arg.lower().endswith(".csv"):
            QTimer.singleShot(0, lambda: win._import_tracks_dialog(arg))
        else:
            QTimer.singleShot(0, lambda: win._open_video(arg))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
