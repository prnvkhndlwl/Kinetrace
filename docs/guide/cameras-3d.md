<sub>Kinetrace guide — [Home](../../README.md) · [Install & run](install.md) · [Workflow: track an animal](workflow.md) · [Segments, silhouettes & skeletons](segments.md) · **Several cameras & 3D** · [Human bodies](bodies.md) · [Export formats](exports.md) · [Projects & autosave](projects.md) · [Keyboard & mouse](shortcuts.md) · [Accuracy](accuracy.md) · [Performance (4K)](performance.md) · [Limitations](limitations.md) · [Testing](testing.md) · [Folder map & design](folder-map.md) · [User manual](../MANUAL.md) · [Licences](../../THIRD_PARTY_LICENSES.md)</sub>

# Several cameras of the same event

Filmed the animal from more than one angle? Put every camera in **one project**
and keep them in sync.

1. Open the first camera's video as usual, then **＋ Add video** in the
   **CAMERAS** panel (top of the right dock) — pick one file or several at once.
   Each camera appears as its own view in a grid above the timeline, showing its
   own tracked points and silhouette; a new camera gets the working camera's
   skeleton, so the landmark names already match. Up to 15 cameras. Adding a
   camera to a calibrated project keeps the calibration, but 3D needs every
   camera calibrated: calibrate again with all of them, or remove the new one
   with its **×** to get 3D back.
2. **Line them up.** The **first camera you load is the reference** — its offset
   is always 0 and is not editable. Every other camera carries a **frame
   offset**: *the frame this camera shows when the reference is at its frame
   0*. A camera switched on N frames **after** the reference has offset **−N**;
   one switched on earlier has +N. That meaning does not change when you switch
   which camera you are working in.
   The quick way is **3D → Sync Cameras (Sound / Motion)…**: from the
   cameras' **sound tracks** (claps, voices — with a noise filter; the usual
   way for hand-started cameras) or from *how much the picture changes*
   frame to frame, each camera gets an offset and a verdict (CLEAR / WEAK /
   NONE). It works on a stretch around the instant on screen, so park the
   playhead near the clap first; then press **Apply the ticked offsets**. The manual way: scrub to something every camera saw — a flash, a
   clap, the first contact — then nudge that camera's offset with **◂ ▸**
   (or type it) until the same moment is on screen in both views, or press
   **Align here** to take what you are looking at as the match. The panel
   then tells you the **overlap** — the frames for which every camera has a
   picture.
3. **Work one camera at a time.** The camera you track is the one selected in the
   panel (or just click its view). Points, silhouette, skeleton, events, undo and
   the timeline are all *that camera's*; the others stay visible but view-only,
   scrolling along with the playhead through their offsets. Switching cameras
   keeps you on the same instant, so you never lose your place. It also disarms
   N and S and drops a half-marked event (they belong to the camera you left),
   while the toggles — Follow, Auto-pause, ROI, Body, the point model — stay as
   you set them.
4. **Everything is saved in the project** (Ctrl+S): every camera's video path,
   tracks, silhouette and offset, and which one you were working in. Reopening a
   project reopens all the videos and asks you to locate any that moved. A
   camera whose video you cannot locate is left out of that session, and the
   project file is then **not** overwritten (autosave goes next to the video and
   Save asks for a new name), so reopening once the drive is back gives you
   every camera again.
5. **Export for 3D**: choose **ALL CAMERAS — DLTdv8 xypts…** (Ctrl+E). It writes one
   file with `pt1_cam1_X, pt1_cam1_Y, pt1_cam2_X, …` columns, matching
   landmarks **by name** across cameras — the input DLTdv / Argus / a
   triangulation script expects. **Row k is frame k of the reference (first)
   camera, counting from 0**; the other cameras are sampled at the same instant
   through their offsets, and `NaN` fills every cell a camera has no data for.
   Pixels are DLTdv8's: top-left origin, first pixel = 1. The
   `*_pointnames.csv` sidecar names the points, the convention, the rows and
   which video is cam1, cam2, …. Give the same landmark the same name in every
   camera and it lines up. The ordinary 2D exports stay per camera and default
   to `<project>_<camera>_tracks.csv`, so one camera's files never overwrite
   another's.

**Ctrl+2** hides the other cameras when you want the full window (and stops them
decoding, which gives scrubbing the whole disk back). Cameras may run at
*different frame rates and resolutions*: a camera at twice the reference rate is
marked **×2** in the panel, steps two of its frames per reference frame, and
carries its offset in its own frames.

### From cameras to 3D (the **3D** menu)

6. **3D → Calibrate a Lens (checkerboard)…** — for wide-angle / action
   cameras (needed) or when unsure (the report says in pixels whether it
   matters). Best with the project of that camera open, so the profile can be
   attached to it at the end; with nothing open the wizard still runs and ends
   by saving a lens file to attach later. Its first page saves the printable
   board (*Save the checkerboard to print…*). Print it, film it with that camera at the **same
   picture size** as its videos (another size is refused, naming both), and
   choose that video on the wizard's second page. The wizard finds the board,
   fits the lens (standard or fisheye model, chosen automatically), shows a
   verdict with reasons and a before/after picture, and **Attach to …** gives
   the profile to the camera; *I already have a lens file…* takes a
   `.klens.json` or an Argus / DLTdv profile instead and skips to the last page.
   The wand wizard's Cameras page opens the same wizard per camera
   (**Calibrate…** / **Load file…**). The wand calibration then straightens
   that camera's points; with every camera profiled it takes their focal
   lengths from the checkerboards, with only some it refines the focal lengths
   from the wand (and, with *lens distortion* ticked, fits the distortion of the
   cameras without a profile).
   **3D → Calibrate Cameras with a Wand…** — calibrate inside the app, no
   MATLAB: track the two ends of a wand of known length (**Add ▾ → Ball marker**
   on each ball; the same landmark names in every camera), optionally a dropped
   ball for the vertical and an independent scale check, and the wizard solves
   the cameras (sparse bundle adjustment, the easyWand principle) and reports a
   plain-language verdict — good / usable / not good enough — with the reasons
   and what to change; **Use this calibration** makes it the project's. It
   opens with one camera too, to review the wand tracking, and says a second
   camera is needed before it can calibrate. **3D → Export Calibration…** writes
   a self-describing `.kcal.json` for your animal projects, a `_dltCoefs.csv`
   (MATLAB 1-based pixels) for DLTdv users and the report as text; when a
   camera has a lens correction the csv is for lens-corrected pixels and a
   `_dltCoefs_README.txt` says so.
   **3D → Import Calibration…** — a `.kcal.json`, or the 11-parameter DLT that
   DLTdv / easyWand / Argus write (`…dltCoefs.csv`, one column per camera), an
   `…easyWandData.mat`, a DLTdv8 `…dvProject.mat` saved as MATLAB v7, which
   also carries DLTdv's lens undistortion (its local-weighted-mean transform is
   rebuilt from the control points stored in the file), or OpenCV-style cameras
   (a K matrix per camera plus R/t, as JSON or text). Map file columns to
   cameras and state the pixel convention (DLTdv/easyWand count from 1, y down
   — the default; a `.kcal.json` or a K + R/t file states its own and the
   choice is locked). A `dltCoefs.csv` records no picture sizes, so match its
   columns to your calibration software's camera order.
7. **3D → Estimate Sub-frame Offsets…** — cameras that were not genlocked differ
   by a *fraction* of a frame; the fraction is found from the tracks themselves
   by minimising the triangulation residual (grid search, then a joint Powell
   refinement) and applied after you confirm. Whole-frame alignment by eye first,
   this afterwards.
8. **3D → Reconstruct 3D Landmarks** (Ctrl+3) — every landmark seen by ≥ 2
   cameras at an instant is triangulated (linear DLT), with DLTdv's residual
   `sqrt(Σe² / (2n−3))` per position. Tracks are interpolated at the fractional
   local frame of each camera. The **3D view** (**3D → Show 3D View**, Ctrl+5)
   opens and follows the playhead, and a verdict comes with the numbers (GOOD /
   USABLE, with care / NOT TRUSTWORTHY, the worst landmark named). Where one
   camera disagrees with the others, a magenta band marks the stretch on that
   camera's timeline lanes. **View → Epipolar guides from the other cameras**
   draws the lines the selected landmark must lie on; right-click it → **Snap to
   the other cameras' rays here** puts it there (one undo step); **3D →
   Re-track Disagreeing Stretches…** does that at the start of every magenta
   stretch, re-tracks it, reconstructs again and asks Keep / Undo with a
   before / after verdict (X or Space stops the whole queue).
9. **3D → Carve Volume at This Frame** (Ctrl+4) — the **visual hull**: the space
   every camera's silhouette agrees on, carved on a voxel grid, meshed (marching
   tetrahedra on the signed distance), with its volume in the caption. It needs
   the animal segmented in at least **three** cameras at that instant (two
   silhouettes carve a long sliver, not the animal, and the app says so).
   **3D → Export Mesh of This Frame…** writes OBJ/PLY. **Ctrl+E → 3D landmarks**
   writes the xyz CSV + residual sidecar, **Ctrl+E → 3D kinematics** smoothed
   velocities and accelerations. Calibration, offsets and 3D live in the
   project. Most 3D entries stay clickable with one camera or no calibration
   yet, and say what is still missing.

You can start tracking from **any frame of interest** — tracking always begins
at the frame you are looking at. Points that have no position at that frame
are skipped (their existing tracks are kept untouched).

**ROI zoom** (the **ROI** toggle in the control bar, on by default): when your points sit in a small part of a
high-resolution frame — say three balls on a stick somewhere in a 4K shot —
the tracker automatically works inside a crop around them, so the model sees
the objects with several times more detail. It engages only when it clearly
helps and follows the points by transparently re-seeding when they near the
crop edge.

**Appearance lock:** right-click a point — either its marker on the video or
its entry in the Points panel — and choose **"Lock to seed appearance"**. For
a distinct, rigid feature this snaps the track back whenever the point's
original appearance is found nearby, catching slow drift. Off by default
because it changes the plain "track the placed location" semantics. It
applies to **plain points only**: a circled region doesn't need it (its
center is already stabilized by the member fit), so the menu shows it
grayed out there.

### Checking a lens calibration

A calibration is a number you cannot eyeball, so the lens wizard shows its
evidence (`boardreview.py`). After the scan you get **every board it found, in
a scrollable gallery**: the detected corners, the first row drawn as a line
with corner 0 ringed (so a flipped or rotated detection is obvious), the
board's own X/Y/Z axes from `solvePnP`, and that view's reprojection error
colour-coded. **Click a board** to open the full-resolution frame and drag a
corner onto where it belongs — the drop is `cornerSubPix`-refined, so a rough
drag still lands sub-pixel. **Tick or untick any image** to choose what the fit
uses, with *Best spread* / *All* / *None* / *Drop the worst* as shortcuts.
After unticking a board or moving a corner, press **Fit the lens from the
ticked boards**: *Next* waits for that refit, so the profile you attach is the
one fitted from what you see. The last page's coverage map lights the boards
the fit actually used.

The automatic choice (`lens.auto_select`) runs twice: a diverse spread over the
picture, near and far, tilted and square-on (a fit from one pose is degenerate
however many frames it has), then a provisional fit and the removal of views
that reproject worse than 3× the median — unless that would leave too few. It
returns the reason in words, and the page shows it.

---

[← Segments, silhouettes & skeletons](segments.md) &nbsp;|&nbsp; [Contents](../../README.md#contents) &nbsp;|&nbsp; [Human bodies →](bodies.md)
