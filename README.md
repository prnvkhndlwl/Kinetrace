# Kinetrace — animal motion tracking

**Kinetrace** is an interactive desktop tool for animal motion tracking from video: Meta's
[CoTracker3](https://github.com/facebookresearch/co-tracker) with a
native-resolution subpixel refinement stage for landmarks, and Meta's
[SAM 3 / SAM 2](https://github.com/facebookresearch/sam3) segmentation for the
animal's silhouette (tail tip, midline, feet and wing tips come from the
silhouette, so thin tails and textureless bodies no longer defeat tracking). Designed for long (40,000+
frame), high-resolution (4K) videos where you watch tracking happen live,
pause at any moment, fix any point with the mouse, and resume.

For **human** subjects there is also a **body layer**: Meta's
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) (or an ungated 2D
keypoint model) finds a person, extracts their joints and joint angles, and
shows them side by side with the footage — see
[Human bodies](#human-bodies-joints-and-joint-angles-the-body-menu).

**Everything installs inside this folder.** No system changes, no registry,
no global Python packages — *deleting the folder is a complete uninstall*.

> **New to motion tracking?** Read **[docs/MANUAL.md](docs/MANUAL.md)** instead of this
> page. It assumes no experience at all: what the ideas mean, a first session
> step by step, how to fix mistakes, and a glossary. It is also inside the app —
> press **F1**, or **Help → User Manual**. This README is the feature summary
> for someone who already knows what tracking is. In the app, hover any menu
> entry or button to see what it does.

## At a glance

| | |
|---|---|
| **Point tracking** | AllTracker (default) or CoTracker3, live on screen, pause anywhere with X / Space, correct with the mouse, resume; sub-pixel refinement at 4K; regions (circle / rectangle / polygon); auto-pause when a point is lost |
| **Silhouettes** | SAM 3 / SAM 2.1 outline the animal on every frame; tail tip, midline points, feet and wing tips come from the outline; named skeletons per species |
| **Ball markers** | click a ball, SAM outlines it, a circle fit gives its centre — wand balls, reflective markers |
| **Several cameras** | one project per shoot; sync by sound (claps), by motion, or to a fraction of a frame from the tracks |
| **3D** | wand and checkerboard calibration wizards with plain-language verdicts, DLT import from DLTdv / easyWand / Argus, triangulation, epipolar guides, automatic re-tracking of disagreeing stretches, visual-hull volume, speeds and accelerations |
| **Human bodies** | SAM 3D Body or ViTPose: joints, joint angles, side-by-side view and video |
| **Exports** | CSV, DeepLabCut, DLTdv8 (one camera or all cameras), MATLAB .mat, overlay video, 3D landmarks and kinematics |
| **Scale** | 40,000+ frame 4K videos, never loaded into memory; autosave every 30 s; projects restore the exact working state |
| **Built for** | biologists with no tracking or calibration background: an in-app manual (F1), a tooltip on every control, and a verdict with every number |

## Licence and third-party software

Kinetrace's own code has **no licence chosen yet** (all rights reserved until a
`LICENSE` file is added). It is built on open models and libraries, each under
its own licence — **[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)** lists
every one, with what it asks of you:

- **CoTracker3 is CC BY-NC 4.0 — non-commercial use only.** The default point
  model, AllTracker, is MIT.
- **SAM 3, SAM 3.1 and SAM 3D Body are under Meta's SAM License** (and DINOv3
  under the DINOv3 License): research and commercial use allowed, **acknowledge
  them in publications**, no military or weapons use; the weights are gated,
  so each lab requests access on Hugging Face itself.
- SAM 2.1, ViTPose and RT-DETR are Apache-2.0; the Python packages are
  BSD / MIT / Apache, except PySide6 (Qt, LGPL-3.0) and the GPL ffmpeg
  executable that `imageio-ffmpeg` carries for the sound sync.

No model weights, third-party code or recorded footage are stored in this
repository; they are downloaded or supplied on each computer.

**Citing:** if you publish results made with Kinetrace, cite the models you
used — CoTracker3 (Karaev et al., 2024) or AllTracker (Harley et al., ICCV
2025) for point tracks, SAM 2 / SAM 3 for silhouettes and ball markers, SAM 3D
Body or ViTPose for human poses (the SAM and DINOv3 licences require the
acknowledgement).

## Install & run

**What you need**

| | Windows 10 / 11 | Ubuntu 22.04 or newer | macOS 14 or newer |
|---|---|---|---|
| Computer | 64-bit PC | 64-bit PC (x86-64 or ARM) | a Mac with **Apple Silicon** (M1 or later); Intel Macs are not supported by PyTorch any more |
| Python | 3.10 – 3.14 (3.12 recommended, from python.org) | the system `python3` (22.04 has 3.10, 24.04 has 3.12) plus `python3-venv` | 3.10 – 3.14 from python.org or Homebrew |
| Graphics | an NVIDIA GPU with a recent driver (R580 or newer) is used automatically; without one, the CPU | the same | the Mac's own GPU (Metal) |
| Disk | about 5 GB for the environment, plus the models you use | the same | about 3 GB, plus models |

**Get the code**, with git (`git clone <this repository's URL>`) or GitHub's
*Code → Download ZIP*, into a folder of your choice.

**Windows:** double-click `run.bat`.

**Ubuntu:** install the few system packages once, then start the launcher:

```bash
sudo apt install python3 python3-venv libxcb-cursor0 libegl1 libxkbcommon-x11-0 libgl1
```

```bash
./run.sh
```

**macOS:** open *Terminal* in the folder and run `./run.sh`. (If macOS refuses
to run it, `chmod +x run.sh` first.)

The **first run** creates a private Python environment in `.venv` inside the
folder and installs everything into it — up to ~4 GB, mostly PyTorch — which
takes a while; later launches start at once. `install.py` chooses the PyTorch
build for the machine: the CUDA 13 build when an NVIDIA GPU is present, the
CPU build on Linux without one, the standard build (Metal GPU) on a Mac. To
force a choice, set `KINETRACE_TORCH=cuda`, `cpu` or `mps` before the first
run (for example the CPU build on a PC whose NVIDIA driver is too old for CUDA
13). If the download is interrupted, run the launcher again: it resumes.

**Models** are downloaded the first time a feature needs them, into `models/`
inside the folder: the AllTracker point model (63 MB, when its code has been
placed in `models/alltracker`) or CoTracker3 (~100 MB), SAM 2.1 for
silhouettes (~620 MB), ViTPose + RT-DETR for human poses (~500 MB). **SAM 3**
and **SAM 3D Body** are gated by Meta: request access on Hugging Face, then
put the weights in `models/sam3` / `models/sam-3d-body-*` (the Settings dialog
and the Body dialog say exactly what is missing). An internet connection is
needed only for these first downloads.

**Everything stays in the folder:** nothing is installed into the operating
system, and deleting the folder removes Kinetrace completely. Kinetrace sends
no usage data anywhere (Hugging Face telemetry is switched off); the only
network traffic is the model downloads above.

The status bar always shows which device (CUDA GPU, Apple GPU or CPU) the app
is using; on the CPU, tracking works but is several times slower.

## Workflow

1. **Open a video** (Ctrl+O) — any common format/codec via OpenCV/FFmpeg.
2. Press **N** (or the **Add** button in the control bar under the timeline; on
   a narrow window its buttons show only their icons — hover one for its name)
   — the cursor becomes a crosshair —
   then **click** the thing you want to track. One press, one placement:
   stray clicks never add or move anything. Points are listed in the right
   panel (double-click to rename, checkbox to show/hide, right-click for
   rename/appearance-lock/delete; Ctrl/Shift+click selects several).
   **Or, while armed, drag a circle** around an object instead: the region is
   tracked as a cloud of internal points and exported as **one** point — its
   robustly fitted center. Regions survive rotation, partial occlusion, and background
   changes far better than a single point (draw the circle to match the
   object, not larger — the tracked points are sampled *inside* it).
   **Balls** — wand balls, reflective markers, a dropped ball — get **Add ▾ →
   Ball marker (SAM circle)** instead: click the ball, and on every frame SAM
   outlines it, a circle is fitted and its centre is the point (balls about
   5 px across still work). Balls tracked in one run share one 960-px window:
   if two are more than about 740 px apart the run stops and says to track
   each ball on its own (select it alone in the POINTS list, then Track).
   **Segment the animal (optional):** press **S** (or the **Segment** button) and **click the animal**.
   Nothing waits for this step — points alone track fine, and for an animal
   only a few pixels across (a small fish or a bird seen from far away) **skip it**:
   there is no silhouette at that size and the model grabs something bigger.
   Place a point with **N** and Track instead; the ROI zoom gives the tracker a
   close-up around it. When you do segment, the
   silhouette appears within a second (the first click loads the model). If it
   grabbed too much, **Shift+click** the wrong part; if it missed a part, click
   it; or **drag a box** around the animal. Right-click a click marker to remove
   it. The **▾ arrow on the Segment button** picks which segmentation model does
   this (SAM 3 / SAM 2.1), each entry showing whether its weights are ready,
   need downloading, or are gated — the same choice as in Settings.
   The silhouette is always the whole animal: that is what the segmentation
   model answers with, and it is the reliable way to use it. For a single body
   part — one wing, one leg — track named points on it instead.
   Then choose **Skeleton ▾** in the right panel: a named landmark set for
   your study animal (lizard/iguana, quadruped, gliding mammal, flying lizard,
   undulating body, bird/bat, or your own). Landmarks marked with a dotted square
   are *derived from the silhouette* (tail tip, midline %, feet, wing tips) and
   need no clicking; the others you place: select one, press **N**, click it.
   Place the head landmark first — it anchors the midline.
3. Press **Track ▶** (or **T**). Tracking runs forward *frame by frame, visibly* —
   the progress bar shows position, the status bar shows speed and ETA, and
   the **timeline panel** fills in live. (**Space** only plays the video; during
   a run it pauses.)
   The Track button's dropdown also picks the **point model**: **AllTracker**
   (the default whenever its code is in `models\alltracker`) or **CoTracker3**. AllTracker tracks every point from the frame you
   started on with a high-resolution dense correlation, so it does not
   accumulate drift along a uniform body — measured on the iguana clip it keeps
   head and hip on the animal for 270 frames where CoTracker3 slides off, and it
   halves the error on a small textured feature. CoTracker3 is about twice as
   fast and slightly sharper on high-contrast markers (0.74 px at 4K after
   refinement). Both get the same silhouette constraint and sub-pixel refinement.
   The Track button's dropdown offers two modes:
   - **Automatic** (default): run to the end of the video.
   - **Semi-automatic**: each press of **F** tracks *one* frame forward and
     pauses. Check the result, fix anything, press **F** again — every step
     re-seeds from what you see, so you supervise the track frame by frame.
     (Where no point has data, F just moves forward as usual; B always just
     steps back.)
4. **Pause any time**: press **X**, Space, or the Track button (it reads
   **Pause ■ (X)** during a run). It stops within a fraction of a second. With
   **Auto-pause** on (default), the app also stops *itself* when it loses a
   point (low confidence for 16 frames in a row): it jumps back to the first
   unreliable frame, selects the point, and **cuts that point's track there** —
   what the run wrote for it from that frame on is cleared, so its data ends
   where it stopped being trustworthy. Ordinary occlusion does not trigger it.
   With a segment and the **Body** toggle on, a landmark that **leaves the
   silhouette** stops the run at that frame as well (Auto-pause on or off): its
   track ends on the frame before. Click it where it really is and press Track
   again.
5. **Review & correct**: scrub by **clicking or dragging anywhere on the
   timeline panel** — it is the scrubber, and each point's lane shows exactly
   which frames are tracked (red tint = the model was unsure there). Drag the
   **splitter above the panel** to make it taller (more point lanes; the
   wheel scrolls any that still overflow). **Shift + +/− (or Ctrl+wheel over
   the panel) zoom the time axis** around the cursor, Premiere-style: the
   ruler relabels itself at sensible 1/2/5-step intervals as you zoom, and a
   **scroll strip appears along the panel's bottom edge** showing where the
   view window sits in the whole video — click or drag it (or middle-drag
   the lanes) to move around; the view also follows the playhead. The zoom
   buttons next to the play controls do the same (the third shows the whole
   video again). Hotkeys work wherever the keyboard focus is, except while you
   type a name. The full reference lives in **Help → Keyboard & Mouse Reference**. Use **F/B** keys (frame forward/back; hold Shift to
   jump by the *step* size in the control panel), arrow keys, or the frame
   box (type a number + Enter).
   Zoom with the mouse wheel or **+/−** (anchored under the pointer); pan
   with middle-drag or the **✋ Pan** tool (**H**) for left-drag panning.
   The view never moves on its own unless you switch **⌖ Follow** on (it is
   off by default): then it keeps tracked points in sight while zoomed in,
   following the *selected* point (holding still across gaps in its track),
   or — with nothing selected — re-framing **all** points as they move. **R**
   always fits the whole frame. If a
   marker hides the exact spot it sits on (a ball close to the camera, or
   zoomed way in), shrink the **marker px** size in the control panel —
   markers keep a fixed on-screen size by design, but you pick that size.
   Fix any point on any frame:
   - **drag** the point marker, or
   - select it, then **Ctrl+click** where it should be, or
   - select it in the POINTS list and simply **click** the video (no N): it is
     placed there on this frame by hand (◆ on the timeline) — the way to
     digitize frames the tracker gets wrong.
   Each of these is one undo step (Ctrl+Z). A landmark derived from the
   silhouette cannot be placed by hand: switch it first with right-click →
   **Data source → Track by appearance**.
6. Press **Track** again: the points **selected in the right panel** re-seed from
   their positions on the current frame (your corrections included) and re-track
   forward, overwriting later frames. Select all, or nothing, to track every point;
   Ctrl/Shift+click to pick several. `Ctrl+Z` (**Edit → Undo Last Run / Edit**) undoes the whole last tracking run — or the last hand edit — if needed.
   **Bulk cleanup:** to wipe a bad stretch, **Shift+drag on the timeline**
   across the frames *and* the lanes you want gone, then press **Delete**.
   The drag is a marquee in both directions, so it picks the target for you:
   drag along the **segment lane** to drop just the **silhouettes** in that
   stretch, along one or more **point lanes** to drop just **their** tracked
   data, or across both to drop both in a single undoable step. The band
   labels what Delete will clear, and right-clicking inside it offers the same
   choices explicitly. Dragging in the **ruler** (or right-click an event →
   *Select this window*) selects the frames without naming lanes: Delete then
   falls back to the points selected in the right panel, asking first if none
   are. Either way the points themselves stay — only that stretch of their
   data goes. Delete with points (and no frame window) selected removes those
   points entirely. All of it is undoable with Ctrl+Z.
7. **Mark events** as you go: press **E** at a frame of interest, scrub, press
   **E** again — the name dialog offers your **existing event types in a
   dropdown**, so tagging another occurrence of "swing" is one click (type a
   new name to create a new type). All occurrences of a type share one color
   on the timeline, and the Events menu groups them ("swing (3×)" opens the
   list of its occurrences). Click a ribbon to select its window and jump
   there; right-click to rename, retime, or delete. Events are saved in the
   project and included in exports.
8. **Export** (Ctrl+E) when happy.

## Several cameras of the same event

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

### Shortcuts

| Key | Action |
|---|---|
| N | Add a point: arms the crosshair — next click places it (drag = a region; Add ▾ picks circle / rectangle / polygon) |
| click (not armed) | Annotate by hand: place the point selected in the list on this frame, replacing the tracker's position (◆ on the timeline) |
| Shift+< / Shift+> | First / last frame the selected point has data on; with nothing selected, the segment's first / last silhouette |
| , / . | Previous / next hand-placed frame of the selected point |
| right-click a point | Rename, appearance lock, data source, hidden-here, *May leave the segment*, delete — plus go to its first / last / first hand-placed / last hand-placed / first doubtful frame; clear its position on this frame, in the selected window, or its whole track; fill the gaps between its hand placements with a smooth curve (or replace everything between them); and, with a calibration, snap it to the other cameras' rays |
| J / Shift+J | Next / previous low-confidence (red) stretch |
| Shift+X | Mark the selected point hidden on this frame: kept, not exported, not used for 3D (again to unmark; timeline selection → right-click marks a window) |
| Shift+N | Note on this frame (green ▲ on the timeline; Edit → Annotator Name records who) |
| O / L | Onion skin (ghosts of the previous / next frame) / loupe (magnifier under the cursor) |
| View → Trails / Display filter | Fading trails with optional upcoming path; contrast / brighten / frame-difference view (display only) |
| S | Segment tool: click the animal (Shift+click = not the animal, drag = box); S again or Esc when done |
| SEGMENT row (right panel) | The tracked segment as its own row: checkbox shows/hides the silhouette, double-click renames, right-click jumps to its first/last silhouette, clears this frame / the selected window / all of them (clicks kept), or removes it |
| Body (toolbar toggle) | Keep tracked points on the segment: a point a few pixels outside the silhouette is nudged back onto it; a landmark that really leaves it **stops the run at that frame** (its track ends on the frame before). Right-click a point → *May leave the segment (free point)* to exempt it |
| Segment ▾ | Pick the segmentation model (SAM 3 / SAM 2.1), with each entry's weight status; last entry opens Settings |
| Ctrl+, | Settings: segmentation model (SAM 3 / SAM 2.1), Hugging Face token, mask opacity — also the last entry of the **Segment ▾** dropdown |
| T | Start tracking / pause (semi-automatic mode: one step) |
| Space | Play / pause preview — pauses tracking while a run is live |
| X | Pause tracking |
| F / B | Frame forward / back (in semi-automatic mode, F *tracks* one frame forward) |
| Shift+F / Shift+B | Jump forward / back by the *step* size (toolbar spinner) |
| H | Pan tool — left-drag moves the view (middle-drag always pans) |
| + / − (or =) | Zoom the video in / out around the pointer |
| Shift+ + / Shift+ − | Zoom the timeline's time axis (also Ctrl+wheel over it) |
| ← / → (Shift: ±step) | Step frames |
| Home / End | First / last frame |
| E | Mark event start / end at the current frame |
| R | Reset view (fit video to window) |
| Shift+C | Clear the frame cache and re-decode the current frame (failsafe for a stale/garbled picture) |
| Shift+drag (timeline) | Select a frame window **and the lanes under the drag** — segment lane, point lanes, or both |
| Delete | Clear the timeline selection (the lanes it covers decide: silhouettes, those points' tracks, or both) — or, with no selection, delete the selected point(s) |
| Esc | In this order: leave the segment tool, cancel point placement (or a half-drawn region), cancel a half-marked event, clear the timeline selection, deselect the point |
| Ctrl+Z | Undo the last tracking run or the last edit (a click, drag, Ctrl+click, Delete, Shift+X, a timeline clear) |
| Ctrl+2 | Show only the working camera (hides the others and stops them decoding) |
| Ctrl+1 | Show / hide the Segment & Points panel |
| Ctrl+3 / Ctrl+4 / Ctrl+5 | Reconstruct 3D landmarks / carve the volume at this frame / show the 3D view |
| Ctrl+6 | Body: the side-by-side view |
| Enter / double-click | Close a polygon region |
| F1 | The user manual (**Help → Keyboard & Mouse Reference** lists every key) |
| Ctrl+O / Ctrl+Shift+O | Open video / project |
| Ctrl+S / Ctrl+Shift+S | Save project / save as |
| Ctrl+E | Export tracks |
| File → Export Overlay Video… | An MP4 with markers, names, skeleton, silhouette, trails, frame counter, events and notes drawn on it |

## Export formats

| Format | Layout | Best for |
|---|---|---|
| **Wide CSV** | one row per frame: `frame, P1_x, P1_y, P1_visible, P2_x, …` (blank cells = untracked) | Excel, quick plotting |
| **Sparse TSV** | one row per tracked (frame, point): `frame  point  x  y  visible` | smallest files, pandas/R |
| **DeepLabCut CSV** | 3 header rows (`scorer`, `bodyparts`, `coords`), `x, y, likelihood` per landmark, one row per frame | DeepLabCut, Anipose, pose notebooks |
| **DLTdv8 xypts CSV** | `pt1_cam1_X, pt1_cam1_Y, …`, one row per frame, `NaN` where untracked, **top-left origin, first pixel = 1** (DLTdv8's own convention) + a `*_pointnames.csv` sidecar naming the points and stating the convention. A separate *bottom-left origin* entry writes the older DLTdv5 / Argus Clicker variant | DLTdv / easyWand 3D workflows |
| **ALL CAMERAS — DLTdv8 xypts** (two or more cameras) | every camera in one file, `pt1_cam1_X, pt1_cam1_Y, pt1_cam2_X, …`, landmarks matched **by name**; row k = frame k of the reference camera (from 0), each camera sampled through its offset, `NaN` where a camera has no data; top-left, first pixel = 1; the sidecar also names the rows and each camera's video | DLTdv / easyWand / Argus / your own triangulation |
| **MATLAB .mat** | `tracks (T×N×2)` with NaN where untracked or marked hidden, `visibility`, `manual`, `occluded`, `confidence`, `point_names`, `point_kind`, `point_source`, `point_spec`, `event_*`, `note_*`, `fps`, `segment_*` (presence, score, bbox, centroid, area, 32-point midline), `skeleton_*`, and `pixel_convention` — these pixels count from **0** (OpenCV); add 1 for MATLAB / DLTdv8 | direct MATLAB analysis |
| **3D landmarks CSV** | `frame, name_X, name_Y, name_Z, …` per reference frame (NaN where < 2 cameras) + a `*_xyzres.csv` sidecar with each position's residual (px), camera count, and how far each camera disagrees (`name_camK_px`) | 3D positions; the same layout as DLTdv's xyzpts |
| **3D kinematics CSV** | a first line starting with `#` (unit, fps, the smoothing used — read it with `comment='#'`), then `frame, time_s` and per landmark the smoothed `X Y Z`, `Vx Vy Vz speed`, `Ax Ay Az acc` (project units per second), with the smoothing cutoff (chosen by residual analysis, or yours); stretches shorter than 12 frames keep their positions with blank velocity and acceleration; plus a plain-English `*_report.txt` with hand checks | velocities and accelerations |
| **Mesh (3D → Export Mesh of This Frame…)** | the carved volume hull of one frame as OBJ or PLY, in the calibration's world units | Blender, MeshLab, MATLAB `stlread`/`readSurfaceMesh` |
| **Everything** | every format of the export dialog with one base name — the DLTdv8 convention once (no bottom-left copy), ALL CAMERAS only with two or more cameras, the 3D files only once there is a reconstruction (the mesh is exported from the 3D menu) | — |

When an animal is defined, CSV/TSV exports also write a `*_segment.csv` sidecar
(`frame, present, score, bbox, centroid, area, mid0_x … mid31_y`: the
32-point body midline for curvature / undulation analysis).

Coordinates are pixels in the native video resolution, x right, y down, with
the centre of the top-left pixel at (0, 0) (OpenCV's convention — your clicks
are stored that way too); only the DLTdv exports add 1. A cell marked hidden
(Shift+X) exports blank, like an untracked one.
`visible = 0` means the model believes the point is occluded on that frame
(its position is still predicted). Silhouette-derived landmarks have data
wherever the animal's silhouette exists; their confidence is the segmentation
model's presence score mapped to 0–1. A point that **leaves the frame** produces
no coordinates at all (blank cells / no rows) until it is re-placed. A region
group exports exactly like a point (its fitted center). `confidence` (in the
.mat) is the model's per-frame track-correctness score — the same signal the
timeline colors red. If you marked events, CSV/TSV exports also write a
`*_events.csv` sidecar (`name,start_frame,end_frame,note,author`, then one
`note` row per frame note); the .mat embeds them directly.

**Continuing a point that left the frame:** when it comes back into view,
select it in the right panel, press **N**, and click its new location (this
continues the *same* point — no new point is created, because the selected
point has no position on that frame), then press Track. Press Esc to
deselect first if you wanted a brand-new point instead.

## Projects, autosave, crash safety

- **File → Save Project** writes a `.cotrk` file (tracks + points + the frame
  you were on). **Open Project** puts you back exactly where you stopped —
  even on another machine, it will ask you to locate the video if the path
  moved.
- The session **autosaves every 30 s** (also during tracking, and on every
  pause/export/exit) to `<video>.cotracker.npz` next to the video, or to your
  project file once you have one. If the app or machine dies mid-run, you
  lose at most 30 seconds of work — reopening the video offers to resume.
- An autosave that cannot be resumed (unreadable, or saved for a video with a
  different number of frames) is never overwritten: it is set aside as
  `<video>.cotracker.<date-time>.bak.npz` and the app says so.
- A Save that fails says so and writes nothing. If the project file cannot be
  written, autosave falls back to `<video>.cotracker.npz` next to the video and
  tells you to use **File → Save Project As…**.

## Accuracy

CoTracker3 (like all learned point trackers) processes video at ~512×384
internally, so its raw localization error grows with resolution: measured
**≈3.7 px mean on 4K** synthetic ground truth. This tool therefore runs a
second stage at native resolution — a gated Lucas-Kanade refinement seeded
from the model's own prediction, chained from your exact seed click and
re-anchored to the model every frame (it can never drift more than a few
pixels from the CoTracker track). Measured result on 4K ground truth:
**≈0.7 px mean / 1.8 px max**. At ≤720p the model is already at native scale
and refinement automatically stays out of the way. ROI zoom attacks the same
root cause from the other side: inside a crop, small objects reach the model
at several times the detail, which is what makes tracking robust to
background changes around them.

Tips for best precision and robustness:
- Click points on visually distinct features when you can — or better, drag a
  circle snugly around the object so it tracks as a region.
- The seed frame keeps your exact clicked coordinates in the export —
  corrections are treated as ground truth.
- Track a few extra "sacrificial" points on the same rigid object: CoTracker
  tracks all points jointly, so they support each other. Ignore the extras at
  export (or just use a region, which does this internally).
- Trust the timeline reds: they mark exactly where the model struggled.

## Segments, silhouettes and skeletons

Point trackers hold textured features (an eye, a snout, a marker) but lose thin
undulating tails and dark textureless bodies — measured on real 4K underwater
iguana footage. The app therefore tracks the **animal as a whole** too:

- **One click defines the animal.** SAM 3 (or SAM 2.1) segments it on that frame
  and, during tracking, on every frame — following it through occlusion with a
  memory of what it looks like. The crop that the point tracker works in follows
  the silhouette, and extra hidden support points sampled inside the silhouette
  anchor the joint tracking to the body.
- **Silhouette landmarks** are geometry, not appearance: the body midline
  (anchored at your head landmark), the tail tip (end of the midline), points at
  25/50/75 % of body length, the centroid, and extremities (the farthest
  protrusions on each side — feet, wing or patagium tips). They are ordinary
  points in the export. Left and right are named **as seen from above** (a
  dorsal view): filmed from below, the animal's left and right swap, so rename
  those landmarks (or swap their rules). Exports made before 2026-09-22 from
  dorsal footage had them the wrong way round.
- **Skeletons** give every landmark a fixed name so exports line up across
  videos and, later, across cameras. Built-in templates cover lizards, generic
  quadrupeds, gliding mammals, flying lizards, undulating bodies and birds/bats;
  *Custom skeleton…* saves yours as JSON in `skeletons/`. Right-click any point
  → **Data source** to switch it between appearance tracking and a silhouette
  rule (it asks before erasing the point's existing track; Ctrl+Z brings it
  back).
- **Points stay on the animal.** With the **Body** toggle on (default), a
  tracked point the model puts just outside the silhouette (within about 3 % of
  the silhouette's diagonal, at least 8 px) is nudged back onto it — that is
  outline jitter. A landmark that goes further has **left** the animal: the run
  **stops at that frame**, its track ends on the frame before, and the app
  takes you there. Pulling it back onto the nearest silhouette pixel would put
  it on some other spot of the body and carry on as if nothing had happened.
  Click it where it really is and press Track. Exempt a point that lives
  elsewhere (a ground marker) with right-click → *May leave the segment (free
  point)*.
- **Corrections are clicks.** Pause, go to the frame, press **S**, click (or
  Shift+click), press **Track**: the correction applies from that frame on. A
  landmark that leaves the silhouette stops the run (above); one that sits off
  the body at low confidence turns red on the timeline; and with Auto-pause on
  the run also stops when a point stays unreliable, or when the animal itself
  is lost or leaves the frame.
- **Segment ▾** picks the segmentation model right on the button — the same way
  **Track ▾** picks the point model — and each entry says whether its weights are
  ready, need downloading (with the size), or are gated behind a token. SAM 3 is
  the default when its weights are in `models\sam3` (best masks, ~13 fps at 4K);
  SAM 2.1 base+ is the fast fallback (~27 fps). Robustness was chosen over speed
  by design. **Settings** (Ctrl+,) holds the same choice plus the Hugging Face
  token and the mask opacity; the last menu entry opens it.

A step-by-step guide for new users is in [docs/MANUAL.md](docs/MANUAL.md)
(also in the app: **F1**).

## Human bodies: joints and joint angles (the **Body** menu)

Everything above follows points *you* choose. The body layer instead finds a
**person** and measures them: joints on every frame, joint angles in degrees,
a side-by-side view, and spreadsheets. Nothing is placed by hand and no
skeleton template is needed.

**Two backends, one interface** (`bodypose.BodyEstimator.step`):

| Backend | Output | Availability |
|---|---|---|
| **SAM 3D Body** ([Meta, 11/2025](https://github.com/facebookresearch/sam-3d-body)) | metric **3D** joints on the 70-joint Momentum Human Rig from a *single* camera, plus the mesh — so the angles are true anatomical angles whichever way the subject faces | checkpoints are **gated** on Hugging Face and the inference code is a GitHub checkout; both are supplied locally, exactly like the gated SAM 3 weights |
| **ViTPose base** (via `transformers`, ungated) | 17 COCO joints **in the image plane** | downloads 425 MB on first use; no new dependency, no gate to accept (Apache-2.0) |

Supplying SAM 3D Body: request access at
`huggingface.co/facebook/sam-3d-body-dinov3`, put `model.ckpt` and
`assets/mhr_model.pt` in `models/sam3d-body-dinov3/`, and clone
`facebookresearch/sam-3d-body` into `models/sam-3d-body/`. Until then the
Body dialog lists it greyed out and *says what is missing and where to get it*
— it never fails at run time.

**Finding the person**: a person detector (RT-DETR v2, 81 MB, ungated) by
default, or the **existing SAM silhouette** from the segment tool — which is
what to use when the detector cannot see your subject, or when several people
are in shot and you want one of them. A frame with no silhouette stays
**blank**: a top-down pose model returns a confident skeleton for any box it is
given, so "no prompt" must never become "pose the background".

**Joint angles** are defined once, in canonical joint names, and resolved
against whichever rig the backend speaks — so the same table serves a 70-joint
mesh model and a 17-joint keypoint model, and an angle whose joints a rig lacks
is dropped rather than exported as a column of NaN. Hips, shoulders, ankles and
the trunk are **signed** about the subject's own left-right axis (positive =
forwards): an unsigned three-point angle cannot tell flexion from extension and
turns a gait cycle into nonsense. Every angle carries the meaning of its zero
all the way into the CSV header, the report and the tooltip. Two examples:
**neck flexion** is 0 with the ears over the shoulders in line with the trunk
(a nod of the head alone barely changes it), and **thigh separation (stride)**
is 0 with the thighs side by side, positive with the left knee ahead. A joint
the model scores under 15 % gives no angle.

**Side-by-side view** (**Ctrl+6**) — the footage with the skeleton on it
(left green, right orange, so a swapped leg is obvious), the pose on its own
(orbitable when the backend gives 3D), and the joint angles against time with a
playhead. The pose panel **stands the subject up** by default
(`bodyview.body_basis`): a monocular model reports joints in CAMERA
coordinates, so a rig camera angled down renders a standing person lying at the
camera's angle — the panel aligns the body's own long axis (pelvis→shoulders,
median over the clip so it cannot rock per stride) to vertical, and its
left-right axis to the screen, giving a canonical front view at azimuth 0.
Verified to 0.000 deg from vertical for camera pitches up to 90 deg. When the
backend returns a mesh it is drawn as a shaded **3D body shape** rather than a
stick figure (`BodyTrack.mesh`, float16, sparse per (frame, person), ~110 kB a
frame; batched depth-sliced painter's algorithm with backface culling — 48 ms
for a 113k-triangle body, against 259 ms for one `fillConvexPoly` per
triangle). `Body → Export Side-by-side Video…` writes exactly that to an MP4
through the same compose function, so the window and the file cannot drift.

**Exports** (**Body → Export Joint Positions… / Export Joint Angles…**): joint
positions (pixels, plus X/Y/Z in metres in the camera's frame — x right, y
down, z away — with an `xyz_frame` column when 3D), joint angles with deg/s,
and a plain-language report — which model, how many frames had a person,
whether the angles are 3D or flat, the range of motion per joint, what each zero
means, and a verdict of *good / ok / poor / nothing found*. The CSVs open with
`#` note lines (read them with `comment='#'`); SAM 3D Body gives no per-joint
score, so its `conf` column is blank.

A 2D result is labelled as image-plane everywhere it appears, because a limb
pointing at the camera foreshortens and reads a smaller angle than it really
has. Film side on for limb angles, or use the 3D backend.

## Limitations

- **Forward-only tracking** (a CoTracker online-mode property): to fix the
  past, scrub back, correct, and re-track forward from there.
- **Regions** can be circles, rectangles or polygons (Add ▾); their members are
  sampled inside the outline. The animal silhouette covers the deformable case.
- **One animal per project.** Several animals with identities are planned (the
  segmentation sessions already support it; the UI does not yet).
- **Variable-frame-rate video** (some phone/screen recordings): frame numbers
  become ambiguous. The app detects and warns; re-encode first:
  `ffmpeg -i in.mp4 -vsync cfr -r 30 -c:v libx264 -crf 18 out.mp4`
- **A file that states no usable frame rate**: the app measures it from the
  frames' timestamps (or, failing that, assumes 30 fps) and says so when the
  video opens. Check that number, because every time, speed and camera sync
  uses it. High-speed rates up to 100 000 fps are read from the file.
- **A damaged frame** stops tracking there ("Tracking stopped: frame N of the
  video could not be decoded") and keeps everything before it; scrubbing onto
  it shows a blank frame and says so (Shift+C retries).
- Tracking speed is usually limited by video *decoding*, not the GPU
  (measured ≈40 fps at 1080p, ≈12–15 fps at 4K with refinement). A full 40k-frame
  4K pass is roughly an hour — pause/resume and autosave make that practical.

## Performance tuning (4K scrubbing)

Because decoding is the bottleneck, a faster GPU does little; these help:

- **Frame cache** — sized automatically to a quarter of your RAM (max 24 GB),
  so a whole working region of 4K frames stays in memory and re-scrubbing a
  stretch you are correcting is instant. Override with the `COTRACKER_CACHE_GB`
  environment variable if you want more or less.
- **Decode backend** — hardware decode roughly doubles *sequential* decode
  (long tracking runs) but is not automatically better for seeking. Measure
  your own footage, then apply the winner:

  ```
  .venv\Scripts\python.exe tools\bench_decode.py "D:\path\to\your4k.mp4"
  ```

  Apply with `COTRACKER_DECODE` = `auto` (default) | `msmf` | `hw` | `ffmpeg`.
  Pick one and keep it for a project: backends agree on frame numbering but
  differ marginally in color conversion.

## Testing

`make_test_video.py` generates synthetic videos with known ground truth and
validates exports against it:

```
python make_test_video.py demo.mp4 --frames 600 --size 1920x1080 --dots 4
python make_test_video.py demo.mp4 --check exported_tracks.csv
```

For a first look, make the demo clip with
`python make_test_video.py test600.mp4 --seed 0` (inside `.venv`), open it,
place the four dots (press **N**, click a dot; once per dot), press Track, and
watch.

The verification suites live in `tests/` (run them with the environment's
Python: `.venv\Scripts\python.exe` on Windows, `.venv/bin/python` on Linux and
macOS — `tests/verify_<name>.py`; each prints `PASSED` and exits non-zero on
failure). `tests/run_suites.py --cpu|--gpu` runs a whole group and generates
the synthetic test videos it needs first. Ball markers have
`verify_balls.py` and the automatic re-track `verify_retrack.py` (both GPU);
`verify_sweep_fixes.py` pins the fixes of the 2026-09-22 release sweep —
camera timing and exports, undo and keyframes, the multi-camera, canvas and
timeline behaviour (offscreen, no GPU, part of `--cpu`). Two audit tools are
not suites: `tests\audit_sweep.py` drives every enabled menu entry, hotkey and
context-menu entry in every state offscreen and must end with 0 exceptions;
`tests/audit_gui_real.py` drives the app with real mouse clicks and key
presses on the real desktop, screenshots every window, dialog, wizard
page and menu into `tests\out\gui_real\`, and checks that nothing raises,
menus show their tooltips, hotkeys work wherever the focus is, the view never
moves on its own, a pause lands within a second and nothing is clipped at
1366×768 (`--no-gpu` skips the segment and tracking states).
The 3D layer has two of its own: `verify_3d.py` (cameras, triangulation,
undistortion, sub-frame sync, visual hull — synthetic ground truth, no GPU)
and `verify_3d_gui.py` (the 3D menu end to end, offscreen). The body layer has
`verify_body.py` (joint angles against a synthetic walker **built from** known
angles, storage, exports, the backend registry, the drawing — no GPU, no
weights) and `verify_body_gui.py` (the real pose model through the app, from
the run dialog to the exported video).

## Appearance

The app uses a quiet, dark, content-first design: the video is the hero,
controls stay out of the way, and one blue accent consistently means
"active / selected / primary" — the filled **Track** button, checked tools,
selections, and the timeline's frame-window highlight are all the same blue.

## Folder map

```
run.bat / run.sh        launchers (Windows / Linux + macOS; create .venv on first run)
install.py              installs PyTorch for this machine, then requirements.txt
requirements.txt        the other dependencies
cotracker_app/          the application (calib.py / hull.py / view3d.py = the 3D layer)
docs/                   the user manual (also in the app, F1) and the developer handbook
tests/                  verification suites and audit tools (tests/out/ is scratch)
tools/                  developer tools (a decode benchmark)
make_test_video.py      synthetic test videos with ground truth
LICENSES/, THIRD_PARTY_LICENSES.md   licences of everything Kinetrace uses
created on your computer (never in the repository):
  .venv/                all Python dependencies
  models/               the point, segmentation and body models
  skeletons/            your own skeleton templates (JSON)
```
