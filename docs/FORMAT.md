# The Kinetrace project (`name.kinetrace/`)

A Kinetrace project is **one folder**, `name.kinetrace`. Everything in it is a
plain file: spreadsheet-style CSV tables and small JSON files for everything a
person reads, edits or another program writes (one CSV per landmark and
camera), and NumPy `.npy` arrays for the bulky model output (silhouette
outlines, body poses). Open any table directly in Excel, MATLAB, R or Python;
there is nothing to unzip. To move a project, move the whole folder: keep the
videos beside it or inside it (in `videos/`) and they are found again on any
computer and operating system.

A save writes **only the files whose content changed**, and a save that is cut
short (a crash, a power cut) is undone the next time the project opens. The
project opens fast because Kinetrace keeps binary copies of the tables in
`.cache/`; a CSV edited by hand is read from its text, so the CSV is always
what counts.

**One file** for e-mail or archiving: File → Export Project as One File (or
`python -m kinetrace.convert pack FOLDER name.kinetrace`) writes the same
files as an ordinary **zip archive** named `name.kinetrace`; Kinetrace opens
it like the folder, and `convert unpack` turns it back into a folder.

Everything here describes **format version 2** (2026-09-29). Format 1 (a zip
with one long `tracks.csv` per camera) still opens; see the end.

## Conventions (the same in every file)

| | |
|---|---|
| Frames | counted from **0** |
| Pixels | OpenCV **pixel centres counted from 0**: the centre of the top-left pixel is (0, 0), x to the right, y **down**. The last pixel of a 1920-wide picture has x = 1919. (DLTdv8 and MATLAB count from 1: subtract 1.) |
| Missing data | an **empty cell** (not 0, not `NaN`) |
| True / false | `1` / `0` (`TRUE` / `FALSE` are accepted when read) |
| Numbers | `.` decimal point, no thousands separator. 32-bit values (positions, confidence) are written as the shortest text that reads back to the same number exactly |
| Text | UTF-8; CSV quoting as usual (`"a, b"` for a comma, `""` for a quote); line ends `\n` (`\r\n` and Excel's BOM are accepted) |
| Colours | `#rrggbb` |

Columns are found **by name**, in any order; unknown columns and unknown files
are ignored. A table saved by a spreadsheet set to a decimal-comma locale
(separated by `;`) is refused with a message saying so — save it with `,`
separators and `.` decimals.

## What is inside

```
kinetrace.json                       format, version, project id, when it was saved, the cameras' folders
README.txt                           what every file is (written by Kinetrace at each save)
project.json                         the cameras: video, frames, frame rate, size, offset, rate; the active one
state.json                           the window: every toggle, panels, splitter, step (restored exactly)
calibration.json                     the camera calibration, if there is one
lenses.json                          lens profiles per camera, if any
reconstruction/meta.json             the last 3D result, if any: its landmarks, unit, first frame
reconstruction/points/<landmark>.csv   its positions: frame, x, y, z, residual, n_cams, <camera>_px
cameras/<folder>/view.json           this camera: frame on screen, selected point, zoom, timeline zoom
cameras/<folder>/points.csv          the points (landmarks) and their settings
cameras/<folder>/tracks/<landmark>.csv   one landmark's positions: one row per frame that has data
cameras/<folder>/events.csv          marked events
cameras/<folder>/notes.csv           notes on frames
cameras/<folder>/ball_prompts.json   ball markers' SAM clicks (only with ball markers)
cameras/<folder>/spots.json          per landmark, the Moving spot point model's settings that
                                     Test the point models on my clicks chose: {"P1": {"cue":
                                     "bright", "radius": 6.0, "speed_gain": 0.0, "sigma": 1.5}}
                                     (only when a test chose them; a landmark not listed is automatic)
cameras/<folder>/skeleton.json       the named skeleton (only with one)
cameras/<folder>/segment.json        the segment's name, colour and SAM clicks / boxes (only with a segment)
cameras/<folder>/silhouette/summary.csv   the segment per frame: area, score, centroid, box
cameras/<folder>/silhouette/*.npy    the segment's outline and midline per frame (binary)
cameras/<folder>/body/*.npy          body poses and mesh (binary; only after a Body run)
exports/                             files for other programs, refreshed at each save (only when chosen)
videos/                              optional: the videos, to keep everything in one folder
.cache/                              binary copies of the tables, for fast opening (safe to delete)
.history/                            the files as they were before the last save
```

Only the files listed above are the project's: a save never touches anything
else in the folder (`videos/`, `exports/` apart from its own files, a note of
your own). The smallest valid project is `kinetrace.json` + `project.json` +
one `cameras/<folder>/tracks/<landmark>.csv` (a landmark file that
`points.csv` does not name is a landmark with default settings, named after
the file).

### `kinetrace.json`

```json
{"format": "kinetrace-project", "format_version": 2, "app_version": "0.3.0",
 "project_id": "5f0c…", "saved_at": "2026-09-29T21:04:11.482113+00:00",
 "cameras": [{"folder": "cam1", "name": "cam1"}], "videos_relative_to": "project"}
```

`project_id` identifies the project wherever the folder is moved (unsaved work
is matched to it); `saved_at` identifies the save: the moment kinetrace.json
is replaced is the moment a save counts. `videos_relative_to` = where the
videos' `relative_path`s start (`project` = the project folder itself;
`container` = the folder holding a single file). A project whose
`format_version` is newer than the program is refused with a message.

### `project.json`

```json
{"cameras": [{"name": "cam1", "folder": "cam1",
              "video": {"path": "D:\\shoot\\cam1.mp4", "relative_path": "../cam1.mp4"},
              "n_frames": 40000, "fps": 239.76, "file_fps": 239.76, "width": 3840, "height": 2160,
              "offset": 0.0, "rate": 1.0}],
 "active_camera": "cam1", "exports_on_save": ["dltdv_all", "dlc"]}
```

* `relative_path` is relative to the project folder, with `/` separators
  (`../cam1.mp4` = beside the project folder, `videos/cam1.mp4` = inside it);
  it is tried first, so a project moved together with its videos opens on any
  computer and operating system. `path` is where the video was when the
  project was saved. Then the same file name is looked for in the project
  folder, its `videos/` folder and beside it, then Kinetrace asks.
* `fps` = the rate the camera really recorded at; `file_fps` = the rate its
  file says (they differ for slow-motion files; Frame rate… in the camera panel).
* `exports_on_save` = the formats written into `exports/` at every save
  (File → Keep Exports Up to Date…): `dltdv_all`, `dltdv`, `dlc`, `mat`, `wide`, `xyz_dltdv`.
* `offset` = the frame this camera shows when the first (reference) camera is
  at its frame 0 (the first camera's is always 0). `rate` = this camera's
  frame rate / the reference camera's. Camera `i`'s frame at reference instant
  `t` is `rate_i · t + offset_i`.

### `cameras/<folder>/points.csv`

`name, color, shown, kind, radius, anchor, source, spec, free, shape, outline, file`

| column | meaning |
|---|---|
| `name` | unique within the camera; the same name in two cameras is the same landmark (that is how 3D matches them) |
| `file` | its positions: `tracks/<file>` (the name itself, with `< > : " / \ \| ? *` as `_`; blank = from the name) |
| `kind` | `point` or `group` (a region tracked as a whole) |
| `radius`, `shape`, `outline` | a region's size and outline (`circle` / `rect` / `polygon`; `outline` = space-separated `x y x y …` in pixels) |
| `anchor` | 1 = the appearance lock is on |
| `source` | `track` (followed by the point tracker), `silhouette` (computed from the segment; `spec` says how) or `ball` |
| `free` | 1 = may leave the animal (not held on its silhouette) |

### `cameras/<folder>/tracks/<landmark>.csv`

`frame, x, y, confidence, visible, hand_placed, hidden` (+ `radius` for a ball marker)

One file per landmark and camera, one row per frame **that has something to
say**, frames ascending; a 40,000-frame video tracked on 5 frames has 5 rows.
`x` / `y` empty = no position on that frame (such a row still carries, for
example, `hidden = 1`). Only `frame, x, y` are required: a missing
`confidence` is 1 where there is a position, a missing `visible` = there is a
position, the rest 0.

| column | meaning |
|---|---|
| `confidence` | 0 – 1, the tracker's belief the position is right (red on the timeline below 0.5); 1 for hand placements |
| `visible` | 0 = the tracker thinks the point is covered on that frame (its position is still predicted) |
| `hand_placed` | 1 = placed by hand on that frame |
| `hidden` | 1 = marked hidden (Shift+X): exported as blank, left out of 3D |
| `radius` | a ball marker's fitted radius, in pixels |

### `events.csv` and `notes.csv`

`events.csv`: `name, start, end, color, note, author` (frames, inclusive).
`notes.csv`: `frame, text, author, time`.

### `state.json` and `view.json`

How the program looked. `state.json` holds `tools` (every toggle: follow,
auto-pause, ROI, marker size, trails, display filter, point model, …) and
`layout` (window rectangle, side panel shown / floating, splitter sizes, solo
mode, step size, the getting-started strip). `view.json` (per camera):
`current_frame`, `selected_point` (by name), `zoom`, `center_x`,
`center_y`, `user_zoomed`, `timeline` (`[first, last]` frame shown). Odd or
missing values fall back to the defaults: this is never data.

### `calibration.json`

```json
{"unit": "m", "source": "wand calibration",
 "cams": [{"width": 2704, "height": 1520, "pixel_origin": 0.0, "y_flip": false,
           "rmse": 0.41, "undistort": {"kind": "opencv", "K": [[…]], "dist": […], "fisheye": false}}],
 "coefs": [[L1, …, L11], …], "origin_shift": [x, y, z], "notes": []}
```

Per camera the 11 DLT coefficients (`coefs`, in camera order) and the
convention they were fitted in: `pixel_origin` 1 = MATLAB pixels (DLTdv,
easyWand), 0 = OpenCV; `y_flip` = y counted up from the bottom edge;
`undistort` = the lens correction applied to raw pixels before the DLT
(`none`, `opencv` with K / dist / optional P, or `lwm` with control points).
To hand the cameras to another program, use **3D → Export Calibration**
(Anipose, OpenCV, MATLAB, Blender, DLTdv — converted and checked), not this
file.

### `lenses.json`

A list, one entry per camera (`null` without a profile): `width`, `height`,
`K`, `dist`, `fisheye`, `rms`, `n_views`, `source`, `report`.

### `cameras/<folder>/silhouette/summary.csv`

`frame, area, score, centroid_x, centroid_y, x0, y0, x1, y1` — the segment per
frame (area in pixels², presence score, centroid, bounding box inclusive), one
row per frame that has one.

### `reconstruction/`

`meta.json`: `t0` (first reference frame), `n_frames`, `unit`, `points` (each
landmark's `name` and `file`), `per_camera_columns`. `points/<landmark>.csv`:
`frame, x, y, z, residual, n_cams, <camera>_px …` — `frame` in the REFERENCE
camera's frames (the first camera's), `residual` = DLTdv's rmse in pixels,
`n_cams` = cameras used, `<camera>_px` = that camera's own reprojection error.

### Binary parts (`.npy`)

`silhouette/`: the outlines as one point list `cpts` (P × 2) cut by `coff`
(offsets) with `cframe` (the frame of each outline), and the 32-point midline
`mpts` for the frames in `mframe`. `body/`: `joints3d`, `joints2d`, `conf`,
`score`, `bbox`, `focal`, `cam_t` (frames × people × joints …), `meta.json`
(rig, people's names, backend), and the mesh.

Read them with `numpy.load` in Python, [`readNPY`](https://github.com/kwikteam/npy-matlab)
in MATLAB, or `RcppCNPy::npyLoad` in R. Readable copies are one export away:
the silhouette as polygons (JSON) or PNG masks, the body joints as CSV.

## Reading a project without Kinetrace

Every table is an ordinary CSV:

```python
import csv

with open("lizard.kinetrace/cameras/cam1/tracks/snout.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["x"]:                                   # empty = no position on that frame
            print(int(r["frame"]), float(r["x"]), float(r["y"]))
```

pandas: `pd.read_csv("lizard.kinetrace/cameras/cam1/tracks/snout.csv")`.
MATLAB: `readtable('lizard.kinetrace/cameras/cam1/tracks/snout.csv')`. R:
`read.csv("lizard.kinetrace/cameras/cam1/tracks/snout.csv")`. The landmarks
of a camera are listed in its `points.csv`. For other programs' own formats
(DeepLabCut, DLTdv, MATLAB) use File → Export Tracks, or let every save keep
them in `exports/` (File → Keep Exports Up to Date…).

## Writing a project for Kinetrace

Make a folder with the three files below, then open its `kinetrace.json` in
Kinetrace (File → Open Project…):

```
myproject.kinetrace/kinetrace.json    {"format": "kinetrace-project", "format_version": 2,
                                       "cameras": [{"folder": "cam1", "name": "cam1"}]}
myproject.kinetrace/project.json      {"cameras": [{"name": "cam1", "folder": "cam1",
                                        "video": {"relative_path": "../clip.mp4"},
                                        "n_frames": 600, "fps": 30, "width": 640, "height": 480}]}
myproject.kinetrace/cameras/cam1/tracks/snout.csv   frame,x,y
                                                    0,120.5,88.25
                                                    1,121.0,88.0
```

`python -m kinetrace.convert check myproject.kinetrace` says what it found, and why not
(exit 0 clean, 1 warnings such as a video that is not found, 2 errors).

## Saving, the previous save, and unsaved work

The project folder changes only when you save, and a save writes only the files
whose content changed (a fingerprint of each file is kept in
`.cache/index.json`). The new files are written aside first (`.saving/`), the
files they replace are moved into `.history/`, and replacing `kinetrace.json`
is the moment the save counts: a save cut short before that is undone the next
time the project opens, and one cut short after it is finished. `.history/`
therefore holds **the previous save** (the files the last save replaced or
removed, and its `kinetrace.json`); `python -m kinetrace.convert previous
FOLDER` puts the folder back as it was then. `.cache/` holds a binary copy of
every table; a copy is used only while its CSV is exactly the file that save
wrote (same size and time) AND the copy's own fingerprint is the one that
save recorded, so a hand-edited CSV is always read from its text and a copy
damaged on disk is never used. Deleting `.cache/` only makes the next open and
save slower.

One save at a time: a save holds `.lock` (process, computer, time) in the
project folder; a second Kinetrace saving the same project is refused with a
sentence, and a lock left by a process that is gone is cleared. A save checks
the free space on the drive before it writes the big part, and on Linux and
macOS makes its renames durable (directory fsync) before and after the commit.

Close the project in Kinetrace before editing its files by hand: a save from
Kinetrace writes what Kinetrace holds. A save that finds a file open in
another program (a CSV in Excel on Windows) stops, undoes itself and says so.

Unsaved work is kept every 30 seconds in a **recovery copy** — the same layout
in one file, with binary tables — in the `recovery` folder inside the
Kinetrace folder (or, when that cannot be written,
`%LOCALAPPDATA%\Kinetrace\recovery`, `~/Library/Application Support/Kinetrace/recovery`,
`~/.local/share/kinetrace/recovery`; `KINETRACE_RECOVERY_DIR` overrides).
It is found by `project_id`, so moving or renaming the project folder does not
lose it. Two copies of Kinetrace working on the same project at once is not
supported.

## Format 1 (projects saved before 2026-09-29)

A format-1 project is one zip file `name.kinetrace` with a long
`cameras/<folder>/tracks.csv` per camera (`frame, point, x, y, confidence,
visible, hand_placed, hidden, radius`: one row per frame and point that has
data), the silhouette's per-frame numbers as `.npy`, the 3D result as `.npy`,
and video paths relative to the folder holding the zip. Kinetrace opens it;
its first Save offers to turn it into a folder of the same name (the single
file is kept as `name.kinetrace.bak`) or to keep saving it as one file.

## Other programs' conventions

Kinetrace converts these for you (File → Import, 3D → Export Calibration,
Ctrl+E, and `python -m kinetrace.convert`). For reference:

| Program | Pixels | What Kinetrace reads / writes |
|---|---|---|
| **DeepLabCut** | top-left, pixel centres on whole numbers (as Kinetrace) | CSV with `scorer` / `bodyparts` / `coords` rows, `x, y, likelihood`, first column = frame (videos analysed by DLC; single animal) |
| **SLEAP** | (0, 0) is the centre of the top-left pixel (as Kinetrace; sleap-io documentation) | SLEAP 1.x analysis CSV (`track, frame_idx, instance.score, node.x, node.y, node.score`) and sleap-io's `sleap` / `instances` / `points` / `frames` CSV layouts; one track |
| **DLTdv8 / easyWand** | first pixel = **1**, top-left | xypts `pt1_cam1_X …` (NaN = none) with a `_pointnames.csv` sidecar; xyzpts `pt1_X …`; dltCoefs.csv (11 rows, a column per camera) |
| **older DLTdv, Argus** | first pixel = 1, y up from the **bottom** edge | xypts (the sidecar's `convention` line says `bottom-left`); Argus lens profile lines `cam f w h cx cy AR k1 k2 t1 t2 k3` (OpenCV pixels) |
| **Anipose / aniposelib** | OpenCV | `calibration.toml`: `[cam_0]` `name`, `size = [w, h]`, `matrix` (K), `distortions` (k1 k2 p1 p2 k3; 4 + `fisheye = true`), `rotation` (Rodrigues vector), `translation`; x_cam = R X + t. 3D output CSV `name_x, name_y, name_z, name_error, name_ncams, name_score, …, fnum` |
| **OpenCV** | 0-based | FileStorage `.yml` / `.json`: per camera `camera_matrix`, `distortion_coefficients`, `image_width`, `image_height`, `rvec`, `tvec`, `R` |
| **MATLAB Computer Vision Toolbox** | first pixel = **1** | `K` (R2022b+) or `IntrinsicMatrix` (= K'), `ImageSize` = [rows cols], `RadialDistortion` [k1 k2 k3], `TangentialDistortion` [p1 p2], `RotationMatrix` (= R', row-vector form), `TranslationVector` (1 × 3); Kinetrace's .mat also carries `K_opencv`, `R`, `t` |
| **Blender** | — | a script: lens = fx · sensor width / image width (horizontal fit), shift from the picture's centre pixel (w − 1) / 2, camera matrix [R' \| −R't] · diag(1, −1, −1) (Blender cameras look down −Z) |

A calibration made by easyWand or DLTdv usually describes a **mirrored**
(left-handed) world, which the K + R/t formats cannot hold: Kinetrace then
exports that world mirrored in Z and says so, and 3D points exported beside
it are mirrored the same way. DLTdv's own lens correction (a local weighted
mean) is not an OpenCV model: an OpenCV model is fitted to it and its worst
error is reported.

Sources: aniposelib `cameras.py` and anipose `triangulate.py`
(github.com/lambdaloop), SLEAP 1.4 `sleap/io/format/csv.py` and the sleap-io
documentation (github.com/talmolab), MathWorks' `cameraParameters`,
`cameraIntrinsics` and `cameraIntrinsicsToOpenCV` pages, the OpenCV
calibration sample, BlenderProc's camera utilities, DLTdv7 / DLTdv8
(github.com/tlhedrick).
