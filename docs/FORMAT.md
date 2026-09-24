# The Kinetrace project file (`.kinetrace`)

A Kinetrace project is **one file**, `name.kinetrace`. It is an ordinary
**zip archive** of plain files: spreadsheet-style CSV tables and small JSON
files for everything a person edits or another program writes, and NumPy
`.npy` arrays for the bulky model output (silhouette outlines, body poses).
Unzip it with any tool and you can read, edit or generate every part of it
without Kinetrace. An unzipped folder with the same layout also opens in
Kinetrace (File → Open Project… accepts the folder's `kinetrace.json`), and
`python -m kinetrace.convert pack FOLDER name.kinetrace` zips it back after
checking it.

Everything here describes **format version 1**.

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
project.json                         the cameras: video, frames, frame rate, size, offset, rate; the active one
state.json                           the window: every toggle, panels, splitter, step (restored exactly)
calibration.json                     the camera calibration, if there is one
lenses.json                          lens profiles per camera, if any
reconstruction/                      the last 3D result, if any (.npy + meta.json)
cameras/<folder>/view.json           this camera: frame on screen, selected point, zoom, timeline zoom
cameras/<folder>/points.csv          the points (landmarks) and their settings
cameras/<folder>/tracks.csv          the positions: one row per frame and point that has data
cameras/<folder>/events.csv          marked events
cameras/<folder>/notes.csv           notes on frames
cameras/<folder>/ball_prompts.json   ball markers' SAM clicks (only with ball markers)
cameras/<folder>/skeleton.json       the named skeleton (only with one)
cameras/<folder>/segment.json        the segment's name, colour and SAM clicks / boxes (only with a segment)
cameras/<folder>/silhouette/*.npy    the segment's outline and midline per frame (binary)
cameras/<folder>/body/*.npy          body poses and mesh (binary; only after a Body run)
```

The smallest valid project is `kinetrace.json` + `project.json` +
`cameras/<folder>/tracks.csv` (points that only `tracks.csv` names are
created with default settings).

### `kinetrace.json`

```json
{"format": "kinetrace-project", "format_version": 1, "app_version": "1.2.0",
 "project_id": "5f0c…", "saved_at": "2026-09-23T21:04:11.482113+00:00",
 "cameras": [{"folder": "cam1", "name": "cam1"}]}
```

`project_id` identifies the project wherever the file is moved (unsaved work
is matched to it); `saved_at` identifies the save. A file whose
`format_version` is newer than the program is refused with a message.

### `project.json`

```json
{"cameras": [{"name": "cam1", "folder": "cam1",
              "video": {"path": "D:\\shoot\\C0004.MP4", "relative_path": "C0004.MP4"},
              "n_frames": 40000, "fps": 239.76, "width": 3840, "height": 2160,
              "offset": 0.0, "rate": 1.0}],
 "active_camera": "cam1"}
```

* `relative_path` is relative to the project file's folder, with `/`
  separators; it is tried first, so a project moved together with its videos
  opens on any computer and operating system. `path` is where the video was
  when the project was saved. Then the same file name is looked for beside
  the project, then Kinetrace asks.
* `offset` = the frame this camera shows when the first (reference) camera is
  at its frame 0 (the first camera's is always 0). `rate` = this camera's
  frame rate / the reference camera's. Camera `i`'s frame at reference instant
  `t` is `rate_i · t + offset_i`.

### `cameras/<folder>/points.csv`

`name, color, shown, kind, radius, anchor, source, spec, free, shape, outline`

| column | meaning |
|---|---|
| `name` | unique within the camera; the same name in two cameras is the same landmark (that is how 3D matches them) |
| `kind` | `point` or `group` (a region tracked as a whole) |
| `radius`, `shape`, `outline` | a region's size and outline (`circle` / `rect` / `polygon`; `outline` = space-separated `x y x y …` in pixels) |
| `anchor` | 1 = the appearance lock is on |
| `source` | `track` (followed by the point tracker), `silhouette` (computed from the segment; `spec` says how) or `ball` |
| `free` | 1 = may leave the animal (not held on its silhouette) |

### `cameras/<folder>/tracks.csv`

`frame, point, x, y, confidence, visible, hand_placed, hidden, radius`

One row per frame and point **that has something to say**; a 40,000-frame
video tracked on 5 frames has 5 rows per point. `x` / `y` empty = no position
on that frame (such a row still carries, for example, `hidden = 1`).

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

### Binary parts (`.npy`)

`silhouette/`: `bbox` (T × 4, −1 where none), `area` (T), `centroid`
(T × 2), `score` (T), the outlines as one point list `cpts` (P × 2) cut by
`coff` (offsets) with `cframe` (the frame of each outline), and the 32-point
midline `mpts` for the frames in `mframe`. `body/`: `joints3d`, `joints2d`,
`conf`, `score`, `bbox`, `focal`, `cam_t` (frames × people × joints …),
`meta.json` (rig, people's names, backend), and the mesh. `reconstruction/`:
`xyz` (T × N × 3), `residual`, `n_cams`, `per_cam`, and `meta.json` (`t0` =
first reference frame, `names`, `unit`).

Read them with `numpy.load` in Python, [`readNPY`](https://github.com/kwikteam/npy-matlab)
in MATLAB, or `RcppCNPy::npyLoad` in R. Readable copies are one export away:
the silhouette as polygons (JSON) or PNG masks, the body joints as CSV, the 3D
points as CSV.

## Reading a project without Kinetrace

Python (standard library only):

```python
import csv, io, zipfile

with zipfile.ZipFile("lizard.kinetrace") as z:
    rows = csv.DictReader(io.TextIOWrapper(z.open("cameras/cam1/tracks.csv"), encoding="utf-8"))
    for r in rows:
        if r["x"]:                                   # empty = no position on that frame
            print(int(r["frame"]), r["point"], float(r["x"]), float(r["y"]))
```

pandas: `pd.read_csv(zipfile.ZipFile(p).open("cameras/cam1/tracks.csv"))`.
MATLAB: unzip it (`unzip('lizard.kinetrace', 'lizard')`), then
`readtable('lizard/cameras/cam1/tracks.csv')`. R:
`read.csv(unz("lizard.kinetrace", "cameras/cam1/tracks.csv"))`.

## Writing a project for Kinetrace

Make a folder with the three files below, then either open its
`kinetrace.json` in Kinetrace or run
`python -m kinetrace.convert pack myproject myproject.kinetrace`:

```
myproject/kinetrace.json          {"format": "kinetrace-project", "format_version": 1,
                                   "cameras": [{"folder": "cam1", "name": "cam1"}]}
myproject/project.json            {"cameras": [{"name": "cam1", "folder": "cam1",
                                    "video": {"relative_path": "../clip.mp4"},
                                    "n_frames": 600, "fps": 30, "width": 640, "height": 480}]}
myproject/cameras/cam1/tracks.csv frame,point,x,y
                                  0,snout,120.5,88.25
                                  1,snout,121.0,88.0
```

`python -m kinetrace.convert check myproject` says what it found, and why not
(exit 0 clean, 1 warnings such as a video that is not found, 2 errors).

## Unsaved work and backups (not part of the file)

The project file changes only when you save. Each save keeps the one before it
beside it as `name.kinetrace.bak`. Unsaved work is kept every 30 seconds in a
**recovery copy** — the same layout with binary tables — in the `recovery`
folder inside the Kinetrace folder (or, when that cannot be written,
`%LOCALAPPDATA%\Kinetrace\recovery`, `~/Library/Application Support/Kinetrace/recovery`,
`~/.local/share/kinetrace/recovery`; `KINETRACE_RECOVERY_DIR` overrides).
It is found by `project_id`, so moving or renaming the project does not lose
it. Two copies of Kinetrace working on the same project file at once is not
supported.

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
