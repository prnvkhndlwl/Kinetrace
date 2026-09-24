<sub>Kinetrace guide — [Home](../../README.md) · [Install & run](install.md) · [Workflow: track an animal](workflow.md) · [Segments, silhouettes & skeletons](segments.md) · [Several cameras & 3D](cameras-3d.md) · [Human bodies](bodies.md) · **Export formats** · [Projects & autosave](projects.md) · [Keyboard & mouse](shortcuts.md) · [Accuracy](accuracy.md) · [Performance (4K)](performance.md) · [Limitations](limitations.md) · [Testing](testing.md) · [Folder map & design](folder-map.md) · [User manual](../MANUAL.md) · [Licences](../../THIRD_PARTY_LICENSES.md)</sub>

# Export formats

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

---

[← Human bodies](bodies.md) &nbsp;|&nbsp; [Contents](../../README.md#contents) &nbsp;|&nbsp; [Projects & autosave →](projects.md)
