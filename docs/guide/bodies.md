<sub>Kinetrace guide — [Home](../../README.md) · [Install & run](install.md) · [Workflow: track an animal](workflow.md) · [Segments, silhouettes & skeletons](segments.md) · [Several cameras & 3D](cameras-3d.md) · **Human bodies** · [Export formats](exports.md) · [Projects & autosave](projects.md) · [Keyboard & mouse](shortcuts.md) · [Accuracy](accuracy.md) · [Performance (4K)](performance.md) · [Limitations](limitations.md) · [Testing](testing.md) · [Folder map & design](folder-map.md) · [User manual](../MANUAL.md) · [Licences](../../THIRD_PARTY_LICENSES.md)</sub>

# Human bodies: joints and joint angles (the **Body** menu)

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

---

[← Several cameras & 3D](cameras-3d.md) &nbsp;|&nbsp; [Contents](../../README.md#contents) &nbsp;|&nbsp; [Export formats →](exports.md)
