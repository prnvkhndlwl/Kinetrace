"""Human body layer, user interface: run a pose model over a range, then look
at the result side by side.

The side-by-side window is the point of the layer. On the left is the footage
with the skeleton drawn on it, so you can see whether the model actually found
the person. On the right is the same pose free of the picture -- orbitable in
3D when the backend produced 3D, or a clean front-on stick figure when it did
not -- so you can see the pose itself. Underneath, the joint angles are
plotted against time with a playhead, and the current frame's values are
listed with the convention each one uses. All three are driven by one frame
number, so what you read is always what you are looking at.

Drawing is plain OpenCV into numpy, as everywhere else in this app: no OpenGL,
nothing to install, and the exact same functions produce the exported video.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QRadioButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from cotracker_app import theme
from cotracker_app.body import MIN_JOINT_CONF, BodyTrack, canon, is_finger
from cotracker_app.hull import _view_rotation

# Left / right / centre, BGR. Colour-coding the sides is the single most
# useful thing a skeleton drawing can do -- it is how you spot the classic
# monocular failure where the model swaps a person's legs.
COL_LEFT = (90, 200, 90)
COL_RIGHT = (80, 140, 250)
COL_MID = (210, 200, 190)
COL_DIM = (110, 110, 120)
BG_PANEL = (24, 26, 30)
PLOT_BG = (20, 22, 26)
TRACE_COLS = [(250, 180, 70), (90, 200, 90), (80, 140, 250), (200, 120, 240),
              (80, 220, 220), (160, 170, 255), (120, 230, 160), (230, 150, 150)]


def _up_basis(up) -> np.ndarray:
    """Rotation taking the rig's up vector onto +y, so `hull._view_rotation`
    (which orbits about +y) can be reused. Without this the 3D panel draws a
    monocular model's output upside down: its camera frame has y pointing
    DOWN, which is the opposite of the renderer's assumption."""
    u = np.asarray(up, np.float64)
    u = u / max(np.linalg.norm(u), 1e-12)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(u, y)
    s = float(np.linalg.norm(v))
    c = float(u @ y)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _median_dir(stack: np.ndarray, rig, a: str, b: str) -> np.ndarray | None:
    """Median unit vector from joint `a` to joint `b` over every frame that
    has both. Median, not mean, so one frame with a swapped limb cannot tip
    the whole clip over."""
    from cotracker_app.body import _resolve
    pa, pb = _resolve(a, stack, rig), _resolve(b, stack, rig)
    if pa is None or pb is None:
        return None
    v = pb - pa
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ok = (n[..., 0] > 1e-9) & np.isfinite(v).all(axis=-1)
    if not ok.any():
        return None
    u = v[ok] / n[ok]
    m = np.median(u, axis=0)
    nm = np.linalg.norm(m)
    return m / nm if nm > 1e-9 else None


def body_basis(stack: np.ndarray, rig) -> np.ndarray | None:
    """Rotation taking the SUBJECT's own axes onto the view's: their long axis
    (pelvis to shoulders) to +y, their left-right axis to +x.

    This is what makes the pose panel stand the person up. A monocular model
    reports joints in CAMERA coordinates, so a camera that looks down at 45
    degrees -- an action camera on a rig, say -- renders a standing person
    lying at 45 degrees, and every reader has to tilt their head. Aligning to
    the body rather than to a guessed gravity vector also means it is right
    for a subject on a slope, and it is what "the body's long axis should be
    vertical" literally asks for.

    Taken over the whole clip (a median), not per frame: a per-frame basis
    would rock the view with every stride. Returns None when the joints
    needed are missing, and the caller falls back to the rig's declared up.
    """
    up = _median_dir(stack, rig, "mid(left_hip,right_hip)",
                     "mid(left_shoulder,right_shoulder)")
    if up is None:
        up = _median_dir(stack, rig, "mid(left_hip,right_hip)", "neck")
    if up is None:
        return None
    lat = _median_dir(stack, rig, "right_hip", "left_hip")
    if lat is None:
        lat = _median_dir(stack, rig, "right_shoulder", "left_shoulder")
    y = up
    if lat is None:
        # no left-right axis: any vector square to the body will do for the
        # spin about it, which the azimuth controls anyway
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(seed @ y)) > 0.9:
            seed = np.array([0.0, 0.0, 1.0])
        lat = seed
    x = lat - float(lat @ y) * y
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        return None
    x = x / nx
    z = np.cross(x, y)
    return np.stack([x, y, z])          # rows are the new axes


def body_basis_2d(stack: np.ndarray, rig) -> np.ndarray | None:
    """The same idea for image-plane joints: a 2x2 rotation putting the
    subject's long axis up the panel. Image rows grow downwards, so "up the
    body" has to land on -y."""
    up = _median_dir(stack, rig, "mid(left_hip,right_hip)",
                     "mid(left_shoulder,right_shoulder)")
    if up is None:
        up = _median_dir(stack, rig, "mid(left_hip,right_hip)", "nose")
    if up is None:
        return None
    phi = -np.pi / 2 - np.arctan2(float(up[1]), float(up[0]))
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def side_of(name: str) -> str:
    n = canon(name)
    return "left" if n.startswith("left_") else "right" if n.startswith("right_") else "mid"


def joint_colour(name: str) -> tuple[int, int, int]:
    return {"left": COL_LEFT, "right": COL_RIGHT}.get(side_of(name), COL_MID)


@dataclass
class PoseDrawOptions:
    """What the skeleton drawing shows. Mirrors `render.OverlayOptions` in
    spirit so the two overlays feel like one feature."""
    bones: bool = True
    joints: bool = True
    names: bool = False
    box: bool = False
    angle_labels: bool = True
    confidence: bool = True      # fade a joint the model was unsure of
    # below this a joint is not drawn at all -- the SAME threshold below which
    # no angle is computed from it (body.MIN_JOINT_CONF), so the picture and
    # the numbers agree about which joints were seen (I86)
    min_conf: float = MIN_JOINT_CONF
    thickness: float = 1.0
    person: int | None = None    # None = every person
    angles_shown: tuple[str, ...] = ()


def _pt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def draw_pose(bgr: np.ndarray, track: BodyTrack, frame: int,
              opts: PoseDrawOptions | None = None, scale: float = 1.0) -> np.ndarray:
    """Draw the 2D skeleton onto a copy of `bgr`. Pure OpenCV, no Qt -- the
    live view, the exported video and the tests all call this one function."""
    opts = opts or PoseDrawOptions()
    # Always a copy: the caller's frame may be the one in the frame cache, and
    # drawing into that would put a skeleton into the footage itself.
    if scale != 1.0:
        img = cv2.resize(bgr, (max(2, int(round(bgr.shape[1] * scale))),
                               max(2, int(round(bgr.shape[0] * scale)))),
                         interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    else:
        img = bgr.copy()
    if track is None or not (0 <= frame < track.n_frames):
        return img
    th = max(1, int(round(2 * opts.thickness * max(0.5, scale))))
    rad = max(2, int(round(3.5 * opts.thickness * max(0.5, scale))))
    people = ([opts.person] if opts.person is not None else range(track.n_people))
    bones = track.bones()
    for p in people:
        if not (0 <= p < track.n_people) or not track.has(frame, p):
            continue
        uv = track.joints2d[frame, p] * scale
        cf = track.conf[frame, p]
        # NaN confidence = the backend gives no score: drawn, never hidden
        ok = np.isfinite(uv).all(axis=1) & ~(cf < opts.min_conf)
        if opts.box:
            b = track.bbox[frame, p] * scale
            if np.isfinite(b).all() and b[0] >= 0:
                cv2.rectangle(img, _pt(b[:2]), _pt(b[2:]), COL_DIM, max(1, th - 1), cv2.LINE_AA)
        if opts.bones:
            for i, j in bones:
                if i < len(ok) and j < len(ok) and ok[i] and ok[j]:
                    c = joint_colour(track.rig.joints[i] if side_of(track.rig.joints[i]) != "mid"
                                     else track.rig.joints[j])
                    cv2.line(img, _pt(uv[i]), _pt(uv[j]), c, th, cv2.LINE_AA)
        if opts.joints:
            for k in np.nonzero(ok)[0]:
                c = joint_colour(track.rig.joints[k])
                if opts.confidence and cf[k] < 0.5:
                    c = tuple(int(v * 0.55) for v in c)      # unsure: drawn faint
                cv2.circle(img, _pt(uv[k]), rad, c, -1, cv2.LINE_AA)
        if opts.names:
            for k in np.nonzero(ok)[0]:
                cv2.putText(img, track.rig.joints[k], (_pt(uv[k])[0] + 5, _pt(uv[k])[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32 * max(1.0, scale), COL_MID, 1,
                            cv2.LINE_AA)
        if opts.angle_labels and opts.angles_shown:
            _label_angles(img, track, frame, p, opts, scale)
    return img


# Where an angle's number is written: the joint it is measured at.
_ANGLE_AT = {
    "elbow": "{side}_elbow", "knee": "{side}_knee", "hip": "{side}_hip",
    "shoulder": "{side}_shoulder", "ankle": "{side}_ankle",
}


def _angle_anchor(track: BodyTrack, name: str) -> int | None:
    low = name.lower()
    side = "left" if low.startswith("left") else "right" if low.startswith("right") else None
    for key, pat in _ANGLE_AT.items():
        if key in low and side:
            return track.rig.index(pat.format(side=side))
    return track.rig.index("neck") or track.rig.index("left_shoulder")


def _label_angles(img, track, frame, person, opts, scale) -> None:
    defs, vals = track.angles(person)
    uv = track.joints2d[frame, person] * scale
    for k, d in enumerate(defs):
        if d.name not in opts.angles_shown or not np.isfinite(vals[frame, k]):
            continue
        a = _angle_anchor(track, d.name)
        if a is None or not np.isfinite(uv[a]).all():
            continue
        txt = f"{vals[frame, k]:+.0f}"
        # Filmed side on, a left joint and its right twin land on the same
        # pixel; nudge the two apart so both numbers stay readable.
        dy = -9 if d.name.lower().startswith("left") else 11
        org = (_pt(uv[a])[0] + 8, _pt(uv[a])[1] + dy)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.44 * max(1.0, scale),
                    (20, 20, 24), 3, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.44 * max(1.0, scale),
                    (235, 235, 240), 1, cv2.LINE_AA)


# --------------------------------------------------------------- pose panel

_SPAN_SAMPLES = 2000     # frames the figure's size is taken from (drawing only)


def _stable_span(stack: np.ndarray, R: np.ndarray) -> tuple[float, float]:
    """How big to draw the figure: the typical (median) extent of ONE pose
    along each view axis, over every frame that has one.

    Per-AXIS, because a standing person is three times taller than they are
    wide and a portrait panel is taller than it is wide; one combined span
    scaled against min(W, H) left the figure filling about half the panel.
    Median rather than maximum so one bad frame cannot shrink the whole clip,
    and per-clip rather than per-frame so the figure does not breathe.

    Vectorised over the frames (I89): the per-frame Python loop cost ~0.2 s
    on every redraw of a 40k-frame track. The extent of a pose does not
    depend on where it is, so no per-frame centring is needed."""
    fin = np.isfinite(stack).all(axis=-1)                    # (T, J)
    rows = np.nonzero(fin.sum(axis=1) >= 2)[0]
    if not len(rows):
        return 1.0, 1.0
    if len(rows) > _SPAN_SAMPLES:
        # the median of an even sample is the same figure size to the eye,
        # and orbiting recomputes this for every new view angle
        rows = rows[np.linspace(0, len(rows) - 1, _SPAN_SAMPLES).astype(np.int64)]
    q = np.asarray(stack[rows], np.float64) @ np.asarray(R, np.float64).T
    q = np.where(fin[rows][..., None], q, np.nan)
    ext = np.nanmax(q, axis=1) - np.nanmin(q, axis=1)        # (n, D)
    return max(float(np.median(ext[:, 0])), 1e-3), max(float(np.median(ext[:, 1])), 1e-3)


MESH_BASE = (196, 168, 150)     # BGR: a cool blue-grey, like SAM 3D Body's own render
BONE_ON_MESH = (196, 214, 232)  # BGR, warm cream -- reads over the blue-grey body
_DEPTH_SLICES = 28              # painter's-algorithm resolution
_SHADE_LEVELS = 18


def _draw_mesh(img, verts, faces, R, c_view, sc, size, pan, flip_y) -> bool:
    """Shaded body mesh in the same projection the skeleton uses, so the two
    line up exactly. Software only, like every other surface in this app.

    SAM 3D Body's mesh is ~37k triangles, and one `fillConvexPoly` per
    triangle measured 259 ms a frame -- too slow to drag the view around, let
    alone to export a video. Two things fix that without changing the picture
    much:

    * **backface culling** -- on a closed body roughly half the triangles face
      away from the viewer and are painted over anyway;
    * **batched painter's algorithm** -- triangles are bucketed into depth
      slices (far to near) and, within a slice, by shade, so one `fillPoly`
      draws hundreds at a time. Ordering between slices is exact; inside a
      slice it is arbitrary, which on a smooth surface whose triangles are a
      couple of pixels across is not visible.
    """
    W, Hh = size
    if verts is None or faces is None or not len(faces):
        return False
    v = verts @ R.T - c_view
    if not np.isfinite(v).all():
        return False
    scr = np.column_stack([W / 2 + v[:, 0] * sc + pan[0],
                           Hh / 2 + flip_y * v[:, 1] * sc + pan[1]])
    tv = v[faces]
    a, b, cc = tv[:, 0], tv[:, 1], tv[:, 2]
    nrm = np.cross(b - a, cc - a)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-12)

    # the viewer looks along -z in view space when flip_y is -1 (3D), +z else
    towards = -1.0 if flip_y < 0 else 1.0
    depth = tv[:, :, 2].mean(axis=1) * -towards          # far first after argsort
    keep = (nrm[:, 2] * towards) > 0                     # backface cull
    if not keep.any():
        keep = np.ones(len(faces), bool)                 # open mesh: draw it all

    poly = np.round(scr[faces]).astype(np.int32)
    # anything entirely off-panel costs a fill for nothing
    on = ((poly[..., 0] >= 0).any(axis=1) & (poly[..., 0] < W).any(axis=1)
          & (poly[..., 1] >= 0).any(axis=1) & (poly[..., 1] < Hh).any(axis=1))
    keep &= on
    idx = np.nonzero(keep)[0]
    if not len(idx):
        return False

    light = np.array([0.35, 0.55, -0.75])
    light = light / np.linalg.norm(light)
    shade = np.clip(0.30 + 0.70 * np.abs(nrm[idx] @ light), 0.0, 1.0)
    level = np.clip((shade * (_SHADE_LEVELS - 1)).astype(int), 0, _SHADE_LEVELS - 1)

    order = np.argsort(depth[idx], kind="stable")
    base = np.asarray(MESH_BASE, np.float64)
    for chunk in np.array_split(order, min(_DEPTH_SLICES, max(1, len(order)))):
        if not len(chunk):
            continue
        lv = level[chunk]
        for q in np.unique(lv):
            sel = chunk[lv == q]
            col = (base * (q / (_SHADE_LEVELS - 1) * 0.70 + 0.30)).astype(int).tolist()
            cv2.fillPoly(img, poly[idx[sel]], col)
    return True


def render_pose_panel(track: BodyTrack, frame: int, person: int, size: tuple[int, int],
                      azimuth: float = 25.0, elevation: float = 12.0, zoom: float = 1.0,
                      names: bool = False, pan: tuple[float, float] = (0.0, 0.0),
                      trail: int = 0, upright: bool = True, mesh: bool = True) -> np.ndarray:
    """The pose on its own, away from the picture.

    With 3D it is an orbitable skeleton in the model's own camera frame; with
    a 2D backend it is the image-plane skeleton re-centred and scaled to fill
    the panel, which still shows the pose clearly and, unlike the video, does
    not move around as the subject walks.
    """
    W, Hh = size
    img = np.full((Hh, W, 3), BG_PANEL, np.uint8)
    if track is None or not track.has(frame, person):
        cv2.putText(img, "no pose on this frame", (16, Hh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DIM, 1, cv2.LINE_AA)
        return img

    # Centre on THIS frame's pose but take the scale from the whole clip: a
    # subject who walks across the shot would otherwise shrink to a dot (the
    # framing box would be the whole path), and a per-frame scale would make
    # the figure breathe with every stride.
    # The whole-clip quantities (upright basis, figure size) are cached on the
    # track until it changes (I89): recomputing them on every redraw made
    # each seek of a 40k-frame video take most of a second.
    if track.has_3d:
        pts = track.joints3d[frame, person].astype(np.float64)
        stack = track.joints3d[:, person]
        base = None
        if upright:
            base = track.cached(f"basis3d{person}", (person,),
                                lambda: body_basis(stack, track.rig))
        R = _view_rotation(azimuth, elevation, up=1) @ (
            base if base is not None else _up_basis(track.rig.up))
        flip_y = -1.0
    else:
        pts = track.joints2d[frame, person].astype(np.float64)
        stack = track.joints2d[:, person]
        R = None
        if upright:
            R = track.cached(f"basis2d{person}", (person,),
                             lambda: body_basis_2d(stack, track.rig))
        if R is None:
            R = np.eye(2)
        flip_y = 1.0                        # image rows already grow downwards
    here = pts[np.isfinite(pts).all(axis=1)]
    if not len(here):
        return img
    span_x, span_y = track.cached(f"span{person}", (person, R.tobytes()),
                                  lambda: _stable_span(stack, R))
    sc = 0.78 * min(W / span_x, Hh / span_y) * zoom
    # Centre the pose's BOUNDING BOX, not its centroid: a third of the COCO
    # joints are in the head, so a centroid sits somewhere in the chest and
    # pushes the feet off the bottom of the panel.
    seen = here @ R.T
    c_view = (seen.min(axis=0) + seen.max(axis=0)) / 2.0
    v = pts @ R.T - c_view
    scr = np.column_stack([W / 2 + v[:, 0] * sc + pan[0],
                           Hh / 2 + flip_y * v[:, 1] * sc + pan[1]])
    unit = track.rig.units if track.has_3d else "px"

    # The body's own 3D shape, when the backend produced one, drawn first so
    # the skeleton reads on top of it.
    drew_mesh = False
    if mesh and track.has_3d:
        mv = track.mesh_at(frame, person)
        if mv is not None:
            drew_mesh = _draw_mesh(img, mv[0].astype(np.float64), mv[1], R, c_view, sc,
                                   (W, Hh), pan, flip_y)
    ok = np.isfinite(scr).all(axis=1) & ~(track.conf[frame, person] < MIN_JOINT_CONF)
    # Over the shaded body the bones are drawn in a neutral cream and the SIDE
    # is carried by the joint dots -- colouring the bones too fights with the
    # surface. On a bare skeleton the coloured bones are the quickest way to
    # spot a model that has swapped someone's legs, so they stay.
    span_px = 0.78 * min(W, Hh)
    jr = max(2, int(round(span_px / 190)))
    fr = max(1, int(round(span_px / 380)))          # finger joints, smaller
    bw = max(1, int(round(span_px / 300)))
    for i, j in track.bones():
        if i < len(ok) and j < len(ok) and ok[i] and ok[j]:
            if drew_mesh:
                c = BONE_ON_MESH
            else:
                c = joint_colour(track.rig.joints[i] if side_of(track.rig.joints[i]) != "mid"
                                 else track.rig.joints[j])
            thin = is_finger(track.rig.joints[i]) and is_finger(track.rig.joints[j])
            cv2.line(img, _pt(scr[i]), _pt(scr[j]), c, max(1, bw - 1) if thin else bw,
                     cv2.LINE_AA)
    for k in np.nonzero(ok)[0]:
        nm = track.rig.joints[k]
        cv2.circle(img, _pt(scr[k]), fr if is_finger(nm) else jr, joint_colour(nm), -1,
                   cv2.LINE_AA)
        if names and not is_finger(nm):
            cv2.putText(img, nm, (_pt(scr[k])[0] + 5, _pt(scr[k])[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL_DIM, 1, cv2.LINE_AA)
    tag = f"3D pose ({unit})" if track.has_3d else "2D pose (image plane)"
    if drew_mesh:
        tag = f"3D body shape ({unit})"
    if upright:
        tag += " - stood upright"
    cv2.putText(img, tag, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, "left", (12, Hh - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_LEFT, 1, cv2.LINE_AA)
    cv2.putText(img, "right", (12, Hh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_RIGHT, 1, cv2.LINE_AA)
    return img


# -------------------------------------------------------------- angle plots

_PLOT_MARGINS = (46, 168, 16, 22)      # left, right, top, bottom


def angle_plot(track: BodyTrack, person: int, frame: int, size: tuple[int, int],
               shown: list[str] | None = None, fps: float = 0.0) -> np.ndarray:
    """Joint angles against time with a playhead, plus the current value of
    each. Degrees on the left, frames along the bottom.

    Everything that does not depend on the frame (grid, traces, legend) is
    drawn once per track version and size and cached on the track (I89): it
    walked every frame of the video in Python on every redraw."""
    W, Hh = size
    if track is None:
        return np.full((Hh, W, 3), PLOT_BG, np.uint8)
    key = (int(person), int(W), int(Hh), None if shown is None else tuple(shown))
    base, defs, vals, pick = track.cached(
        f"plot{person}", key, lambda: _plot_layer(track, person, W, Hh, shown))
    img = base.copy()
    if not pick:
        return img
    left, right, top, bot = _PLOT_MARGINS
    pw, ph = max(10, W - left - right), max(10, Hh - top - bot)
    T = max(1, track.n_frames - 1)
    for n, k in enumerate(pick):
        col = TRACE_COLS[n % len(TRACE_COLS)]
        v = vals[:, k]
        cur = v[frame] if 0 <= frame < len(v) else np.nan
        val = f"{cur:+6.1f}" if np.isfinite(cur) else "   --"
        yy = top + 12 + n * 14
        cv2.putText(img, val, (W - 46, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
    x = int(left + pw * max(0, min(track.n_frames - 1, frame)) / T)
    cv2.line(img, (x, top), (x, top + ph), (240, 240, 245), 1, cv2.LINE_AA)
    lab = f"frame {frame}" + (f"   {frame / fps:.2f} s" if fps > 0 else "")
    cv2.putText(img, lab, (left, Hh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COL_DIM, 1, cv2.LINE_AA)
    return img


def _plot_layer(track: BodyTrack, person: int, W: int, Hh: int, shown):
    """(static image, defs, vals, picked columns) for `angle_plot`."""
    img = np.full((Hh, W, 3), PLOT_BG, np.uint8)
    defs, vals = track.angles(person)
    pick = [k for k, d in enumerate(defs) if shown is None or d.name in shown]
    pick = [k for k in pick if np.isfinite(vals[:, k]).any()][:len(TRACE_COLS)]
    left, right, top, bot = _PLOT_MARGINS
    pw, ph = max(10, W - left - right), max(10, Hh - top - bot)
    if not pick:
        cv2.putText(img, "no angles to plot", (left, Hh // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, COL_DIM, 1, cv2.LINE_AA)
        img.flags.writeable = False
        return img, defs, vals, pick
    sel = vals[:, pick]
    lo = float(np.nanmin(sel)) if np.isfinite(sel).any() else -1.0
    hi = float(np.nanmax(sel)) if np.isfinite(sel).any() else 1.0
    if hi - lo < 1e-6:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    T = max(1, track.n_frames - 1)

    def Y(v):
        return top + ph * (1.0 - (v - lo) / (hi - lo))

    # grid: zero line plus a rounded step
    step = max(10.0, round((hi - lo) / 4.0 / 10.0) * 10.0)
    g = np.ceil(lo / step) * step
    while g <= hi:
        y = int(Y(g))
        cv2.line(img, (left, y), (left + pw, y), (44, 46, 52), 1, cv2.LINE_AA)
        cv2.putText(img, f"{g:.0f}", (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    COL_DIM, 1, cv2.LINE_AA)
        g += step
    if lo < 0 < hi:
        y = int(Y(0.0))
        cv2.line(img, (left, y), (left + pw, y), (86, 90, 98), 1, cv2.LINE_AA)

    # Break the line across gaps so missing frames never look measured -- but
    # a run that sampled every Nth frame has a gap of N between every pair of
    # real samples, and those SHOULD be joined. `track.step` is what the run
    # asked for, so anything wider than it is a real gap.
    gap = max(1, int(getattr(track, "step", 1) or 1))
    for n, k in enumerate(pick):
        col = TRACE_COLS[n % len(TRACE_COLS)]
        v = vals[:, k]
        idx = np.nonzero(np.isfinite(v))[0]
        if len(idx):
            # int() truncation, as the per-frame loop this replaces did
            xs = (left + pw * idx / T).astype(np.int32)
            ys = (top + ph * (1.0 - (v[idx] - lo) / (hi - lo))).astype(np.int32)
            cuts = np.nonzero(np.diff(idx) > gap)[0] + 1
            for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(idx)]):
                if b - a > 1:
                    cv2.polylines(img, [np.column_stack([xs[a:b], ys[a:b]])], False, col, 1,
                                  cv2.LINE_AA)
                else:
                    cv2.circle(img, (int(xs[a]), int(ys[a])), 1, col, -1, cv2.LINE_AA)
        txt = f"{defs[k].name[:22]}"
        yy = top + 12 + n * 14
        cv2.line(img, (left + pw + 8, yy - 4), (left + pw + 20, yy - 4), col, 2, cv2.LINE_AA)
        cv2.putText(img, txt, (left + pw + 25, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (198, 200, 206), 1, cv2.LINE_AA)
    cv2.putText(img, "degrees", (4, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL_DIM, 1,
                cv2.LINE_AA)
    img.flags.writeable = False
    return img, defs, vals, pick


def _plot_height(h: int) -> int:
    return max(90, int(round(h * 0.28)))


def side_by_side_scale(frame_w: int, frame_h: int, width: int | None = None,
                       max_height: int | None = None, plot: bool = True) -> float:
    """The scale the footage is drawn at so the composite (two panels side by
    side, the plot under them) is `width` wide and no taller than
    `max_height`."""
    s = float(width) / (2.0 * frame_w) if width else 1.0
    if max_height:
        if plot:
            s_h = max_height / (1.28 * frame_h)
            if 0.28 * frame_h * s_h < 90:           # the plot's 90 px floor
                s_h = (max_height - 90) / float(frame_h)
        else:
            s_h = max_height / float(frame_h)
        s = min(s, s_h)
    return max(s, 16.0 / max(frame_w, frame_h))


def compose_side_by_side(bgr: np.ndarray, track: BodyTrack, frame: int, person: int,
                         opts: PoseDrawOptions | None = None, plot: bool = True,
                         azimuth: float = 25.0, elevation: float = 12.0,
                         fps: float = 0.0, width: int | None = None,
                         upright: bool = True, mesh: bool = True,
                         max_height: int | None = None) -> np.ndarray:
    """video+skeleton | pose panel, with the angle plot underneath. This is
    what the window shows and what the exported video contains, so the two can
    never drift apart.

    Drawn directly at the OUTPUT size: the footage is scaled first and the
    pose panel and the plot are rendered at their final size. (I91) Drawing
    at native 4K and shrinking the 7680-px composite to a 1200-1600 px window
    or video made every caption, legend and angle number 1-3 px tall; it was
    also most of the cost."""
    h0, w0 = bgr.shape[:2]
    s = side_by_side_scale(w0, h0, width, max_height, plot)
    left = draw_pose(bgr, track, frame, opts, scale=s)
    h, w = left.shape[:2]
    right = render_pose_panel(track, frame, person, (w, h), azimuth, elevation,
                              upright=upright, mesh=mesh)
    top = np.hstack([left, right])
    cv2.line(top, (w, 0), (w, h), (70, 74, 82), 2)
    out = top
    if plot and track is not None:
        shown = list(opts.angles_shown) if opts and opts.angles_shown else None
        out = np.vstack([top, angle_plot(track, person, frame, (top.shape[1], _plot_height(h)),
                                         shown, fps)])
    d = int(width) - out.shape[1] if width else 0
    if d and abs(d) <= 2:
        # rounding the frame to whole pixels can leave the composite a pixel
        # or two off: pad or trim the edge rather than resample the picture
        # (a max_height limit makes it narrower on purpose; that is left alone)
        out = (np.hstack([out, np.full((out.shape[0], d, 3), BG_PANEL, np.uint8)]) if d > 0
               else np.ascontiguousarray(out[:, :int(width)]))
    return out


# ------------------------------------------------------------------ worker

def lens_intrinsics(focal: float) -> np.ndarray:
    """The intrinsics the run dialog hands a 3D backend from a measured focal
    length. The principal point is left NaN on purpose: the estimator puts it
    at each frame's centre (`Sam3DBodyEstimator.cam_int_for`), which is where
    the model's own projection assumes it. (I88) It used to be 0, i.e. the
    optical centre at the top-left pixel."""
    f = float(focal)
    return np.array([[f, 0.0, np.nan], [0.0, f, np.nan], [0.0, 0.0, 1.0]])


@dataclass
class BodyRunOptions:
    backend: str = ""
    start: int = 0
    end: int = 0
    step: int = 1
    max_people: int = 1
    use_detector: bool = True
    use_masks: bool = False       # take the person box from the segment silhouette
    detector_threshold: float = 0.4
    intrinsics: np.ndarray | None = None
    store_mesh: bool = True       # keep the body shape, not only the joints


class BodyPoseWorker(QThread):
    """Runs a pose backend over a frame range on its own thread, with its own
    VideoCapture (one capture per thread is a hard rule in this app) and its
    own torch import. Emits progress, then finished_ok or error."""

    progress = Signal(int, int, str)     # done, total, note
    finished_ok = Signal(object)         # BodyTrack
    error = Signal(str)

    def __init__(self, video_path: str, n_frames: int, opts: BodyRunOptions,
                 masks=None, fps: float = 0.0, target=None):
        super().__init__()
        self.video_path = str(video_path)
        self.n_frames = int(n_frames)
        self.opts = opts
        self.masks = masks               # a MaskTrack, when driving from the silhouette
        self.fps = float(fps)
        # (I83) What the result BELONGS to -- the caller's session for the view
        # the run was started on. Never touched here; the receiver must write
        # the result into this, not into whatever view is active when the run
        # ends (the progress dialog is non-modal, so the user can switch).
        self.target = target
        self._cancel = False
        self.track: BodyTrack | None = None

    def result_fits(self, session) -> bool:
        """True when a finished track may be stored in `session`: it is the
        session the run was started for and has the same number of frames.
        Without a target, the video path has to match instead."""
        if session is None:
            return False
        if int(getattr(session, "n_frames", -1)) != self.n_frames:
            return False
        if self.target is not None:
            return session is self.target
        vp = getattr(session, "video_path", None)
        if vp:
            try:
                return Path(str(vp)).resolve() == Path(self.video_path).resolve()
            except OSError:
                return str(vp) == self.video_path
        return True

    def request_cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        from cotracker_app import bodypose
        from cotracker_app.video_source import open_capture
        cap = None
        try:
            est = bodypose.make_estimator(
                self.opts.backend, max_people=self.opts.max_people,
                use_detector=self.opts.use_detector and not self.opts.use_masks,
                detector_threshold=self.opts.detector_threshold,
                intrinsics=self.opts.intrinsics)
            track = BodyTrack(self.n_frames, est.rig, self.opts.max_people)
            track.backend = bodypose.BACKENDS[self.opts.backend].label
            track.examined = np.zeros(0, np.int64)
            if not est.gives_3d:
                track.notes = ("2D backend: angles are measured in the picture, not in "
                               "space. Film the subject side on for limb angles.")
            elif self.opts.intrinsics is not None:
                f_lens = float(np.asarray(self.opts.intrinsics, np.float64).reshape(-1)[0])
                track.notes = (f"Focal length taken from this camera's lens calibration "
                               f"({f_lens:.0f} px); the frames were not undistorted first.")
            else:
                track.notes = ("No lens calibration was used: the model assumed its own "
                               "focal length, so absolute 3D distances may be off by the "
                               "ratio of that guess to the real lens (angles are not affected).")
            matcher = bodypose.PersonMatcher(self.opts.max_people)
            cap = open_capture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("could not open " + self.video_path)
            f0, f1 = int(self.opts.start), int(self.opts.end)
            step = max(1, int(self.opts.step))
            frames = list(range(f0, f1 + 1, step))
            total = len(frames)
            # what the run asked for, so the report judges it on that and the
            # angle plot knows which gaps are real
            track.n_requested = total
            track.step = step
            track.runs = [(f0, f1, step)]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            nxt = f0
            done = 0
            bad = 0                 # frames the model choked on
            first_error = ""
            # (I82) the frames actually put through the model: a merge
            # replaces exactly these and keeps every other frame's pose
            examined: list[int] = []
            for f in frames:
                if self._cancel:
                    track.stopped_early = True
                    break
                while nxt < f:                       # step > 1: skip cheaply
                    if not cap.grab():
                        break
                    nxt += 1
                ok, bgr = cap.read()
                nxt = f + 1
                if not ok:
                    track.stopped_early = True
                    break
                boxes = masks_in = None
                if self.opts.use_masks:
                    if self.masks is None or not self.masks.has(f):
                        # No silhouette here means nothing to aim at. Leaving
                        # the frame blank is the whole point: a top-down pose
                        # model hands back a confident skeleton for ANY box,
                        # so "no prompt" must not become "pose the background".
                        done += 1
                        continue
                    masks_in = [self.masks.rasterize(f, bgr.shape[0], bgr.shape[1])]
                try:
                    people = est.step(bgr, boxes=boxes, masks=masks_in)
                except Exception as exc:              # noqa: BLE001
                    # One awkward frame must not throw away a long run
                    # (robustness over speed). A systemic failure looks
                    # different -- nothing works from the very first frame --
                    # and is reported instead of grinding through the video.
                    bad += 1
                    first_error = first_error or str(exc)
                    if bad >= 5 and not track.n_posed():
                        raise RuntimeError(
                            f"the pose model failed on the first {bad} frames: "
                            f"{first_error}") from exc
                    done += 1
                    continue
                examined.append(f)
                for i, col in enumerate(matcher.assign(people, frame=f)):
                    if col < 0:
                        continue
                    p = people[i]
                    track.set_person(f, col, joints2d=p.joints2d, joints3d=p.joints3d,
                                     conf=p.conf, score=p.score, bbox=p.bbox, focal=p.focal,
                                     vertices=p.vertices if self.opts.store_mesh else None,
                                     faces=getattr(est, "faces", None), cam_t=p.cam_t)
                done += 1
                if done % 3 == 0 or done == total:
                    self.progress.emit(done, total, f"frame {f}")
            est.close()
            if bad:
                note = (f"{bad} frame{'s' if bad != 1 else ''} could not be processed; "
                        f"{'they keep' if bad != 1 else 'it keeps'} any earlier pose, "
                        f"else {'stay' if bad != 1 else 'stays'} blank ({first_error}).")
                track.notes = (track.notes + "  " + note).strip()
            track.examined = np.asarray(examined, np.int64)
            self.track = track
            self.finished_ok.emit(track)
        except Exception as exc:                     # noqa: BLE001 - reported to the user
            self.error.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()


class SideBySideRenderer(QThread):
    """Write the side-by-side view to an mp4."""
    progress = Signal(int, int)
    finished_ok = Signal(str, str)
    error = Signal(str)

    def __init__(self, video_path: str, track: BodyTrack, out_path: str, start: int, end: int,
                 person: int, opts: PoseDrawOptions, fps: float, width: int = 1600,
                 azimuth: float = 25.0, elevation: float = 12.0, plot: bool = True,
                 upright: bool = True, mesh: bool = True):
        super().__init__()
        self.video_path, self.out_path = str(video_path), str(out_path)
        self.track, self.person, self.opts = track, int(person), opts
        # NOT self.start: that name is QThread.start, and shadowing it makes
        # the thread un-startable with a baffling "int object is not callable"
        self.f0, self.f1 = int(start), int(end)
        self.fps, self.width = float(fps or 30.0), int(width)
        self.azimuth, self.elevation, self.plot = azimuth, elevation, plot
        self.upright, self.mesh = bool(upright), bool(mesh)
        self._cancel = False
        # (I90) A run that sampled every Nth frame has a pose on one frame in
        # N; writing every frame strobed the skeleton on and "no pose" off.
        # Such a run writes only its posed frames, at fps / N, so the video
        # still plays in real time. `note` says so, for the caller's message.
        step = max(1, int(getattr(track, "step", 1) or 1))
        if step > 1:
            fr = track.frames()
            self.frames = [int(f) for f in fr if self.f0 <= f <= self.f1]
            self.out_fps = max(1.0, self.fps / step)
            self.note = (f"only the {len(self.frames)} posed frames (the run sampled every "
                         f"{step}th frame), played at {self.out_fps:g} fps so it keeps "
                         f"real time")
        else:
            self.frames = list(range(self.f0, self.f1 + 1))
            self.out_fps = self.fps
            self.note = ""

    def request_cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        from cotracker_app.render import open_writer
        from cotracker_app.video_source import open_capture
        cap = vw = None
        try:
            cap = open_capture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("could not open " + self.video_path)
            if not self.frames:
                raise RuntimeError("no posed frames in %d-%d" % (self.f0, self.f1))
            first = self.frames[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError("could not read frame %d" % first)
            probe = compose_side_by_side(bgr, self.track, first, self.person, self.opts,
                                         self.plot, self.azimuth, self.elevation, self.fps,
                                         self.width, self.upright, self.mesh)
            size = (probe.shape[1], probe.shape[0])
            vw, codec = open_writer(self.out_path, self.out_fps, size)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            nxt = first
            total = max(1, len(self.frames))
            done = 0
            for f in self.frames:
                if self._cancel:
                    break
                while nxt < f:                       # sampled run: skip cheaply
                    if not cap.grab():
                        break
                    nxt += 1
                ok, bgr = cap.read()
                nxt = f + 1
                if not ok:
                    break
                img = compose_side_by_side(bgr, self.track, f, self.person, self.opts,
                                           self.plot, self.azimuth, self.elevation, self.fps,
                                           self.width, self.upright, self.mesh)
                if (img.shape[1], img.shape[0]) != size:
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                vw.write(img)
                done += 1
                if done % 5 == 0 or done == total:
                    self.progress.emit(done, total)
            vw.release()
            vw = None
            if self._cancel:
                Path(self.out_path).unlink(missing_ok=True)
                self.error.emit("cancelled")
                return
            self.finished_ok.emit(self.out_path, codec)
        except Exception as exc:                     # noqa: BLE001
            self.error.emit(str(exc))
        finally:
            if vw is not None:
                vw.release()
            if cap is not None:
                cap.release()


# ------------------------------------------------------------------ dialogs

class BodyRunDialog(QDialog):
    """Ask what to run and over what. Written for someone who has never heard
    of human mesh recovery: every choice says what it costs and what it gives."""

    def __init__(self, parent, n_frames: int, current: int, sel_range, has_masks: bool,
                 default_backend: str = "", lens_focal: float | None = None,
                 existing: BodyTrack | None = None):
        super().__init__(parent)
        from cotracker_app import bodypose
        self.setWindowTitle("Find people and measure their joints")
        self.result_options: BodyRunOptions | None = None
        self._bp = bodypose
        # the body track this view already has: a new run is merged into it,
        # unless it uses another joint set / number of people (asked first)
        self._existing = existing if existing is not None and existing.n_posed() else None
        lay = QVBoxLayout(self)

        intro = QLabel(
            "This finds a person on every frame of the range and works out where their "
            "joints are. With a 3D model you also get real joint angles in space; with the "
            "2D model you get angles as they appear in the picture.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(intro)

        form = QFormLayout()
        self.cmb_backend = QComboBox()
        for key, spec in bodypose.BACKENDS.items():
            state, why = bodypose.backend_status(key)
            tag = {"ready": "", "download": "  (downloads on first use)",
                   "needs-code": "  - code not installed",
                   "needs-weights": "  - weights not installed"}[state]
            self.cmb_backend.addItem(spec.label + tag, key)
            self.cmb_backend.setItemData(self.cmb_backend.count() - 1, why, Qt.ToolTipRole)
            if state in ("needs-code", "needs-weights"):
                self.cmb_backend.model().item(self.cmb_backend.count() - 1).setEnabled(False)
        want = default_backend or bodypose.preferred_backend()
        i = self.cmb_backend.findData(want)
        self.cmb_backend.setCurrentIndex(max(0, i))
        form.addRow("Model", self.cmb_backend)
        self.lbl_state = QLabel("")
        self.lbl_state.setWordWrap(True)
        self.lbl_state.setStyleSheet(f"color: {theme.TEXT_DIM};")
        form.addRow("", self.lbl_state)
        self.cmb_backend.currentIndexChanged.connect(self._refresh_state)

        self.spin_people = QSpinBox()
        self.spin_people.setRange(1, 8)
        self.spin_people.setValue(1)
        self.spin_people.setToolTip("How many people to follow. Each one keeps its own "
                                    "column in the results.")
        form.addRow("People", self.spin_people)

        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 50)
        self.spin_step.setValue(1)
        self.spin_step.setToolTip("1 = every frame. Raise it to sample a long clip quickly; "
                                  "the frames in between stay blank.")
        form.addRow("Every Nth frame", self.spin_step)
        lay.addLayout(form)

        box = QGroupBox("Which frames")
        bl = QVBoxLayout(box)
        self.rb_all = QRadioButton(f"The whole video  (0 - {n_frames - 1})")
        self.rb_sel = QRadioButton("The timeline selection")
        self.rb_here = QRadioButton(f"This frame only  ({current})")
        self._sel = sel_range
        self.rb_sel.setEnabled(sel_range is not None)
        if sel_range is not None:
            self.rb_sel.setText(f"The timeline selection  ({sel_range[0]} - {sel_range[1]})")
        for r in (self.rb_all, self.rb_sel, self.rb_here):
            bl.addWidget(r)
        # checked only once the three share the box: radio buttons are exclusive
        # among siblings, and checking two while still parentless left BOTH shown
        # as chosen -- a click on "The whole video" then did nothing and the run
        # used the selection (G7)
        (self.rb_sel if sel_range is not None else self.rb_all).setChecked(True)
        lay.addWidget(box)

        who = QGroupBox("How to find the person")
        wl = QVBoxLayout(who)
        self.rb_det = QRadioButton("Automatically (a person detector runs first)")
        self.rb_mask = QRadioButton("Use the segment silhouette already in this view")
        self.rb_mask.setEnabled(bool(has_masks))
        if not has_masks:
            self.rb_mask.setToolTip("There is no silhouette in this view yet. Draw one with "
                                    "the segment tool (S) first.")
        for r in (self.rb_det, self.rb_mask):
            wl.addWidget(r)
        self.rb_det.setChecked(True)          # after parenting, like the frame choice (G7)
        lay.addWidget(who)

        self.chk_mesh = QCheckBox("Keep the 3D body shape, not just the joints")
        self.chk_mesh.setChecked(True)
        self.chk_mesh.setToolTip(
            "A 3D model returns a whole body surface. Keeping it lets the side-by-side "
            "view show the shape itself instead of a stick figure, and it is what gets "
            "saved in the project. It costs about 110 kB per frame per person (55 MB per "
            "500 frames), and every save of the project writes it again; turn it off for "
            "a very long run. The 2D model has no shape to keep.")
        lay.addWidget(self.chk_mesh)

        self._focal = lens_focal
        self.chk_lens = QCheckBox("Use this camera's measured focal length")
        self.chk_lens.setChecked(lens_focal is not None)
        self.chk_lens.setEnabled(lens_focal is not None)
        self.chk_lens.setToolTip(
            "A 3D model has to guess how wide the lens is unless you tell it. If you have "
            "calibrated this camera, using the real value makes the 3D distances right. "
            "Only the focal length is used (the frames are not undistorted first)."
            if lens_focal is not None else
            "No lens calibration for this camera. 3D → Calibrate a Lens measures one.")
        lay.addWidget(self.chk_lens)

        self.n_frames, self.current = n_frames, current
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Run")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._refresh_state()

    def _refresh_state(self) -> None:
        key = self.cmb_backend.currentData()
        if key:
            self.lbl_state.setText(self._bp.backend_status(key)[1])

    def _confirm_replace(self, key) -> bool:
        """A new run is merged into the poses already here -- only the frames
        it processes change. (I82) When it cannot be (another joint set or
        number of people), say that it will replace every earlier pose and ask
        now, rather than after a run that may take hours."""
        old = self._existing
        spec = self._bp.BACKENDS.get(key)
        rig = spec.rig if spec is not None else old.rig.name
        people = self.spin_people.value()
        if old.rig.name == rig and old.n_people == people:
            return True
        why = (f"joints from {spec.label if spec else key}" if old.rig.name != rig
               else f"{people} {'person' if people == 1 else 'people'} instead of "
                    f"{old.n_people}")
        ans = QMessageBox.question(
            self, "Replace the earlier body poses?",
            f"This view already has body poses on {old.n_posed()} frames "
            f"({old.rig.label}, {old.n_people} "
            f"{'person' if old.n_people == 1 else 'people'}). This run measures {why}, so "
            f"it cannot be merged with them: it will REPLACE the earlier poses on every "
            f"frame, including frames outside the range you chose.\n\nCtrl+Z after the run "
            f"brings them back. Run anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return ans == QMessageBox.Yes

    def _accept(self) -> None:
        key = self.cmb_backend.currentData()
        if self.rb_sel.isChecked() and self._sel is not None:
            f0, f1 = int(self._sel[0]), int(self._sel[1])
        elif self.rb_here.isChecked():
            f0 = f1 = int(self.current)
        else:
            f0, f1 = 0, self.n_frames - 1
        K = None
        if self.chk_lens.isChecked() and self._focal:
            K = lens_intrinsics(self._focal)
        if self._existing is not None and not self._confirm_replace(key):
            return
        self.result_options = BodyRunOptions(
            backend=key, start=f0, end=f1, step=self.spin_step.value(),
            max_people=self.spin_people.value(),
            use_detector=self.rb_det.isChecked(), use_masks=self.rb_mask.isChecked(),
            intrinsics=K, store_mesh=self.chk_mesh.isChecked())
        self.accept()


class BodySideBySide(QWidget):
    """The side-by-side window. Follows the app's current frame; asks the app
    to seek when the user drags the plot."""

    closed = Signal()
    seek_requested = Signal(int)
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Body pose - side by side")
        self.resize(1180, 760)
        self.track: BodyTrack | None = None
        self.frame = 0
        self.fps = 0.0
        self._bgr: np.ndarray | None = None
        self.azimuth, self.elevation = 25.0, 12.0
        self._drag = None
        self.last_image: np.ndarray | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.cmb_person = QComboBox()
        self.cmb_person.setMinimumWidth(110)
        self.cmb_person.currentIndexChanged.connect(lambda _: self.request_render())
        top.addWidget(QLabel("Person"))
        top.addWidget(self.cmb_person)
        self.chk_names = QCheckBox("Joint names")
        self.chk_angles = QCheckBox("Angle numbers")
        self.chk_angles.setChecked(True)
        self.chk_plot = QCheckBox("Angle plot")
        self.chk_plot.setChecked(True)
        self.chk_upright = QCheckBox("Stand upright")
        self.chk_upright.setChecked(True)
        self.chk_upright.setToolTip(
            "Turn the pose so the body's long axis is vertical, whatever angle the "
            "camera was at. Off shows it in the camera's own frame, which is how the "
            "model reports it.")
        self.chk_shape = QCheckBox("Body shape")
        self.chk_shape.setChecked(True)
        self.chk_shape.setToolTip("Draw the 3D body surface the model returned, instead "
                                  "of only the skeleton")
        for c in (self.chk_names, self.chk_angles, self.chk_plot, self.chk_upright,
                  self.chk_shape):
            c.toggled.connect(lambda _=False: self.request_render())
            top.addWidget(c)
        self.btn_export = QPushButton("Save video…")
        self.btn_export.setToolTip("Write exactly this view to an mp4")
        self.btn_export.clicked.connect(lambda: self.export_requested.emit())
        top.addStretch(1)
        self.info = QLabel("")
        self.info.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        top.addWidget(self.info, 1)
        top.addWidget(self.btn_export)
        lay.addLayout(top)

        self.canvas = QLabel()
        self.canvas.setMinimumSize(420, 300)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet("background: #16181c;")
        lay.addWidget(self.canvas, 1)
        hint = QLabel("drag the right-hand pose to orbit it · double-click to reset · "
                      "the plot follows the video")
        hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
        hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(hint)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._render)

    # ---- data
    def set_track(self, track: BodyTrack | None, fps: float = 0.0) -> None:
        self.track = track
        self.fps = float(fps)
        cur = self.cmb_person.currentIndex()
        self.cmb_person.blockSignals(True)
        self.cmb_person.clear()
        if track is not None:
            for i in range(track.n_people):
                self.cmb_person.addItem(track.person_label(i), i)
        self.cmb_person.setCurrentIndex(max(0, min(cur, self.cmb_person.count() - 1)))
        self.cmb_person.blockSignals(False)
        self.info.setText(track.summary() if track is not None else "no body pose yet")
        self.chk_shape.setEnabled(track is not None and track.has_mesh())
        self.request_render()

    def set_frame(self, frame: int, bgr: np.ndarray | None) -> None:
        self.frame = int(frame)
        if bgr is not None:
            self._bgr = bgr
        self.request_render()

    def person(self) -> int:
        return max(0, self.cmb_person.currentIndex())

    def draw_options(self) -> PoseDrawOptions:
        shown = ()
        if self.track is not None and self.chk_angles.isChecked():
            defs, _ = self.track.angles(self.person())
            shown = tuple(d.name for d in defs
                          if d.group in ("arm", "leg") and "flexion" in d.name)
        return PoseDrawOptions(names=self.chk_names.isChecked(),
                               angle_labels=self.chk_angles.isChecked(),
                               person=self.person(), angles_shown=shown)

    def request_render(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def _render(self) -> None:
        w = max(200, self.canvas.width())
        h = max(150, self.canvas.height())
        if self._bgr is None:
            img = np.full((h, w, 3), BG_PANEL, np.uint8)
            cv2.putText(img, "open a video and run a pose estimate", (20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DIM, 1, cv2.LINE_AA)
        else:
            # drawn to fit the canvas both ways, so nothing is shrunk afterwards (I91)
            img = compose_side_by_side(self._bgr, self.track, self.frame, self.person(),
                                       self.draw_options(), self.chk_plot.isChecked(),
                                       self.azimuth, self.elevation, self.fps, width=w,
                                       upright=self.chk_upright.isChecked(),
                                       mesh=self.chk_shape.isChecked(), max_height=h)
            if img.shape[0] > h:
                sc = h / img.shape[0]
                img = cv2.resize(img, (max(2, int(img.shape[1] * sc)), h),
                                 interpolation=cv2.INTER_AREA)
        self.last_image = img
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self.canvas.setPixmap(QPixmap.fromImage(qi.copy()))

    def save_png(self, path: str) -> None:
        if self.last_image is None:
            self._render()
        cv2.imwrite(str(path), self.last_image)

    # ---- interaction
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.request_render()

    def mousePressEvent(self, ev):
        self._drag = (ev.position().x(), ev.position().y())

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        dx, dy = ev.position().x() - self._drag[0], ev.position().y() - self._drag[1]
        self.azimuth = (self.azimuth + dx * 0.5) % 360
        self.elevation = float(np.clip(self.elevation + dy * 0.5, -89, 89))
        self._drag = (ev.position().x(), ev.position().y())
        self.request_render()

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def mouseDoubleClickEvent(self, ev):
        self.azimuth, self.elevation = 25.0, 12.0
        self.request_render()

    def closeEvent(self, ev):
        self.closed.emit()
        super().closeEvent(ev)
