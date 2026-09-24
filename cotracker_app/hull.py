"""Visual hull: the 3D volume carved from the cameras' silhouettes.

Every camera that sees the animal cuts a cone out of space — the set of world
points that project INSIDE its silhouette. The intersection of those cones is
the **visual hull**, the tightest shape consistent with every outline. It is
an over-estimate of the body (concavities that no camera looks into survive),
which is the right property for a volume proxy: it never loses part of the
animal, and adding cameras only tightens it.

Pipeline (all numpy + OpenCV, no GPU):

1. `undistorted_mask` — each view's silhouette (Kinetrace's `MaskTrack` at the
   frame nearest the instant) is warped into the calibration's undistorted
   image space once per frame (`cv2.remap` with maps the calibration caches),
   so the DLT projection of a voxel can be looked up directly.
2. `carve` — a regular voxel grid over a bounding box; every voxel centre is
   projected into every camera with the DLT; a voxel survives when it falls
   inside the (dilated) silhouette of every camera that can see it, up to
   `tolerance` misses. Dilation absorbs segmentation error at the outline
   (a 2 px halo at 1080p costs far less volume than a 2 px bite loses).
3. `surface` — marching tetrahedra on the signed distance of the occupancy
   (inside positive), so the mesh sits at the sub-voxel boundary rather than
   on voxel faces; vertices are welded and the normals point outward.
4. `mesh_volume` — divergence-theorem volume of the closed mesh, alongside the
   plain voxel count, both in the calibration's world units cubed.

Verified in tests/verify_3d.py on a synthetic ellipsoid seen by six DLT
cameras: hull volume within a few percent of the analytic volume, and the
mesh volume within 1% of the voxel volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from cotracker_app.calib import CameraCalibration, dlt_project, front_sign

MAP_STEP = 8          # px between LWM evaluations when building undistortion maps

# ------------------------------------------------------------- silhouettes


def undistort_maps(cal: CameraCalibration, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Remap tables from the UNDISTORTED canvas (same size as the raw frame,
    Kinetrace 0-based pixels) to raw pixel coordinates, for `cv2.remap`. The
    canvas pixel (i, j) holds the raw pixel that lands at calibration-frame
    coordinate (i, j) after undistortion. Cached on the calibration."""
    key = (int(width), int(height))
    cache = getattr(cal, "_maps", None)
    if cache is not None and cache[0] == key and len(cache) == 4:
        return cache[1], cache[2]
    # The lens map is smooth, so it is evaluated on a coarse grid (MAP_STEP px)
    # and interpolated linearly to every pixel: 30k LWM evaluations instead
    # of 2M at 1080p (3 s instead of ~60 s per camera). Measured against the
    # dense map on real DLTdv LWM profiles: max 0.13 px at 1080p, 0.69 px at
    # 480p (both at the very edge of the control-point coverage; the mask is
    # looked up at whole pixels and dilated by 2 px, so this is invisible),
    # and a <= MAP_STEP px band at that edge becomes "not seen". Finer steps
    # did not reduce the deviation (the LWM itself is only piecewise smooth).
    gx = np.unique(np.concatenate([np.arange(0, width, MAP_STEP), [width - 1]])).astype(np.float64)
    gy = np.unique(np.concatenate([np.arange(0, height, MAP_STEP), [height - 1]])).astype(np.float64)
    xs, ys = np.meshgrid(gx, gy)
    canvas = np.column_stack([xs.ravel(), ys.ravel()])
    # canvas (0-based undistorted) -> calibration frame -> raw canvas
    p = canvas + cal.pixel_origin
    if cal.y_flip:
        p[:, 1] = cal.height - p[:, 1] + 2 * cal.pixel_origin - 1
    raw = cal.from_calib_frame(p)
    from scipy.interpolate import RegularGridInterpolator
    full_y, full_x = np.mgrid[0:height, 0:width]
    q = np.column_stack([full_y.ravel().astype(np.float64), full_x.ravel().astype(np.float64)])
    map_x = RegularGridInterpolator((gy, gx), raw[:, 0].reshape(len(gy), len(gx)), bounds_error=False,
                                    fill_value=np.nan)(q).reshape(height, width).astype(np.float32)
    map_y = RegularGridInterpolator((gy, gx), raw[:, 1].reshape(len(gy), len(gx)), bounds_error=False,
                                    fill_value=np.nan)(q).reshape(height, width).astype(np.float32)
    bad = ~(np.isfinite(map_x) & np.isfinite(map_y))
    map_x[bad] = -1.0
    map_y[bad] = -1.0
    # (I94) canvas pixels the camera did not see: no lens inverse there (LWM
    # outside its control points), or a raw position off the picture. remap
    # writes 0 there, which `carve` must not read as "seen, and outside the
    # silhouette" -- that carved the body away along the coverage boundary
    # (a synthetic ellipsoid came out at 55 % of its true volume).
    valid = ~bad & (map_x >= -0.5) & (map_x < width - 0.5) & (map_y >= -0.5) & (map_y < height - 0.5)
    cal._maps = (key, map_x, map_y, valid)
    return map_x, map_y


def undistort_valid(cal: CameraCalibration, width: int, height: int) -> np.ndarray:
    """(height, width) bool: the undistorted-canvas pixels this camera actually
    saw (see `undistort_maps`); anywhere else it has no vote."""
    undistort_maps(cal, width, height)
    return cal._maps[3]


def undistorted_mask(cal: CameraCalibration, mask: np.ndarray, dilate_px: int = 0) -> np.ndarray:
    """Warp a raw-frame bool mask into the undistorted canvas (uint8 0/1),
    dilated by `dilate_px` first so outline error erodes nothing."""
    m = np.asarray(mask, np.uint8)
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
    h, w = m.shape
    map_x, map_y = undistort_maps(cal, w, h)
    if cal.undistort.kind == "none" and not cal.y_flip and cal.pixel_origin == 0.0:
        return m
    return cv2.remap(m, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def calib_canvas_xy(cal: CameraCalibration, uv: np.ndarray) -> np.ndarray:
    """Calibration-frame (undistorted) coordinates -> undistorted-canvas
    pixel coordinates (float), the frame `undistorted_mask` is in."""
    p = np.array(uv, np.float64, copy=True)
    if cal.y_flip:
        p[:, 1] = cal.height - p[:, 1] + 2 * cal.pixel_origin - 1
    return p - cal.pixel_origin


# ------------------------------------------------------------------ carving


@dataclass
class Hull:
    """The carved volume: a bool occupancy grid and where it sits."""
    occupancy: np.ndarray          # (nx, ny, nz) bool
    origin: np.ndarray             # world position of voxel (0, 0, 0)'s centre
    voxel: float                   # edge length, world units
    n_views: int                   # cameras that took part
    seen: np.ndarray               # (nx, ny, nz) int8: views whose image contained the voxel

    @property
    def n_voxels(self) -> int:
        return int(self.occupancy.sum())

    def volume(self) -> float:
        return self.n_voxels * self.voxel ** 3

    def centroid(self) -> np.ndarray | None:
        idx = np.argwhere(self.occupancy)
        if len(idx) == 0:
            return None
        return self.origin + idx.mean(axis=0) * self.voxel

    def extent(self) -> tuple[np.ndarray, np.ndarray] | None:
        idx = np.argwhere(self.occupancy)
        if len(idx) == 0:
            return None
        return (self.origin + idx.min(axis=0) * self.voxel, self.origin + idx.max(axis=0) * self.voxel)


def bounds_from_points(xyz: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box around finite points, padded by `margin` on each side."""
    p = np.asarray(xyz, np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) == 0:
        raise ValueError("no finite points to bound")
    return p.min(axis=0) - margin, p.max(axis=0) + margin


def carve(cams: list[CameraCalibration], masks: list[np.ndarray | None],
          lo: np.ndarray, hi: np.ndarray, voxel: float,
          dilate_px: int = 2, tolerance: int = 0, min_views: int = 2,
          chunk: int = 400_000) -> Hull:
    """Carve the box [lo, hi] at resolution `voxel` with the given cameras and
    their raw silhouettes (None = this camera has no mask at this instant and
    does not vote). A voxel survives when at least `min_views` silhouettes
    contain it and at most `tolerance` of the cameras that can see it do not."""
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    n = np.maximum(np.ceil((hi - lo) / voxel).astype(int) + 1, 1)
    if n.prod() > 60_000_000:
        raise ValueError(f"grid of {n} voxels is too large; use a coarser voxel or tighter bounds")
    views = [(cal, undistorted_mask(cal, m, dilate_px)) for cal, m in zip(cams, masks) if m is not None]
    valids = [undistort_valid(cal, um.shape[1], um.shape[0]) for cal, um in views]
    inside = np.zeros(n.prod(), np.int16)
    seen = np.zeros(n.prod(), np.int16)
    if views:
        gx, gy, gz = np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]), indexing="ij")
        centres = lo + np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]) * voxel
        # only points IN FRONT of a camera vote. The DLT denominator is the
        # depth up to a sign that depends on the calibration's handedness, so
        # the sign is read off a point known to be in view: the centre of the
        # working volume (easyWand/DLTdv coefficients on a real rig can have
        # det M < 0 with the animal at a positive denominator — the old det
        # rule called every voxel "behind" and carved nothing)
        probe = (lo + hi) / 2.0
        fronts = [front_sign(cal.coefs, probe) for cal, _ in views]
        for s in range(0, len(centres), chunk):
            X = centres[s:s + chunk]
            for (cal, um), sgn, valid in zip(views, fronts, valids):
                h, w = um.shape
                uv = calib_canvas_xy(cal, dlt_project(cal.coefs, X))
                den = cal.coefs[8] * X[:, 0] + cal.coefs[9] * X[:, 1] + cal.coefs[10] * X[:, 2] + 1.0
                front = sgn * den > 0
                xi = np.round(uv[:, 0]).astype(np.int64)
                yi = np.round(uv[:, 1]).astype(np.int64)
                ok = front & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                ok[ok] = valid[yi[ok], xi[ok]]          # an unmapped canvas pixel does not vote (I94)
                seen[s:s + chunk] += ok.astype(np.int16)
                hit = np.zeros(len(X), bool)
                hit[ok] = um[yi[ok], xi[ok]] > 0
                inside[s:s + chunk] += hit.astype(np.int16)
    occ = (inside >= min_views) & (inside >= seen - tolerance) & (seen > 0)
    return Hull(occ.reshape(tuple(n)), lo, float(voxel), len(views),
                seen.reshape(tuple(n)).astype(np.int8))


# ----------------------------------------------------------------- surface

# the six tetrahedra of a cube sharing the (0, 7) diagonal; corner k has
# coordinates (k & 1, (k >> 1) & 1, (k >> 2) & 1)
_TETS = np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]])
_CORNER = np.array([[k & 1, (k >> 1) & 1, (k >> 2) & 1] for k in range(8)], np.int64)


def _tet_tables():
    """For each of the 16 inside/outside codes of a tetrahedron: the triangles
    as lists of edges (pairs of local vertex ids) whose crossing points form
    them. Derived programmatically so nothing is hand-copied."""
    tables = {}
    for code in range(16):
        ins = [i for i in range(4) if code >> i & 1]
        outs = [i for i in range(4) if not code >> i & 1]
        tris = []
        if len(ins) == 1:
            a = ins[0]
            tris.append([(a, outs[0]), (a, outs[1]), (a, outs[2])])
        elif len(ins) == 3:
            a = outs[0]
            tris.append([(a, ins[0]), (a, ins[1]), (a, ins[2])])
        elif len(ins) == 2:
            a, b = ins
            c, d = outs
            tris.append([(a, c), (a, d), (b, d)])
            tris.append([(a, c), (b, d), (b, c)])
        tables[code] = tris
    return tables


_TABLES = _tet_tables()


def surface(field: np.ndarray, origin: np.ndarray, voxel: float, iso: float = 0.0
            ) -> tuple[np.ndarray, np.ndarray]:
    """Marching tetrahedra on a scalar field (inside > iso): welded vertices
    (V, 3) in world units and outward-facing triangles (F, 3)."""
    f = np.asarray(field, np.float64)
    nx, ny, nz = f.shape
    if min(nx, ny, nz) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    cx, cy, cz = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
    base = np.column_stack([cx.ravel(), cy.ravel(), cz.ravel()])          # (C, 3)
    corners = base[:, None, :] + _CORNER[None, :, :]                        # (C, 8, 3)
    vals = f[corners[..., 0], corners[..., 1], corners[..., 2]]            # (C, 8)
    tri_pts = []
    for tet in _TETS:
        v = vals[:, tet]                                                    # (C, 4)
        p = corners[:, tet, :].astype(np.float64)                           # (C, 4, 3)
        code = ((v[:, 0] > iso) | ((v[:, 1] > iso) << 1) | ((v[:, 2] > iso) << 2)
                | ((v[:, 3] > iso) << 3)).astype(int)
        for c in range(1, 15):
            sel = np.nonzero(code == c)[0]
            if len(sel) == 0:
                continue
            vv, pp = v[sel], p[sel]
            for tri in _TABLES[c]:
                pts = []
                for (a, b) in tri:
                    va, vb = vv[:, a], vv[:, b]
                    t = (iso - va) / np.where(np.abs(vb - va) < 1e-12, 1e-12, vb - va)
                    t = np.clip(t, 0.0, 1.0)[:, None]
                    pts.append(pp[:, a] + t * (pp[:, b] - pp[:, a]))
                tri_xyz = np.stack(pts, axis=1)                              # (S, 3, 3)
                # orient outward: the normal must point away from the inside corners
                ins_c = np.array([i for i in range(4) if c >> i & 1])
                inside_centroid = pp[:, ins_c].mean(axis=1)
                normal = np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0])
                flip = np.einsum("ij,ij->i", normal, inside_centroid - tri_xyz.mean(axis=1)) > 0
                tri_xyz[flip] = tri_xyz[flip][:, ::-1]
                tri_pts.append(tri_xyz)
    if not tri_pts:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    soup = np.concatenate(tri_pts).reshape(-1, 3)                           # (3F, 3) grid units
    key = np.round(soup * 1024.0).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    verts = uniq.astype(np.float64) / 1024.0 * voxel + np.asarray(origin, np.float64)
    faces = inv.reshape(-1, 3)
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    return verts, faces


def signed_distance(occ: np.ndarray) -> np.ndarray:
    """Inside-positive signed distance (voxel units) of a bool grid, padded by
    one empty voxel so the surface always closes."""
    o = np.pad(np.asarray(occ, bool), 1)
    if not o.any():
        return -np.ones(o.shape)
    return ndimage.distance_transform_edt(o) - ndimage.distance_transform_edt(~o)


def hull_mesh(h: Hull, smooth: int = 3, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Mesh of a hull: marching tetrahedra on its signed distance (Gaussian-
    blurred by `sigma` voxels so the iso-surface is not a voxel staircase),
    then a few passes of Laplacian smoothing."""
    sdf = signed_distance(h.occupancy)
    if sigma > 0:
        sdf = ndimage.gaussian_filter(sdf, sigma)
    verts, faces = surface(sdf, h.origin - h.voxel, h.voxel, 0.0)
    if smooth > 0 and len(faces):
        verts = smooth_mesh(verts, faces, smooth)
    return verts, faces


def smooth_mesh(verts: np.ndarray, faces: np.ndarray, iterations: int = 2, lam: float = 0.5) -> np.ndarray:
    """Umbrella Laplacian smoothing (moves each vertex toward its neighbours'
    mean by `lam`); cheap and enough to take the voxel texture off."""
    v = np.array(verts, np.float64, copy=True)
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    deg = np.bincount(i, minlength=len(v)).astype(np.float64)
    for _ in range(int(iterations)):
        acc = np.zeros_like(v)
        np.add.at(acc, i, v[j])
        mean = acc / np.maximum(deg, 1)[:, None]
        v = np.where(deg[:, None] > 0, v + lam * (mean - v), v)
    return v


def mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Volume enclosed by a closed, outward-oriented mesh (divergence theorem)."""
    if len(faces) == 0:
        return 0.0
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = np.zeros_like(verts)
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- file I/O


def save_obj(path, verts: np.ndarray, faces: np.ndarray, comment: str = "") -> None:
    lines = [f"# {comment}"] if comment else []
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def save_ply(path, verts: np.ndarray, faces: np.ndarray) -> None:
    hdr = ["ply", "format ascii 1.0", f"element vertex {len(verts)}",
           "property float x", "property float y", "property float z",
           f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header"]
    body = [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts] + [f"3 {a} {b} {c}" for a, b, c in faces]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(hdr + body) + "\n")


# ------------------------------------------------------------ quick render


def render_mesh(verts: np.ndarray, faces: np.ndarray, size: tuple[int, int] = (900, 700),
                azimuth: float = 35.0, elevation: float = 25.0, points: np.ndarray | None = None,
                cameras: np.ndarray | None = None, up: int = 2, color=(120, 190, 240),
                bg=(24, 26, 30)) -> np.ndarray:
    """Software-rendered BGR image of the mesh (painter's algorithm, Lambert
    shading), optionally with landmark points and camera positions. Used for
    report figures and the in-app 3D view; no OpenGL needed."""
    W, H = size
    img = np.full((H, W, 3), bg, np.uint8)
    if len(verts) == 0:
        return img
    R = _view_rotation(azimuth, elevation, up)
    allpts = [verts]
    if points is not None and len(points):
        allpts.append(np.asarray(points, np.float64).reshape(-1, 3))
    centre = np.concatenate(allpts).mean(axis=0) if points is None else np.asarray(verts).mean(axis=0)
    span = np.linalg.norm(verts - centre, axis=1).max() * 2.2 + 1e-9
    scale = min(W, H) / span

    def to_screen(p):
        q = (np.asarray(p, np.float64).reshape(-1, 3) - centre) @ R.T
        sx = W / 2 + q[:, 0] * scale
        sy = H / 2 - q[:, 1] * scale
        return np.column_stack([sx, sy]), q[:, 2]

    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    light = np.array([0.3, 0.5, 0.8])
    light /= np.linalg.norm(light)
    shade = np.clip(0.25 + 0.75 * np.clip((fn @ R.T) @ light, 0, 1), 0, 1)
    scr, depth = to_screen(verts)
    fd = depth[faces].mean(axis=1)
    order = np.argsort(fd)                          # far first
    col = np.array(color, np.float64)
    for k in order:
        tri = np.round(scr[faces[k]]).astype(np.int32)
        c = (col[::-1] * shade[k]).astype(int).tolist()   # BGR
        cv2.fillConvexPoly(img, tri, c, lineType=cv2.LINE_AA)
    if cameras is not None and len(cameras):
        cs, _ = to_screen(cameras)
        for (x, y) in cs:
            if 0 <= x < W and 0 <= y < H:
                cv2.drawMarker(img, (int(x), int(y)), (80, 80, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
    if points is not None and len(points):
        ps, _ = to_screen(points)
        for (x, y) in ps:
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(img, (int(x), int(y)), 4, (60, 230, 120), -1, cv2.LINE_AA)
    return img


def _view_rotation(azimuth: float, elevation: float, up: int = 2) -> np.ndarray:
    """Rotation taking world axes to screen axes (x right, y up, z toward the
    viewer) for a camera orbiting the `up` axis."""
    az, el = np.radians(azimuth), np.radians(elevation)
    # bring the chosen up axis to y
    perm = {2: np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64),
            1: np.eye(3), 0: np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], np.float64)}[up]
    ry = np.array([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]])
    rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    return rx @ ry @ perm
