"""Silhouette geometry: body midline and extremities from a binary mask.

Model-free landmarks for the parts appearance tracking cannot hold — thin
undulating tails, wing membranes, textureless bodies. Given an object mask (from
the segmentation layer) and optionally a head anchor (a tracked point), we:

* build a pixel graph over the largest connected component, with edge costs
  inversely weighted by the distance transform, so the cheapest path runs down
  the *middle* of the shape (a "centered geodesic" — a cheap medial axis);
* midline = cheapest path from the head anchor to the geodesically farthest
  pixel (the tail tip). Without an anchor we use the graph diameter and orient
  the thicker end as the head;
* extremities = outline pixels that stick out far beyond the local half-width
  measured from their nearest midline pixel (feet, hands, wing tips), the
  farthest one per protruding piece of the outline (one per limb).

Everything runs at an analysis scale chosen so the component has at most
MAX_NODES pixels (a 4K lizard silhouette resolves in ~10–30 ms).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

MAX_NODES = 30_000          # analysis-resolution pixel budget per component
MIN_COMPONENT_PX = 12       # smaller silhouettes carry no usable geometry
EXTREMITY_MAX = 8           # feet/hands/wing tips returned at most
SMOOTH_WIN = 5              # moving-average window on the midline path
ANCHOR_COMPONENT_FRAC = 0.25  # the anchor's own component is used only if at least this share of the largest


@dataclass
class Midline:
    """All coordinates in the pixel frame of the mask passed to `midline()`."""
    path: np.ndarray                # (M, 2) float32, head -> tip
    arc: np.ndarray                 # (M,) cumulative arc length in px
    width: np.ndarray               # (M,) local full width (2 x distance transform)
    extremities: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    scale: float = 1.0              # analysis scale used (for diagnostics)

    @property
    def length(self) -> float:
        return float(self.arc[-1]) if len(self.arc) else 0.0

    @property
    def head(self) -> np.ndarray:
        return self.path[0]

    @property
    def tip(self) -> np.ndarray:
        return self.path[-1]

    def sample(self, fractions) -> np.ndarray:
        """Points at the given fractions of the arc length (0 = head, 1 = tip).
        A fraction outside [0, 1] gives NaN, not the nearest end (I62: a spec
        'midline:50', meant as 50 %, exported a copy of the tail tip)."""
        f = np.asarray(fractions, np.float64).reshape(-1)
        bad = ~((f >= -1e-6) & (f <= 1.0 + 1e-6))
        fr = np.clip(np.nan_to_num(f), 0.0, 1.0) * self.length
        x = np.interp(fr, self.arc, self.path[:, 0])
        y = np.interp(fr, self.arc, self.path[:, 1])
        out = np.stack([x, y], axis=1).astype(np.float32)
        out[bad] = np.nan
        return out


def resample(path: np.ndarray, n: int = 32) -> np.ndarray:
    """`n` points evenly spaced along the polyline `path` (M, 2)."""
    path = np.asarray(path, np.float32)
    if len(path) < 2:
        return np.repeat(path[:1], n, axis=0) if len(path) else np.zeros((0, 2), np.float32)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.linspace(0.0, arc[-1], n)
    return np.stack([np.interp(want, arc, path[:, 0]), np.interp(want, arc, path[:, 1])],
                    axis=1).astype(np.float32)


def oriented(ml: "Midline", prev_head: np.ndarray | None) -> "Midline":
    """Keep head/tip labelling consistent over time when there is no anchor:
    flip the midline if its head end is closer to where the tip was."""
    if prev_head is None or len(ml.path) < 2:
        return ml
    if np.linalg.norm(ml.head - prev_head) > np.linalg.norm(ml.tip - prev_head):
        path = ml.path[::-1].copy()
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
        return Midline(path, arc, ml.width[::-1].copy(), ml.extremities, ml.scale)
    return ml


def extremity_roles(ml: "Midline") -> dict[str, np.ndarray]:
    """Name the extremities by where they attach to the midline.

    Keys: "L"/"R" (the farthest-protruding extremity on each side, e.g. wing
    or patagium tips) and "FL"/"FR"/"HL"/"HR" (fore/hind on each side, e.g.
    feet: of the two extremities sticking out farthest on a side, the first
    along the body is fore, the second hind). Sides are the animal's own
    sides in a dorsal (top-down) view; a ventral view (camera below) mirrors
    them.
    """
    out: dict[str, np.ndarray] = {}
    ext = ml.extremities
    if len(ext) == 0 or len(ml.path) < 3:
        return out
    path, arc, L = ml.path, ml.arc, ml.length
    d = np.linalg.norm(ext[:, None, :] - path[None, :, :], axis=2)
    idx = np.argmin(d, axis=1)
    per_side: dict[str, list[tuple[float, float, np.ndarray]]] = {"L": [], "R": []}
    for e, i in zip(ext, idx):
        i0, i1 = max(i - 2, 0), min(i + 2, len(path) - 1)
        t = path[i1] - path[i0]
        o = e - path[i]
        cross = float(t[0] * o[1] - t[1] * o[0])
        lateral = abs(cross) / max(np.linalg.norm(t), 1e-6)
        # image rows grow DOWN: with the tangent pointing head -> tail, a positive
        # cross product is the animal's LEFT seen from above (head up the screen,
        # its right side is screen right). This read "R" and mirrored every
        # ext:L/R/FL/FR/HL/HR for dorsal footage (I54).
        per_side["L" if cross > 0 else "R"].append((float(arc[i]), lateral, e))
    split_arc: dict[str, float] = {}
    for side, items in per_side.items():
        if not items:
            continue
        out[side] = max(items, key=lambda it: it[1])[2]
        # fore / hind = the two that stick out farthest on this side, in order
        # along the body (I55): a third, smaller bump is not a foot
        two = sorted(sorted(items, key=lambda it: -it[1])[:2], key=lambda it: it[0])
        if len(two) == 2:
            out["F" + side], out["H" + side] = two[0][2], two[1][2]
            split_arc[side] = 0.5 * (two[0][0] + two[1][0])
    for side, items in per_side.items():
        if len(items) == 1:
            # one foot on this side: fore or hind by the other side's split when
            # that side shows both, else by where it attaches (front third = fore)
            split = split_arc.get("R" if side == "L" else "L", 0.35 * L)
            out[("F" if items[0][0] < split else "H") + side] = items[0][2]
    return out


def analysis_scale(area_px: int) -> float:
    """Downscale factor keeping a component under the node budget (never upscales)."""
    return float(min(1.0, np.sqrt(MAX_NODES / max(int(area_px), 1))))


def largest_component(mask_u8: np.ndarray, anchor: tuple[float, float] | None = None) -> np.ndarray | None:
    """8-connected component containing `anchor` (nearest one if the anchor is
    off-mask) when that is a real part of the body (>= ANCHOR_COMPONENT_FRAC of
    the largest), else the largest. Returns a uint8 0/1 image or None if empty."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    pick = 1 + int(np.argmax(areas))
    if anchor is not None:
        ax, ay = int(round(anchor[0])), int(round(anchor[1]))
        h, w = lab.shape
        if 0 <= ax < w and 0 <= ay < h and lab[ay, ax] > 0:
            at = int(lab[ay, ax])
        else:  # nearest component pixel to the anchor
            ys, xs = np.nonzero(lab)
            d = (xs - anchor[0]) ** 2 + (ys - anchor[1]) ** 2
            k = int(np.argmin(d))
            at = int(lab[ys[k], xs[k]])
        # a speck SAM left by the snout (or that the head point was snapped onto)
        # is not the body: taking it made midline() return None, or a 3-px
        # "midline", on a frame whose silhouette was intact (I63). The largest
        # component then serves, and midline() starts it nearest the anchor.
        if areas[at - 1] >= max(MIN_COMPONENT_PX, ANCHOR_COMPONENT_FRAC * areas[pick - 1]):
            pick = at
    comp = (lab == pick).astype(np.uint8)
    return comp if int(areas[pick - 1]) >= MIN_COMPONENT_PX else None


def _pixel_graph(comp: np.ndarray):
    """Two sparse undirected graphs over the component pixels sharing one edge
    set: `centered` costs step x mean(1/dt) of the edge ends (paths are pulled
    onto the medial axis); `plain` costs the step length (physical distance,
    used to find the longest path when no anchor says where the head is)."""
    h, w = comp.shape
    ys, xs = np.nonzero(comp)
    n = len(xs)
    idx = np.full((h, w), -1, np.int64)
    idx[ys, xs] = np.arange(n)
    dt = cv2.distanceTransform(comp, cv2.DIST_L2, 3).astype(np.float32)
    inv = 1.0 / np.maximum(dt[ys, xs], 0.5)
    rows, cols, vals, plain = [], [], [], []
    for dy, dx, step in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        v = np.full(n, -1, np.int64)
        v[ok] = idx[ny[ok], nx[ok]]
        ok &= v >= 0
        u = np.nonzero(ok)[0]
        rows.append(u)
        cols.append(v[u])
        vals.append(step * 0.5 * (inv[u] + inv[v[u]]))
        plain.append(np.full(len(u), step))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    centered = coo_matrix((np.concatenate(vals).astype(np.float64), (rows, cols)), shape=(n, n)).tocsr()
    plain_g = coo_matrix((np.concatenate(plain).astype(np.float64), (rows, cols)), shape=(n, n)).tocsr()
    return centered, plain_g, xs, ys, dt


def _backtrack(pred: np.ndarray, src: int, dst: int) -> np.ndarray:
    out = [dst]
    cur = dst
    while cur != src and cur >= 0:
        cur = int(pred[cur])
        out.append(cur)
    return np.asarray(out[::-1], np.int64)


def _smooth(path: np.ndarray, win: int = SMOOTH_WIN) -> np.ndarray:
    if len(path) < win or win < 3:
        return path
    k = np.ones(win) / win
    pad = win // 2
    padded = np.pad(path, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, i], k, mode="valid") for i in range(2)], axis=1).astype(np.float32)


def midline(mask: np.ndarray, anchor: tuple[float, float] | None = None,
            want_extremities: bool = True) -> Midline | None:
    """Head -> tail-tip midline of the silhouette in `mask` (bool or 0/1 uint8).

    anchor: (x, y) pixel of the head end (e.g. a tracked head point). Without it,
    the thicker end of the longest internal path is taken as the head.
    """
    m8 = (np.asarray(mask) > 0).astype(np.uint8)
    area = int(m8.sum())
    if area < MIN_COMPONENT_PX:
        return None
    s = analysis_scale(area)
    sxy = np.ones(2)
    if s < 1.0:
        h, w = m8.shape
        sw, sh = max(2, int(round(w * s))), max(2, int(round(h * s)))
        # pixel-centre sampling and the TRUE per-axis factors (I57): INTER_NEAREST
        # samples source pixel floor(i / s), a cell corner, and the rounded size
        # makes each axis's factor differ from s; mapped back with the centre
        # formula that biased every derived landmark of a large silhouette
        # ~0.9 working px down-right (3.4 native px at 4K)
        sxy = np.array([sw / w, sh / h])
        small = cv2.resize(m8, (sw, sh), interpolation=cv2.INTER_NEAREST_EXACT)
        # recover thin structures the nearest-neighbour shrink may have severed
        small = cv2.dilate(small, np.ones((3, 3), np.uint8)) & cv2.resize(
            cv2.dilate(m8, np.ones((3, 3), np.uint8)), (sw, sh), interpolation=cv2.INTER_NEAREST_EXACT)
        anc = None if anchor is None else ((anchor[0] + 0.5) * sxy[0] - 0.5, (anchor[1] + 0.5) * sxy[1] - 0.5)
    else:
        small, anc = m8, anchor
    comp = largest_component(small, anc)
    if comp is None:
        return None
    g, g_plain, xs, ys, dt = _pixel_graph(comp)
    n = len(xs)
    if n < 2:
        return None

    if anc is not None:
        # the tip is the costliest point to reach down the centred graph: thin
        # appendages are expensive per pixel, so a long thin tail wins over a leg
        src = int(np.argmin((xs - anc[0]) ** 2 + (ys - anc[1]) ** 2))
        d, pred = dijkstra(g, directed=False, indices=src, return_predecessors=True)
        d[~np.isfinite(d)] = -1
        dst = int(np.argmax(d))
        nodes = _backtrack(pred, src, dst)
    else:
        # no head known: take the physically longest internal path (graph
        # diameter in pixel length), then thread it down the middle
        d0 = dijkstra(g_plain, directed=False, indices=0)
        d0[~np.isfinite(d0)] = -1
        a = int(np.argmax(d0))
        da = dijkstra(g_plain, directed=False, indices=a)
        da[~np.isfinite(da)] = -1
        b = int(np.argmax(da))
        d, pred = dijkstra(g, directed=False, indices=a, return_predecessors=True)
        nodes = _backtrack(pred, a, b)
        k = max(3, len(nodes) // 6)   # thicker end first (= head)
        if dt[ys[nodes[:k]], xs[nodes[:k]]].mean() < dt[ys[nodes[-k:]], xs[nodes[-k:]]].mean():
            nodes = nodes[::-1]

    path_s = np.stack([xs[nodes], ys[nodes]], axis=1).astype(np.float32)
    width_s = 2.0 * dt[ys[nodes], xs[nodes]]

    ext_s = np.zeros((0, 2), np.float32)
    if want_extremities and len(nodes) >= 3:
        dm, _, srcs = dijkstra(g, directed=False, indices=nodes, min_only=True, return_predecessors=True)
        # dm is geodesic *cost* (dt-weighted); convert to a plain pixel distance
        # by re-running a cheap euclidean proxy: distance to the nearest midline
        # node in pixels, which is what "sticks out" means to a user
        px = np.stack([xs, ys], axis=1).astype(np.float32)
        near = px[srcs]
        dpix = np.linalg.norm(px - near, axis=1)
        half_w = dt[ys[srcs], xs[srcs]]
        boundary = (comp - cv2.erode(comp, np.ones((3, 3), np.uint8))).astype(bool)[ys, xs]
        # an extremity protrudes well beyond the local half-width AND by a
        # length that matters at body scale (outline wobble on a bent tail
        # protrudes by a few pixels; a foot or wing tip by a fraction of the body)
        body_len = float(np.sum(np.linalg.norm(np.diff(path_s, axis=0), axis=1)))
        protrusion = dpix - half_w
        cand = np.nonzero(boundary & (dpix > 1.4 * half_w + 2.0)
                          & (protrusion > max(2.0, 0.08 * body_len)))[0]
        if len(cand):
            # ONE extremity per protrusion (I55): spacing alone kept two or more
            # outline points along any limb longer than ~13 % of the body, and the
            # hind-foot role then landed on the limb's shaft or on the fore limb.
            # The pixels beyond the body core form one connected piece per limb
            # (foot, wing tip); each piece is represented by its farthest point.
            outer = dpix > 1.4 * half_w + 2.0
            out_img = np.zeros_like(comp)
            out_img[ys[outer], xs[outer]] = 1
            piece = cv2.connectedComponents(out_img, connectivity=8)[1][ys, xs]
            done: set[int] = set()
            order = cand[np.argsort(-dpix[cand])]
            keep: list[int] = []
            sep = max(6.0, 0.05 * body_len)
            ends = np.array([path_s[0], path_s[-1]])
            for c in order:
                if int(piece[c]) in done:
                    continue
                done.add(int(piece[c]))
                p = px[c]
                if np.min(np.linalg.norm(ends - p, axis=1)) < sep:
                    continue
                if keep and np.min(np.linalg.norm(px[keep] - p, axis=1)) < sep:
                    continue
                keep.append(int(c))
                if len(keep) >= EXTREMITY_MAX:
                    break
            ext_s = px[keep] if keep else ext_s

    path_s = _smooth(path_s)
    if s < 1.0:  # back to the caller's pixel frame (cell centres, per axis)
        path = (path_s + 0.5) / sxy - 0.5
        ext = (ext_s + 0.5) / sxy - 0.5 if len(ext_s) else ext_s
        width = width_s / float(sxy.mean())
    else:
        path, ext, width = path_s, ext_s, width_s
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.zeros(0)
    arc = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
    return Midline(path.astype(np.float32), arc, width.astype(np.float32),
                   ext.astype(np.float32), s)
