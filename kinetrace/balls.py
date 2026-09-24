"""Ball markers: round objects (wand balls, reflective markers, a dropped ball)
tracked by SEGMENTING them, not by appearance matching.

Design: use SAM to segment every ball on the wand, fit a circle, and take the
centre of the circle as the tracked point for that ball.
A point tracker placed on a uniform ball has nothing to hold and a colour
threshold fails on a red shirt or a blue tarp; a SAM mask of a ball is robust,
and a circle fitted to its outline gives a sub-pixel centre even when a rod or
a hand covers part of it.

* One SAM video session tracks EVERY ball as its own object, simultaneously.
* SAM runs on a FIXED crop around the balls at native resolution: a 2704-px
  frame would be downscaled 2.6x to the model's working size and a 30-px ball
  would become 11 px. When a ball nears the crop edge the session restarts on
  a new crop, each ball re-prompted with a click + box at its extrapolated
  centre (a mask prompt is taken literally and lags a frame), so nothing is lost.
* Balls can be added on ANY frame, several at once: `segmenter.SegSession`
  works around a transformers bug that dropped all but the last object
  prompted on one frame (both SAM 2 and SAM 3 raised "maskmem_features in
  conditioning outputs cannot be empty"). Restarts happen only for the crop.
* Per frame and per ball: a RANSAC circle through the mask outline (robust
  to the rod the ball sits on, a finger, a shadow), quality = how much of the
  rim the outline follows x sqrt(the share of the mask inside the disc), the
  centre = the landmark, the confidence = quality x SAM's presence.
* The mean colour inside the disc must stay within 35 of the ball's own (a
  slow running reference from the first accepted fit), measured in OpenCV's
  8-bit Lab units (L* x 2.55, a* b* + 128), NOT true CIELAB: a lightness step
  weighs 2.55x a chroma step. SAM's mask of a red ball migrated onto a
  red-shirted man's FACE for 3000 frames of real footage with every shape
  guard satisfied, and the lit forehead differs from that ball by 54 in these
  units but by only 29 in true CIELAB (I60, sampled from the footage) - a
  true-CIELAB 35 would have let it through. The price: an abrupt lightness
  step of more than ~14 L* (a white ball entering deep shade) is rejected
  too, the ball is dropped and a click re-seeds it.
* A jump beyond 4 radii (and 5x the ball's recent step) is a swap onto another
  round thing and is rejected - also after missed frames, with an allowance
  that grows by a radius (or two recent steps) per missed frame (I56); two
  balls fitted to one circle keep only the one that was there before; a
  radius outside 0.65-1.45x the ball's slow running radius (or over a quarter
  of the crop) is not the ball.
* A ball whose circle is rejected for `lost_frames` frames in a row is dropped
  (its data stops there - the worker's auto-pause rules take over); a click on
  a later frame prompts it again. `drop_reason[obj]` says why: "lost", or
  "apart" when it was outside the one shared crop because the balls are too
  far apart to fit it together (I58) - clicking again cannot help then; each
  ball has to be tracked on its own run.

Pure numpy / cv2 apart from the segmenter it drives. No Qt. Used by the
tracking worker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from kinetrace.segmenter import Prompt, score_to_confidence

CROP = 960              # native px, the fixed SAM window (working size = native when <= 1024)
CROP_MARGIN = 110       # restart when an accepted ball centre comes this close to the crop edge
LOST_FRAMES = 12        # frames without an accepted circle before a ball is dropped
MIN_QUALITY = 0.45      # rim coverage x sqrt(mask share inside the disc): a half-hidden ball passes
R_RANGE = (2.5, 160.0)  # plausible ball radii, native px
R_JUMP = (0.5, 2.0)     # accepted radius relative to the ball's previous radius
JUMP_RADII = 4.0        # a centre moving more than this many radii (and 5x its recent step) in one frame is a swap
R_DRIFT = (0.65, 1.45)  # accepted radius relative to the ball's SLOW running radius: a smeared mask of a fast ball grew 20 -> 36 px
COLOUR_DE = 35.0        # max Lab distance (OpenCV 8-bit units, see guard_lab) between a fit's mean colour and the ball's (a red ball is not a face)
RIM_BIN_PX = 0.75       # rim-coverage bins are at least this much arc wide: 36 bins (as before) from r = 4.3 px up


@dataclass
class CircleFit:
    x: float            # native px
    y: float
    r: float
    quality: float      # IoU of the mask with the fitted disc, 0..1
    score: float        # SAM presence logit

    @property
    def confidence(self) -> float:
        return ball_confidence(self.quality, self.score)


def ball_confidence(quality: float, score: float) -> float:
    """quality x presence, on the point tracker's 0..1 scale."""
    return float(np.clip(quality, 0, 1) * score_to_confidence(score))


def _circumcircle(p1, p2, p3):
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy, float(np.hypot(ax - ux, ay - uy))


def _kasa(p):
    x, y = p[:, 0], p[:, 1]
    A = np.column_stack([x, y, np.ones(len(p))])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    return float(cx), float(cy), float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9)))


def fit_circle(mask: np.ndarray, trials: int = 240) -> tuple[float, float, float, float] | None:
    """A circle through the OUTLINE of a bool mask, robust to whatever else the
    mask includes (the rod the ball sits on, a finger, a shadow): RANSAC over
    three-point circles (inliers = outline points within a thin band of the
    rim, radius bounded by the mask's AREA), then least squares on the inliers.
    A plain least-squares fit put the centre 30 px off on a disc with a rod
    twice its diameter attached.

    Returns (cx, cy, r, quality) in mask pixels. quality = the fraction of the
    rim the outline actually follows (up to 36 angular bins) x sqrt(the share of the
    mask inside the disc): a full ball with a rod reads ~0.8-0.9, a half-hidden
    ball ~0.5, a blob that is not a circle < 0.4.
    None when the mask is empty or too small."""
    m8 = np.asarray(mask).astype(np.uint8)
    if m8.sum() < 12:
        return None
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    pts = max(cs, key=len).reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 8:
        return None
    # the ball's disc cannot hold more pixels than the whole mask: a large circle
    # grazing a straight rod edge and half the rim once out-voted the true rim
    # (radius 73 for a 22-px ball) - the area bound is what rules it out
    # (x 1.6 so a HALF-hidden ball, whose visible area is half its disc, still fits)
    r_max = max(3.0, 1.6 * float(np.sqrt(m8.sum() / np.pi)))
    rng = np.random.default_rng(n)
    best, best_n = None, -1
    w = max(6, min(n // 5, 40))             # local triplets: three points within ~120 deg of a 20-px ball's rim
    for t in range(trials):
        if t % 2 == 0:
            # local triplets land on ONE structure (the rim, not rim + rod), spread
            # over enough arc to fix the circle; with a thin rod twice the ball's
            # diameter attached, thirds-of-the-outline triplets never all hit the rim
            i = int(rng.integers(n))
            idx = [i, (i + w // 2) % n, (i + w) % n]
        else:
            idx = rng.choice(n, 3, replace=False).tolist()
        c = _circumcircle(pts[idx[0]], pts[idx[1]], pts[idx[2]])
        if c is None or not (2.0 <= c[2] <= r_max):
            continue
        res = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - c[2])
        n_in = int((res <= max(1.5, 0.06 * c[2])).sum())
        if n_in > best_n:
            best, best_n = c, n_in
    if best is None:
        cx, cy, r = _kasa(pts)
    else:
        cx, cy, r = best
    inl = pts
    for _ in range(2):
        res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        keep = res <= max(1.5, 0.08 * r)
        if keep.sum() < 8:
            break
        inl = pts[keep]
        cx, cy, r = _kasa(inl)
    # rim coverage: how much of the circle the outline follows. At most 36 bins,
    # each at least RIM_BIN_PX of arc (I59): the whole outline of a small ball has
    # fewer pixels than 36 bins, so a perfect r = 3 disc read 0.4 and could never
    # pass MIN_QUALITY although R_RANGE admits it and its centre was exact. Now a
    # perfect disc reads ~0.7 at every r below 4.3 (0.72+ above, unchanged). At
    # these sizes a partly hidden ball fits as a SMALLER whole circle, so the
    # tracker's radius guards (R_DRIFT), not this number, are what reject it.
    nb = int(np.clip(np.floor(2 * np.pi * r / RIM_BIN_PX), 8, 36))
    ang = np.arctan2(inl[:, 1] - cy, inl[:, 0] - cx)
    bins = np.unique(((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb)
    coverage = len(bins) / float(nb)
    # precision: the share of the mask that lies inside the disc (a rod or a hand
    # in the mask lowers it; a hidden part of the ball does not - the centre of a
    # half-visible ball is still exact, so it is not punished twice)
    yy, xx = np.nonzero(m8)
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r + 1.0) ** 2
    precision = float(inside.mean()) if len(xx) else 0.0
    q = coverage * float(np.sqrt(precision))
    return float(cx), float(cy), float(r), float(np.clip(q, 0, 1))


def guard_lab(rgb: np.ndarray) -> np.ndarray:
    """The colour guard's Lab image of a uint8 RGB picture: OpenCV's 8-bit Lab
    (L* x 2.55, a* b* + 128) as float32. Deliberately not true CIELAB (I60): the
    weight on lightness is what rejected the forehead on that real stretch
    (see the module notes), and COLOUR_DE was verified in these units."""
    return cv2.cvtColor(np.ascontiguousarray(rgb, np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def _disc_colour(lab: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray | None:
    """Mean Lab colour (`guard_lab` units) inside 0.7 r of the fitted centre
    (the rim's anti-aliasing and the rod's root are left out)."""
    H, W = lab.shape[:2]
    rr = max(1.0, 0.7 * r)
    x0, y0, x1, y1 = int(max(0, cx - rr)), int(max(0, cy - rr)), int(min(W, cx + rr + 1)), int(min(H, cy + rr + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr * rr
    if m.sum() < 4:
        return None
    return lab[y0:y1, x0:x1][m].mean(axis=0)


@dataclass
class BallPrompt:
    """The user's click(s) on one ball on one frame, native px. `radius` (if
    known) adds a box around the click; `mask` (native-res bool) re-seeds."""
    obj: int
    points: np.ndarray | None = None
    labels: np.ndarray | None = None
    radius: float | None = None
    mask: np.ndarray | None = None


@dataclass
class _State:
    xy: tuple[float, float]
    r: float | None
    miss: int = 0
    mask: np.ndarray | None = None      # crop-resolution bool mask of the last accepted frame
    fits: int = 0
    prev_xy: tuple[float, float] | None = None   # the frame before, for velocity extrapolation
    r_ref: float | None = None                   # slow running radius: a mask that inflates frame by frame is caught
    lab_ref: np.ndarray | None = None            # slow running mean Lab colour inside the disc (guard_lab units)


class BallTracker:
    """One SAM session over a fixed native-resolution crop, every ball its own
    object. Call `step` once per consecutive frame with the FULL frame."""

    def __init__(self, seg, native_size: tuple[int, int], crop: int = CROP, margin: int = CROP_MARGIN,
                 lost_frames: int = LOST_FRAMES, min_quality: float = MIN_QUALITY):
        self.seg = seg
        self.W, self.H = int(native_size[0]), int(native_size[1])
        self.crop_size = int(crop)
        self.margin = int(margin)
        self.lost_frames = int(lost_frames)
        self.min_quality = float(min_quality)
        self.session = None
        self.crop: tuple[int, int, int, int] | None = None      # x0, y0, x1, y1 native
        self.active: dict[int, _State] = {}
        self.n_restarts = 0
        self.last: dict[int, CircleFit] = {}
        # why each dropped ball was dropped: "lost" (SAM lost it) or "apart" (it lay
        # outside the one shared crop because the balls do not fit it together, I58)
        self.drop_reason: dict[int, str] = {}
        self._outside: set[int] = set()      # balls not given to SAM: outside the crop

    # ------------------------------------------------------------ helpers
    def has(self, obj: int) -> bool:
        return obj in self.active

    def drop(self, obj: int) -> None:
        self.active.pop(obj, None)
        self._outside.discard(obj)

    def _bbox_crop(self, centres) -> tuple[int, int, int, int]:
        xs = [c[0] for c in centres]
        ys = [c[1] for c in centres]
        cx = 0.5 * (min(xs) + max(xs))          # bbox centre: one far ball does not drag the others out
        cy = 0.5 * (min(ys) + max(ys))
        w, h = min(self.crop_size, self.W), min(self.crop_size, self.H)
        x0 = int(np.clip(round(cx - w / 2), 0, self.W - w))
        y0 = int(np.clip(round(cy - h / 2), 0, self.H - h))
        return x0, y0, x0 + w, y0 + h

    def _crop_for(self, centres) -> tuple[int, int, int, int]:
        crop = self._bbox_crop(centres)
        if not all(self._inside(crop, c) for c in centres):
            # balls farther apart than the crop (I58): the bbox centre can leave EVERY
            # ball outside it (two wand ends 1100 px apart). Hold the first one (the
            # newest click, else the oldest ball); the others are tracked if they fit
            crop = self._bbox_crop(centres[:1])
        return crop

    def fits_one_crop(self, centres=None) -> bool:
        """Can these centres (default: every active ball) share one crop with
        none of them in its restart margin? False = too far apart (I58)."""
        centres = [st.xy for st in self.active.values()] if centres is None else list(centres)
        if len(centres) < 2:
            return True
        crop = self._bbox_crop(centres)
        return not any(not self._inside(crop, c) or self._near_edge_in(crop, c) for c in centres)

    @staticmethod
    def _inside(crop, xy) -> bool:
        x0, y0, x1, y1 = crop
        return bool(x0 <= xy[0] < x1 and y0 <= xy[1] < y1)

    def _near_edge(self, xy) -> bool:
        return self._near_edge_in(self.crop, xy)

    def _near_edge_in(self, crop, xy) -> bool:
        x0, y0, x1, y1 = crop
        m = self.margin
        return bool((xy[0] - x0 < m and x0 > 0) or (x1 - xy[0] < m and x1 < self.W)
                    or (xy[1] - y0 < m and y0 > 0) or (y1 - xy[1] < m and y1 < self.H))

    def _click_prompt(self, obj: int, xy, r: float | None, x0: int, y0: int) -> Prompt:
        bx, by = xy[0] - x0, xy[1] - y0
        box = None
        if r is not None and r > 0:
            box = (bx - 1.1 * r, by - 1.1 * r, bx + 1.1 * r, by + 1.1 * r)
        return Prompt(obj, points=np.array([[bx, by]]), labels=np.array([1]), box=box)

    # --------------------------------------------------------------- step
    def step(self, rgb: np.ndarray, frame_idx: int, prompts: list[BallPrompt] | None = None
             ) -> dict[int, CircleFit]:
        """`rgb` = the full native frame. Returns the accepted circles this frame."""
        prompts = list(prompts or [])
        centres = []
        for p in prompts:
            if p.points is not None and len(p.points):
                pt = np.asarray(p.points, np.float64).reshape(-1, 2)
                centres.append((float(pt[:, 0].mean()), float(pt[:, 1].mean())))
            elif p.mask is not None and np.any(p.mask):
                ys, xs = np.nonzero(p.mask)
                centres.append((float(xs.mean()), float(ys.mean())))
        # the newest clicks FIRST: when the balls do not fit one crop, _crop_for holds
        # centres[0], so a (re)started session always has a prompt inside its picture
        centres += [st.xy for st in self.active.values()]
        if not centres:
            return {}
        # a session that skipped frames (nothing to track for a while) or lost every
        # ball is stale: SAM's streaming session must see consecutive frames
        stale = self.session is not None and (self.session.next_frame != frame_idx or not self.active)
        restart = self.session is None or stale or any(self._near_edge(c) for c in centres)
        if restart and self.session is not None and not stale:
            # balls too far apart for one crop would restart on EVERY frame (measured:
            # 191 restarts in 500 frames with a false third ball); only move the crop
            # when the new one actually frees the balls from the margin
            new_crop = self._crop_for(centres)
            if new_crop == self.crop or any(self._near_edge_in(new_crop, c) for c in centres):
                restart = False
        sam_prompts: list[Prompt] = []
        if restart:
            old_crop = self.crop
            self.crop = self._crop_for(centres)
            x0, y0, x1, y1 = self.crop
            self.session = self.seg.new_session(frame_idx, (x1 - x0, y1 - y0))
            self.n_restarts += 1
            # carry the tracked balls over with a click + box at their velocity-
            # extrapolated centre: a MASK prompt is taken literally by SAM (the
            # previous frame's mask becomes this frame's output) and the object
            # lagged for several frames after every restart - measured 2.4 px
            # median error on a 4 px/frame ball against 0.1 px without restarts
            for obj, st in self.active.items():
                if any(p.obj == obj for p in prompts):
                    continue                      # a fresh click on it replaces the guess
                if not self._inside(self.crop, st.xy):
                    self._outside.add(obj)        # never click SAM outside its picture (I58)
                    continue
                xy = st.xy
                if st.prev_xy is not None:
                    xy = (2 * st.xy[0] - st.prev_xy[0], 2 * st.xy[1] - st.prev_xy[1])
                    xy = (float(np.clip(xy[0], x0 + 1, x1 - 2)), float(np.clip(xy[1], y0 + 1, y1 - 2)))
                sam_prompts.append(self._click_prompt(obj, xy, (st.r or 0) * 1.25, x0, y0))
        x0, y0, x1, y1 = self.crop
        for p in prompts:
            # a click outside the crop (the balls are too far apart for one crop) is
            # never handed to SAM in coordinates outside its picture (I58): the ball
            # is registered, misses, and is dropped with drop_reason "apart"
            given = False
            if p.mask is not None and np.any(p.mask):
                m = np.asarray(p.mask, bool)[y0:y1, x0:x1]
                ys, xs = np.nonzero(m)
                if len(xs):
                    sam_prompts.append(Prompt(p.obj, mask=m))
                    given = True
                    xy = (float(xs.mean()) + x0, float(ys.mean()) + y0)
                    r = float(np.sqrt(len(xs) / np.pi))
                else:
                    fy, fx = np.nonzero(p.mask)
                    xy, r = (float(fx.mean()), float(fy.mean())), p.radius
            else:
                pt = np.asarray(p.points, np.float64).reshape(-1, 2)
                lab = np.asarray(p.labels if p.labels is not None else np.ones(len(pt)), int).ravel()
                pos = pt[lab > 0] if (lab > 0).any() else pt
                xy = (float(pos[:, 0].mean()), float(pos[:, 1].mean()))
                r = p.radius
                box = None
                if r is not None and r > 0:
                    box = (xy[0] - x0 - 1.1 * r, xy[1] - y0 - 1.1 * r, xy[0] - x0 + 1.1 * r, xy[1] - y0 + 1.1 * r)
                sel = np.array([self._inside(self.crop, q) for q in pt], bool)
                if self._inside(self.crop, xy) and (sel[lab > 0].any() if (lab > 0).any() else sel.any()):
                    sam_prompts.append(Prompt(p.obj, points=pt[sel] - [x0, y0], labels=lab[sel], box=box))
                    given = True
            self.active[p.obj] = _State(xy=xy, r=r, miss=0, mask=None)
            self.drop_reason.pop(p.obj, None)
            if given:
                self._outside.discard(p.obj)
            else:
                self._outside.add(p.obj)
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        if restart and not sam_prompts:
            # nothing inside the new crop to prompt (only when the balls do not fit
            # one crop, I58): no SAM step this frame, every ball misses
            self.session, fm = None, None
        else:
            fm = self.session.step(crop, frame_idx, sam_prompts or None)
        lab = None                      # Lab of the crop (guard_lab), computed once and only when a fit needs it
        out: dict[int, CircleFit] = {}
        for obj in list(self.active):
            st = self.active[obj]
            if fm is None or obj not in fm.obj_ids:
                st.miss += 1
                continue
            m = fm.mask(obj)
            score = float(fm.scores[fm.index(obj)])
            fit = fit_circle(m) if (score > 0 and m.any()) else None
            ok = fit is not None and fit[3] >= self.min_quality and R_RANGE[0] <= fit[2] <= R_RANGE[1]
            if ok and st.r is not None and st.fits > 0 and not (R_JUMP[0] * st.r <= fit[2] <= R_JUMP[1] * st.r):
                ok = False
            # a ball never fills a quarter of the crop, and its size cannot drift far from
            # what it was when it was clicked (SAM's mask of a ball that LEFT the picture
            # inflated into an 80-px blob over a few frames, each step within R_JUMP)
            if ok and fit[2] > 0.25 * min(x1 - x0, y1 - y0):
                ok = False
            if ok and st.r_ref is not None and not (R_DRIFT[0] * st.r_ref <= fit[2] <= R_DRIFT[1] * st.r_ref):
                ok = False
            # colour guard: SAM's mask migrated from a red ball onto a face (both roundish,
            # both reddish) for 3000 frames of real footage; the mean colour inside the disc
            # must stay close to the ball's own (guard_lab units, slow reference)
            col = None
            if ok:
                if lab is None:
                    lab = guard_lab(crop)
                col = _disc_colour(lab, fit[0], fit[1], fit[2])
                if col is not None and st.lab_ref is not None and float(np.linalg.norm(col - st.lab_ref)) > COLOUR_DE:
                    ok = False
            if ok and st.fits > 0:
                # identity swap guard: SAM latched onto ANOTHER round thing (measured: a
                # fast ball's mask jumped 130 px onto its neighbour with quality 1.0). It
                # stays on after missed frames (I56): it used to be skipped after ANY miss,
                # its own rejection included, so a persistent swap - or one after a frame
                # hidden by a hand - was accepted from the next frame at full confidence.
                # Per missed frame the allowance grows by a radius or two recent steps,
                # which covers a ball that kept moving while it was not seen.
                step = float(np.hypot(fit[0] + x0 - st.xy[0], fit[1] + y0 - st.xy[1]))
                recent = (float(np.hypot(st.xy[0] - st.prev_xy[0], st.xy[1] - st.prev_xy[1]))
                          if st.prev_xy is not None else 0.0)
                rr = float(st.r or fit[2])
                if step > max(JUMP_RADII * rr, 5.0 * recent) + st.miss * max(rr, 2.0 * recent):
                    ok = False
            if ok:
                cx, cy, r, q = fit
                st.prev_xy = st.xy if st.fits > 0 else None
                st.xy, st.r, st.miss, st.mask = (cx + x0, cy + y0), r, 0, m.copy()
                self._outside.discard(obj)
                st.r_ref = r if st.r_ref is None else 0.97 * st.r_ref + 0.03 * r
                if col is not None:
                    st.lab_ref = col if st.lab_ref is None else 0.95 * st.lab_ref + 0.05 * col
                st.fits += 1
                out[obj] = CircleFit(cx + x0, cy + y0, r, q, score)
            else:
                st.miss += 1
        # two balls fitted to one circle: the one that was already there keeps it,
        # the other missed this frame (SAM merged them or swapped onto its neighbour)
        objs = list(out)
        for i_a in range(len(objs)):
            for i_b in range(i_a + 1, len(objs)):
                a, b = objs[i_a], objs[i_b]
                if a not in out or b not in out:
                    continue
                fa, fb = out[a], out[b]
                if np.hypot(fa.x - fb.x, fa.y - fb.y) < 0.8 * min(fa.r, fb.r):
                    sa, sb = self.active[a], self.active[b]
                    da = np.hypot(fa.x - sa.prev_xy[0], fa.y - sa.prev_xy[1]) if sa.prev_xy is not None else 0.0
                    db = np.hypot(fb.x - sb.prev_xy[0], fb.y - sb.prev_xy[1]) if sb.prev_xy is not None else 0.0
                    loser = a if da > db else b
                    del out[loser]
                    st = self.active[loser]
                    st.miss += 1
                    if st.prev_xy is not None:
                        st.xy, st.prev_xy = st.prev_xy, None      # forget the merged position
        gone = [o for o, st in self.active.items() if st.miss > self.lost_frames]
        if gone:
            # "apart": the ball sat outside the shared crop, or in its margin while the
            # balls could not share one crop - re-clicking it cannot help (I58)
            fits_all = self.fits_one_crop()
            for obj in gone:
                st = self.active.pop(obj)
                apart = obj in self._outside or (self.crop is not None and not fits_all
                                                 and (not self._inside(self.crop, st.xy) or self._near_edge(st.xy)))
                self.drop_reason[obj] = "apart" if apart else "lost"
                self._outside.discard(obj)
        if not self.active:
            self.session = None          # nothing left to track: the next prompt starts afresh
        self.last = out
        return out
