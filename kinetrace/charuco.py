"""ChArUco boards: the board, its printable page, finding it in a picture, and the frame
rules a ChArUco calibration uses (I321 lens, I322 cameras, I323 Argus files).

A ChArUco board is a checkerboard with an ArUco marker in every white square. Each inner
corner therefore has an id read from its neighbouring markers: a board half out of the
picture, or partly covered by a hand, still gives corners that are known by name. A plain
checkerboard has to be seen whole and its corner order guessed (`lens.orient_board`).

Credit
------
The ChArUco workflow follows Argus (argus_gui, utils/calibrate_charuco.py at commit
f7a8f0c, https://github.com/backyardbiomech/argus_gui; Jackson, B.E., Evangelista, D.J.,
Ray, D.D. and Hedrick, T.L. (2016), "3D for the people: multi-camera motion capture in the
field with consumer-grade cameras and open source software", Biology Open 5(9),
1334-1342, doi:10.1242/bio.018713), adapted with the authors' permission. No Argus code
is copied: detection is OpenCV's (`cv2.aruco.CharucoDetector`), the fits are Kinetrace's
own (`lens.calibrate_lens`, `wand.calibrate_wand`). The rules taken from Argus, named
where they are used:
  R1  a camera's frame counts when it sees at least 4 ChArUco corners;
  R2  a frame pairs cameras when they share at least 6 corner ids;
  R3  at most 100 candidate frames per camera, ranked by n + 10 x (sx / W + sy / H) / 2
      (n corners, sx / sy the population spread of their x / y), then back in time order;
  R4  the "wand" of a frame = the first and last corner of a long board edge (both long
      edges tried, the first seen by at least two cameras), its length (long-side inner
      corners - 1) x square; every other corner seen by at least two cameras is an extra
      point;
  R5  frames sampled evenly over the chosen stretch, 50 by default.
Argus chose R3's frames per camera, which mis-pairs cameras in its pairwise stereo step;
here R3 only picks each camera's OWN lens boards, and the cameras are paired on the
frames they share (R2, R4).

Conventions: 0-based pixels, top-left origin (Kinetrace's own). Corner id k sits in row
k // cols, column k % cols of the inner corners (cols = squares across - 1): the same
row-major order as `lens._object_points`, so a ChArUco view is a checkerboard view of the
whole inner grid with NaN where a corner was not seen.
"""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass

import cv2
import numpy as np

# the dictionaries a printed board can use, by OpenCV's names (the ArUco families; AprilTag
# and the original ArUco set are left out: ChArUco boards are not made with them in practice)
DICTIONARIES = tuple(f"DICT_{b}X{b}_{n}" for b in (4, 5, 6, 7) for n in (50, 100, 250, 1000))
DEFAULT_DICTIONARY = "DICT_4X4_50"
# 7 x 5 squares of 30 mm with 22 mm markers: 210 x 150 mm, fits A4 and Letter with a margin
DEFAULT_SQUARES = (7, 5)
DEFAULT_SQUARE_MM = 30.0
DEFAULT_MARKER_MM = 22.0
MIN_CORNERS_SEEN = 4            # R1 (Argus)
MIN_SHARED_IDS = 6              # R2 (Argus)
MAX_CANDIDATES = 100            # R3 (Argus)
DEFAULT_FRAMES = 50             # R5 (Argus --frames)
MIN_CORNERS_LENS = 6            # a lens board needs a pose: 6 corners not all on one line


@dataclass(frozen=True)
class BoardSpec:
    """The printed board: squares across and down, the square and marker side in metres,
    the ArUco dictionary by name, and `legacy` = the corner layout OpenCV used before 4.6
    (a board with an even number of rows printed by an older program)."""
    squares_x: int = DEFAULT_SQUARES[0]
    squares_y: int = DEFAULT_SQUARES[1]
    square_m: float = DEFAULT_SQUARE_MM / 1000.0
    marker_m: float = DEFAULT_MARKER_MM / 1000.0
    dictionary: str = DEFAULT_DICTIONARY
    legacy: bool = False

    @property
    def pattern(self) -> tuple[int, int]:
        """Inner corners (columns, rows): what `lens` calls the pattern."""
        return (int(self.squares_x) - 1, int(self.squares_y) - 1)

    @property
    def n_corners(self) -> int:
        c, r = self.pattern
        return c * r

    def problem(self) -> str:
        """Why this board cannot exist, in a sentence ("" = it can)."""
        if self.squares_x < 3 or self.squares_y < 3 or self.n_corners < MIN_CORNERS_LENS:
            return (f"a ChArUco board needs at least 3 squares across and 3 down, and {MIN_CORNERS_LENS} inner "
                    "corners (4 x 3 squares or more)")
        if not (0.0 < self.marker_m < self.square_m):
            return "the marker must be smaller than the square it sits in"
        if self.dictionary not in DICTIONARIES:
            return f"{self.dictionary} is not one of the ArUco dictionaries Kinetrace knows"
        need = (self.squares_x * self.squares_y) // 2
        if need > int(self.dictionary.rsplit("_", 1)[1]):
            return (f"a {self.squares_x} x {self.squares_y} board needs {need} markers; {self.dictionary} "
                    "has fewer - pick a dictionary with more markers")
        return ""

    def words(self) -> str:
        return (f"ChArUco {self.squares_x} x {self.squares_y} squares of {1000 * self.square_m:g} mm, markers "
                f"{1000 * self.marker_m:g} mm, {self.dictionary}" + (", older OpenCV layout" if self.legacy else ""))

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "BoardSpec":
        return BoardSpec(int(d["squares_x"]), int(d["squares_y"]), float(d["square_m"]), float(d["marker_m"]),
                         str(d["dictionary"]), bool(d.get("legacy", False)))


@functools.lru_cache(maxsize=8)
def make_board(spec: BoardSpec):
    """The OpenCV board for `spec` (cached: one per spec)."""
    bad = spec.problem()
    if bad:
        raise ValueError(bad)
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    board = cv2.aruco.CharucoBoard((int(spec.squares_x), int(spec.squares_y)), float(spec.square_m),
                                   float(spec.marker_m), d)
    board.setLegacyPattern(bool(spec.legacy))
    return board


@functools.lru_cache(maxsize=8)
def _detector(spec: BoardSpec):
    return cv2.aruco.CharucoDetector(make_board(spec))


# ---------------------------------------------------------------- printing


def board_image(spec: BoardSpec, square_px: int = 100, margin_px: int = 100) -> np.ndarray:
    """The board as a uint8 grayscale picture, white margin."""
    w = spec.squares_x * square_px + 2 * margin_px
    h = spec.squares_y * square_px + 2 * margin_px
    return make_board(spec).generateImage((w, h), marginSize=margin_px, borderBits=1)


def save_board_png(path, spec: BoardSpec, dpi: int = 300) -> tuple[float, float]:
    """Write the board at `dpi` so that printing at 100 % gives the typed square size;
    returns the printed page size (width_mm, height_mm). Same file path handling as the
    checkerboard (`lens.save_checkerboard_png`, I222)."""
    from pathlib import Path

    from kinetrace.lens import _png_with_dpi
    px_per_mm = dpi / 25.4
    sq = int(round(1000.0 * spec.square_m * px_per_mm))
    margin = int(round(12.0 * px_per_mm))
    img = board_image(spec, sq, margin)
    txt = (f"Kinetrace {spec.words()}. Print at 100% / 'actual size' and check a square with a ruler.")
    cv2.putText(img, txt, (margin, img.shape[0] - margin // 3), cv2.FONT_HERSHEY_SIMPLEX,
                0.9 * dpi / 300.0, 0, max(1, dpi // 150), cv2.LINE_AA)
    p = Path(path)
    if p.suffix.lower() != ".png":
        p = p.with_suffix(".png")
    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise OSError(f"could not encode the ChArUco board as a PNG ({p})")
    p.write_bytes(_png_with_dpi(bytes(enc), dpi))
    return img.shape[1] / px_per_mm, img.shape[0] / px_per_mm


# ---------------------------------------------------------------- detection


def detect(gray: np.ndarray, spec: BoardSpec) -> tuple[np.ndarray, np.ndarray] | None:
    """The board's corners in one picture: ((M, 2) float64 sub-pixel corners, (M,) int ids),
    sorted by id, or None when fewer than R1's 4 corners are seen."""
    cc, ci, _mc, _mi = _detector(spec).detectBoard(gray)
    if cc is None or ci is None or len(ci) < MIN_CORNERS_SEEN:
        return None
    ids = np.asarray(ci, np.int64).ravel()
    pts = np.asarray(cc, np.float64).reshape(-1, 2)
    order = np.argsort(ids, kind="stable")
    return pts[order], ids[order]


def padded(corners: np.ndarray, ids: np.ndarray, spec: BoardSpec) -> np.ndarray:
    """(N, 2) for the whole inner grid, NaN where a corner was not seen: the view `lens`
    fits from."""
    out = np.full((spec.n_corners, 2), np.nan)
    out[np.asarray(ids, np.int64)] = np.asarray(corners, np.float64).reshape(-1, 2)
    return out


def seen_ids(view: np.ndarray) -> np.ndarray:
    """The ids of a padded view's corners that were seen."""
    return np.nonzero(np.isfinite(np.asarray(view, np.float64)).all(axis=1))[0]


def usable_for_lens(view: np.ndarray, spec: BoardSpec) -> bool:
    """A padded view a lens fit can use: at least 6 corners, not all on one row or column
    of the board (one line of corners has no pose)."""
    ids = seen_ids(view)
    if len(ids) < MIN_CORNERS_LENS:
        return False
    cols = spec.pattern[0]
    return len(set((ids // cols).tolist())) >= 2 and len(set((ids % cols).tolist())) >= 2


def object_points(spec: BoardSpec, ids) -> np.ndarray:
    """(M, 3) board coordinates in metres of corner `ids` (corner 0 at the origin, x along
    the rows, y down the columns, z = 0): `lens._object_points` rows for those ids."""
    ids = np.asarray(ids, np.int64)
    cols = spec.pattern[0]
    out = np.zeros((len(ids), 3), np.float64)
    out[:, 0] = (ids % cols) * float(spec.square_m)
    out[:, 1] = (ids // cols) * float(spec.square_m)
    return out


def guess_legacy(grays: list[np.ndarray], spec: BoardSpec) -> tuple[BoardSpec, str]:
    """The layout (current or OpenCV-before-4.6) that finds more corners in these pictures,
    and a sentence when it is not the one typed ("" otherwise)."""
    other = BoardSpec(spec.squares_x, spec.squares_y, spec.square_m, spec.marker_m, spec.dictionary,
                      not spec.legacy)

    def count(s):
        n = 0
        for g in grays:
            d = detect(g, s)
            n += 0 if d is None else len(d[1])
        return n
    a, b = count(spec), count(other)
    if b > a:
        return other, ("The board's corners were found with OpenCV's OLDER ChArUco layout (boards printed before "
                       f"OpenCV 4.6 with an even number of rows): {b} corners against {a}. That layout is used.")
    return spec, ""


# ---------------------------------------------------------------- frame rules


def quality(view: np.ndarray, size: tuple[int, int]) -> float:
    """R3 (Argus): n + 10 x (sx / W + sy / H) / 2 for one view, n = corners seen, sx / sy the
    population standard deviation of their x / y."""
    p = np.asarray(view, np.float64).reshape(-1, 2)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) == 0:
        return 0.0
    sx, sy = p.std(axis=0)
    return float(len(p) + 10.0 * (sx / size[0] + sy / size[1]) / 2.0)


def best_frames(views: list[np.ndarray], size: tuple[int, int], keep: int = MAX_CANDIDATES) -> list[int]:
    """R3 (Argus): the indices of the `keep` best views by `quality`, in their original
    (time) order."""
    if len(views) <= keep:
        return list(range(len(views)))
    q = np.array([quality(v, size) for v in views])
    top = np.argsort(-q, kind="stable")[:keep]
    return sorted(int(i) for i in top)


def long_edge_pairs(spec: BoardSpec) -> list[tuple[int, int]]:
    """R4 (Argus): the candidate wand ends - the first and last corner id along each long
    edge of the inner grid, in the order they are tried."""
    cols, rows = spec.pattern
    if cols >= rows:               # wide board: the top and bottom rows
        return [(0, cols - 1), ((rows - 1) * cols, rows * cols - 1)]
    return [(0, (rows - 1) * cols), (cols - 1, rows * cols - 1)]   # tall board: the left and right columns


def wand_length_m(spec: BoardSpec) -> float:
    """R4: the distance between a pair's two corners."""
    cols, rows = spec.pattern
    return (max(cols, rows) - 1) * float(spec.square_m)


@dataclass
class RigScan:
    """Board detections in every camera at the same reference instants.

    `views[f][c]` = camera c's padded (N, 2) view at reference instant `ref_frames[f]`
    (all NaN where the board was not found or fewer than R1's corners were seen);
    `local_frames[f][c]` camera c's frame nearest that instant (between two frames both were read
    and the corners interpolated, see `scan_rig`); `failed[c]` = the first frame of camera c that
    did not decode (None = every one did) - reported, never skipped in silence; `cameras[c]` =
    the project camera of scan camera c; `spec` = the board as found (its layout included), `note`
    a sentence when the layout is not the one typed."""
    spec: BoardSpec
    sizes: list[tuple[int, int]]
    names: list[str]
    ref_frames: list[int]
    local_frames: list[list[float]]
    views: list[list[np.ndarray]]
    failed: list[int | None]
    note: str = ""
    cameras: list[int] | None = None
    # per scan camera {local frame: padded view} of every picture READ and the board found in it - the
    # real detections; `views` may interpolate two of them at a fractional offset
    frame_views: list[dict] | None = None

    def seen(self, c: int) -> int:
        """Frames in which camera c saw the board (R1)."""
        return sum(1 for row in self.views if np.isfinite(row[c]).all(axis=1).sum() >= MIN_CORNERS_SEEN)

    def camera_views(self, c: int) -> tuple[list[np.ndarray], list[int]]:
        """Camera c's lens boards (views usable for a lens fit, R1 + a pose) and the frames they are:
        the pictures actually read (`frame_views`), never an interpolated instant - a lens is fitted on
        corners one picture shows, and the frames it records can be detected again."""
        vs, fr = [], []
        src = self.frame_views[c] if self.frame_views is not None else {}
        for k in sorted(src):
            if usable_for_lens(src[k], self.spec):
                vs.append(src[k])
                fr.append(int(k))
        return vs, fr


def _interp_views(a: np.ndarray | None, b: np.ndarray | None, w: float, n: int) -> np.ndarray:
    """A padded view at a fraction `w` of the way from frame view `a` to `b`: corners seen in
    both, linearly (what Reconstruct does between two frames, I253); NaN elsewhere."""
    out = np.full((n, 2), np.nan)
    if a is None or b is None:
        return out
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    out[ok] = (1.0 - w) * a[ok] + w * b[ok]
    return out


def shared_window(project, offsets=None, views=None) -> tuple[int, int]:
    """The integer reference instants every camera `views` (None = all) covers at `offsets`
    ((0, -1) when they share none): the stretch a rig scan can look at."""
    from kinetrace.wanddata import RunView
    return _window(RunView(project, views, offsets))


def scan_rig(project, spec: BoardSpec, offsets=None, first: int | None = None, last: int | None = None,
             n: int = DEFAULT_FRAMES, progress=None, should_cancel=None, views=None) -> RigScan:
    """Find the board in the cameras `views` (None = every camera) of `project` at `n` reference
    instants spread evenly over first..last (R5; default: the stretch they all cover) at `offsets`
    (None = the project's own). Each camera's picture at an instant is its frame there; when the
    instant falls between two frames (a fractional offset) both are read and the corners seen in
    both are interpolated, so a moving board is not seen half a frame early or late. The board's
    layout (current or older OpenCV) is first read from the first camera (`guess_legacy`; the scan's
    `spec` and `note` say which). Opens its own captures: never on the GUI thread. Frames are read in
    increasing order per camera. The scan's cameras are `views`, in that order."""
    from kinetrace.calib import even_indices
    from kinetrace.project import half_up
    from kinetrace.video_source import FrameReader
    from kinetrace.wanddata import RunView

    rv = RunView(project, views, offsets)
    C = rv.n_views
    lo, hi = _window(rv)
    if first is not None:
        lo = max(lo, int(first))
    if last is not None:
        hi = min(hi, int(last))
    if hi < lo:
        raise ValueError("the cameras share no stretch of time at these offsets: set the offsets first "
                         "(3D -> Sync), or widen the frames to look at")
    count = hi - lo + 1
    refs = (lo + even_indices(count, min(int(n), count))).tolist()
    # the printed layout, from up to 8 of the first camera's pictures at these instants
    note = ""
    reader = FrameReader(rv.sessions[0].video_path, 0)
    try:
        grays = []
        for t in [refs[i] for i in even_indices(len(refs), min(8, len(refs)))]:
            k = int(half_up(float(rv.local_frame(0, t))))
            if 0 <= k < rv.sessions[0].n_frames:
                reader.reset(k)
                bgr = reader.read(k)
                if bgr is not None:
                    grays.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    finally:
        reader.release()
    if grays:
        spec, note = guess_legacy(grays, spec)
    N = spec.n_corners
    sizes = [(int(s.width), int(s.height)) for s in rv.sessions]
    out_views = [[np.full((N, 2), np.nan) for _ in range(C)] for _ in refs]
    local = [[float("nan")] * C for _ in refs]
    frame_views: list[dict] = [{} for _ in range(C)]
    failed: list[int | None] = [None] * C
    done, total = 0, len(refs) * C
    for c in range(C):
        reader = None
        try:
            reader = FrameReader(rv.sessions[c].video_path, 0)
            last_read = -1
            cache: dict[int, np.ndarray | None] = {}
            for f, t in enumerate(refs):
                if should_cancel is not None and should_cancel():
                    raise InterruptedError("stopped")
                x = float(rv.local_frame(c, t))
                k0 = int(np.floor(x + 1e-9))
                w = x - k0
                want = [k0] if w < 1e-6 else [k0, k0 + 1]
                got = {}
                for k in want:
                    if k < 0 or k >= rv.sessions[c].n_frames:
                        got[k] = None
                        continue
                    if k not in cache:
                        if k - last_read > 60 or k <= last_read:
                            reader.reset(k)
                        bgr = reader.read(k)
                        if bgr is None:
                            if failed[c] is None:       # the frame that did not decode (a skip may stop earlier)
                                failed[c] = int(reader.failed if reader.failed is not None else k)
                            cache[k] = None
                            reader.reset(k + 1)
                        else:
                            d = detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), spec)
                            cache[k] = None if d is None else padded(d[0], d[1], spec)
                            if cache[k] is not None:
                                frame_views[c][int(k)] = cache[k]
                        last_read = k
                    got[k] = cache[k]
                if len(want) == 1:
                    v = got[k0]
                    out_views[f][c] = v if v is not None else np.full((N, 2), np.nan)
                else:
                    out_views[f][c] = _interp_views(got[k0], got[k0 + 1], w, N)
                local[f][c] = float(int(half_up(x)))
                for k in [k for k in cache if k < k0]:
                    del cache[k]
                done += 1
                if progress is not None:
                    progress(done / total, f"{rv.name(c)}: instant {f + 1} of {len(refs)}")
        finally:
            if reader is not None:
                reader.release()
    return RigScan(spec, sizes, list(rv.names), [int(r) for r in refs], local, out_views, failed, note,
                   list(rv.views), frame_views)


def _window(rv) -> tuple[int, int]:
    """Integer reference instants every camera of run view `rv` covers (its offsets)."""
    lo, hi = -np.inf, np.inf
    for k, s in enumerate(rv.sessions):
        lo = max(lo, float(rv.reference_time(k, 0)))
        hi = min(hi, float(rv.reference_time(k, s.n_frames - 1)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return (0, -1)
    return int(np.ceil(lo)), int(np.floor(hi))


def pairs_and_extras(scan: RigScan) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """What the wand solver takes, from a rig scan (R2, R4):
    (wand_uv (F, C, 2, 2), wand length in metres, bg_uv (B, C, 2), frame_ids (F,) reference frames,
    bg_frames (B,) the reference frame of each extra point). A frame counts when at least two
    cameras saw the board (R1: >= 4 corners each) and some PAIR of them shares at least R2's 6 ids
    (I355: it used to need 6 ids shared by EVERY such camera, so one camera grazing the board vetoed a
    frame two others saw whole); only the cameras that share 6 ids with another one take part in that
    frame. Its wand is the first long-edge pair (R4) seen by two or more of them; every other corner
    seen by two or more of them is an extra point. RAW (lens-uncorrected) pixels, NaN where unseen."""
    spec = scan.spec
    pairs = long_edge_pairs(spec)
    C = len(scan.sizes)
    wand, frames, bg, bgf = [], [], [], []
    for f, row in enumerate(scan.views):
        seen = np.stack([np.isfinite(v).all(axis=1) for v in row])          # (C, N)
        cams = np.nonzero(seen.sum(axis=1) >= MIN_CORNERS_SEEN)[0]
        if len(cams) < 2:
            continue
        # (I355) R2 per pair: the cameras linked to another by >= 6 shared ids; nobody else takes part
        common = (seen[cams][:, None, :] & seen[cams][None, :, :]).sum(axis=2)            # (k, k) shared ids
        np.fill_diagonal(common, 0)
        linked = cams[(common >= MIN_SHARED_IDS).any(axis=1)]
        if len(linked) < 2:
            continue
        keep = np.zeros(C, bool)
        keep[linked] = True
        seen = seen & keep[:, None]
        pick = None
        for a, b in pairs:
            both = seen[:, a] & seen[:, b]
            if int(both.sum()) >= 2:
                pick = (a, b)
                break
        if pick is None:
            continue
        a, b = pick
        w = np.full((C, 2, 2), np.nan)
        for c in range(C):
            if seen[c, a] and seen[c, b]:
                w[c, 0] = row[c][a]
                w[c, 1] = row[c][b]
        wand.append(w)
        frames.append(scan.ref_frames[f])
        multi = np.nonzero(seen.sum(axis=0) >= 2)[0]
        for k in multi:
            if k in (a, b):
                continue
            p = np.full((C, 2), np.nan)
            for c in range(C):
                if seen[c, k]:
                    p[c] = row[c][k]
            bg.append(p)
            bgf.append(scan.ref_frames[f])
    wand_uv = np.asarray(wand, np.float64).reshape(-1, C, 2, 2)
    bg_uv = np.asarray(bg, np.float64).reshape(-1, C, 2)
    return (wand_uv, wand_length_m(spec), bg_uv, np.asarray(frames, np.int64), np.asarray(bgf, np.int64))
