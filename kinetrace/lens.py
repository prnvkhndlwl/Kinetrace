"""Lens (intrinsic) calibration from a printed checkerboard — pure numpy/cv2.

Why it exists
-------------
A wand calibration (`wand.py`) recovers where the cameras are and their focal
lengths, but it models each lens as a pinhole with the principal point at the
image centre. That is fine for an ordinary lens. A wide-angle or action-camera
lens (GoPro and friends) bends straight lines by tens of pixels near the
edges; the wand alone estimates that only weakly, so the tracks must be
UNDISTORTED with a separately measured lens model before the wand solve and
before triangulation. easyWand users do this with a camera profile from
MATLAB / Argus; this module does it inside Kinetrace.

What the user does: prints the checkerboard (`save_checkerboard_png`), films
it with the camera at the experiment's settings, and points the lens wizard
at that video. `scan_video` finds the board in a spread of frames,
`select_diverse` keeps the most varied views, `calibrate_lens` fits the
OpenCV pinhole + radial model (or the fisheye model) and writes a plain-
language report, and `LensProfile` is what the project stores per camera.

Conventions: 0-based pixels, top-left origin (Kinetrace's own). The profile's
`undistort_model()` returns a `calib.OpenCVUndistort` that maps raw pixels
into a SQUARE camera frame (fx = fy = (fx + fy) / 2, same principal point),
which is what the wand core's single-focal camera model needs; the focal and
principal point it reports (`f_square`, `principal`) are in that frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from kinetrace.calib import OpenCVUndistort

DEFAULT_PATTERN = (9, 6)        # inner corners (columns, rows) = 10 x 7 squares
DEFAULT_SQUARE_MM = 24.0
LENS_SUFFIX = ".klens.json"
# A border pixel whose undistorted ray leaves the optical axis by more than
# this is "beyond the model": no real lens on a real sensor sees past it, and a
# rectilinear picture puts a 90-degree ray at infinity.
MAX_RAY_DEG = 80.0
MAX_VIEWS = 40                  # boards a fit is made from (`select_diverse` thins a longer list to this)
# The "3 x the median" rule that sets a view aside, with the floor each call site has always
# used (R19: one rule, the floors kept so no fit changes): the fit's own robust pass, the
# automatic choice of views, and the review board's "Drop the worst".
OUTLIER_FLOOR_FIT_PX = 0.45
OUTLIER_FLOOR_PICK_PX = 0.5
QUARTER_TURNS = (0, 90, 180, 270)   # the turns a video's rotation tag can ask for


def px_scale(width: float, height: float) -> float:
    """How many times a 1080p pixel one of this picture's pixels is, by the
    LONGER side (I250: a portrait 4K clip was judged ~1.8 x stricter than the
    same lens in landscape by max(1, width / 1920))."""
    return max(1.0, max(float(width), float(height)) / 1920.0)


def rms_limits(width: float, height: float) -> tuple[float, float]:
    """(good, ok) limits of one reprojection error in px for a picture of this
    size: at or under the first reads good, over the second poor. The ONE rule
    behind the report's fit verdict and the review board's tile colours."""
    s = px_scale(width, height)
    return 0.6 * s, 1.2 * s


def outlier_limit(errors, floor: float) -> float:
    """The error above which a view is set aside: 3 x the median of `errors`
    (NaN ignored), never below `floor`."""
    e = np.asarray(errors, np.float64)
    e = e[np.isfinite(e)]
    return max(3.0 * float(np.median(e)) if len(e) else 0.0, float(floor))


# ------------------------------------------------------------------ profile


@dataclass
class LensProfile:
    """One lens (intrinsic) calibration. Identical cameras (same model, lens,
    zoom and recording mode) may share one: `same_profile` tells (G40)."""
    width: int
    height: int
    K: np.ndarray                      # (3, 3)
    dist: np.ndarray                   # OpenCV (k1, k2, p1, p2, k3) or fisheye (k1..k4)
    fisheye: bool = False
    rms: float = float("nan")          # reprojection RMS of the checkerboard fit, px
    n_views: int = 0
    source: str = ""
    report: dict = field(default_factory=dict)
    # how far the decoder turned the board video's stored pictures clockwise (0 / 90 / 180 / 270,
    # `video_source.applied_rotation`) before the lens was measured on them: what `fit_profile`
    # turns the profile by for a camera whose video is turned otherwise. None = not known (a file
    # from another program, or a profile saved before Kinetrace recorded it)
    rotation: int | None = None

    # ---- derived ------------------------------------------------------
    @property
    def f_square(self) -> float:
        return float(0.5 * (self.K[0, 0] + self.K[1, 1]))

    @property
    def principal(self) -> tuple[float, float]:
        return float(self.K[0, 2]), float(self.K[1, 2])

    def K_square(self) -> np.ndarray:
        f = self.f_square
        cx, cy = self.principal
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    def undistort_model(self) -> OpenCVUndistort:
        """Raw pixels -> undistorted pixels in the square camera frame."""
        return OpenCVUndistort(self.K, self.dist, self.fisheye, self.K_square())

    def border_check(self, n: int = 64) -> dict:
        """Walk the picture's border through the model and say where it holds.

        A distortion polynomial is only known out to where the board went; in
        the corners it is extrapolated, and a fisheye fit can RUN AWAY there:
        the inverse does not converge and OpenCV returns the sentinel
        (-1000000, -1000000), which once reached the report as "the edges
        are bent by 1417199 px" on a real GoPro clip (the board had
        reached 69 % of the way to the corners). A border pixel counts as
        described by the model when its undistorted position is finite, not
        the sentinel, comes back within a few pixels when re-distorted, and
        its ray leaves the optical axis by less than `MAX_RAY_DEG` (a
        rectilinear picture puts a 90-degree ray at infinity).

        Returns {"bend_px": largest displacement over the VALID border,
                 "valid_frac": share of border pixels the model describes,
                 "runaway": valid_frac < 1,
                 "fov_diag_deg": the field of view the model implies across
                 the diagonal (nan when a corner is not described)}.
        """
        w, h = self.width, self.height
        xs = np.linspace(0, w - 1, n)
        ys = np.linspace(0, h - 1, n)
        border = np.concatenate([np.column_stack([xs, np.zeros(n)]), np.column_stack([xs, np.full(n, h - 1.0)]),
                                 np.column_stack([np.zeros(n), ys]), np.column_stack([np.full(n, w - 1.0), ys])])
        model = OpenCVUndistort(self.K, self.dist, self.fisheye)
        und = model.undistort(border)
        with np.errstate(invalid="ignore", over="ignore"):
            back = model.distort(und)
            finite = np.isfinite(und).all(axis=1)
            sane = finite & (np.abs(und).max(axis=1) < 1e5)
            rt = np.linalg.norm(np.nan_to_num(back, nan=1e9) - border, axis=1)
            cx, cy = self.principal
            r_und = np.hypot(und[:, 0] - cx, und[:, 1] - cy)
            theta = np.degrees(np.arctan(r_und / max(self.f_square, 1e-9)))
            valid = sane & (rt < max(4.0, 0.003 * w)) & (theta < MAX_RAY_DEG)
            d = np.linalg.norm(und - border, axis=1)
        bend = float(np.max(d[valid])) if valid.any() else float("nan")
        corners_idx = [0, n - 1, n, 2 * n - 1]          # (0,0) (w,0) (0,h) (w,h)
        corner_ok = valid[corners_idx]
        fov = float(2.0 * np.mean(theta[corners_idx])) if corner_ok.all() else float("nan")
        frac = float(valid.mean())
        return {"bend_px": bend, "valid_frac": frac, "runaway": bool(frac < 1.0), "fov_diag_deg": fov}

    def distortion_at_border(self, n: int = 64) -> float:
        """Largest displacement (px) between a raw border pixel and where the
        undistorted picture puts it — how much this lens bends the edges —
        over the part of the border the model describes (`border_check`)."""
        return self.border_check(n)["bend_px"]

    # ---- files ----------------------------------------------------------
    def to_json(self) -> dict:
        return {"kinetrace_lens": 1, "width": int(self.width), "height": int(self.height),
                "K": np.asarray(self.K, np.float64).tolist(), "dist": np.asarray(self.dist, np.float64).tolist(),
                "fisheye": bool(self.fisheye), "rms": (None if not np.isfinite(self.rms) else float(self.rms)),
                "n_views": int(self.n_views), "source": self.source, "report": self.report,
                "rotation": None if self.rotation is None else int(self.rotation)}

    @staticmethod
    def from_json(d: dict) -> "LensProfile":
        if "kinetrace_lens" not in d:
            raise ValueError("not a Kinetrace lens file")
        rms = d.get("rms")
        rot = d.get("rotation")
        rot = int(rot) % 360 if isinstance(rot, (int, float)) and np.isfinite(rot) else None
        return LensProfile(int(d["width"]), int(d["height"]), np.asarray(d["K"], np.float64).reshape(3, 3),
                           np.asarray(d.get("dist", []), np.float64).ravel(), bool(d.get("fisheye", False)),
                           float("nan") if rms is None else float(rms), int(d.get("n_views", 0)),
                           str(d.get("source", "")), dict(d.get("report") or {}),
                           rot if rot in QUARTER_TURNS else None)

    def save(self, path: str | Path) -> str:
        p = Path(path)
        if not p.name.lower().endswith(LENS_SUFFIX):
            p = p.with_name(p.stem + LENS_SUFFIX)
        p.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")
        return str(p)

    @staticmethod
    def load(path: str | Path) -> "LensProfile":
        return LensProfile.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        """One line for the UI: focal length, distortion strength, quality."""
        chk = self.border_check()
        bend = chk["bend_px"]
        kind = "fisheye model" if self.fisheye else "standard model"
        q = f", fit error {self.rms:.2f} px over {self.n_views} views" if np.isfinite(self.rms) else ""
        # (G144) the curvature is the LENS's (how far an edge pixel sits from where a straight-line lens
        # would put it), not an error: "bends the edges by 1076 px" beside "fit 0.90 px" read as one
        if chk["runaway"]:
            where = "at least " + f"{bend:.0f} px" if np.isfinite(bend) else "an unknown amount"
            return (f"f = {self.f_square:.0f} px, lens curvature at the picture edges {where}, and the model "
                    f"RUNS AWAY in the picture corners ({kind}{q})")
        return f"f = {self.f_square:.0f} px, lens curvature at the picture edges {bend:.0f} px ({kind}{q})"


def lens_label(prof: LensProfile | None) -> str:
    """A camera's lens profile in a few words (where it came from + its verdict), for lists and
    questions: GoPro's nominal model, else the profile's source; "none" without a profile."""
    if prof is None:
        return "none"
    rep = prof.report or {}
    if rep.get("gopro_nominal"):
        return "GoPro's lens model (nominal)"
    v = rep.get("verdict")
    return (prof.source or "lens profile") + (f", {str(v).upper()}" if v else "")


def _parse_argus(path: str | Path) -> list[tuple[int, LensProfile]]:
    """(camera number from the file's first column, profile) per camera line."""
    out = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [q for q in line.replace(",", " ").split() if q]
        if len(parts) < 6:
            continue
        try:
            vals = [float(v) for v in parts]
        except ValueError:
            continue
        cam, f, w, h, cx, cy = vals[:6]
        ar = vals[6] if len(vals) > 6 else 1.0
        k = (vals[7:12] + [0.0] * 5)[:5]
        # (I75) The principal point is used AS WRITTEN: Argus fits its profiles
        # with OpenCV and consumes them the same way (argus_gui undistort.py
        # `calibParse` and tools.undistort_pts put cx, cy straight into the
        # camera matrix of cv2.initUndistortRectifyMap / cv2.undistortPoints on
        # the raw 0-based image), and its database centres are (w - 1) / 2,
        # e.g. 1351.5 x 759.5 for 2704 x 1520. A -1 "1-based" shift here moved
        # the distortion centre a pixel up and left of where it was fitted.
        K = np.array([[f, 0.0, cx], [0.0, f * (ar or 1.0), cy], [0.0, 0.0, 1.0]])
        n = int(round(cam))
        out.append((n, LensProfile(int(w), int(h), K, np.array([k[0], k[1], k[2], k[3], k[4]]), False,
                                   float("nan"), 0, f"Argus profile {Path(path).name}, camera {n}")))
    if not out:
        raise ValueError(f"{Path(path).name}: no camera lines (expected 'cam f w h cx cy AR k1 k2 t1 t2 k3')")
    return out


def same_profile(a: LensProfile | None, b: LensProfile | None) -> bool:
    """The same lens calibration (one profile shared by several identical
    cameras, G40) - compared by value, so it holds after a project reload,
    where each camera gets its own copy; a copy turned for a camera filmed on
    its side (`turn_profile`, I269) is the same lens."""
    if a is None or b is None:
        return False
    if a is b:
        return True

    def equal(x, y):
        return (int(x.width) == int(y.width) and int(x.height) == int(y.height) and bool(x.fisheye) == bool(y.fisheye)
                and np.shape(x.dist) == np.shape(y.dist) and np.allclose(x.K, y.K, rtol=0, atol=1e-9)
                and np.allclose(x.dist, y.dist, rtol=0, atol=1e-12))
    if equal(a, b):
        return True
    for cw in (90, 180, 270):
        try:
            if equal(turn_profile(a, cw), b):
                return True
        except ValueError:          # a model that cannot be turned is only ever itself
            return False
    return False


def profile_key(prof: LensProfile | None):
    """A hashable value that names a profile by what it CONTAINS (None for no
    profile): what a settings key must hold, where `id(prof)` changes with a
    reload or a copy and stays the same after an in-place edit (R20)."""
    if prof is None:
        return None
    return (int(prof.width), int(prof.height), bool(prof.fisheye),
            tuple(float(v) for v in np.asarray(prof.K, np.float64).ravel()),
            tuple(float(v) for v in np.asarray(prof.dist, np.float64).ravel()))


def read_lens_for(path: str | Path, view: int, cam_name: str = "") -> tuple[LensProfile | None, str]:
    """THE lens-file dispatch of both wizards (R19): a Kinetrace .klens.json or
    an OpenCV .yml / .json (`calibio.read_lens`), else an Argus / DLTdv profile
    text, where the file's own camera column decides which line camera `view`
    gets (`argus_profile_for`, I80). -> (profile or None, a sentence: the line
    that was used, or why there is none)."""
    from kinetrace import calibio
    suf = Path(path).suffix.lower()
    if suf in (".json", ".yml", ".yaml", ".xml"):
        return calibio.read_lens_file(path), ""
    try:
        return argus_profile_for(path, view, cam_name)
    except ValueError:                  # no camera lines: an OpenCV file under another name is read by content
        if suf == ".txt":
            raise
        return calibio.read_lens_file(path), ""


def load_argus_profile(path: str | Path) -> list[LensProfile]:
    """An Argus / DLTdv camera profile text file: one camera per line,
    `cam f w h cx cy AR k1 k2 t1 t2 k3` (pinhole + OpenCV distortion). Pixels
    are OpenCV's: 0-based, the principal point is taken as written (I75).
    The profiles come back in file order; `argus_profile_for` picks the line
    for one camera."""
    return [p for _, p in _parse_argus(path)]


def argus_profile_for(path: str | Path, view: int, cam_name: str = "") -> tuple[LensProfile | None, str]:
    """The line of an Argus profile file that belongs to camera `view`
    (0-based project order), and a sentence saying which line was used -- or
    (None, the reason) when the file has no line for that camera (I80).

    A one-line file is one lens, used for whichever camera it is loaded on.
    With several lines, the file's own camera column decides (camera number
    = view + 1), as Argus's DLC converter reads it; only when those numbers
    are not distinct is the row order used, as argus-wand does. A camera past
    the file's last line is refused instead of silently given the last line.
    """
    rows = _parse_argus(path)
    name = Path(path).name
    who = cam_name or f"camera {view + 1}"
    if len(rows) == 1:
        n, prof = rows[0]
        return prof, f" (its only line, camera {n})"
    nums = [n for n, _ in rows]
    if len(set(nums)) == len(nums):
        for n, prof in rows:
            if n == view + 1:
                return prof, f" (the line for camera {n})"
        return None, (f"{name} lists cameras {', '.join(str(n) for n in nums)} but has no line for camera "
                      f"{view + 1} ({who}). Load a profile that has a line for this camera, or a single-camera "
                      "profile (one line), or choose the camera this file's lines belong to.")
    if 0 <= view < len(rows):
        n, prof = rows[view]
        return prof, f" (line {view + 1} of {len(rows)}; its camera numbers repeat, so the line order was used)"
    return None, (f"{name} has {len(rows)} camera lines, so it has no line for camera {view + 1} ({who}). "
                  "Load a profile that has a line for this camera, or a single-camera profile (one line).")


def size_mismatch(size: tuple[int, int], cam_size: tuple[int, int], cam_name: str = "this camera",
                  what: str = "This lens profile was measured on") -> str | None:
    """A sentence when a lens measured on pictures of `size` (w, h) is about to
    be attached to a camera recording `cam_size`, else None.

    A profile's focal length, principal point and distortion are in pixels of
    the picture it was measured on. On a camera recording another size they
    are simply wrong -- the wand solve would then hold that wrong focal length
    and centre fixed -- and a rescale is not safe either: many cameras CROP the
    sensor for a smaller mode rather than shrink the picture."""
    w, h = int(size[0]), int(size[1])
    cw, ch = int(cam_size[0]), int(cam_size[1])
    if (w, h) == (cw, ch):
        return None
    return (f"{what} {w} x {h} pictures, but {cam_name} records {cw} x {ch}. A lens profile only fits "
            "the resolution and recording mode it was measured at: film the board with the camera set exactly "
            "as for the experiment, or choose the camera this lens belongs to.")


_TURNED = {0: "as the camera stored them", 90: "turned 90° clockwise from how the camera stored them",
           180: "turned upside down from how the camera stored them",
           270: "turned 90° counter-clockwise from how the camera stored them"}
_TURN = {90: "90° clockwise", 180: "upside down", 270: "90° counter-clockwise"}


def turn_needed(size: tuple[int, int], rotation: int | None, cam_size: tuple[int, int], cam_rotation: int | None,
                cam_name: str = "this camera", what: str = "This lens profile was measured on"
                ) -> tuple[int | None, str | None]:
    """How a lens measured on pictures of `size` (w, h), turned `rotation` degrees clockwise by the
    decoder, fits a camera whose pictures are `cam_size`, turned `cam_rotation`: (the clockwise
    turn to give the profile, None) or (None, the sentence that says why it cannot be used).

    The size rule (`size_mismatch`, I31) with one exception: a camera filmed on its side records
    the SAME pictures and its video only says "show them turned", so a profile measured with the
    camera level fits it once turned by the difference -- when both turns are known. When either
    is not, a quarter-turned size is refused (which way to turn it is unknown), and an equal size is
    taken as it is, as always (an upside-down camera cannot be told then)."""
    w, h = int(size[0]), int(size[1])
    cw, ch = int(cam_size[0]), int(cam_size[1])
    if rotation is not None and cam_rotation is not None:
        turn = (int(cam_rotation) - int(rotation)) % 360
        tw, th = (h, w) if turn in (90, 270) else (w, h)
        if (tw, th) == (cw, ch) and turn in QUARTER_TURNS:
            return turn, None
        return None, size_mismatch((w, h), (cw, ch), cam_name, what) or (
            f"{what} {w} x {h} pictures {_TURNED.get(int(rotation) % 360, '')}, and {cam_name}'s pictures are "
            f"{_TURNED.get(int(cam_rotation) % 360, '')}: turned to match, they are not the same size.")
    elif (w, h) == (ch, cw) and (w, h) != (cw, ch):
        unknown = ("the profile does not record which way the pictures of its checkerboard video were turned "
                   "(a file from another program, or one made before Kinetrace recorded it)" if rotation is None
                   else f"the video of {cam_name} does not say which way it is turned (this decoder cannot tell)")
        fix = ("Measure the lens in Kinetrace from the checkerboard video (3D → Calibrate a Lens): a profile made "
               "now records the turn, and is turned for this camera by itself." if rotation is None else
               "Open the videos with the default decoder (KINETRACE_DECODE not set), which reads the turn.")
        return None, (f"{what} {w} x {h} pictures and {cam_name} records {cw} x {ch}: the same pictures turned a "
                      f"quarter turn, as when a camera films on its side. The lens is the same and the profile "
                      f"could be turned to fit, but {unknown}, so it is not known which way. {fix}")
    bad = size_mismatch((w, h), (cw, ch), cam_name, what)
    return (None, bad) if bad else (0, None)


def _turn_cw_once(prof: LensProfile) -> LensProfile:
    """`prof` for its pictures turned 90 degrees clockwise: pixel (x, y) moves to (h - 1 - y, x),
    the ray (x, y, 1) to (-y, x, 1). Radial distortion does not change; the tangential pair goes
    (p1, p2) -> (p2, -p1) and the thin-prism terms (s1..s4) -> (-s3, -s4, s1, s2). Exact."""
    import copy
    w, h = int(prof.width), int(prof.height)
    K = np.asarray(prof.K, np.float64)
    if abs(K[0, 1]) > 1e-9:
        raise ValueError("its camera matrix has a skew term, which a turned picture cannot keep in that form")
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    K2 = np.array([[fy, 0.0, (h - 1) - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]])
    d = np.asarray(prof.dist, np.float64).ravel().copy()
    if not prof.fisheye:                 # Kannala-Brandt (fisheye) is radial only: nothing to move
        if d.size >= 14 and np.any(np.abs(d[12:14]) > 0):
            raise ValueError("it has tilted-sensor terms, which Kinetrace cannot turn")
        if d.size >= 4:
            p1, p2 = float(d[2]), float(d[3])
            d[2], d[3] = p2, -p1
        if d.size >= 12:
            s1, s2, s3, s4 = (float(v) for v in d[8:12])
            d[8:12] = [-s3, -s4, s1, s2]
    rep = copy.deepcopy(prof.report or {})
    if "focal_px" in rep:
        rep["focal_px"] = [fy, fx]
    if "principal_px" in rep:
        rep["principal_px"] = [float(K2[0, 2]), float(K2[1, 2])]
    if "width" in rep or "height" in rep:
        rep["width"], rep["height"] = h, w
    if "dist" in rep:
        rep["dist"] = d.tolist()
    if isinstance(rep.get("centre_fixed"), list) and len(rep["centre_fixed"]) == 2:
        rep["centre_fixed"] = rep["centre_fixed"][::-1]           # left-right <-> up-down
    if isinstance(rep.get("centre_split"), dict):
        cs = rep["centre_split"]
        cs["dx_px"], cs["dy_px"] = cs.get("dy_px"), cs.get("dx_px")
    return LensProfile(h, w, K2, d, prof.fisheye, prof.rms, prof.n_views, prof.source, rep,
                       None if prof.rotation is None else (int(prof.rotation) + 90) % 360)


def turn_profile(prof: LensProfile, cw_deg: int) -> LensProfile:
    """The same lens for its pictures turned `cw_deg` (0 / 90 / 180 / 270) degrees clockwise -- a
    new profile (the given one is not changed; 0 returns it). Raises ValueError for another
    angle or a model that cannot be turned (skew, tilted sensor)."""
    cw = int(cw_deg) % 360
    if cw not in QUARTER_TURNS:
        raise ValueError(f"a lens profile can only be turned by quarter turns, not {cw_deg} degrees")
    if cw == 0:
        return prof
    out = prof
    for _ in range(cw // 90):
        out = _turn_cw_once(out)
    out.source = f"{prof.source or 'lens profile'}, turned {_TURN[cw]}"
    reasons = out.report.get("verdict_reasons")
    if isinstance(reasons, list):
        reasons.append(f"The profile was measured on {int(prof.width)} x {int(prof.height)} pictures and turned "
                       f"{_TURN[cw]} for a camera whose video is turned that way (the same lens; its numbers "
                       "moved to the turned picture's pixels, an exact change).")
    return out


def fit_profile(prof: LensProfile, cam_size: tuple[int, int], cam_rotation: int | None,
                cam_name: str = "this camera", what: str = "This lens profile was measured on"
                ) -> tuple[LensProfile | None, str]:
    """THE rule for putting lens profile `prof` on a camera recording `cam_size` pictures turned
    `cam_rotation` degrees clockwise by the decoder (`video_source.display_rotation`): -> (the
    profile to attach, "" or a sentence saying it was turned) or (None, why it cannot be used).
    `turn_needed` decides; a turned profile is a new one that records the camera's turn."""
    turn, bad = turn_needed((prof.width, prof.height), prof.rotation, cam_size, cam_rotation, cam_name, what)
    if bad:
        return None, bad
    if not turn:
        return prof, ""
    try:
        out = turn_profile(prof, turn)
    except ValueError as e:
        return None, (f"{what} {int(prof.width)} x {int(prof.height)} pictures and {cam_name}'s video is turned "
                      f"{_TURN[turn]} from them, so the profile would have to be turned, but {e}. Measure the lens "
                      "in Kinetrace from the checkerboard video (3D → Calibrate a Lens).")
    return out, (f"The lens was measured on {int(prof.width)} x {int(prof.height)} pictures "
                 f"{_TURNED[int(prof.rotation)]} and {cam_name}'s pictures are {_TURNED[int(cam_rotation) % 360]}, "
                 f"so the profile was turned {_TURN[turn]} to fit its {int(cam_size[0])} x {int(cam_size[1])} "
                 "pictures (exact: the same lens, its numbers moved to the turned picture's pixels).")


# ---------------------------------------------------------- checkerboard


def checkerboard_image(cols_inner: int = DEFAULT_PATTERN[0], rows_inner: int = DEFAULT_PATTERN[1],
                       square_px: int = 100, margin_px: int = 100) -> np.ndarray:
    """A printable checkerboard: (cols_inner + 1) x (rows_inner + 1) squares,
    white margin, uint8 grayscale."""
    cols, rows = cols_inner + 1, rows_inner + 1
    img = np.full((rows * square_px + 2 * margin_px, cols * square_px + 2 * margin_px), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y0, x0 = margin_px + r * square_px, margin_px + c * square_px
                img[y0:y0 + square_px, x0:x0 + square_px] = 0
    return img


def save_checkerboard_png(path: str | Path, cols_inner: int = DEFAULT_PATTERN[0],
                          rows_inner: int = DEFAULT_PATTERN[1], square_mm: float = DEFAULT_SQUARE_MM,
                          dpi: int = 300) -> tuple[float, float]:
    """Write the board at `dpi` so that printing at 100 % gives `square_mm`
    squares (10 x 7 squares of 24 mm fit both A4 and Letter). Returns the
    printed size (width_mm, height_mm) of the whole page."""
    px_per_mm = dpi / 25.4
    sq = int(round(square_mm * px_per_mm))
    margin = int(round(12.0 * px_per_mm))
    img = checkerboard_image(cols_inner, rows_inner, sq, margin)
    txt = (f"Kinetrace checkerboard: {cols_inner + 1} x {rows_inner + 1} squares of {square_mm:g} mm "
           f"({cols_inner} x {rows_inner} inner corners). Print at 100% / 'actual size' and check a "
           "square with a ruler.")
    cv2.putText(img, txt, (margin, img.shape[0] - margin // 3), cv2.FONT_HERSHEY_SIMPLEX,
                0.9 * dpi / 300.0, 0, max(1, dpi // 150), cv2.LINE_AA)
    p = Path(path)
    if p.suffix.lower() != ".png":
        p = p.with_suffix(".png")
    # PNG with the dpi recorded so viewers print it at the right size. (I222) encoded in memory and
    # written with Python's own file API: cv2.imwrite cannot open a path with a non-ASCII character
    # on Windows ("José"), and answered False with no reason
    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise OSError(f"could not encode the checkerboard as a PNG ({p})")
    p.write_bytes(_png_with_dpi(bytes(enc), dpi))
    return img.shape[1] / px_per_mm, img.shape[0] / px_per_mm


def _png_with_dpi(data: bytes, dpi: int) -> bytes:
    """The PNG `data` with a pHYs chunk so it carries its print resolution."""
    import struct
    import zlib
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return data
    ppm = int(round(dpi / 0.0254))
    body = struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", len(body)) + b"pHYs" + body
    chunk += struct.pack(">I", zlib.crc32(b"pHYs" + body) & 0xFFFFFFFF)
    # after IHDR (8 signature + 25 IHDR bytes)
    return data[:33] + chunk + data[33:]


# ----------------------------------------------------------------- detect


def _detect_raw(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """Corners as the detectors return them, in whatever order they chose."""
    pat = (int(pattern[0]), int(pattern[1]))
    ok, corners = cv2.findChessboardCornersSB(gray, pat, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if ok and corners is not None and len(corners) == pat[0] * pat[1]:
        return corners.reshape(-1, 2).astype(np.float64)
    # The SB detector gives up when a hand or the frame edge eats the board's
    # white border (measured on real hand-held GoPro footage: SB found
    # a third of the boards the classic detector found). The classic detector
    # is the fallback -- and it orders the corners DIFFERENTLY (by square
    # colour, where SB goes by the picture's top-left), which is why every
    # result goes through `orient_board` before anyone sees it.
    ok, corners = cv2.findChessboardCorners(gray, pat, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok or corners is None or len(corners) != pat[0] * pat[1]:
        return None
    corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (7, 7), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    return corners.reshape(-1, 2).astype(np.float64)


def detect_board_info(gray: np.ndarray, pattern: tuple[int, int]) -> tuple[np.ndarray, str] | None:
    """Sub-pixel inner corners (N, 2) float64 in the CANONICAL order (see
    `orient_board`) plus how corner 0 was chosen ("colour" or "image"), or None."""
    raw = _detect_raw(gray, pattern)
    if raw is None:
        return None
    return orient_board(gray, raw, pattern)


def detect_board(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """Sub-pixel inner corners (N, 2) float64 in the canonical row-major order, or None."""
    info = detect_board_info(gray, pattern)
    return None if info is None else info[0]


# ------------------------------------------------------- corner order


def board_is_asymmetric(pattern: tuple[int, int]) -> bool:
    """True when turning the board 180 degrees swaps which squares are black,
    i.e. one inner-corner count is odd and the other even (the printed 9 x 6:
    10 x 7 squares). Only such a board lets corner 0 be recognised by colour
    on every frame; an 8 x 6 board looks identical upside down."""
    cols, rows = int(pattern[0]), int(pattern[1])
    return (cols + rows) % 2 == 1


# Two candidate "first squares" must differ by at least this many grey levels
# before colour is trusted to say which way up the board is.
ORIENT_MIN_CONTRAST = 10.0


def _square_mean(gray: np.ndarray, quad: np.ndarray) -> float:
    """Mean grey of a small patch in the middle of the square bounded by the
    four inner corners `quad` (4, 2). The patch is a quarter of the way to the
    corners, so it stays inside the square under any tilt the detector accepts."""
    c = quad.mean(axis=0)
    r = max(1.0, 0.25 * float(np.median(np.linalg.norm(quad - c, axis=1))))
    h, w = gray.shape[:2]
    x0 = int(np.clip(round(c[0] - r), 0, w - 1))
    x1 = int(np.clip(round(c[0] + r), 0, w - 1))
    y0 = int(np.clip(round(c[1] - r), 0, h - 1))
    y1 = int(np.clip(round(c[1] + r), 0, h - 1))
    return float(gray[y0:y1 + 1, x0:x1 + 1].mean())


def orient_board(gray: np.ndarray, corners: np.ndarray, pattern: tuple[int, int]
                 ) -> tuple[np.ndarray, str]:
    """Put detected corners into ONE canonical order, whichever detector and
    whichever way the board was held.

    OpenCV's two detectors disagree: `findChessboardCornersSB` starts at the
    corner nearest the picture's top-left, the classic detector starts by
    square colour. Mixed in one scan (SB fails on part-covered boards, the
    classic one takes over) the ring and the axes on the review board jumped
    between opposite corners of the board from frame to frame.

    Canonical order: row-major with `cols` corners per row; the first row is
    the board's X axis, the first column its Y axis, X x Y right-handed in
    image coordinates (x right, y down) -- so a board seen from the front
    keeps that handedness under any rotation and Z (= into the board, as
    OpenCV has it) is drawn pointing OUT towards the camera; and **corner 0
    is the inner corner whose own inner square (towards corners 1 and cols)
    is black**. On an asymmetric board (cols + rows odd) that names one
    physical corner on every frame. On a symmetric board colour cannot tell a
    180 degree turn apart, so corner 0 is the candidate nearer the picture's
    top-left and the tag says "image" -- the lens fit does not care, the axes
    will flip when the board is turned.

    Returns (corners (N, 2) float64, how) with how in {"colour", "image"}.
    """
    cols, rows = int(pattern[0]), int(pattern[1])
    g = np.asarray(corners, np.float64).reshape(rows, cols, 2)
    # 1. handedness: the four consistent grid orders are two rotations and two
    #    reflections; reversing the ROWS turns a reflection into a rotation.
    vx = g[0, 1] - g[0, 0]
    vy = g[1, 0] - g[0, 0]
    if vx[0] * vy[1] - vx[1] * vy[0] < 0:
        g = g[::-1, :]
    # 2. which of the two rotations: the black first square wins
    how = "image"
    if board_is_asymmetric(pattern):
        flipped = g[::-1, ::-1]
        a = _square_mean(gray, np.array([g[0, 0], g[0, 1], g[1, 0], g[1, 1]]))
        b = _square_mean(gray, np.array([flipped[0, 0], flipped[0, 1], flipped[1, 0], flipped[1, 1]]))
        if abs(a - b) >= ORIENT_MIN_CONTRAST:
            if a > b:
                g = flipped
            how = "colour"
    if how == "image":
        p0, pn = g[0, 0], g[-1, -1]
        if p0[0] + p0[1] > pn[0] + pn[1]:
            g = g[::-1, ::-1]
    return np.ascontiguousarray(g.reshape(-1, 2)), how


def orientation_note(pattern: tuple[int, int]) -> str:
    """A warning for the wizard when the typed board cannot show which way up
    it is; empty for a board that can."""
    if board_is_asymmetric(pattern):
        return ""
    cols, rows = int(pattern[0]), int(pattern[1])
    return (f"A board with {cols} x {rows} inner corners looks the same turned upside down, so "
            "the program cannot tell which corner is which from frame to frame. The lens "
            "calibration is not affected, but the drawn axes will flip when the board is "
            "turned, and such a board cannot be used to line cameras up. For that, use a board "
            "with one odd and one even count -- the printed board is 9 x 6.")


def corner_zero_text(how: str, pattern: tuple[int, int]) -> str:
    """What corner 0 (the ringed dot) IS on ONE board, from how it was chosen
    (`orient_board`'s tag: "colour" / "image", "" = unknown) - the corner editor's
    hint (G138: it said "the same physical corner on every board" for a symmetric
    board and for one too faint to tell by colour, where that is false)."""
    if how == "colour":
        return ("corner 0 - the inner corner beside the black square, the same physical corner on every "
                "board whose corner 0 was found this way.")
    if how == "image":
        if not board_is_asymmetric(pattern):
            return ("corner 0 - simply the corner nearest the top-left of the picture: this board looks the same "
                    "turned upside down, so it is NOT the same physical corner from board to board.")
        return ("corner 0 - chosen from the picture's top-left, because this board had too little contrast "
                "to tell the black square: it may not be the same physical corner as on other boards.")
    return "corner 0, where the corner order starts."


def orientation_summary(orient: list[str], pattern: tuple[int, int]) -> str:
    """One plain sentence for the review board: how corner 0 was recognised."""
    n = len(orient)
    if n == 0:
        return ""
    cols, rows = int(pattern[0]), int(pattern[1])
    if not board_is_asymmetric(pattern):
        return (f"This board ({cols} x {rows} inner corners) looks the same turned 180 degrees, "
                "so corner 0 (the ring) is simply the corner nearest the top-left of the picture "
                "and the axes will flip when the board is turned. That does not affect the lens.")
    k = sum(1 for o in orient if o == "colour")
    if k == n:
        return (f"Corner 0 (the ring) is the corner beside the black square on all {n} boards, "
                "so the axes point the same way along the board in every picture.")
    return (f"Corner 0 (the ring) is the corner beside the black square on {k} of {n} boards; "
            f"{n - k} had too little contrast to tell and were ordered from the picture's "
            "top-left -- check that their rings sit in the same place on the board, and "
            "untick any that do not.")


def board_features(corners: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A small descriptor of one view for diversity selection: centre (x, y)
    normalised, apparent size, and the two aspect/skew ratios of the board."""
    w, h = size
    c = corners.mean(axis=0)
    span = corners.max(axis=0) - corners.min(axis=0)
    cov = np.cov(corners.T)
    ev = np.sort(np.linalg.eigvalsh(cov))
    skew = float(np.sqrt(max(ev[0], 1e-9) / max(ev[1], 1e-9)))
    return np.array([c[0] / w, c[1] / h, float(np.hypot(*span) / np.hypot(w, h)), skew])


def select_diverse(corner_list: list[np.ndarray], size: tuple[int, int], keep: int) -> list[int]:
    """Greedy farthest-point selection in feature space: views spread over
    the picture, near and far, tilted and square-on."""
    n = len(corner_list)
    if n <= keep:
        return list(range(n))
    F = np.array([board_features(c, size) for c in corner_list])
    F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)
    chosen = [int(np.argmax(np.linalg.norm(F - F.mean(axis=0), axis=1)))]
    d = np.linalg.norm(F - F[chosen[0]], axis=1)
    while len(chosen) < keep:
        k = int(np.argmax(d))
        if d[k] <= 1e-9:
            break           # (I179) what is left is a copy of a pose already chosen: never the same index twice
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(F - F[k], axis=1))
    return sorted(chosen)


# two views of the board are "the same pose" when it sits in the same place, at the same size and
# the same tilt: centre within 1.5 % of the picture, size within 3 %, aspect / skew within 0.03
POSE_CENTRE_TOL, POSE_SIZE_TOL, POSE_SKEW_TOL = 0.015, 0.03, 0.03


def pose_clusters(corner_list: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """A label per view; views holding the board (nearly) still share one
    (I179: ten poses held for eight frames each are ten views of the board, not
    eighty). Greedy: a view joins the first earlier pose it matches."""
    labels = np.zeros(len(corner_list), int)
    reps: list[np.ndarray] = []
    for i, c in enumerate(corner_list):
        f = board_features(np.asarray(c, np.float64).reshape(-1, 2), size)
        for k, r in enumerate(reps):
            if (np.abs(f[:2] - r[:2]).max() <= POSE_CENTRE_TOL
                    and abs(f[2] - r[2]) <= POSE_SIZE_TOL * max(r[2], 1e-9)
                    and abs(f[3] - r[3]) <= POSE_SKEW_TOL):
                labels[i] = k
                break
        else:
            labels[i] = len(reps)
            reps.append(f)
    return labels


THUMB_MAX_W = 480       # per-board preview kept in memory for the review board


@dataclass
class ScanResult:
    size: tuple[int, int]
    fps: float
    n_frames: int
    n_scanned: int
    frames: list[int]
    corners: list[np.ndarray]
    sample_bgr: np.ndarray | None = None       # one frame with the board, for the preview
    # One small BGR image per FOUND board, so the user can look at every one
    # of them without the scan holding whole 4K frames: 100 boards at 480 px
    # wide is about 40 MB, the same frames at native 4K would be 2.5 GB. The
    # corner editor re-reads the full frame from the video when it needs it.
    thumbs: list[np.ndarray] = field(default_factory=list)
    video: str = ""
    # per FOUND board: how corner 0 was chosen, "colour" (beside the black
    # square -- the same physical corner on every frame) or "image" (nearest
    # the picture's top-left: a symmetric board, or one too faint to tell)
    orient: list[str] = field(default_factory=list)
    # how far the decoder turned the video's stored pictures clockwise (`video_source.applied_rotation`):
    # the profile fitted from these boards records it (`LensProfile.rotation`)
    rotation: int | None = None

    def thumb_scale(self, i: int) -> float:
        """Pixels of thumbnail per pixel of video, for drawing corners on it."""
        if i >= len(self.thumbs) or self.thumbs[i] is None:
            return 1.0
        return float(self.thumbs[i].shape[1]) / max(1, self.size[0])


def scan_video(path: str | Path, pattern: tuple[int, int], max_candidates: int = 240,
               progress=None, should_cancel=None, downscale_max: int = 1280) -> ScanResult:
    """Look for the board in up to `max_candidates` frames spread over the
    video. Detection runs on a downscaled copy for speed, then the corners
    are refined at full resolution. Own VideoCapture (one per thread)."""
    from kinetrace.video_source import applied_rotation, open_capture
    cap = open_capture(str(path))
    if not cap.isOpened():
        raise OSError(f"could not open {path}")
    try:
        rotation = applied_rotation(cap)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if n <= 0:
            raise OSError("the video reports no frames")
        idx = np.unique(np.linspace(0, n - 1, min(n, int(max_candidates))).round().astype(int))
        frames, corners, thumbs, orient = [], [], [], []
        sample = None
        scale = min(1.0, downscale_max / max(w, h))
        for k, f in enumerate(idx.tolist()):
            if should_cancel is not None and should_cancel():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, bgr = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
            info = detect_board_info(small, pattern)
            if info is not None:
                c, how = info
                orient.append(how)
                if scale < 1:
                    c32 = (c / scale).astype(np.float32).reshape(-1, 1, 2)
                    c = cv2.cornerSubPix(gray, c32, (9, 9), (-1, -1),
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
                                         ).reshape(-1, 2).astype(np.float64)
                frames.append(int(f))
                corners.append(c)
                tw = min(THUMB_MAX_W, bgr.shape[1])
                thumbs.append(cv2.resize(bgr, (tw, max(1, round(bgr.shape[0] * tw / bgr.shape[1]))),
                                         interpolation=cv2.INTER_AREA))
                if sample is None:
                    sample = bgr.copy()
            if progress is not None:
                progress((k + 1) / len(idx), f"frame {f}: {len(frames)} boards found")
        return ScanResult((w, h), fps, n, len(idx), frames, corners, sample,
                          thumbs, str(path), orient, rotation)
    finally:
        cap.release()


# ------------------------------------------------------------- calibrate


def _object_points(pattern: tuple[int, int], square: float) -> np.ndarray:
    cols, rows = pattern
    obj = np.zeros((rows * cols, 3), np.float64)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square)
    return obj


def view_pose(corners: np.ndarray, pattern: tuple[int, int], square: float,
              prof: "LensProfile") -> tuple[np.ndarray, np.ndarray] | None:
    """Where the board was, for one view: (rvec, tvec) by solvePnP against the
    fitted lens. This is what lets the review board draw the board's own axes
    on each picture -- the single clearest way to see a view whose corners
    were found in the wrong order, because its axes point somewhere silly."""
    obj = _object_points(pattern, square)
    img = np.asarray(corners, np.float64).reshape(-1, 1, 2)
    if prof.fisheye:
        # undistort to pinhole pixels first; solvePnP has no fisheye model
        img = cv2.fisheye.undistortPoints(img, prof.K, prof.dist.reshape(4, 1)[:4],
                                          P=prof.K)
        dist = np.zeros(5)
    else:
        dist = prof.dist.astype(np.float64).ravel()
    ok, rvec, tvec = cv2.solvePnP(obj.reshape(-1, 1, 3), img, prof.K.astype(np.float64),
                                  dist, flags=cv2.SOLVEPNP_ITERATIVE)
    return (rvec, tvec) if ok else None


def project_with(prof: "LensProfile", obj: np.ndarray, rvec, tvec) -> np.ndarray:
    """Project object points through this lens, distortion and all."""
    obj = np.asarray(obj, np.float64).reshape(-1, 1, 3)
    if prof.fisheye:
        p, _ = cv2.fisheye.projectPoints(obj.reshape(1, -1, 3), rvec, tvec, prof.K,
                                         prof.dist.reshape(4, 1)[:4])
    else:
        p, _ = cv2.projectPoints(obj, rvec, tvec, prof.K, prof.dist.astype(np.float64).ravel())
    return np.asarray(p, np.float64).reshape(-1, 2)


def per_view_errors(corner_list: list[np.ndarray], pattern: tuple[int, int], square: float,
                    prof: "LensProfile") -> np.ndarray:
    """RMS reprojection error in pixels for EVERY view, including ones the fit
    did not use. This is the number the review board ranks images by: a view
    that reprojects badly is a view whose corners are wrong, or that was
    blurred, and it drags the whole calibration with it."""
    obj = _object_points(pattern, square)
    out = np.full(len(corner_list), np.nan)
    for i, c in enumerate(corner_list):
        try:
            pose = view_pose(c, pattern, square, prof)
            if pose is None:
                continue
            proj = project_with(prof, obj, pose[0], pose[1])
            d = proj - np.asarray(c, np.float64).reshape(-1, 2)
            out[i] = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
        except cv2.error:
            continue
    return out


def draw_board_review(bgr: np.ndarray, corners: np.ndarray, pattern: tuple[int, int],
                      square: float = 0.024, prof: "LensProfile | None" = None,
                      scale: float = 1.0, axes: bool = True, every: int = 1) -> np.ndarray:
    """One board, marked up for a human to check: the detected corners, the
    order they were found in (corner 0 ringed, a line along the first row),
    and the board's own X/Y/Z axes when a lens is available.

    Drawn on a COPY. `scale` converts native video pixels to this image's.
    """
    img = bgr.copy()
    c = np.asarray(corners, np.float64).reshape(-1, 2) * float(scale)
    cols, rows = pattern
    # Size the markers from the spacing between neighbouring corners, or on a
    # thumbnail of a distant board the dots merge into one green smear and
    # hide the very thing they are there to let you check.
    step = (float(np.median(np.linalg.norm(np.diff(c[:cols], axis=0), axis=1)))
            if len(c) >= max(2, cols) else 8.0)
    r = int(max(1, min(4, round(step / 4.0))))
    if len(c) >= cols:
        # the first row, so a flipped or rotated detection is obvious
        cv2.polylines(img, [np.round(c[:cols]).astype(np.int32)], False, (250, 210, 90),
                      max(1, r - 1), cv2.LINE_AA)
    for k in range(0, len(c), max(1, int(every))):
        p = (int(round(c[k, 0])), int(round(c[k, 1])))
        cv2.circle(img, p, r, (70, 230, 120), -1, cv2.LINE_AA)
    if len(c):
        p0 = (int(round(c[0, 0])), int(round(c[0, 1])))
        cv2.circle(img, p0, int(max(4, r * 2.5)), (60, 120, 255), max(1, r - 1),
                   cv2.LINE_AA)                                      # corner 0
    if axes and prof is not None:
        for lab, a, b in board_axes(corners, pattern, square, prof):
            o = tuple(np.round(a * float(scale)).astype(int))
            q = tuple(np.round(b * float(scale)).astype(int))
            col = AXIS_COLORS_BGR[lab]
            cv2.arrowedLine(img, o, q, col, 2, cv2.LINE_AA, tipLength=0.25)
            cv2.putText(img, lab, (q[0] + 3, q[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, col, 1, cv2.LINE_AA)
    return img


AXIS_COLORS_BGR = {"X": (60, 60, 240), "Y": (60, 220, 60), "Z": (240, 160, 60)}


def board_axes(corners: np.ndarray, pattern: tuple[int, int], square: float,
               prof: "LensProfile") -> list:
    """The board's own X / Y / Z axes, 3 squares long, as (label, start, end)
    in VIDEO pixels (empty when the pose cannot be solved): what the review
    board draws on a thumbnail and the corner editor over the full frame."""
    try:
        pose = view_pose(corners, pattern, square, prof)
        if pose is None:
            return []
        L = 3.0 * float(square)
        pts = project_with(prof, np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, -L]]),
                           pose[0], pose[1])
    except cv2.error:
        return []
    return [("X", pts[0], pts[1]), ("Y", pts[0], pts[2]), ("Z", pts[0], pts[3])]


def auto_select(corner_list: list[np.ndarray], pattern: tuple[int, int], square: float,
                size: tuple[int, int], model: str = "auto", keep: int = MAX_VIEWS, base: LensProfile | None = None
                ) -> tuple[list[int], np.ndarray, str]:
    """Pick the views worth calibrating from, in two passes.

    1. a diverse spread (`select_diverse`) -- near and far, centre and edge,
       square-on and tilted, because a calibration fitted from one pose is
       degenerate however many frames it has;
    2. fit on that spread, measure EVERY view against it, and drop the ones
       that reproject far worse than the rest (blurred frames, part-occluded
       boards, corners found in the wrong order).

    Returns (chosen indices, per-view error for all views, a sentence saying
    what happened). The user can override any of it on the review board.
    """
    n = len(corner_list)
    if n == 0:
        return [], np.zeros(0), "no boards were found in that video"
    spread = select_diverse(corner_list, size, min(keep, n))
    if n < 4:
        return spread, np.full(n, np.nan), f"only {n} boards found -- using all of them"
    try:
        prof = calibrate_lens([corner_list[i] for i in spread], pattern, square, size, model, base=base)
    except (ValueError, cv2.error) as exc:
        return spread, np.full(n, np.nan), f"a first fit was not possible ({exc}); using the spread"
    err = per_view_errors(corner_list, pattern, square, prof)
    ok = np.isfinite(err)
    if not ok.any():
        return spread, err, "the spread could not be scored; using it as it is"
    limit = outlier_limit(err, OUTLIER_FLOOR_PICK_PX)
    good =[i for i in spread if ok[i] and err[i] <= limit]
    dropped = len(spread) - len(good)
    if len(good) < max(8, len(spread) // 3):      # too aggressive: keep the spread
        return spread, err, (f"{len(spread)} of {n} boards chosen for a good spread "
                             f"(none dropped -- too few would be left)")
    why = (f"{len(good)} of {n} boards chosen: a spread over the picture, near and far, "
           f"tilted and square-on")
    if dropped:
        why += f"; {dropped} dropped for reprojecting worse than {limit:.2f} px"
    return good, err, why


CENTRE_SPLIT_FRAC = 0.003   # (G146) two halves of the boards may disagree on the centre by this x the long side
CENTRE_SPLIT_MIN_PX = 4.0   # ... and never less than this


def _fit_shape(corner_list, pattern, square, base: "LensProfile", fix: tuple = (False, False)):
    """(G146) A lens with `base`'s CURVE held fixed (GoPro's own lens model: its fisheye
    coefficients) and only the focal length -- one, the pixels are square -- and the centre fitted
    to the boards; an axis in `fix` keeps `base`'s centre. Same returns as `_fit`. Sparse least
    squares over (f, cx, cy) + six pose numbers per board; each board's pose starts from solvePnP
    on its corners undistorted with `base`."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    obj = _object_points(pattern, square)
    obj3 = obj.reshape(-1, 1, 3)
    D = np.asarray(base.dist, np.float64).reshape(4, 1)
    K0 = base.K_square()
    imgs = [np.asarray(c, np.float64).reshape(-1, 2) for c in corner_list]
    x0 = [K0[0, 0]] + ([] if fix[0] else [K0[0, 2]]) + ([] if fix[1] else [K0[1, 2]])
    n_in = len(x0)
    for c in imgs:
        und = cv2.fisheye.undistortPoints(c.reshape(-1, 1, 2), K0, D).reshape(-1, 2)
        _ok, rv, tv = cv2.solvePnP(obj, und, np.eye(3), None)
        x0 += np.asarray(rv, np.float64).ravel().tolist() + np.asarray(tv, np.float64).ravel().tolist()
    npt = obj.shape[0]

    def K_of(x):
        cx = K0[0, 2] if fix[0] else x[1]
        cy = K0[1, 2] if fix[1] else x[1 + (0 if fix[0] else 1)]
        return np.array([[x[0], 0.0, cx], [0.0, x[0], cy], [0.0, 0.0, 1.0]])

    def res(x):
        K = K_of(x)
        out = []
        for v, c in enumerate(imgs):
            q = x[n_in + 6 * v:n_in + 6 * v + 6]
            pr, _ = cv2.fisheye.projectPoints(obj3, q[:3], q[3:], K, D)
            out.append((pr.reshape(-1, 2) - c).ravel())
        return np.concatenate(out)
    S = lil_matrix((2 * npt * len(imgs), len(x0)), dtype=int)
    for v in range(len(imgs)):
        rows = slice(2 * npt * v, 2 * npt * (v + 1))
        S[rows, :n_in] = 1
        S[rows, n_in + 6 * v:n_in + 6 * v + 6] = 1
    sol = least_squares(res, np.asarray(x0, np.float64), jac_sparsity=S, x_scale="jac", method="trf")
    K = K_of(sol.x)
    r = sol.fun.reshape(-1, npt, 2)
    per_view = [float(np.sqrt(np.mean(np.sum(rv ** 2, axis=1)))) for rv in r]
    rms = float(np.sqrt(np.mean(np.sum(r.reshape(-1, 2) ** 2, axis=1))))
    tilts = []
    for v in range(len(imgs)):
        R, _ = cv2.Rodrigues(sol.x[n_in + 6 * v:n_in + 6 * v + 3])
        tilts.append(float(np.degrees(np.arccos(min(1.0, abs(float(R[2, 2])))))))
    return rms, K, np.asarray(base.dist, np.float64).ravel(), per_view, tilts


def _calibrate_on_curve(corner_list, pattern, square, size, base: "LensProfile", source: str,
                        max_views: int) -> "LensProfile":
    """(G146) `calibrate_lens(model="gopro")`: GoPro's curve, the focal length and centre from the
    boards. The robust pass of the other models, then a stability check of the centre: the kept
    boards are split into two interleaved halves and each fitted alone; an axis on which the two
    disagree by more than CENTRE_SPLIT_FRAC x the long side is NOT pinned down by these boards
    (measured on a real HERO12 video: two disjoint sets gave +10.6 and -5.0 px vertically) and
    keeps GoPro's centre."""
    if base is None or not base.fisheye:
        raise ValueError("the GoPro lens model of the video is needed for this choice")
    idx = select_diverse(corner_list, size, max_views)
    use = [corner_list[i] for i in idx]
    fit = _fit_shape(use, pattern, float(square), base)
    used = list(idx)
    pv = np.asarray(fit[3])
    bad = pv > outlier_limit(pv, OUTLIER_FLOOR_FIT_PX)
    dropped = 0
    if bad.any() and (len(use) - int(bad.sum())) >= 8:
        use = [c for c, b in zip(use, bad) if not b]
        used = [i for i, b in zip(idx, bad) if not b]
        dropped = int(bad.sum())
        fit = _fit_shape(use, pattern, float(square), base)
    fix = (False, False)
    split = None
    if len(use) >= 12:
        fa = _fit_shape(use[0::2], pattern, float(square), base)
        fb = _fit_shape(use[1::2], pattern, float(square), base)
        lim = max(CENTRE_SPLIT_MIN_PX, CENTRE_SPLIT_FRAC * max(size))
        dx, dy = abs(fa[1][0, 2] - fb[1][0, 2]), abs(fa[1][1, 2] - fb[1][1, 2])
        fix = (bool(dx > lim), bool(dy > lim))
        split = {"dx_px": float(dx), "dy_px": float(dy), "limit_px": float(lim)}
        if any(fix):
            fit = _fit_shape(use, pattern, float(square), base, fix)
    rms, K, dist, per_view, tilts = fit
    prof = LensProfile(int(size[0]), int(size[1]), K, dist, True, rms, len(per_view), source)
    prof.report = lens_report(prof, use, per_view, tilts, {"gopro": rms}, {}, "gopro", curve_known=True)
    prof.report["views_used"] = [int(i) for i in used]
    f0 = float(base.f_square)
    reasons = prof.report["verdict_reasons"]
    reasons.insert(1, f"The lens curve is GoPro's own model of this lens, read from the video: it holds over the "
                      f"whole picture, so the corners need no board. Only the focal length and the centre were "
                      f"measured from the boards: focal length {K[0, 0]:.1f} px ({100 * (K[0, 0] / f0 - 1):+.1f} % "
                      f"against GoPro's nominal {f0:.1f} px), centre {K[0, 2] - base.K[0, 2]:+.1f}, "
                      f"{K[1, 2] - base.K[1, 2]:+.1f} px from the picture centre.")
    if any(fix):
        axes = " and ".join(a for a, f in zip(("left-right", "up-down"), fix) if f)
        reasons.append(f"The {axes} position of the centre could not be pinned down by these boards (two halves of "
                       f"them disagree by {max(split['dx_px'] if fix[0] else 0, split['dy_px'] if fix[1] else 0):.0f} px), "
                       "so GoPro's (the picture centre) is kept there. Tilting the board more (left-right for one, "
                       "up-down for the other) measures it.")
    prof.report["centre_split"] = split
    prof.report["centre_fixed"] = list(fix)
    prof.report["gopro_focal_nominal_px"] = f0
    if dropped:
        prof.report["views_set_aside"] = dropped
        reasons.append(f"{dropped} view(s) fitted far worse than the rest (blur or a mis-detected board) and were "
                       "set aside before the final fit.")
    return prof


def _fisheye_flags() -> int:
    """The fisheye solver's flags, by NAME: they moved between OpenCV 4 (cv2.fisheye.*) and 5
    (cv2.*) and their values differ, so a name this build does not know is an error, not a
    fallback to OpenCV 4's numbers (R19)."""
    flags = 0
    for name in ("CALIB_RECOMPUTE_EXTRINSIC", "CALIB_FIX_SKEW"):
        v = getattr(cv2.fisheye, name, getattr(cv2, name, None))
        if v is None:
            raise ValueError(f"this OpenCV ({cv2.__version__}) does not name the fisheye flag {name}, so the "
                             "fisheye lens model cannot be fitted here")
        flags |= int(v)
    return flags


def _fit(corner_list, pattern, square, size, fisheye: bool):
    obj = _object_points(pattern, square)
    if fisheye:
        # the fisheye solver wants every view as a (1, N, k) float64 array
        objs = [obj.reshape(1, -1, 3) for _ in corner_list]
        imgs = [c.reshape(1, -1, 2).astype(np.float64) for c in corner_list]
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        flags = _fisheye_flags()
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objs, imgs, size, K, D, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-7))
        dist = D.ravel()
        per_view = []
        for o, im, r, t in zip(objs, imgs, rvecs, tvecs):
            proj, _ = cv2.fisheye.projectPoints(o, r, t, K, D)
            per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - im.reshape(-1, 2)) ** 2, axis=1)))))
    else:
        objs = [obj.astype(np.float32) for _ in corner_list]
        imgs = [c.astype(np.float32) for c in corner_list]
        flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, size, None, None, flags=flags)
        dist = D.ravel()
        per_view = []
        for o, im, r, t in zip(objs, imgs, rvecs, tvecs):
            proj, _ = cv2.projectPoints(o, r, t, K, D)
            per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - im.reshape(-1, 2)) ** 2, axis=1)))))
    tilts = []
    for r in rvecs:
        R, _ = cv2.Rodrigues(np.asarray(r, np.float64))
        # the board normal in camera coordinates vs the optical axis
        nz = abs(float(R[2, 2]))
        tilts.append(float(np.degrees(np.arccos(min(1.0, nz)))))
    return float(rms), np.asarray(K, np.float64), np.asarray(dist, np.float64), per_view, tilts


def calibrate_lens(corner_list: list[np.ndarray], pattern: tuple[int, int], square: float,
                   size: tuple[int, int], model: str = "auto", source: str = "checkerboard (Kinetrace)",
                   max_views: int = MAX_VIEWS, base: LensProfile | None = None) -> LensProfile:
    """Fit the lens from detected boards. `model`: "standard" (pinhole +
    k1, k2), "fisheye" (Kannala-Brandt, wide-angle / action cameras) or
    "auto" (fit both, keep the clearly better one, prefer standard). A list
    longer than `max_views` is thinned to a diverse spread of that many (the
    review board passes the length of the boards the user ticked, so a hand-made
    choice is fitted whole, G116). `model="gopro"` (G146): `base`'s curve (GoPro's lens model from the
    video, `gpmf.lens_profile`) with the focal length and centre fitted to the boards."""
    if len(corner_list) < 3:
        raise ValueError(f"only {len(corner_list)} usable views of the board; at least 3 are needed, "
                         "15 or more for a trustworthy result")
    if model == "gopro":
        return _calibrate_on_curve(corner_list, pattern, square, size, base,
                                   "GoPro lens curve + checkerboard (focal length and centre)", max_views)
    idx = select_diverse(corner_list, size, max_views)
    use = [corner_list[i] for i in idx]
    fits = {}
    errors = {}
    dropped = {}
    fitted = {}             # per model: indices into corner_list its final fit used
    for kind in (("standard", "fisheye") if model == "auto" else (model,)):
        try:
            fit = _fit(use, pattern, float(square), size, kind == "fisheye")
            used = list(idx)
            # robust pass: a blurred or mis-detected view fits far worse than
            # the rest; set it aside and refit once (needs enough views left)
            pv = np.asarray(fit[3])
            bad = pv > outlier_limit(pv, OUTLIER_FLOOR_FIT_PX)
            if bad.any() and (len(use) - int(bad.sum())) >= 8:
                keep = [c for c, b in zip(use, bad) if not b]
                used = [i for i, b in zip(idx, bad) if not b]
                fit = _fit(keep, pattern, float(square), size, kind == "fisheye")
                dropped[kind] = int(bad.sum())
            fits[kind] = fit
            fitted[kind] = used
        except (cv2.error, ValueError) as e:      # the fisheye solver is picky about degenerate views
            errors[kind] = str(e).splitlines()[-1][:160]
    if not fits:
        raise ValueError("the lens fit failed: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))
    scores = {k: v[0] for k, v in fits.items()}
    runaway = None
    if model == "auto" and "fisheye" in fits and "standard" in fits:
        # (I178) both final models are scored on the SAME views: each one's robust pass drops its own
        # worst views, so comparing their own fit errors let the model that discarded what it cannot
        # follow win (standard kept at 0.138 px over 35 views where, on the same 40, the fisheye fits
        # 0.145 against its 0.281)
        scores = _same_view_scores(corner_list, fits, fitted, pattern, square, size)
        chosen = "fisheye" if scores["fisheye"] < 0.8 * scores["standard"] else "standard"
        if chosen == "fisheye":
            # a fisheye fit that runs away in the corners (its inverse does not
            # converge where the board never went) is not "better" however it
            # scores on the boards: it cannot undistort part of the picture
            fk, fd = fits["fisheye"][1], fits["fisheye"][2]
            chk = LensProfile(int(size[0]), int(size[1]), fk, fd, True).border_check()
            if chk["runaway"]:
                chosen = "standard"
                runaway = {"fisheye_score_px": scores["fisheye"], "standard_score_px": scores["standard"],
                           "border_valid_frac": float(chk["valid_frac"])}      # (G117) said in the report
    else:
        chosen = next(iter(fits))
    rms, K, dist, per_view, tilts = fits[chosen]
    n_used = len(per_view)
    prof = LensProfile(int(size[0]), int(size[1]), K, dist, chosen == "fisheye", rms, n_used, source)
    # (I77) the report describes the views the final fit used: the robust pass's
    # set-aside views (typically the blurred ones swung into a corner) must not
    # count towards the view total, the coverage or the reach into the corners
    views = [corner_list[i] for i in fitted[chosen]]
    prof.report = lens_report(prof, views, per_view, tilts, scores, errors, model, fisheye_runaway=runaway)
    # which of the given views were fitted, so the evidence shown matches (I78)
    prof.report["views_used"] = [int(i) for i in fitted[chosen]]
    if dropped.get(chosen):
        prof.report["views_set_aside"] = int(dropped[chosen])
        prof.report["verdict_reasons"].append(
            f"{dropped[chosen]} view(s) fitted far worse than the rest (blur, a mis-detected board, or a lens "
            "this model cannot follow near the edges) and were set aside before the final fit.")
    return prof


def _same_view_scores(corner_list, fits: dict, fitted: dict, pattern, square, size) -> dict:
    """Per model, the TYPICAL (median) reprojection error of its final lens on the
    views EITHER model's final fit used - the same boards for both, so the two
    numbers can be compared (I178). The median, not the pooled rms: one blurred
    board in the set would otherwise decide a comparison it has nothing to say about."""
    union = sorted(set(fitted["standard"]) | set(fitted["fisheye"]))
    views = [corner_list[i] for i in union]
    out = {}
    for kind, fit in fits.items():
        tmp = LensProfile(int(size[0]), int(size[1]), fit[1], fit[2], kind == "fisheye")
        e = per_view_errors(views, pattern, float(square), tmp)
        e = e[np.isfinite(e)]
        out[kind] = float(np.median(e)) if len(e) else float(fit[0])
    return out


def coverage_pct(corner_list: list[np.ndarray], size: tuple[int, int]) -> float:
    """Convex-hull area of every detected corner as % of the picture."""
    if not corner_list:
        return 0.0
    pts = np.concatenate(corner_list).astype(np.float32)
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    return float(100.0 * cv2.contourArea(hull) / (size[0] * size[1]))


def edge_reach_pct(corner_list: list[np.ndarray], size: tuple[int, int]) -> float:
    """How far towards the picture corners the board got: the farthest corner
    from the centre as % of the half-diagonal."""
    if not corner_list:
        return 0.0
    pts = np.concatenate(corner_list)
    c = np.array([(size[0] - 1) / 2.0, (size[1] - 1) / 2.0])
    return float(100.0 * np.max(np.linalg.norm(pts - c, axis=1)) / np.hypot(*c))


def lens_report(prof: LensProfile, views: list[np.ndarray], per_view: list[float], tilts: list[float],
                rms_by_model: dict, errors: dict, model_requested: str,
                fisheye_runaway: dict | None = None, curve_known: bool = False) -> dict:
    w, h = prof.width, prof.height
    scale = px_scale(w, h)                                  # (I250) by the longer side
    cov = coverage_pct(views, (w, h))
    reach = edge_reach_pct(views, (w, h))
    # (I179) views of one held pose are ONE pose: the view and tilt rules count distinct poses
    labels = pose_clusters(views, (w, h)) if len(views) else np.zeros(0, int)
    n_total = len(views)
    n = int(len(set(labels.tolist())))
    tilt_mask = np.asarray(tilts) >= 20.0
    tilted = int(len(set(labels[tilt_mask[:len(labels)]].tolist()))) if len(labels) else 0
    bend = prof.distortion_at_border()
    worst = float(np.max(per_view)) if per_view else float("nan")
    reasons = []
    good = ok = True
    r_lim_good, r_lim_ok = rms_limits(w, h)
    if prof.rms <= r_lim_good:
        reasons.append(f"The corner detections agree with the lens model to {prof.rms:.2f} px: a clean fit.")
    elif prof.rms <= r_lim_ok:
        good = False
        reasons.append(f"Fit error {prof.rms:.2f} px is acceptable but not tight: a sharper, steadier video "
                       "of a flat board (glued to something rigid) usually halves it.")
    else:
        good = ok = False
        reasons.append(f"Fit error {prof.rms:.2f} px is too high. Usual causes: the square count typed does "
                       "not match the printed board, the board is not flat, motion blur, or a wrong "
                       "lens model (try the other one).")
    held = (f" ({n_total} frames, but the board was held still in some of them)" if n < n_total else "")
    if n >= 15:
        reasons.append(f"{n_total} views of the board were used"
                       + (f", {n} of them distinct poses (the rest hold the board still)" if n < n_total else "") + ".")
    elif n >= 8:
        good = False
        reasons.append(f"Only {n} usable views{held}: film the board in more positions and tilts (20 or more is "
                       "comfortable).")
    else:
        good = ok = False
        reasons.append(f"Only {n} usable views{held}: not enough to pin the lens down. Film a longer, slower pass "
                       "with the board sharp in every frame.")
    if curve_known:
        # (G146) the curve comes from the camera's own lens model: the boards only measure the focal length
        # and the centre, which a spread over the middle of the picture does
        if cov >= 20.0:
            reasons.append(f"The board covered {cov:.0f}% of the picture and reached {reach:.0f}% of the way into "
                           "the corners: enough for the focal length and the centre.")
        else:
            good = False
            reasons.append(f"The board covered only {cov:.0f}% of the picture: move it around more (near and far, "
                           "left and right) so the focal length and the centre are well measured.")
    elif cov >= 55.0 and reach >= 85.0:
        reasons.append(f"The board covered {cov:.0f}% of the picture and reached {reach:.0f}% of the way "
                       "into the corners: the distortion is measured where it matters.")
    else:
        if cov < 55.0:
            good = False
            if cov < 35.0:
                ok = False
            reasons.append(f"The board covered only {cov:.0f}% of the picture (55% or more is wanted): hold it "
                           "near every edge and in every corner, not only in the middle.")
        if reach < 85.0:
            good = False
            reasons.append(f"The board reached only {reach:.0f}% of the way into the picture corners: the "
                           "correction there is extrapolated. Film it again with the board pushed into the corners.")
    if tilted >= max(3, n // 5):
        reasons.append(f"{tilted} views show the board tilted by 20 degrees or more: good for the focal length.")
    else:
        good = False
        reasons.append("The board was always nearly square-on to the camera: tilt it (lean it left, right, "
                       "up and down by 20-40 degrees) so the focal length is well determined.")
    if np.isfinite(worst) and worst > 2.5 * max(prof.rms, 0.1):
        reasons.append(f"One view fits much worse than the rest ({worst:.2f} px): probably blurred; the "
                       "result still holds.")
    chk = prof.border_check()
    model_word = "fisheye" if prof.fisheye else "standard"
    if chk["runaway"]:
        # The model is extrapolated past where the board went and its inverse
        # does not converge there: a pixel in that part of the picture would
        # be placed at infinity. That is worse than a wrong number -- it is no
        # number -- so it can never be "good", and "poor" once the corners
        # are lost wholesale.
        good = False
        lost = 100.0 * (1.0 - chk["valid_frac"])
        if chk["valid_frac"] < 0.75:
            ok = False
        msg = (f"The {model_word} model RUNS AWAY in the picture corners: {lost:.0f}% of the picture's border "
               "cannot be undistorted at all (a point there would be placed at infinity). The board reached "
               f"only {reach:.0f}% of the way into the corners, so out there the model is extrapolating a "
               "curve it never measured. Film the board pushed into every corner")
        if prof.fisheye and "standard" in rms_by_model:
            msg += (f", or use the standard model, which fits these boards about as well "
                    f"({rms_by_model['standard']:.2f} px against {rms_by_model['fisheye']:.2f} px)")
        elif prof.fisheye:
            msg += (", or try the standard model (choose 'Ordinary lens' or 'Not sure' on the video "
                    "page), which is far less prone to this")
        reasons.append(msg + ".")
        if np.isfinite(bend):
            reasons.append(f"Where the model does hold, the lens curves the picture edges by at least {bend:.0f} px "
                           "(the lens's own curvature, not an error).")
    elif np.isfinite(bend):
        # (G144) a curvature, said as one: the fit error above is what measures the calibration
        if bend >= 8.0 * scale:
            reasons.append(f"This lens curves the edges of the picture by up to {bend:.0f} px: an edge pixel sits "
                           "that far from where a straight-line lens would put it (the wide-angle look, not an "
                           "error; the fit error above measures the calibration). Without this correction, "
                           "anything tracked near the edges would be placed wrongly in 3D by a similar amount: "
                           "attach this profile to the camera.")
        else:
            reasons.append(f"This lens curves the edges by only {bend:.0f} px: a nearly ideal lens. The "
                           "correction is small but free.")
    if np.isfinite(chk["fov_diag_deg"]):
        reasons.append(f"Hand check: the model implies a field of view of {chk['fov_diag_deg']:.0f} degrees "
                       "corner to corner. Compare with the camera's own figure (a GoPro 'Wide' is roughly "
                       "120-150, 'Linear' about 90-100, a phone 70-85, a camcorder 40-70).")
    if len(rms_by_model) == 2:
        reasons.append("Both lens models were tried: standard {:.2f} px, fisheye {:.2f} px (typical error on the "
                       "same boards); the {} model was kept.".format(
                           rms_by_model["standard"], rms_by_model["fisheye"], "fisheye" if prof.fisheye else "standard"))
    if fisheye_runaway:
        # (G117) the 0.8 x rule picked the fisheye model, the border check overruled it: say so, or the
        # sentence above contradicts the rule it quotes
        lost = 100.0 * (1.0 - fisheye_runaway["border_valid_frac"])
        reasons.append(f"The fisheye model fitted these boards better ({fisheye_runaway['fisheye_score_px']:.2f} px "
                       f"against {fisheye_runaway['standard_score_px']:.2f} px) but it runs away in the picture "
                       f"corners: {lost:.0f}% of the picture's border cannot be undistorted with it. The standard "
                       "model was kept; film the board pushed into every corner to let the fisheye model be "
                       "measured there.")
    for k, v in errors.items():
        reasons.append(f"The {k} model could not be fitted ({v}).")
    verdict = "good" if good else ("ok" if ok else "poor")
    return {"verdict": verdict, "verdict_reasons": reasons, "rms_px": float(prof.rms), "n_views": n_total,
            "n_poses": n, "fisheye_runaway_rejected": fisheye_runaway,
            "coverage_pct": cov, "edge_reach_pct": reach, "tilted_views": tilted,
            "worst_view_px": worst, "distortion_border_px": bend, "model": "fisheye" if prof.fisheye else "standard",
            "border_valid_frac": float(chk["valid_frac"]), "border_runaway": bool(chk["runaway"]),
            "fov_diag_deg": float(chk["fov_diag_deg"]),
            "model_requested": model_requested, "rms_by_model": {k: float(v) for k, v in rms_by_model.items()},
            "focal_px": [float(prof.K[0, 0]), float(prof.K[1, 1])],
            "principal_px": [float(prof.K[0, 2]), float(prof.K[1, 2])],
            "dist": np.asarray(prof.dist, np.float64).tolist(), "width": w, "height": h}


# ---------------------------------------------------------------- preview


def undistort_image(bgr: np.ndarray, prof: LensProfile) -> np.ndarray:
    """The picture as the lens model straightens it (same size, same K)."""
    h, w = bgr.shape[:2]
    if prof.fisheye:
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(prof.K, prof.dist.reshape(-1, 1)[:4], np.eye(3), prof.K,
                                                     (w, h), cv2.CV_16SC2)
    else:
        m1, m2 = cv2.initUndistortRectifyMap(prof.K, prof.dist, np.eye(3), prof.K, (w, h), cv2.CV_16SC2)
    return cv2.remap(bgr, m1, m2, cv2.INTER_LINEAR)


def coverage_image(scan: ScanResult, chosen: list[int] | None = None, max_w: int = 640) -> np.ndarray:
    """Every detected corner drawn on a dark canvas of the picture's shape,
    the views that were used brighter: the user sees where the board went."""
    w, h = scan.size
    sc = min(1.0, max_w / w)
    img = np.full((int(h * sc), int(w * sc), 3), 30, np.uint8)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (90, 90, 95), 1)
    use = set(chosen) if chosen is not None else set(range(len(scan.corners)))
    for i, c in enumerate(scan.corners):
        col = (60, 220, 120) if i in use else (110, 110, 115)
        for x, y in c:
            cv2.circle(img, (int(x * sc), int(y * sc)), 1, col, -1)
    return img
