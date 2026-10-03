"""Moving spot: a model-free point model for small, fast, featureless targets (I160).

Track ▾ -> Point model: Moving spot. For what the appearance trackers lose
without a word: a spot a few pixels across that moves more than its own size
per frame over a background that moves too (waves, ripples, leaves) -- a
squid's head spot filmed from a ship, a bat over a river, an insect. On such
footage AllTracker and CoTracker3 follow the background with high confidence
(measured on two flying squid: lost within 2 frames, confidence 0.74-0.98,
so auto-pause never fires; AUDIT I160).

On every frame the spot is searched near where its speed puts it (a
constant-velocity prediction) with ONE detector, its "cue":

  bright  the strongest bright blob of the WHITENESS channel (min of R, G, B):
          a white or pale spot on blue water, sky or foliage
  dark    the strongest dark blob of the DARKNESS channel (255 - max of R, G, B):
          a dark animal against the sky
  change  the blob that changes much more than those pixels usually do (each
          pixel against its own last 30 frames): an animal crossing a
          background that is never still

The run STOPS at the first frame where nothing is found or two candidates are
equally likely (owner, 2026-10-01): the data ends where it stops being
reliable and the user clicks the spot again. A spot whose prediction leaves
the picture simply ends (out of frame = no data).

The second half is the scorer behind Track ▾ -> Test the point models on my
clicks: every setting is started from the first of the user's hand-placed
frames, re-placed from the click wherever it drifts or stops (what the user
would do), and counted. Its result is for ONE point: "Use ..." sets that
point's tracker (G62), not the project's default.

No Qt, no torch: numpy + cv2 on small crops around the prediction.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

CUES = ("bright", "dark", "change")
CUE_LABELS = {"auto": "automatic", "bright": "bright spot", "dark": "dark spot",
              "change": "unusual change"}
DOG_RATIO = 4.0 / 1.5          # outer / inner sigma of the difference of Gaussians (the squid's 1.5 / 4.0)
SIGMAS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 15.0, 21.0)   # spot scales tried at the seed
#   (diameter ~ 2.8 sigma: 3 to ~60 px across; I162 -- it stopped at ~23 px)
SMALL_SIGMAS = SIGMAS[:7]      # enough for the tiny-spot hint (<= 12 px), cheap on the GUI thread
MOTION_MARGIN = (1.5, 2.0)     # the test's smallest search = 1.5 x the 90th-percentile miss of the
                               # prediction replayed over the clicks + 2 px (I162)
DEFAULT_SIGMA = 1.5
MIN_RADIUS = 6.0               # px: the search radius never goes below this
MAX_RADIUS = 400.0
FIRST_RADIUS = 25.0            # px: the first search when the speed is unknown (squid: 20-30 px worked)
FIRST_SIMILAR = 0.3            # ... where two candidates this alike in strength (log ratio) = ambiguous
VEL_KEEP = 0.5                 # velocity = VEL_KEEP x the old one + the rest x the last step
REF_FRAMES = 8                 # the spot's reference strength = median of its last accepted frames
WEAK_FRAC = 0.25               # a candidate under this x the reference strength is not the spot ...
NOISE_K = 3.0                  # ... nor one under this x the DoG noise of the water / sky around it ON
                               # THAT FRAME (the owner's squid: clicked frames >= 3.3, after the spot
                               # fades into the ripples 1.6-3.0; one start-frame noise let it drift on)
NOISE_HALF = 24.0              # px around the 3-sigma support: the area that noise is measured over
AMBIG_RATIO = 0.85             # a second candidate this strong = two equally likely spots: stop
MERGE_RATIO = 1.8              # a spot suddenly this much stronger = two on top of each other (or a
                               # glint on it): where it is cannot be told, stop
AUTO_MIN_CONTRAST = 4.0        # "automatic" picks bright / dark only when the seed stands out this much
# unusual change: each pixel against its own usual change (the bat prototype)
HIST_FRAMES = 30               # the background = the frames up to this far back ...
HIST_STEP = 2                  # ... every 2nd of them (15 frames)
MIN_HIST = 5
HIST_GAP = 3                   # frames this close to the current one are not background: the
                               # target's own wake (an 11 px bat at 10 px / frame overlaps itself)
CHANGE_K = 2.5                 # unusual = |frame - median| beyond 2.5 x that pixel's usual change ...
CHANGE_FLOOR = 6.0             # ... + 6 grey levels
CHANGE_BLUR = 1.5
CHANGE_THRESH = 2.0
CHANGE_MIN_AREA = 4
CHANGE_WEAK_FRAC = 0.1
# the "this looks like a tiny spot" hint (G58)
SPOT_MAX_DIAMETER = 12.0       # px across at most ...
SPOT_MIN_CONTRAST = 6.0        # ... and standing this far out of its background's DoG noise
# the one rule the app gives everywhere for which point model to use (G60)
POINT_TARGET_MAX = 20.0        # px across: up to this, a target can be followed as a single point
WHICH_MODEL = ("Moving spot is for a target small enough to be ONE point: a dot a few pixels to about 20 px "
               "across, with no shape you could put a second landmark on (a squid's head spot seen from a ship, "
               "a distant bat, bird or insect). If you can see a body, a head, legs, wings or an outline, use "
               "AllTracker (with Segment for the outline): it needs no extra clicking.")
# the test on the user's clicks (G57)
TEST_MIN_FRAMES = 20           # hand-placed frames needed (owner, 2026-10-01) ...
TEST_MAX_GAP = 2               # ... in one stretch with no gap longer than this


# ------------------------------------------------------------------ settings

@dataclass
class SpotSettings:
    """How one point is searched. 0 / "auto" = decided at the run's first
    frame from the spot itself (`resolve`)."""
    cue: str = "auto"          # "auto" | "bright" | "dark" | "change"
    radius: float = 0.0        # search radius around the prediction, px
    speed_gain: float = -1.0   # the radius grows by this x the speed (px / frame); -1 = automatic
    sigma: float = 0.0         # the spot's scale (DoG inner sigma), px

    def to_dict(self) -> dict:
        return {"cue": self.cue, "radius": round(float(self.radius), 3),
                "speed_gain": round(float(self.speed_gain), 3), "sigma": round(float(self.sigma), 3)}

    @staticmethod
    def from_dict(d) -> "SpotSettings":
        """Anything odd falls back to automatic (a hand-edited or foreign file)."""
        if not isinstance(d, dict):
            return SpotSettings()

        def num(k, lo, hi):
            try:
                v = float(d.get(k, 0.0))
            except (TypeError, ValueError):
                return 0.0
            return v if math.isfinite(v) and lo <= v <= hi else 0.0
        cue = d.get("cue", "auto")
        try:
            g = float(d.get("speed_gain", -1.0))
        except (TypeError, ValueError):
            g = -1.0
        gain = g if math.isfinite(g) and 0.0 <= g <= 20.0 else -1.0
        return SpotSettings(cue if cue in ("auto",) + CUES else "auto",
                            num("radius", 0.0, MAX_RADIUS), gain, num("sigma", 0.0, 50.0))

    @property
    def automatic(self) -> bool:
        return self.cue == "auto" and self.radius == 0 and self.sigma == 0 and self.speed_gain < 0

    @property
    def resolved(self) -> bool:
        return self.cue in CUES and self.radius > 0 and self.sigma > 0 and self.speed_gain >= 0

    def describe(self) -> str:
        if self.cue == "auto":
            return "automatic"
        s = f"{CUE_LABELS[self.cue]}, search {self.radius:.0f} px"
        if self.speed_gain > 0:
            s += f" + {self.speed_gain:g} x speed"
        return s


@dataclass
class SpotLook:
    """What a spot at a click looks like (`measure_spot`)."""
    cue: str                   # "bright" | "dark": the channel it stands out in
    sigma: float               # its scale (DoG inner sigma), px
    contrast: float            # its DoG peak / the surrounding DoG noise
    at_limit: bool = False     # the largest scale tried fitted best: the target may be bigger still

    @property
    def diameter(self) -> float:
        return spot_diameter(self.sigma)


def spot_diameter(sigma: float) -> float:
    """A spot's diameter in px from its DoG scale (~2.8 sigma)."""
    return 2.0 * math.sqrt(2.0) * sigma


def default_radius(cue: str, sigma: float) -> float:
    d = spot_diameter(sigma)
    return max(15.0, 2.0 * d) if cue == "change" else max(MIN_RADIUS, 1.5 * d)


def default_gain(cue: str) -> float:
    # a bright / dark spot is searched tightly (a wider search grabs glints on
    # water: the squid); a change blob is a turning animal (the bats)
    return 1.5 if cue == "change" else 0.0


# ------------------------------------------------------------------ image helpers

def _crop(img: np.ndarray, cx: float, cy: float, half: float):
    """img[...] around (cx, cy) clipped to the picture -> (crop, x0, y0)."""
    h, w = img.shape[:2]
    x0 = max(0, int(math.floor(cx - half)))
    y0 = max(0, int(math.floor(cy - half)))
    x1 = min(w, int(math.ceil(cx + half)) + 1)
    y1 = min(h, int(math.ceil(cy + half)) + 1)
    return img[y0:max(y0, y1), x0:max(x0, x1)], x0, y0


def channel(rgb: np.ndarray, cue: str) -> np.ndarray:
    """The detector's channel as float32: whiteness, darkness or grey."""
    if rgb.ndim == 2:
        g = rgb.astype(np.float32)
        return 255.0 - g if cue == "dark" else g
    if cue == "bright":
        return rgb.min(axis=2).astype(np.float32)
    if cue == "dark":
        return 255.0 - rgb.max(axis=2).astype(np.float32)
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)


def _dog(ch: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(ch, (0, 0), sigma) - cv2.GaussianBlur(ch, (0, 0), sigma * DOG_RATIO)


def _noise(dog: np.ndarray) -> float:
    """Robust spread of a DoG image (1.4826 x the median absolute deviation)."""
    med = float(np.median(dog))
    return max(1e-3, 1.4826 * float(np.median(np.abs(dog - med))))


def _peaks(dog: np.ndarray):
    """Local maxima (5 x 5) above zero -> (ys, xs, values)."""
    dil = cv2.dilate(dog, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero((dog == dil) & (dog > 0))
    return ys, xs, dog[ys, xs]


def _subpix(dog: np.ndarray, y: int, x: int) -> tuple[float, float]:
    """Parabola vertex through the peak and its neighbours, per axis (+-0.5 px)."""
    dx = dy = 0.0
    h, w = dog.shape
    if 0 < x < w - 1:
        l, c, r = float(dog[y, x - 1]), float(dog[y, x]), float(dog[y, x + 1])
        den = l - 2.0 * c + r
        if den < 0:
            dx = float(np.clip(0.5 * (l - r) / den, -0.5, 0.5))
    if 0 < y < h - 1:
        u, c, d = float(dog[y - 1, x]), float(dog[y, x]), float(dog[y + 1, x])
        den = u - 2.0 * c + d
        if den < 0:
            dy = float(np.clip(0.5 * (u - d) / den, -0.5, 0.5))
    return dx, dy


def measure_spot(rgb: np.ndarray, xy, cues=("bright", "dark"), sigmas=SIGMAS) -> SpotLook | None:
    """The scale and channel at which the click stands out most: the strongest
    DoG response within a few px of it over `SIGMAS` (a DoG is a scale-
    normalised Laplacian, so its values compare across scales). None when
    the click is outside the picture or nothing is brighter / darker there."""
    x, y = float(xy[0]), float(xy[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    h, w = rgb.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    half = 3.0 * sigmas[-1] * DOG_RATIO + 12.0
    crop, x0, y0 = _crop(rgb, x, y, half)
    if crop.shape[0] < 5 or crop.shape[1] < 5:
        return None
    cx, cy = x - x0, y - y0
    best = None
    for cue in cues:
        ch = channel(crop, cue)
        for s in sigmas:
            dog = _dog(ch, s)
            r = max(3.0, s)
            sub = _crop(dog, cx, cy, r)[0]
            if sub.size == 0:
                continue
            v = float(sub.flat[int(np.argmax(sub))])
            if v > 0 and (best is None or v > best[0]):
                best = (v, cue, s, dog)
    if best is None:
        return None
    v, cue, s, dog = best
    return SpotLook(cue, float(s), v / _noise(dog), s >= sigmas[-1])


def looks_like_small_spot(rgb: np.ndarray, xy) -> SpotLook | None:
    """The tiny-spot hint (G58): the look when the click sits on an isolated
    blob at most `SPOT_MAX_DIAMETER` px across, else None."""
    look = measure_spot(rgb, xy, sigmas=SMALL_SIGMAS)
    if (look is None or look.at_limit or look.diameter > SPOT_MAX_DIAMETER
            or look.contrast < SPOT_MIN_CONTRAST):
        return None
    return look


def resolve(settings: SpotSettings, rgb: np.ndarray, xy) -> tuple[SpotSettings, SpotLook | None]:
    """Fill in what is automatic from the spot at `xy` on its first frame."""
    s = settings if isinstance(settings, SpotSettings) else SpotSettings()
    look = measure_spot(rgb, xy, (s.cue,) if s.cue in ("bright", "dark") else ("bright", "dark"))
    cue = s.cue
    if cue == "auto":
        cue = look.cue if look is not None and look.contrast >= AUTO_MIN_CONTRAST else "change"
    sigma = s.sigma if s.sigma > 0 else (look.sigma if look is not None else DEFAULT_SIGMA)
    radius = s.radius if s.radius > 0 else default_radius(cue, sigma)
    gain = s.speed_gain if s.speed_gain >= 0 else default_gain(cue)
    return SpotSettings(cue, float(min(MAX_RADIUS, max(MIN_RADIUS, radius))), float(gain), float(sigma)), look


# ------------------------------------------------------------------ change history

class ChangeHistory:
    """Grey frames before the current one, for the "unusual change" cue: every
    `HIST_STEP`-th frame of the last `HIST_FRAMES`. Look-ahead frames stand in
    only while fewer than `MIN_HIST` past frames exist (a run near the start
    of the video)."""

    def __init__(self):
        self.past: deque = deque(maxlen=HIST_FRAMES // HIST_STEP)
        self.ahead: list = []

    def push(self, idx: int, gray: np.ndarray) -> None:
        if self.past and idx - self.past[-1][0] < HIST_STEP:
            return
        self.past.append((int(idx), gray))
        if len(self.past) == self.past.maxlen:
            self.ahead = []

    def add_ahead(self, idx: int, gray: np.ndarray) -> None:
        self.ahead.append((int(idx), gray))

    def seed(self, past, ahead, first: int) -> None:
        """The background before a run's first frame `first` (I194, ONE rule for the
        worker, the test and the scorer): the frames before it, and the look-ahead
        frames only when the frames the cue can really use at `first` (`frames_for`: the
        `HIST_GAP` frames beside the current one are left out) are fewer than
        `MIN_HIST`. Counting the frames merely read let a run from frame 9 or 10 start
        with 4 usable ones and stop at once."""
        for k, g in past:
            self.push(k, g)
        if len(self.frames_for(first)) < MIN_HIST:
            for k, g in ahead:
                self.add_ahead(k, g)

    def frames_for(self, idx: int) -> list:
        past = [g for k, g in self.past if idx - HIST_FRAMES <= k <= idx - HIST_GAP]
        if len(past) >= MIN_HIST:
            return past
        return past + [g for k, g in self.ahead if k >= idx + HIST_GAP]

    def stats(self, idx: int, x0: int, y0: int, x1: int, y1: int):
        """(median, usual change) of the box over the history, or None."""
        frames = self.frames_for(idx)
        if len(frames) < MIN_HIST:
            return None
        stack = np.stack([g[y0:y1, x0:x1] for g in frames]).astype(np.float32)
        med = np.median(stack, axis=0)
        spread = np.percentile(np.abs(stack - med), 90, axis=0)
        return med, spread


def _grey(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)


def read_background(get_frame, first: int, n_video: int | None = None):
    """The change cue's background for a run that starts at `first` (I194): `get_frame(k)`
    -> RGB or None. -> (past, ahead) = [(k, grey)] before the start (every `HIST_STEP`-th of
    the last `HIST_FRAMES`) and, only when the usable past is short (`ChangeHistory.seed`),
    the frames after it, read up to the VIDEO's end (`n_video`, or until `get_frame` has
    none): the run's own end used to cut them (a semi-automatic step from frame 0-8 got
    none and stopped at once)."""
    past = []
    for k in range(max(0, first - HIST_FRAMES), first, HIST_STEP):
        rgb = get_frame(k)
        if rgb is not None:
            past.append((k, _grey(rgb)))
    probe = ChangeHistory()
    for k, g in past:
        probe.push(k, g)
    ahead = []
    if len(probe.frames_for(first)) < MIN_HIST:
        stop = first + HIST_FRAMES + 1
        if n_video is not None:
            stop = min(stop, int(n_video))
        for k in range(first + HIST_STEP, stop, HIST_STEP):
            rgb = get_frame(k)
            if rgb is None:
                break                       # the video's end
            ahead.append((k, _grey(rgb)))
    return past, ahead


# ------------------------------------------------------------------ one spot

@dataclass
class SpotFix:
    x: float
    y: float
    confidence: float
    strength: float = 0.0


class SpotTracker:
    """One spot, frame by frame: `start` on its first frame (the seed is kept
    exactly), `step` on each following frame. After a stop, `stopped` =
    (frame, "missing" | "ambiguous" | "left") and every step returns None."""

    def __init__(self, settings: SpotSettings, xy, vel=None, size=None):
        self.settings = settings if isinstance(settings, SpotSettings) else SpotSettings()
        self.pos = np.asarray(xy, np.float64).reshape(2).copy()
        v = None if vel is None else np.asarray(vel, np.float64).reshape(2)
        self.vel = v if v is not None and np.isfinite(v).all() else None
        self.size = size                      # (w, h) native px
        self.ref: deque = deque(maxlen=REF_FRAMES)
        self.stopped: tuple[int, str] | None = None

    # -- geometry
    @property
    def diameter(self) -> float:
        return spot_diameter(max(0.5, self.settings.sigma))

    def search_radius(self) -> float:
        s = self.settings
        if self.vel is None:
            return max(FIRST_RADIUS, 3.0 * s.radius)
        return min(MAX_RADIUS, s.radius + s.speed_gain * float(np.hypot(*self.vel)))

    def prediction(self) -> np.ndarray:
        return self.pos + (self.vel if self.vel is not None else 0.0)

    def _border_dist(self, p) -> float:
        w, h = self.size
        return float(min(p[0], p[1], w - 1 - p[0], h - 1 - p[1]))

    # -- the run
    def start(self, idx: int, rgb: np.ndarray, gray=None, history=None) -> SpotFix:
        h, w = rgb.shape[:2]
        if self.size is None:
            self.size = (w, h)
        if not self.settings.resolved:
            self.settings = resolve(self.settings, rgb, self.pos)[0]
        cue = self.settings.cue
        if cue in ("bright", "dark"):
            # the spot's own strength at the seed (the crop `_spot_candidates` measures noise in)
            s = self.settings.sigma
            crop, x0, y0 = _crop(rgb, self.pos[0], self.pos[1], 3.0 * s * DOG_RATIO + NOISE_HALF)
            if crop.shape[0] >= 3 and crop.shape[1] >= 3:
                dog = _dog(channel(crop, cue), s)
                ys, xs, vals = _peaks(dog)
                d = np.hypot(xs + x0 - self.pos[0], ys + y0 - self.pos[1])
                near = d <= max(3.0, 0.5 * self.diameter + 2.0)
                if near.any():
                    self.ref.append(float(vals[near].max()))
        elif history is not None and gray is not None:
            got = self._change_candidates(gray, self.pos, max(4.0, self.diameter), history, idx)
            if got:
                self.ref.append(got[0][3])
        return SpotFix(float(self.pos[0]), float(self.pos[1]), 1.0, self.ref[-1] if self.ref else 0.0)

    def step(self, idx: int, rgb: np.ndarray, gray=None, history=None) -> SpotFix | None:
        if self.stopped is not None:
            return None
        pred = self.prediction()
        R = self.search_radius()
        w, h = self.size
        if not (0 <= pred[0] <= w - 1 and 0 <= pred[1] <= h - 1):
            self.stopped = (int(idx), "left")
            return None
        if self.settings.cue == "change":
            fix, why = self._step_change(idx, gray, pred, R, history)
        else:
            fix, why = self._step_spot(rgb, pred, R)
        if fix is None:
            # nothing there right beside the border = it is leaving the picture
            self.stopped = (int(idx), "left" if why == "missing"
                            and self._border_dist(pred) <= max(3.0, self.diameter) + 0.5 * R else why)
            return None
        new = np.array([fix.x, fix.y], np.float64)
        step = new - self.pos
        self.vel = step if self.vel is None else VEL_KEEP * self.vel + (1.0 - VEL_KEEP) * step
        self.pos = new
        return fix

    # -- bright / dark spot
    def _spot_candidates(self, rgb, pred, R):
        """Peaks within R of `pred`, strongest first: ([(x, y, strength), ...],
        the DoG noise of the area around them on this frame)."""
        s = self.settings.sigma
        sup = 3.0 * s * DOG_RATIO
        crop, x0, y0 = _crop(rgb, pred[0], pred[1], max(R + sup + 3.0, sup + NOISE_HALF))
        if crop.shape[0] < 3 or crop.shape[1] < 3:
            return [], 0.0
        dog = _dog(channel(crop, self.settings.cue), s)
        ys, xs, vals = _peaks(dog)
        d = np.hypot(xs + x0 - pred[0], ys + y0 - pred[1])
        keep = d <= R
        out = []
        for y, x, v in sorted(zip(ys[keep], xs[keep], vals[keep]), key=lambda t: -t[2]):
            dx, dy = _subpix(dog, int(y), int(x))
            out.append((x0 + x + dx, y0 + y + dy, float(v)))
        return out, _noise(dog)

    def _step_spot(self, rgb, pred, R):
        if not self.ref:
            # No reference strength: the click sat on no peak (a few px off the spot). Taking
            # the strongest candidate as the reference would keep a glint as data at
            # confidence >= 0.5, so stop and let the user click the spot itself (I187).
            return None, "missing"
        cands, noise = self._spot_candidates(rgb, pred, R)
        if not cands:
            return None, "missing"
        if self.vel is None:
            # one click, speed unknown: the wide first search takes the candidate
            # most like the clicked spot, not the strongest (a glint on water
            # often is); two alike = ambiguous (click the next frame too)
            ref0 = float(np.median(self.ref))
            sim = sorted(cands, key=lambda c: abs(math.log(max(c[2], 1e-6) / ref0)))
            x, y, v = sim[0]
            d0 = abs(math.log(max(v, 1e-6) / ref0))
            if d0 > math.log(1.0 / WEAK_FRAC) or v < NOISE_K * noise:
                return None, "missing"
            sep = max(3.0, 2.0 * self.settings.sigma)
            rival = next((c for c in sim[1:] if np.hypot(c[0] - x, c[1] - y) > sep), None)
            if rival is not None and abs(math.log(max(rival[2], 1e-6) / ref0)) < d0 + FIRST_SIMILAR:
                return None, "ambiguous"
            self.ref.append(v)
            return SpotFix(float(x), float(y), float(max(0.05, min(1.0, math.exp(-d0)))), v), ""
        x, y, v = cands[0]
        ref = float(np.median(self.ref))
        if v < WEAK_FRAC * ref or v < NOISE_K * noise:
            return None, "missing"
        if v > MERGE_RATIO * ref:
            return None, "ambiguous"
        sep = max(3.0, 2.0 * self.settings.sigma)
        second = next((c for c in cands[1:] if np.hypot(c[0] - x, c[1] - y) > sep), None)
        if second is not None and second[2] >= AMBIG_RATIO * v:
            return None, "ambiguous"
        self.ref.append(v)
        conf = min(1.0, v / max(ref, 1e-6)) * (1.0 - 0.5 * (second[2] / v if second is not None else 0.0))
        return SpotFix(float(x), float(y), float(max(0.05, conf)), v), ""

    # -- unusual change
    def _change_candidates(self, gray, pred, R, history, idx):
        """Unusual blobs within R of `pred`, nearest first: [(dist, x, y, weight), ...]."""
        if gray is None or history is None:
            return []
        half = R + 3.0 * CHANGE_BLUR + 4.0
        cur, x0, y0 = _crop(gray, pred[0], pred[1], half)
        if cur.shape[0] < 3 or cur.shape[1] < 3:
            return []
        st = history.stats(idx, x0, y0, x0 + cur.shape[1], y0 + cur.shape[0])
        if st is None:
            return []
        med, spread = st
        score = np.abs(cur.astype(np.float32) - med) - (CHANGE_K * spread + CHANGE_FLOOR)
        score = cv2.GaussianBlur(np.clip(score, 0, None), (0, 0), CHANGE_BLUR)
        n, lab, stats, _ = cv2.connectedComponentsWithStats((score > CHANGE_THRESH).astype(np.uint8), 8)
        out = []
        for j in range(1, n):
            if stats[j, cv2.CC_STAT_AREA] < CHANGE_MIN_AREA:
                continue
            ys, xs = np.nonzero(lab == j)
            wts = score[ys, xs]
            tot = float(wts.sum())
            if tot <= 0:
                continue
            cx = x0 + float((xs * wts).sum()) / tot
            cy = y0 + float((ys * wts).sum()) / tot
            dist = float(np.hypot(cx - pred[0], cy - pred[1]))
            if dist <= R:
                out.append((dist, cx, cy, tot))
        out.sort(key=lambda t: t[0])
        return out

    def _step_change(self, idx, gray, pred, R, history):
        cands = self._change_candidates(gray, pred, R, history, idx)
        if not cands:
            return None, "missing"
        d1, x, y, wgt = cands[0]
        ref = float(np.median(self.ref)) if self.ref else wgt
        if wgt < CHANGE_WEAK_FRAC * ref:
            return None, "missing"
        if len(cands) > 1 and cands[1][0] < 1.3 * d1 + 5.0 and cands[1][3] > 0.5 * wgt:
            return None, "ambiguous"
        self.ref.append(wgt)
        conf = min(1.0, wgt / max(ref, 1e-6))
        if len(cands) > 1:
            conf *= 1.0 - 0.5 * min(1.0, cands[1][3] / wgt)
        return SpotFix(float(x), float(y), float(max(0.05, conf)), wgt), ""


# ------------------------------------------------------------------ a run's spots

class SpotRun:
    """Every spot of one tracking run (the worker's helper): one SpotTracker
    each, one shared change history. `resolve(rgb)` on the start frame decides
    the automatic settings (and so whether the history is needed), `step` is
    called once per frame in order, the start frame included."""

    def __init__(self, spots, size, start_frame: int):
        # spots: [(pid, xy, vel | None, SpotSettings | dict | None)]
        self.start_frame = int(start_frame)
        self.size = size
        self.trackers: dict[int, SpotTracker] = {}
        for pid, xy, vel, st in spots:
            st = st if isinstance(st, SpotSettings) else SpotSettings.from_dict(st or {})
            self.trackers[int(pid)] = SpotTracker(st, xy, vel, size)
        self.history: ChangeHistory | None = None

    def resolve(self, rgb: np.ndarray) -> None:
        for t in self.trackers.values():
            t.settings = resolve(t.settings, rgb, t.pos)[0]
        if any(t.settings.cue == "change" for t in self.trackers.values()):
            self.history = ChangeHistory()

    def prefill(self, get_frame, n_video: int | None = None) -> None:
        """The change cue's background before the run's first frame (`read_background`):
        `get_frame(k)` -> RGB or None; `n_video` = the VIDEO's frame count when known (near
        the video's start the look-ahead reads up to it, not up to the run's end, I194)."""
        if self.history is None:
            return
        s0 = self.start_frame
        past, ahead = read_background(get_frame, s0, n_video)
        self.history.seed(past, ahead, s0)

    def step(self, idx: int, rgb: np.ndarray) -> dict[int, SpotFix]:
        gray = _grey(rgb) if self.history is not None else None
        out = {}
        for pid, t in self.trackers.items():
            fix = (t.start(idx, rgb, gray, self.history) if idx == self.start_frame
                   else t.step(idx, rgb, gray, self.history))
            if fix is not None:
                out[pid] = fix
        if self.history is not None:
            self.history.push(idx, gray)
        return out


# ------------------------------------------------------------------ the test on the user's clicks

def clicked_stretch(frames, max_gap: int = TEST_MAX_GAP) -> list[int]:
    """The longest run of hand-placed frames with no gap over `max_gap`."""
    fs = sorted(int(f) for f in frames)
    best, cur = [], []
    for f in fs:
        if cur and f - cur[-1] > max_gap + 1:
            cur = []
        cur.append(f)
        if len(cur) > len(best):
            best = list(cur)
    return best


def click_requirement(frames) -> tuple[bool, list[int], str]:
    """(enough?, the stretch the test will use, a sentence for the user)."""
    st = clicked_stretch(frames)
    n = len(st)
    if n >= TEST_MIN_FRAMES:
        return True, st, (f"{n} hand-placed frames, {st[0]}-{st[-1]}: enough for the test "
                          f"(it needs {TEST_MIN_FRAMES}).")
    more = TEST_MIN_FRAMES - n
    where = (f"Your longest stretch has {n} (frames {st[0]}-{st[-1]}); click {more} more frame"
             f"{'s' if more != 1 else ''} right after it." if n else "It has no hand-placed frames yet.")
    return False, st, (f"The test needs the point placed by hand on at least {TEST_MIN_FRAMES} frames in a "
                       f"row (skipping one or two frames is fine). {where} Select the point, step one "
                       f"frame with F and click the spot on each frame.")


def drift_limit(width: int, diameter: float) -> float:
    """How far off a position may be before the user would correct it."""
    return float(max(6.0, 1.5 * diameter, 0.002 * width))


def click_motion(clicks: dict) -> list[tuple[float, float]]:
    """Moving spot's own constant-velocity prediction replayed over the clicks:
    [(how far the next click was from the prediction, the speed then)] in px,
    for every clicked frame whose two predecessors are clicked (a gap restarts
    the speed, as a new run would). It is how wide a search the animal's
    turning and speeding up need -- click noise included (I162)."""
    out, vel, prev = [], None, None
    for f in sorted(clicks):
        p = np.asarray(clicks[f], np.float64)
        if prev is not None and f == prev[0] + 1:
            if vel is not None:
                out.append((float(np.hypot(*(p - (prev[1] + vel)))), float(np.hypot(*vel))))
            step = p - prev[1]
            vel = step if vel is None else VEL_KEEP * vel + (1.0 - VEL_KEEP) * step
        else:
            vel = None
        prev = (f, p)
    return out


def motion_radius(clicks: dict) -> float:
    """The search the clicks call for: 1.5 x the 90th-percentile miss + 2 px
    (0 when the clicks have no three frames in a row)."""
    m = click_motion(clicks or {})
    if not m:
        return 0.0
    k, add = MOTION_MARGIN
    return float(k * np.percentile([e for e, _ in m], 90) + add)


def candidate_settings(sigma: float, clicks: dict | None = None) -> list[SpotSettings]:
    """The settings the test tries: each cue at its search radii x three speed
    gains, at the spot's measured scale. The smallest radius is the spot's
    size floor (bright / dark: max(6, 1.5 x its diameter); change: max(10, its
    diameter)); when the clicks show more turning than that covers
    (`motion_radius`), the search they call for and 1.6 / 2.5 x it are tried
    as well -- before, the radii came from the size alone, and the change
    cue's were fixed at 10 / 16 / 25 px (I162)."""
    d = spot_diameter(sigma)
    need = motion_radius(clicks) if clicks else 0.0
    out = []
    for cue in CUES:
        floor = max(10.0, d) if cue == "change" else max(MIN_RADIUS, 1.5 * d)
        gains = (0.5, 1.5, 2.5) if cue == "change" else (0.0, 0.5, 1.5)
        base = max(floor, need)
        radii = [floor] if base < 1.1 * floor else [floor, base]
        radii += [1.6 * base, 2.5 * base]
        for r in radii:
            for g in gains:
                out.append(SpotSettings(cue, round(min(MAX_RADIUS, r) * 2) / 2, g, float(sigma)))
    return out


@dataclass
class TestResult:
    """One point model / setting scored against the clicks."""
    label: str
    model: str                         # "spot" | "alltracker" | "cotracker3"
    settings: SpotSettings | None = None
    frames: int = 0                    # clicked frames scored (after the first) -- only ones the run really reached
    first_ok: int = 0                  # clicked frames followed before the first correction
    corrections: int = 0
    drifts: int = 0                    # ... where it was silently off by more than the limit
    stops: int = 0                     # ... where it stopped and said so
    errors: list = field(default_factory=list)
    error: str = ""                    # could not run (a sentence)
    unread: int | None = None          # the video stopped delivering frames here: the clicked frames from it on were not scored (I251)

    @property
    def median_error(self) -> float:
        return float(np.median(self.errors)) if self.errors else float("nan")

    def key(self):
        if self.error:
            return (1, 10 ** 9, 10 ** 9, float("inf"))
        m = self.median_error
        return (0, self.corrections, self.drifts, m if math.isfinite(m) else float("inf"))


def click_velocity(clicks: dict, f: int):
    """The speed a run started at click `f` knows: from the click before, else
    from the click after (the guidance: click two frames in a row, track from
    the first). None when neither neighbour is clicked."""
    cur = np.asarray(clicks[f], float)
    if f - 1 in clicks:
        return cur - np.asarray(clicks[f - 1], float)
    if f + 1 in clicks:
        return np.asarray(clicks[f + 1], float) - cur
    return None


class _Cand:
    """One setting run through the clicks in step with the others."""

    def __init__(self, settings: SpotSettings, clicks: dict, size):
        self.settings = settings
        self.clicks = clicks
        self.size = size
        self.res = TestResult(f"Moving spot: {settings.describe()}", "spot", settings)
        self.trk: SpotTracker | None = None
        self.waiting = False

    def seed(self, f, rgb, gray, hist):
        vel = click_velocity(self.clicks, f)
        self.trk = SpotTracker(SpotSettings(**self.settings.to_dict()), self.clicks[f], vel, self.size)
        self.trk.start(f, rgb, gray, hist)
        self.waiting = False

    def frame(self, f, rgb, gray, hist, limit):
        r = self.res
        if self.trk is None:
            self.seed(f, rgb, gray, hist)
            return
        if self.waiting:
            if f in self.clicks:
                self.seed(f, rgb, gray, hist)
            return
        fix = self.trk.step(f, rgb, gray, hist)
        clicked = f in self.clicks
        if fix is None:
            r.corrections += 1
            r.stops += 1
            if clicked:
                self.seed(f, rgb, gray, hist)
            else:
                self.waiting = True
            return
        if not clicked:
            return
        e = float(np.hypot(fix.x - self.clicks[f][0], fix.y - self.clicks[f][1]))
        if e > limit:
            r.corrections += 1
            r.drifts += 1
            self.seed(f, rgb, gray, hist)
            return
        r.errors.append(e)
        if r.corrections == 0:
            r.first_ok += 1


def score_spot_settings(frames, clicks: dict, settings_list, size, limit: float,
                        history_frames=(), ahead_frames=(), cancel=None, progress=None) -> list[TestResult]:
    """Run every setting through the clicked stretch in ONE pass over the
    frames. `frames` yields (frame index, RGB) from the first clicked frame to
    the last; `history_frames` = (index, grey) before it, for the change cue,
    and `ahead_frames` the ones after its first frame that stand in near the
    video's start (as `SpotRun.prefill`); `clicks` = {frame: (x, y)}. Each
    setting starts from the first click, and is re-placed from the click
    wherever it drifts more than `limit` or stops (a correction)."""
    cands = [_Cand(s, clicks, size) for s in settings_list]
    first = min(clicks)
    hist = ChangeHistory()
    hist.seed(history_frames, ahead_frames, first)       # I194: counts the frames the cue can use
    n = 0
    total = max(clicks) - first + 1
    last_seen = first - 1
    cancelled = False
    for f, rgb in frames:
        if cancel is not None and cancel():
            cancelled = True
            break
        gray = _grey(rgb)
        for c in cands:
            c.frame(f, rgb, gray, hist, limit)
        hist.push(f, gray)
        n += 1
        last_seen = f
        if progress is not None:
            progress(n, total)
    # the clicked frames the run really reached; when the video stopped delivering frames
    # before the last click, say from which frame (I251: it counted every click)
    unread = None if cancelled or last_seen >= max(clicks) else last_seen + 1
    for c in cands:
        c.res.frames = sum(1 for f in clicks if first < f <= last_seen)
        c.res.unread = unread
    return [c.res for c in cands]


def corrected_protocol(run_from, clicks: dict, limit: float, label: str, model: str,
                       cancel=None) -> TestResult:
    """The same protocol for a point model that runs as a whole (AllTracker /
    CoTracker3): `run_from(f, xy)` -> {frame: (x, y)} from frame f on (absent
    = no data). Re-started from the click wherever it drifts or has no data."""
    res = TestResult(label, model)
    frames = sorted(clicks)
    res.frames = len(frames) - 1
    f = frames[0]
    while True:
        if cancel is not None and cancel():
            break
        tr = run_from(f, clicks[f])
        nxt = None
        for g in frames:
            if g <= f:
                continue
            p = tr.get(g)
            if p is None or not np.isfinite(p).all():
                nxt, kind = g, "stop"
                break
            e = float(np.hypot(p[0] - clicks[g][0], p[1] - clicks[g][1]))
            if e > limit:
                nxt, kind = g, "drift"
                break
            res.errors.append(e)
            if res.corrections == 0:
                res.first_ok += 1
        if nxt is None:
            break
        res.corrections += 1
        if kind == "drift":
            res.drifts += 1
        else:
            res.stops += 1
        f = nxt
    return res


def best_spot(results: list[TestResult]) -> dict[str, TestResult]:
    """The best setting of each cue (the order of `candidate_settings` breaks ties)."""
    out: dict[str, TestResult] = {}
    for r in results:
        cue = r.settings.cue if r.settings is not None else ""
        if cue not in out or r.key() < out[cue].key():
            out[cue] = r
    return out


def recommend(results: list[TestResult]) -> TestResult | None:
    """The winner; a tie goes to AllTracker, then CoTracker3 (no extra clicks
    needed with them), then Moving spot."""
    order = {"alltracker": 0, "cotracker3": 1, "spot": 2}
    ok = [r for r in results if not r.error]
    if not ok:
        return None
    return min(ok, key=lambda r: (r.key()[:3], round(r.key()[3], 1), order.get(r.model, 3)))


def verdict_text(point: str, results: list[TestResult]) -> str:
    """The recommendation in plain words."""
    win = recommend(results)
    if win is None:
        return "No point model could be run on these clicks."
    n = win.frames
    others = [r for r in results if r is not win and not r.error]

    def corr(r):
        if r.corrections == 0:
            return "no correction"
        silent = f", {r.drifts} of them silent drifts" if r.drifts else ""
        return f"{r.corrections} correction{'s' if r.corrections != 1 else ''}{silent}"
    lead = (f"{win.label} followed {point} on {win.first_ok if win.corrections else n} of {n} clicked frames "
            f"{'with no correction' if win.corrections == 0 else 'before its first correction'}"
            f"{'' if not win.errors else f' (median {win.median_error:.1f} px from your clicks)'}"
            f"{'' if win.corrections == 0 else f' and needed {corr(win)} in all'}.")
    if others:
        cmp = "; ".join(f"{r.label}: {corr(r)}" for r in sorted(others, key=lambda r: r.key())[:3])
        lead += f" The others: {cmp}."
    if win.unread is not None:
        lead += (f" Frame {win.unread} of the video could not be read, so only the clicked frames before it "
                 "were tested.")
    # G118: since G62 the button changes only THIS point's tracker, not the project's
    if win.model == "spot":
        lead += (f" Use Moving spot for {point}: this target behaves like a single point, which is what Moving "
                 "spot follows, and it stops (instead of drifting) where it loses it.")
    else:
        lead += (f" Keep {win.label} for {point}: it needs no extra clicks. Moving spot is only for a target "
                 "small enough to be one point, where the others lose it.")
    return lead
