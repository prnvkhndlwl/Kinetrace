"""Object layer: promptable streaming video segmentation (SAM 2.1 / SAM 3 via
transformers) plus compact per-frame mask storage.

Why this exists (learned on real underwater lizard footage): appearance point
tracking holds textured features (the head) but loses thin undulating tails
and dark textureless bodies. A silhouette per frame gives (1) an object ROI
for high-resolution point tracking, (2) geometry-defined landmarks — tail tip,
midline fractions, extremities — and (3) a presence signal that survives
occlusion via the model's memory bank.

Design constraints inherited from the app:
* videos are 40k+ frames at 4K → strictly streaming, one frame in, one mask
  out, with the session pruned to a bounded memory window (an unpruned
  session grows ~9 MB per frame);
* masks are post-processed at a working resolution (max dim MASK_WORK_MAX_DIM)
  — an outline at 1024 px is indistinguishable on screen from one at 4K and
  the tail-tip landmark lands within ~2 native px, while the 4K upsample +
  transfer + contour pass would cost more than the model itself;
* self-contained folder → weights live in models/hf (HF_HOME is pinned before
  transformers is imported); a Hugging Face token for gated weights (SAM 3)
  lives in models/hf/token;
* the GUI thread never touches torch → callers run this in a worker.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
HF_DIR = MODELS_DIR / "hf"
os.environ.setdefault("HF_HOME", str(HF_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# backend id -> (hub repo, family, human label, approx download). A local copy
# of a repo placed in models/<backend id>/ (config.json + *.safetensors) is used
# in preference to the Hub — that is how gated SAM 3 weights are supplied.
BACKENDS: dict[str, tuple[str, str, str, str]] = {
    "sam2.1-base-plus": ("facebook/sam2.1-hiera-base-plus", "sam2", "SAM 2.1 base+ (fast, 617 MB)", "617 MB"),
    "sam2.1-large": ("facebook/sam2.1-hiera-large", "sam2", "SAM 2.1 large (1.7 GB, ~2x slower)", "1.7 GB"),
    "sam3": ("facebook/sam3", "sam3", "SAM 3 tracker (best masks, ~2x slower, 3.4 GB)", "3.4 GB"),
}
DEFAULT_BACKEND = "sam2.1-base-plus"   # what downloads on a fresh install (ungated, fast)


def preferred_backend() -> str:
    """SAM 3 when its weights are already inside the folder (measured on the
    real 4K lizard clip: 12.8 vs 27.3 fps for SAM 2.1 base+, visibly better masks),
    else the fast ungated default."""
    return "sam3" if local_dir("sam3") is not None else DEFAULT_BACKEND


def local_dir(backend: str) -> Path | None:
    """models/<backend>/ when it holds a usable transformers snapshot."""
    d = MODELS_DIR / backend
    if (d / "config.json").exists() and any(d.glob("*.safetensors")):
        return d
    return None
MEMORY_WINDOW = 32          # non-conditioning frames kept: > num_maskmem (7) + object pointers (16)
MASK_WORK_MAX_DIM = 1024    # masks are produced at this resolution, outlines scaled to native
CONTOUR_EPS_WORK = 0.8      # px at working res, polygon simplification for stored outlines
MIDLINE_SAMPLES = 32        # stored midline polyline length

_lock = threading.Lock()
_engines: dict[str, "Segmenter"] = {}


def repo_of(backend: str) -> str:
    return BACKENDS.get(backend, BACKENDS[DEFAULT_BACKEND])[0]


def model_is_cached(backend: str = DEFAULT_BACKEND) -> bool:
    """True if the weights are already inside the tool folder (no internet needed)."""
    if local_dir(backend) is not None:
        return True
    from kinetrace.downloads import hf_cached
    return hf_cached(repo_of(backend))


def load_path(backend: str) -> str:
    """What to hand to from_pretrained: the local folder if present, else the Hub repo."""
    d = local_dir(backend)
    return str(d) if d is not None else repo_of(backend)


def token_path() -> Path:
    return HF_DIR / "token"


def has_token() -> bool:
    return bool(os.environ.get("HF_TOKEN")) or token_path().exists()


def hf_cache_args() -> dict:
    """`from_pretrained` arguments that make a load read THIS folder's models/hf whatever
    HF_HOME the user's environment has (I195). `os.environ.setdefault("HF_HOME")` above keeps
    a value the user already set, so a load without these looked in THEIR cache (with
    local_files_only once the pinned snapshot was here): SAM failed to load after a
    successful download, and the token saved in Settings was never read. The hub folder is
    where `downloads.hf_snapshot` puts the weights; the token is models/hf/token unless
    HF_TOKEN is set."""
    kw: dict = {"cache_dir": str(HF_DIR / "hub")}
    if not os.environ.get("HF_TOKEN") and token_path().exists():
        tok = token_path().read_text(encoding="utf-8").strip()
        if tok:
            kw["token"] = tok
    return kw


def save_token(token: str) -> None:
    """Store a Hugging Face access token inside the tool folder (read by
    huggingface_hub because HF_HOME points here)."""
    token = token.strip()
    HF_DIR.mkdir(parents=True, exist_ok=True)
    if token:
        token_path().write_text(token, encoding="utf-8")
        if os.name != "nt":
            os.chmod(token_path(), 0o600)       # a secret: readable by this user only (I158)
    elif token_path().exists():
        token_path().unlink()


def backend_status(backend: str) -> tuple[str, str]:
    """('ready' | 'download' | 'gated', human explanation) without touching the network."""
    repo, family, label, size = BACKENDS[backend]
    if model_is_cached(backend):
        return "ready", f"{label}: weights cached in models/hf"
    if family == "sam3" and not has_token():
        return "gated", (f"{label}: the weights are gated on Hugging Face. Request access at "
                         f"https://huggingface.co/{repo}, then paste an access token in "
                         "Settings — it is stored in models/hf/token inside this folder.")
    return "download", f"{label}: first use downloads {size} into models/hf (internet needed once)"


def get_segmenter(backend: str = DEFAULT_BACKEND, progress=None, cancel=lambda: False) -> "Segmenter":
    """Process-wide singleton per backend. Safe to call from any thread.
    A first use downloads the weights with progress(label, done_bytes, total_bytes)
    (`downloads.hf_snapshot`, at the pinned commit; G45, I154)."""
    with _lock:
        eng = _engines.get(backend)
        if eng is None:
            if local_dir(backend) is None:
                from kinetrace import downloads
                downloads.hf_snapshot(repo_of(backend), f"the segmentation model ({BACKENDS[backend][2]})",
                                      progress, cancel)
            eng = Segmenter(backend)
            _engines[backend] = eng
        return eng


def loaded_backends() -> list[str]:
    return list(_engines)


@dataclass
class Prompt:
    """User input for one object on one frame. Points are native pixels; labels
    are 1 (this is the object) / 0 (this is not). A box is (x0, y0, x1, y1).
    A mask (native-res bool array) seeds a session from a previous result."""
    obj_id: int
    points: np.ndarray | None = None
    labels: np.ndarray | None = None
    box: tuple[float, float, float, float] | None = None
    mask: np.ndarray | None = None


@dataclass
class FrameMasks:
    """Masks of all tracked objects on one frame, at the WORKING resolution
    (`work_size`); `scale` maps working px -> native px."""
    frame: int
    obj_ids: list[int]
    masks: np.ndarray       # (n, h, w) bool
    scores: np.ndarray      # (n,) presence logits; > 0 means the object is visible
    work_size: tuple[int, int]      # (w, h)
    native_size: tuple[int, int]    # (w, h)

    @property
    def scale(self) -> float:
        return self.native_size[0] / self.work_size[0]

    @property
    def scale_xy(self) -> tuple[float, float]:
        """(native / working) per axis. `working_size` rounds each axis on its
        own, so on 2704x1520 x is 2.6406 and y 2.6389: using `scale` for y put
        the bottom rows 1 native px low (I61). Positions want this; lengths and
        areas can keep `scale`."""
        return (self.native_size[0] / self.work_size[0], self.native_size[1] / self.work_size[1])

    def index(self, obj_id: int) -> int:
        return self.obj_ids.index(obj_id)

    def mask(self, obj_id: int) -> np.ndarray:
        return self.masks[self.index(obj_id)]

    def present(self, obj_id: int) -> bool:
        return bool(self.scores[self.index(obj_id)] > 0)

    def native_mask(self, obj_id: int) -> np.ndarray:
        """Full-resolution bool mask (nearest upsample) — only when really needed."""
        m = self.mask(obj_id).astype(np.uint8)
        w, h = self.native_size
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def score_to_confidence(score: float) -> float:
    """Presence logit -> [0, 1] confidence comparable to the point tracker's.
    ~0.998 at a typical steady score of 8, 0.5 at 2, 0.12 at 0."""
    return float(1.0 / (1.0 + np.exp(-(float(score) - 2.0))))


def working_size(native_w: int, native_h: int) -> tuple[int, int]:
    s = min(1.0, MASK_WORK_MAX_DIM / max(native_w, native_h))
    return max(2, int(round(native_w * s))), max(2, int(round(native_h * s)))


class Segmenter:
    """Loaded promptable video model + processor. Create sessions with `new_session`."""

    def __init__(self, backend: str = DEFAULT_BACKEND):
        import torch
        self.backend = backend
        self.repo, self.family, self.label, _ = BACKENDS[backend]
        from kinetrace.device import pick_device, sam_dtype
        # one rule for every model: CUDA, else Apple's GPU, else the CPU;
        # bfloat16 only on CUDA (Metal / CPU bfloat16 kernels are partial or slow)
        self.device = pick_device()[0]
        self.dtype = sam_dtype(torch, self.device)
        if self.family == "sam3":
            from transformers import Sam3TrackerVideoModel as M, Sam3TrackerVideoProcessor as P
        else:
            from transformers import Sam2VideoModel as M, Sam2VideoProcessor as P
        src = load_path(backend)
        from kinetrace.downloads import hf_load_args
        # a local folder, or the Hub repo at the commit every test ran against (I154): with
        # that commit's files in models/hf nothing is asked of the Hub
        rev = {} if local_dir(backend) is not None else {**hf_load_args(src), **hf_cache_args()}
        self.model = M.from_pretrained(src, dtype=self.dtype, **rev).to(self.device).eval()
        self.proc = P.from_pretrained(src, **rev)
        self._torch = torch
        self.lock = threading.Lock()   # one inference at a time per model

    def new_session(self, start_frame: int, native_size: tuple[int, int]) -> "SegSession":
        return SegSession(self, start_frame, native_size)


class SegSession:
    """One forward streaming pass from `start_frame`: call `step` once per
    consecutive frame. Prompts may be given on any frame (the first call must
    carry at least one), which is how corrections work: pause, click, resume."""

    def __init__(self, seg: Segmenter, start_frame: int, native_size: tuple[int, int]):
        self.seg = seg
        self.start = int(start_frame)
        self.next_frame = int(start_frame)
        self.obj_ids: list[int] = []
        self.native_size = (int(native_size[0]), int(native_size[1]))
        self.work = working_size(*self.native_size)
        self.session = seg.proc.init_video_session(inference_device=seg.device, dtype=seg.dtype)

    def _to_work(self, xy) -> list[float]:
        # per axis: the two axes of the working size are rounded separately (I61)
        sx = self.work[0] / self.native_size[0]
        sy = self.work[1] / self.native_size[1]
        return [float(xy[0]) * sx, float(xy[1]) * sy]

    def step(self, frame_rgb: np.ndarray, frame_idx: int,
             prompts: list[Prompt] | None = None) -> FrameMasks:
        if frame_idx != self.next_frame:
            raise ValueError(f"streaming session expects frame {self.next_frame}, got {frame_idx}")
        torch, proc, model = self.seg._torch, self.seg.proc, self.seg.model
        local = frame_idx - self.start
        ww, wh = self.work
        h, w = frame_rgb.shape[:2]
        if (w, h) != (ww, wh):
            frame_rgb = cv2.resize(frame_rgb, (ww, wh), interpolation=cv2.INTER_AREA)
        with self.seg.lock:
            inputs = proc(images=frame_rgb, device=self.seg.device, return_tensors="pt")
            prompted: list[int] = []
            for p in prompts or []:
                kw = {}
                if p.points is not None and len(p.points):
                    kw["input_points"] = [[[self._to_work(pt) for pt in np.asarray(p.points, np.float64)]]]
                    kw["input_labels"] = [[[int(v) for v in np.asarray(p.labels).ravel()]]]
                if p.box is not None:
                    x0, y0 = self._to_work(p.box[:2])
                    x1, y1 = self._to_work(p.box[2:])
                    kw["input_boxes"] = [[[x0, y0, x1, y1]]]
                if p.mask is not None:
                    m = np.asarray(p.mask).astype(np.uint8)
                    if m.shape != (wh, ww):
                        m = cv2.resize(m, (ww, wh), interpolation=cv2.INTER_NEAREST)
                    kw["input_masks"] = [m.astype(np.float32)]
                if not kw:
                    continue
                proc.add_inputs_to_inference_session(
                    inference_session=self.session, frame_idx=local, obj_ids=int(p.obj_id),
                    original_size=inputs.original_sizes[0], **kw)
                prompted.append(int(p.obj_id))
                if p.obj_id not in self.obj_ids:
                    self.obj_ids.append(int(p.obj_id))
            if len(prompted) > 1:
                # transformers (SAM 2 and SAM 3 video processors) ASSIGNS
                # `obj_with_new_inputs = obj_ids` on every call, so prompting two
                # objects on one frame leaves only the last one flagged; the
                # first then has no conditioning output and the model raises
                # "maskmem_features in conditioning outputs cannot be empty".
                # Restore the union (measured 2026-09-19: two clicks on frame 0
                # failed on both backends, and worked with this line).
                self.session.obj_with_new_inputs = list(dict.fromkeys(prompted))
            if not self.obj_ids:
                raise ValueError("the first step of a session needs at least one prompt")
            with torch.inference_mode():
                out = model(inference_session=self.session,
                            frame=inputs.pixel_values[0].to(self.seg.device, self.seg.dtype),
                            frame_idx=local)
            masks = proc.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes,
                                            binarize=True)[0]                     # (n, 1, h, w)
            m = masks[:, 0].cpu().numpy().astype(bool)
            scores = (out.object_score_logits.flatten().float().cpu().numpy()
                      if out.object_score_logits is not None else np.ones(len(m), np.float32))
            ids = [int(i) for i in out.object_ids] if out.object_ids is not None else list(self.obj_ids)
            self._prune(local)
        self.next_frame += 1
        return FrameMasks(frame_idx, ids, m, scores.astype(np.float32), self.work, self.native_size)

    def _prune(self, local: int) -> None:
        """Bound the session: the model only attends to the last num_maskmem
        memories and max_object_pointers_in_encoder pointers, so everything
        older than MEMORY_WINDOW frames (and every past preprocessed frame) can go."""
        s = self.session
        pf = s.processed_frames
        if pf:
            for k in [k for k in pf if k < local]:
                del pf[k]
        for d in s.output_dict_per_obj.values():
            nc = d.get("non_cond_frame_outputs", {})
            for k in [k for k in nc if k < local - MEMORY_WINDOW]:
                del nc[k]

    def memory_footprint(self) -> tuple[int, int]:
        """(processed frames held, max non-conditioning outputs held per object) — for tests."""
        s = self.session
        n_pf = len(s.processed_frames or {})
        n_nc = max((len(d.get("non_cond_frame_outputs", {})) for d in s.output_dict_per_obj.values()), default=0)
        return n_pf, n_nc


def _scale_xy(scale) -> tuple[float, float]:
    """A working -> native scale given as one number or as (sx, sy)
    (`FrameMasks.scale_xy`: the working size rounds each axis on its own, I61)."""
    if np.ndim(scale) == 0:
        return float(scale), float(scale)
    sx, sy = scale
    return float(sx), float(sy)


def outline_polygons(mask_work: np.ndarray, scale, eps: float = CONTOUR_EPS_WORK) -> list[np.ndarray]:
    """External outlines of a working-res mask as native-px int32 polygons.
    `scale` = native / working, one number or (sx, sy)."""
    sxy = np.array(_scale_xy(scale))
    cs, _ = cv2.findContours(np.asarray(mask_work).astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cs:
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
        if len(c) >= 3:
            polys.append(np.round((c + 0.5) * sxy - 0.5).astype(np.int32))
    return polys


def summarize_mask(mask_work: np.ndarray, scale, score: float,
                   midline: np.ndarray | None = None) -> dict:
    """Everything the session keeps about one frame's mask, in native px:
    bbox (inclusive), area, centroid, presence score, outline polygons, and
    an optional midline polyline. `area == 0` means no mask. `scale` = native
    / working, one number or (sx, sy) (pass `FrameMasks.scale_xy`, I61)."""
    sx, sy = _scale_xy(scale)
    m8 = np.asarray(mask_work).astype(np.uint8)
    area_w = int(m8.sum())
    d = {"bbox": (-1, -1, -1, -1), "area": 0, "centroid": (np.nan, np.nan),
         "score": float(score), "polys": [], "midline": None}
    if area_w == 0:
        return d
    ys, xs = np.nonzero(m8)
    d["bbox"] = (int(round(xs.min() * sx)), int(round(ys.min() * sy)),
                 int(round((xs.max() + 1) * sx) - 1), int(round((ys.max() + 1) * sy) - 1))
    d["area"] = int(round(area_w * sx * sy))
    d["centroid"] = ((float(xs.mean()) + 0.5) * sx - 0.5, (float(ys.mean()) + 0.5) * sy - 0.5)
    d["polys"] = outline_polygons(m8, (sx, sy))
    if midline is not None and len(midline):
        d["midline"] = np.asarray(midline, np.float32).reshape(-1, 2)
    return d


IDENTITY_BODY_K = 2.0     # a silhouette back after a gap may lie this many body sizes from where the animal was heading
IDENTITY_SPEED_K = 1.0    # ... plus this many times the distance its speed covers over the gap
IDENTITY_JUMP_K = 4.0     # no gap: a one-frame jump beyond this x (body size + speed) is another object
IDENTITY_HISTORY = 6      # present frames the speed and the body size are taken over


class SegmentIdentity:
    """(I260) Is this frame's silhouette still the animal the user clicked? SAM 3 (and SAM 2.1)
    keep a tracked object alive through a loss by re-acquiring whatever looks most like it: on a
    clip of several bats, the clicked bat left the picture, SAM found nothing for 12 frames and
    then took another bat -- and again each time that one left -- under the animal-lost rule's 16
    frames. Here a silhouette that comes back after a gap must lie near where the animal was
    heading (`IDENTITY_BODY_K` body sizes + `IDENTITY_SPEED_K` x the distance its speed covers),
    and a silhouette with no gap may not jump more than `IDENTITY_JUMP_K` x (body size + speed) in
    one frame; anything else is ANOTHER object. Body size = the median bbox diagonal over the last
    `IDENTITY_HISTORY` present frames (a frame where SAM merged two animals does not inflate it).
    A frame the user prompted (a click / box) is trusted and starts the history afresh. Pure
    geometry on `summarize_mask`'s native-px summaries."""

    def __init__(self):
        self._hist: list[tuple[int, float, float, float]] = []    # (frame, cx, cy, bbox diagonal)

    def check(self, frame: int, summ: dict, prompted: bool = False) -> int | None:
        """None = the same animal (or nothing to judge: an empty frame, the first one); else the
        first frame that is no longer the animal -- the first empty frame of the gap it came back
        after, or this frame."""
        if summ["area"] <= 0:
            return None
        cx, cy = (float(v) for v in summ["centroid"])
        x0, y0, x1, y1 = summ["bbox"]
        size = float(np.hypot(x1 - x0 + 1, y1 - y0 + 1))
        if not (np.isfinite(cx) and np.isfinite(cy)):
            return None
        if prompted or not self._hist:
            self._hist = [(frame, cx, cy, size)]
            return None
        fl, lx, ly, _ = self._hist[-1]
        f0, x0h, y0h, _ = self._hist[0]
        vx, vy = ((lx - x0h) / (fl - f0), (ly - y0h) / (fl - f0)) if fl > f0 else (0.0, 0.0)
        speed = float(np.hypot(vx, vy))
        body = float(np.median([h[3] for h in self._hist]))
        step = frame - fl                                   # 1 = no gap
        dist = float(np.hypot(cx - (lx + vx * step), cy - (ly + vy * step)))
        limit = (IDENTITY_JUMP_K * (body + speed) if step <= 1
                 else IDENTITY_BODY_K * body + IDENTITY_SPEED_K * speed * step)
        if dist > limit:
            return fl + 1
        self._hist = (self._hist + [(frame, cx, cy, size)])[-IDENTITY_HISTORY:]
        return None


class MaskTrack:
    """One object's silhouette over the whole video, stored compactly: outline
    polygons (not bitmaps), a 32-point midline, and per-frame bbox / area /
    centroid / score. A 40k-frame 4K video costs tens of MB, not gigabytes."""

    def __init__(self, n_frames: int, native_w: int = 0, native_h: int = 0):
        self.n_frames = int(n_frames)
        # native size, needed by rasterize() at other resolutions; set by the
        # session. INSTANCE attributes that copy() carries (I53): they were class
        # defaults of 0, so the undo snapshot's copy lost them and a resume after
        # Ctrl+Z seeded SAM with an empty or misplaced mask at 4K.
        self.native_w = int(native_w)
        self.native_h = int(native_h)
        self.bbox = np.full((n_frames, 4), -1, np.int32)          # x0, y0, x1, y1 inclusive, native px
        self.area = np.zeros(n_frames, np.int32)                   # native px^2
        self.centroid = np.full((n_frames, 2), np.nan, np.float32)
        self.score = np.full(n_frames, np.nan, np.float32)         # presence logit
        self.contours: dict[int, list[np.ndarray]] = {}          # frame -> [(M, 2) int32 native]
        self.midline: dict[int, np.ndarray] = {}                 # frame -> (MIDLINE_SAMPLES, 2) f32 native

    # ---------------------------------------------------------------- writes
    def set_summary(self, frame: int, d: dict) -> None:
        """Store a `summarize_mask` result (what the tracking worker emits)."""
        if d["area"] <= 0:
            self._blank(frame)
            self.score[frame] = float(d["score"])
            return
        self.bbox[frame] = d["bbox"]
        self.area[frame] = int(d["area"])
        self.centroid[frame] = d["centroid"]
        self.score[frame] = float(d["score"])
        self.contours[frame] = [np.asarray(p, np.int32) for p in d["polys"]]
        if d.get("midline") is not None and len(d["midline"]):
            self.midline[frame] = np.asarray(d["midline"], np.float32).reshape(-1, 2)
        else:
            self.midline.pop(frame, None)

    def set_from_work(self, frame: int, mask_work: np.ndarray, scale, score: float,
                      midline: np.ndarray | None = None) -> None:
        """Store a working-resolution mask (outline scaled to native px;
        `scale` = native / working, one number or (sx, sy))."""
        self.set_summary(frame, summarize_mask(mask_work, scale, score, midline))

    def set(self, frame: int, mask: np.ndarray, score: float = 1.0,
            midline: np.ndarray | None = None) -> None:
        """Store a native-resolution mask."""
        self.set_from_work(frame, mask, 1.0, score, midline)

    def _blank(self, frame: int) -> None:
        self.contours.pop(frame, None)
        self.midline.pop(frame, None)
        self.bbox[frame] = -1
        self.area[frame] = 0
        self.centroid[frame] = np.nan
        self.score[frame] = np.nan

    def clear(self, start: int, end: int) -> None:
        for f in range(max(0, start), min(self.n_frames - 1, end) + 1):
            self._blank(f)

    def copy(self) -> "MaskTrack":
        mt = MaskTrack(self.n_frames, self.native_w, self.native_h)
        mt.bbox = self.bbox.copy()
        mt.area = self.area.copy()
        mt.centroid = self.centroid.copy()
        mt.score = self.score.copy()
        mt.contours = {f: [p.copy() for p in ps] for f, ps in self.contours.items()}
        mt.midline = {f: m.copy() for f, m in self.midline.items()}
        return mt

    # ----------------------------------------------------------------- reads
    def has(self, frame: int) -> bool:
        return 0 <= frame < self.n_frames and self.area[frame] > 0

    def frames(self) -> np.ndarray:
        return np.nonzero(self.area > 0)[0]

    def n_masked(self) -> int:
        return int((self.area > 0).sum())

    def rasterize(self, frame: int, height: int, width: int) -> np.ndarray:
        """Bool mask (at the requested size) rebuilt from the stored native
        outline. Without a known native size the outline is taken to be in the
        requested resolution already."""
        out = np.zeros((height, width), np.uint8)
        polys = self.contours.get(frame)
        if polys:
            sx = width / self.native_w if self.native_w else 1.0
            sy = height / self.native_h if self.native_h else 1.0
            # the inverse of outline_polygons' pixel-centre mapping; p * s shifted a
            # 4K outline by +0.37 working px
            scaled = [np.round((p.astype(np.float64) + 0.5) * [sx, sy] - 0.5).astype(np.int32).reshape(-1, 1, 2)
                      for p in polys]
            cv2.fillPoly(out, scaled, 1)
        return out.astype(bool)

    # ----------------------------------------------------------- persistence
    def to_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        frames = sorted(self.contours)
        pts, off, fr = [], [0], []
        for f in frames:
            for poly in self.contours[f]:
                pts.append(poly)
                off.append(off[-1] + len(poly))
                fr.append(f)
        mid_frames = sorted(self.midline)
        return {
            f"{prefix}bbox": self.bbox, f"{prefix}area": self.area,
            f"{prefix}centroid": self.centroid, f"{prefix}score": self.score,
            f"{prefix}cpts": (np.concatenate(pts) if pts else np.zeros((0, 2), np.int32)).astype(np.int32),
            f"{prefix}coff": np.asarray(off, np.int64),
            f"{prefix}cframe": np.asarray(fr, np.int32),
            f"{prefix}mframe": np.asarray(mid_frames, np.int32),
            f"{prefix}mpts": (np.stack([self.midline[f] for f in mid_frames])
                              if mid_frames else np.zeros((0, MIDLINE_SAMPLES, 2), np.float32)).astype(np.float32),
        }

    @classmethod
    def from_arrays(cls, prefix: str, arrays) -> "MaskTrack":
        bbox = np.asarray(arrays[f"{prefix}bbox"])
        mt = cls(len(bbox))
        mt.bbox = bbox.astype(np.int32)
        mt.area = np.asarray(arrays[f"{prefix}area"]).astype(np.int32)
        mt.centroid = np.asarray(arrays[f"{prefix}centroid"]).astype(np.float32)
        mt.score = np.asarray(arrays[f"{prefix}score"]).astype(np.float32)
        # one conversion, then views: a per-outline astype cost ~0.25 s on a
        # 40k-frame silhouette. Outlines are replaced, never edited in place.
        pts = np.asarray(arrays[f"{prefix}cpts"], np.int32); off = np.asarray(arrays[f"{prefix}coff"])
        fr = np.asarray(arrays[f"{prefix}cframe"])
        for i, f in enumerate(fr.tolist()):
            mt.contours.setdefault(f, []).append(pts[off[i]:off[i + 1]])
        if f"{prefix}mframe" in getattr(arrays, "files", arrays):
            mf = np.asarray(arrays[f"{prefix}mframe"]); mp = np.asarray(arrays[f"{prefix}mpts"], np.float32)
            for i, f in enumerate(mf.tolist()):
                mt.midline[f] = mp[i]
        return mt
