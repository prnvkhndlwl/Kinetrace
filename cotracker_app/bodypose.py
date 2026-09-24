"""Human body layer, model half: turn a video frame into people with joints.

Two backends, one interface (`BodyEstimator.step`):

* **SAM 3D Body** (Meta, 11/2025) -- single-image full-body mesh recovery on
  the Momentum Human Rig. It is the reason this layer exists: it returns
  metric 3D joints, per-joint global rotations and a mesh from ONE camera, so
  joint angles are real angles in space rather than angles in a picture. Its
  checkpoints are gated on Hugging Face and its inference code is a GitHub
  checkout, not a pip package, so it is wired exactly the way SAM 3 and
  AllTracker already are: a vendored repo under `models/sam-3d-body/` and
  weights under `models/<backend>/`, both supplied by the user, and the
  feature reports honestly when they are absent instead of failing at run
  time.
* **ViTPose** (2D keypoints, via `transformers`, ungated) -- a person detector
  (RT-DETR v2) followed by a top-down keypoint model. No new dependency, no
  licence to accept, ~425 MB, and it runs today. It gives 17 COCO joints in
  the image plane, which is enough for stride, cadence and image-plane angles,
  and -- when the project is calibrated and the same person is digitised in
  two or more views -- enough for true 3D by triangulation.

Everything here obeys the app's standing rules: the GUI thread never touches
torch (callers run this on a worker), nothing is written outside the tool
folder (`HF_HOME` and `torch.hub` are pinned to `models/`), and frames go in
one at a time so a 40k-frame video costs no more memory than a 1-frame one.
"""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cotracker_app.body import BodyRig, RIGS, canon, rig_of

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
HF_DIR = MODELS_DIR / "hf"
os.environ.setdefault("HF_HOME", str(HF_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Where a SAM 3D Body GitHub checkout goes. `demo.py` there uses pyrootutils to
# put the repo root on sys.path; we do the same thing explicitly.
S3DB_REPO = MODELS_DIR / "sam-3d-body"

DETECTOR_REPO = "PekingU/rtdetr_v2_r18vd"       # 81 MB, COCO, person = class 0
DETECTOR_PERSON_CLASS = 0
MAX_WORK_DIM = 1280     # frames are downscaled for detection; boxes scale back


@dataclass
class BackendSpec:
    key: str
    label: str
    kind: str            # "sam3d_body" | "vitpose"
    repo: str            # Hugging Face repo id
    rig: str             # key into body.RIGS
    size: str
    gated: bool = False
    needs_code: bool = False     # requires the vendored GitHub checkout
    note: str = ""


BACKENDS: dict[str, BackendSpec] = {
    "sam-3d-body-dinov3": BackendSpec(
        "sam-3d-body-dinov3", "SAM 3D Body - DINOv3-H+ (best, 840M)", "sam3d_body",
        "facebook/sam-3d-body-dinov3", "mhr70", "3.4 GB", gated=True, needs_code=True,
        note="Meta's single-image 3D human mesh model. Real 3D joints and joint angles "
             "from one camera."),
    "sam-3d-body-vith": BackendSpec(
        "sam-3d-body-vith", "SAM 3D Body - ViT-H (631M)", "sam3d_body",
        "facebook/sam-3d-body-vith", "mhr70", "2.6 GB", gated=True, needs_code=True,
        note="Same outputs as the DINOv3 checkpoint, a little lighter."),
    "vitpose-base": BackendSpec(
        "vitpose-base", "ViTPose base - 2D keypoints (no licence needed, 425 MB)",
        "vitpose", "usyd-community/vitpose-base-simple", "coco17", "425 MB",
        note="17 joints in the picture, not in space. Angles are image-plane angles "
             "unless you triangulate two or more calibrated cameras."),
}
DEFAULT_BACKEND = "vitpose-base"

_lock = threading.Lock()


# --------------------------------------------------------------- availability

def _candidate_dirs(key: str):
    """Where a backend's files might sit. The folder name people end up with
    is whatever `hf download <repo> --local-dir ...` was pointed at, so both
    the repo spelling (sam-3d-body-dinov3) and the compact one are accepted."""
    seen = []
    for name in (key, key.replace("sam-3d-body-", "sam3d-body-"),
                 key.replace("sam3d-body-", "sam-3d-body-")):
        if name not in seen:
            seen.append(name)
    return [MODELS_DIR / n for n in seen]


def local_dir(key: str) -> Path | None:
    """models/<key>/ when it holds a usable snapshot. This is how gated
    weights are supplied without the app ever touching the network -- the
    same arrangement `segmenter.local_dir` uses for SAM 3."""
    spec = BACKENDS.get(key)
    for d in _candidate_dirs(key):
        if not d.is_dir():
            continue
        if spec is not None and spec.kind == "sam3d_body":
            # The SAM 3D Body release is a lightning checkpoint plus the MHR
            # asset, not a transformers snapshot.
            if any(d.glob("*.ckpt")):
                return d
            continue
        if (d / "config.json").exists() and any(d.glob("*.safetensors")):
            return d
    return None


def s3db_parts(key: str) -> tuple[Path | None, Path | None]:
    """(model checkpoint, MHR asset) for a SAM 3D Body backend; either may be
    None. They are separate downloads from the same gated repo and a partial
    `hf download` very easily brings one without the other, so the two are
    reported apart rather than as one yes/no."""
    d = local_dir(key)
    if d is None:
        return None, None
    ckpt = d / "model.ckpt"
    if not ckpt.exists():
        cands = sorted(d.glob("*.ckpt"))
        ckpt = cands[0] if cands else None
    mhr = None
    for cand in (d / "assets" / "mhr_model.pt", d / "mhr_model.pt"):
        if cand.exists():
            mhr = cand
            break
    if mhr is None:
        # the asset is identical across the checkpoints, so a copy sitting
        # beside the other one will do
        for other in BACKENDS:
            if other == key or BACKENDS[other].kind != "sam3d_body":
                continue
            od = local_dir(other)
            if od is None:
                continue
            for cand in (od / "assets" / "mhr_model.pt", od / "mhr_model.pt"):
                if cand.exists():
                    mhr = cand
                    break
            if mhr is not None:
                break
    return ckpt, mhr


def s3db_checkpoint(key: str) -> tuple[Path, Path] | None:
    ckpt, mhr = s3db_parts(key)
    return None if (ckpt is None or mhr is None) else (ckpt, mhr)


def code_available(key: str) -> bool:
    """True when the vendored SAM 3D Body checkout is present and importable."""
    if not BACKENDS[key].needs_code:
        return True
    return (S3DB_REPO / "sam_3d_body" / "sam_3d_body_estimator.py").exists()


def hub_cached(repo: str) -> bool:
    snaps = HF_DIR / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not snaps.exists():
        return False
    return any(any(p.glob("*.safetensors")) or any(p.glob("*.ckpt"))
               for p in snaps.iterdir() if p.is_dir())


def backend_status(key: str) -> tuple[str, str]:
    """('ready' | 'download' | 'needs-code' | 'needs-weights', plain English),
    without touching the network. Every branch says what the user must DO."""
    spec = BACKENDS[key]
    if spec.needs_code and not code_available(key):
        return ("needs-code",
                f"{spec.label} also needs Meta's inference code. Clone "
                f"https://github.com/facebookresearch/sam-3d-body into "
                f"{S3DB_REPO} (the folder must contain sam_3d_body/).")
    if spec.kind == "sam3d_body":
        ckpt, mhr = s3db_parts(key)
        if ckpt is not None and mhr is not None:
            return ("ready", f"{spec.label} is ready ({spec.size} already in "
                             f"{ckpt.parent}).")
        if ckpt is not None and mhr is None:
            # By far the likeliest half-installed state: model.ckpt is 2 GB and
            # obvious, the rig asset is small and sits in a subfolder.
            return ("needs-weights",
                    f"{spec.label}: model.ckpt is here but the Momentum Human Rig asset "
                    f"is not. The model cannot be built without it -- it is the part that "
                    f"turns the predicted parameters into a body. Download "
                    f"assets/mhr_model.pt from https://huggingface.co/{spec.repo} into "
                    f"{ckpt.parent / 'assets'}.")
        return ("needs-weights",
                f"{spec.label} weights are gated. Request access at "
                f"https://huggingface.co/{spec.repo}, then download model.ckpt and "
                f"assets/mhr_model.pt into {MODELS_DIR / key}.")
    if local_dir(key) is not None or hub_cached(spec.repo):
        return ("ready", f"{spec.label} is ready.")
    return ("download", f"{spec.label} will download {spec.size} on first use.")


def detector_ready() -> bool:
    return hub_cached(DETECTOR_REPO) or (MODELS_DIR / "rtdetr").is_dir()


def available_backends() -> list[str]:
    """Backends that could run right now (ready, or only need a download)."""
    return [k for k in BACKENDS if backend_status(k)[0] in ("ready", "download")]


def preferred_backend() -> str:
    """SAM 3D Body when the user has supplied code and weights -- it is the
    only backend that gives 3D from one camera -- else the ungated 2D one."""
    for k in ("sam-3d-body-dinov3", "sam-3d-body-vith"):
        if backend_status(k)[0] == "ready":
            return k
    return DEFAULT_BACKEND


# ------------------------------------------------------------------- results

@dataclass
class PersonPose:
    """One person on one frame, in the estimator's own rig order."""
    joints2d: np.ndarray                      # (J, 2) native video pixels
    conf: np.ndarray                          # (J,)
    bbox: np.ndarray                          # (4,) x0, y0, x1, y1 native px
    score: float = 1.0
    # (J, 3) metres, camera AXES (x right, y down, z forward) but centred on
    # the body: SAM 3D Body poses with zero global translation, and the
    # position in the camera's frame is joints3d + cam_t (upstream adds
    # pred_cam_t before projecting). The mesh `vertices` share that origin.
    joints3d: np.ndarray | None = None
    vertices: np.ndarray | None = None        # (V, 3) mesh, when the backend has one
    focal: float = float("nan")               # px, the backend's own estimate
    cam_t: np.ndarray | None = None           # (3,) metres, body origin in the camera frame


def _clip_box(box, w: int, h: int, pad: float = 0.0) -> np.ndarray:
    x0, y0, x1, y1 = (float(v) for v in box)
    if pad:
        dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
        x0, y0, x1, y1 = x0 - dx, y0 - dy, x1 + dx, y1 + dy
    return np.array([max(0.0, x0), max(0.0, y0), min(w - 1.0, x1), min(h - 1.0, y1)],
                    np.float32)


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ub = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return float(inter / (ua + ub - inter)) if (ua + ub - inter) > 0 else 0.0


def mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    """Tight box around a bool mask, or None when it is empty. Used to drive
    the pose model from the app's existing SAM silhouette."""
    ys, xs = np.nonzero(np.asarray(mask).astype(bool))
    if not len(xs):
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)


# ---------------------------------------------------------------- estimators

class BodyEstimator:
    """One frame in, a list of people out. Subclasses own their model."""
    rig: BodyRig
    backend: str = ""
    gives_3d: bool = False

    def step(self, bgr: np.ndarray, boxes: list | None = None,
             masks: list | None = None) -> list[PersonPose]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _PersonDetector:
    """RT-DETR v2 restricted to the COCO person class. Detection runs on a
    downscaled copy (boxes are scaled back) -- a 4K frame costs the detector
    nothing extra in accuracy for a person that fills a useful part of it."""

    def __init__(self, device, threshold: float = 0.4):
        import torch
        from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection
        self._torch = torch
        local = MODELS_DIR / "rtdetr"
        path = str(local) if local.is_dir() and (local / "config.json").exists() else DETECTOR_REPO
        self.proc = AutoImageProcessor.from_pretrained(path)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(path).to(device).eval()
        self.device = device
        self.threshold = float(threshold)

    def detect(self, bgr: np.ndarray, max_people: int) -> list[tuple[np.ndarray, float]]:
        h, w = bgr.shape[:2]
        sc = min(1.0, MAX_WORK_DIM / max(h, w))
        small = cv2.resize(bgr, (int(round(w * sc)), int(round(h * sc))),
                           interpolation=cv2.INTER_AREA) if sc < 1.0 else bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        inp = self.proc(images=rgb, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model(**inp)
        res = self.proc.post_process_object_detection(
            out, target_sizes=[(small.shape[0], small.shape[1])], threshold=self.threshold)[0]
        keep = []
        for box, lab, sco in zip(res["boxes"].cpu().numpy(), res["labels"].cpu().numpy(),
                                 res["scores"].cpu().numpy()):
            if int(lab) != DETECTOR_PERSON_CLASS:
                continue
            keep.append((np.asarray(box, np.float32) / max(sc, 1e-9), float(sco)))
        keep.sort(key=lambda t: -t[1])
        return [(_clip_box(b, w, h), s) for b, s in keep[:max_people]]


class ViTPoseEstimator(BodyEstimator):
    """Top-down 2D keypoints. The box comes from the caller (the app's SAM
    silhouette, or a hand-drawn region) when there is one, else from the
    detector, else the whole frame."""

    gives_3d = False

    def __init__(self, spec: BackendSpec, device, max_people: int = 1,
                 detector_threshold: float = 0.4, use_detector: bool = True,
                 allow_full_frame: bool = False):
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        self._torch = torch
        self.backend = spec.key
        path = str(local_dir(spec.key) or spec.repo)
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = VitPoseForPoseEstimation.from_pretrained(path).to(device).eval()
        self.device = device
        self.max_people = max(1, int(max_people))
        self.detector = _PersonDetector(device, detector_threshold) if use_detector else None
        # Falling back to "the person fills the frame" has to be asked for. A
        # top-down model ALWAYS returns 17 keypoints for whatever box it is
        # given, so an implicit fallback turns every frame with no prompt into
        # a confident pose of the background.
        self.allow_full_frame = bool(allow_full_frame)
        self.rig = self._rig_from_config(spec)

    def _rig_from_config(self, spec: BackendSpec) -> BodyRig:
        """Take the joint names from the checkpoint rather than trusting the
        backend table: a different ViTPose checkpoint (Halpe, WholeBody) has a
        different joint set, and a silently mis-ordered rig would put the
        elbow angle on the knee."""
        lab = getattr(self.model.config, "id2label", None) or {}
        names = [canon(lab[i]) for i in sorted(lab)] if lab else []
        base = rig_of(spec.rig)
        if names and names != base.joints:
            return BodyRig(f"{spec.rig}-custom", f"{spec.label} ({len(names)} joints)",
                           names, base.bone_names, up=base.up, units="px")
        return base

    def step(self, bgr, boxes=None, masks=None) -> list[PersonPose]:
        h, w = bgr.shape[:2]
        found: list[tuple[np.ndarray, float]] = []
        if boxes:
            found = [(_clip_box(b, w, h), 1.0) for b in boxes if b is not None][:self.max_people]
        elif masks:
            for m in masks[:self.max_people]:
                bb = mask_bbox(m)
                if bb is not None:
                    found.append((_clip_box(bb, w, h, pad=0.08), 1.0))
        elif self.detector is not None:
            found = self.detector.detect(bgr, self.max_people)
        if not found:
            if not self.allow_full_frame:
                return []
            found = [(np.array([0, 0, w - 1, h - 1], np.float32), 1.0)]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # the processor wants COCO xywh, one list of boxes per image
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
                for b, _ in found]
        inp = self.proc(rgb, boxes=[xywh], return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model(**inp)
        res = self.proc.post_process_pose_estimation(out, boxes=[xywh])[0]
        people = []
        for k, r in enumerate(res):
            kp = np.asarray(r["keypoints"], np.float32).reshape(-1, 2)
            sc = np.asarray(r["scores"], np.float32).reshape(-1)
            people.append(PersonPose(joints2d=kp, conf=sc, bbox=found[k][0],
                                     score=float(found[k][1])))
        return people


class Sam3DBodyEstimator(BodyEstimator):
    """Meta's SAM 3D Body behind the same one-frame interface.

    This is a thin adapter over the upstream `SAM3DBodyEstimator`
    (`sam_3d_body/sam_3d_body_estimator.py`), which returns, per detected
    person, `pred_keypoints_3d` / `pred_keypoints_2d` on the 70-joint MHR set
    plus the mesh, the focal length and the per-joint global rotations. Two
    upstream details are load-bearing and are handled here:

    * `process_one_image` expects RGB when it is handed an array, and it
      prints a warning saying so -- the conversion happens here, once.
    * mask-conditioned inference asserts that boxes were supplied as well, so
      a mask prompt always travels with its box.

    NOTE: this path is written against the released upstream API and is
    exercised by the test suite through a stand-in that reproduces that API.
    It has NOT been run against the real gated checkpoints in this
    environment, because they cannot be downloaded without gated
    Hugging Face access. Everything downstream of it -- joints, angles,
    storage, views, exports -- is verified against ground truth.
    """

    gives_3d = True

    def __init__(self, spec: BackendSpec, device, max_people: int = 1,
                 detector_threshold: float = 0.4, use_detector: bool = True,
                 intrinsics: np.ndarray | None = None, allow_full_frame: bool = False):
        import torch
        self._torch = torch
        self.backend = spec.key
        self.rig = rig_of(spec.rig)
        self.device = device
        self.max_people = max(1, int(max_people))
        self.intrinsics = None if intrinsics is None else np.asarray(intrinsics, np.float64)
        self.bbox_thr = float(detector_threshold)
        self.allow_full_frame = bool(allow_full_frame)

        paths = s3db_checkpoint(spec.key)
        if paths is None:
            raise RuntimeError(backend_status(spec.key)[1])
        if not code_available(spec.key):
            raise RuntimeError(backend_status(spec.key)[1])
        ckpt, mhr = paths
        if str(S3DB_REPO) not in sys.path:
            sys.path.insert(0, str(S3DB_REPO))
        torch.hub.set_dir(str(MODELS_DIR))      # nothing outside the tool folder
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator  # noqa: E402

        model, model_cfg = load_sam_3d_body(str(ckpt), device=device, mhr_path=str(mhr))
        # Meta's own detector is ViTDet through detectron2, which has no wheel
        # and must be built from source. Kinetrace already carries RT-DETR v2
        # for the 2D backend, so both backends share ONE person detector: no
        # extra dependency, and a box that works for one works for the other.
        # The upstream estimator therefore never gets a detector of its own --
        # we always hand it boxes.
        detector = None
        self.detector = _PersonDetector(device, detector_threshold) if use_detector else None
        self._est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg,
                                       human_detector=detector, human_segmentor=None,
                                       fov_estimator=None)
        self.faces = getattr(self._est, "faces", None)

    def cam_int_for(self, w: int, h: int) -> np.ndarray:
        """(1, 3, 3) intrinsics for a w x h frame from the supplied focal length.

        (I88) Two things upstream requires: the matrix is BATCHED -- it
        indexes `batch["cam_int"]` as (B, 3, 3), and an unbatched (3, 3) made
        every frame raise, so the run aborted after five -- and the principal
        point is the frame CENTRE, which is what upstream's own default uses
        and what its final 2D projection assumes (sam3d_body.py projects with
        width / 2, height / 2 whatever cam_int says). The dialog used to put it
        at (0, 0), the top-left pixel. Only the focal length is taken from the
        caller; the frame is not undistorted first."""
        K = np.asarray(self.intrinsics, np.float64).reshape(-1, 3, 3)[0].copy()
        K[0, 1] = K[1, 0] = 0.0
        K[0, 2], K[1, 2] = w / 2.0, h / 2.0
        K[2] = (0.0, 0.0, 1.0)
        return K[None]

    def step(self, bgr, boxes=None, masks=None) -> list[PersonPose]:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        kw = {}
        det_scores: list[float] = []
        if boxes:
            kw["bboxes"] = np.asarray([_clip_box(b, w, h) for b in boxes], np.float32)
        elif masks:
            bbs, ms = [], []
            for m in masks[:self.max_people]:
                bb = mask_bbox(m)
                if bb is None:
                    continue
                bbs.append(_clip_box(bb, w, h, pad=0.08))
                ms.append(np.asarray(m).astype(np.uint8))
            if bbs:
                # upstream asserts boxes are present whenever masks are
                kw["bboxes"] = np.asarray(bbs, np.float32)
                kw["masks"] = np.stack(ms)
        elif self.detector is not None:
            found = self.detector.detect(bgr, self.max_people)
            if not found:
                return []
            kw["bboxes"] = np.asarray([b for b, _ in found], np.float32)
            det_scores = [s for _, s in found]
        if "bboxes" not in kw and not self.allow_full_frame:
            # same rule as the 2D backend: no prompt must never become a pose
            return []
        if self.intrinsics is not None:
            kw["cam_int"] = self._torch.as_tensor(self.cam_int_for(w, h),
                                                  dtype=self._torch.float32)
        out = self._est.process_one_image(rgb, bbox_thr=self.bbox_thr, **kw) or []

        people = []
        for i, d in enumerate(out[:self.max_people]):
            kp2 = np.asarray(d["pred_keypoints_2d"], np.float32).reshape(-1, 2)
            kp3 = np.asarray(d["pred_keypoints_3d"], np.float32).reshape(-1, 3)
            J = self.rig.n_joints
            # the model emits the full 308-keypoint MHR set; the first 70 are
            # the body and the rest are face detail we do not store
            kp2, kp3 = kp2[:J], kp3[:J]
            # (I85) the model gives no per-joint confidence. NaN says exactly
            # that; 1.0 exported a hidden, hallucinated limb as fully certain.
            conf = np.full(len(kp2), np.nan, np.float32)
            bb = np.asarray(d.get("bbox", [0, 0, w - 1, h - 1]), np.float32).reshape(4)
            verts = d.get("pred_vertices")
            people.append(PersonPose(
                joints2d=kp2, conf=conf, bbox=_clip_box(bb, w, h),
                # the DETECTOR's confidence, not 1.0: this model returns a
                # convincing body for any box it is given, so the only signal
                # that a column is really a person is how sure the detector was
                score=float(det_scores[i]) if i < len(det_scores) else 1.0,
                joints3d=kp3,
                vertices=None if verts is None else np.asarray(verts, np.float32),
                focal=float(np.ravel(d.get("focal_length", [np.nan]))[0]),
                cam_t=None if d.get("pred_cam_t") is None
                else np.asarray(d["pred_cam_t"], np.float32).reshape(-1)))
        return people


def make_estimator(backend: str, device=None, max_people: int = 1,
                   use_detector: bool = True, detector_threshold: float = 0.4,
                   intrinsics: np.ndarray | None = None,
                   allow_full_frame: bool = False) -> BodyEstimator:
    """Build the estimator for `backend`. Raises RuntimeError with the same
    plain-English text `backend_status` gives when it cannot."""
    spec = BACKENDS.get(backend)
    if spec is None:
        raise RuntimeError(f"unknown body backend {backend!r}")
    state, why = backend_status(backend)
    if state in ("needs-code", "needs-weights"):
        raise RuntimeError(why)
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.hub.set_dir(str(MODELS_DIR))
    with _lock:
        if spec.kind == "sam3d_body":
            return Sam3DBodyEstimator(spec, device, max_people, detector_threshold,
                                      use_detector, intrinsics, allow_full_frame)
        return ViTPoseEstimator(spec, device, max_people, detector_threshold, use_detector,
                                allow_full_frame)


# ------------------------------------------------------- identity over time

class PersonMatcher:
    """Keep the same person in the same column from frame to frame.

    A column that loses its person RESERVES their last box for `patience`
    looks (calls), so somebody who is briefly missed or occluded comes back
    into their OWN column instead of swapping with the other person. Per
    call, each detection goes to:

    1. the reserved column whose last box it overlaps most (IoU >= min_iou);
    2. else a reserved column whose last box is NEAR -- centre distance under
       `reach` box diagonals, plus `reach_growth` diagonals for every frame
       elapsed after the first, so a sampled run (every Nth frame) or a fast
       mover is not cut off;
    3. else a column never used or no longer reserved, the longest-empty
       first (biggest person first);
    4. else nowhere (-1): a stranger is never written into a reserved column.

    (I84) Leftover detections used to take the lowest free column, which
    could be one whose person was seen a frame ago: with one column, the
    frame the subject was missed a bystander (or a spurious camera-rig box
    detected at 51 % on a real clip) was written in as the subject.

    This is deliberately simple -- the app's own rule is one subject at a
    time, and a full re-identification model would be a new dependency for a
    case nobody has needed yet.
    """

    def __init__(self, n_people: int, patience: int = 30, min_iou: float = 0.2,
                 reach: float = 1.0, reach_growth: float = 0.25):
        self.n = max(1, int(n_people))
        self.patience = int(patience)
        self.min_iou = float(min_iou)
        self.reach = float(reach)
        self.reach_growth = float(reach_growth)
        self.last: list[np.ndarray | None] = [None] * self.n
        self.last_frame: list[int | None] = [None] * self.n
        self.age: list[int] = [10 ** 9] * self.n

    def _reserved(self, c: int) -> bool:
        return self.last[c] is not None and self.age[c] <= self.patience

    def assign(self, people: list[PersonPose], frame: int | None = None) -> list[int]:
        """Column index for each detected person; -1 when there is no room.
        `frame` (the video frame number) lets the nearness test grow with
        the frames actually elapsed; without it each call counts as one."""
        out = [-1] * len(people)
        free = set(range(self.n))
        pairs = []
        for i, p in enumerate(people):
            for c in range(self.n):
                if not self._reserved(c):
                    continue
                v = _iou(p.bbox, self.last[c])
                if v >= self.min_iou:
                    pairs.append((v, i, c))
        pairs.sort(reverse=True)
        taken_p = set()
        for _, i, c in pairs:
            if i in taken_p or c not in free:
                continue
            out[i], taken_p, free = c, taken_p | {i}, free - {c}
        # 2. near a reserved column's last box
        near = []
        for i, p in enumerate(people):
            if out[i] >= 0:
                continue
            b = np.asarray(p.bbox, np.float64)
            for c in free:
                if not self._reserved(c):
                    continue
                lb = self.last[c].astype(np.float64)
                diag = float(np.hypot(lb[2] - lb[0], lb[3] - lb[1]))
                if diag <= 0:
                    continue
                if frame is not None and self.last_frame[c] is not None:
                    elapsed = max(1, int(frame) - int(self.last_frame[c]))
                else:
                    elapsed = self.age[c] + 1
                gate = diag * (self.reach + self.reach_growth * (elapsed - 1))
                d = float(np.hypot((b[0] + b[2] - lb[0] - lb[2]) / 2.0,
                                   (b[1] + b[3] - lb[1] - lb[3]) / 2.0))
                if d <= gate:
                    near.append((d / gate, i, c))
        near.sort()
        for _, i, c in near:
            if out[i] >= 0 or c not in free:
                continue
            out[i], free = c, free - {c}
        # 3. unreserved columns, longest-empty first, biggest person first
        rest = sorted((i for i in range(len(people)) if out[i] < 0),
                      key=lambda i: -float((people[i].bbox[2] - people[i].bbox[0]) *
                                           (people[i].bbox[3] - people[i].bbox[1])))
        open_cols = sorted((c for c in free if not self._reserved(c)),
                           key=lambda c: (-self.age[c], c))
        for i, c in zip(rest, open_cols):
            out[i] = c
        for c in range(self.n):
            self.age[c] += 1
        for i, c in enumerate(out):
            if c >= 0:
                self.last[c] = np.asarray(people[i].bbox, np.float32)
                self.last_frame[c] = None if frame is None else int(frame)
                self.age[c] = 0
        return out
