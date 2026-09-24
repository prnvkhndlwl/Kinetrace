"""AllTracker (Harley et al., ICCV 2025, MIT) as a streaming point tracker.

AllTracker estimates dense flow from ONE query frame to every other frame with
a 16-frame sliding window whose state (flow + features of the overlapping
half) carries across windows. Its reference `forward_sliding` keeps the whole
video and the whole dense output in memory; this wrapper re-implements the
same loop frame-chunk by frame-chunk and samples the dense flow only at the
query points, so a 40k-frame video costs the same memory as a 16-frame one.

Semantics differ from CoTracker3: every point is tracked *from the query
frame*, so there is no per-window re-seeding and no drift accumulation while
the query frame's appearance still matches. The app's segment loop still
re-anchors on ROI restarts and user corrections.

Windowing contract (verified against nets/alltracker.py):
  * frame 0 of a stream is the query frame; the first window is frames 0..15;
  * every further window advances by STRIDE=8 frames and rewrites its 16 rows
    (the overlapping first half is refined), exactly like CoTracker3;
  * tails shorter than 8 frames are padded by repeating the last frame, as in
    the reference implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPO_DIR = MODELS_DIR / "alltracker"
CHECKPOINT = MODELS_DIR / "checkpoints" / "alltracker.pth"
CHECKPOINT_URL = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
WINDOW = 16
STRIDE = 8

_model = None
_device = None


def available() -> bool:
    return (REPO_DIR / "nets" / "alltracker.py").exists()


def is_cached() -> bool:
    return CHECKPOINT.exists()


def get_alltracker():
    """Process-wide singleton (model, device). Downloads the 63 MB checkpoint once."""
    global _model, _device
    if _model is not None:
        return _model, _device
    import torch
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    from nets.alltracker import Net  # noqa: E402  (vendored repo)
    from kinetrace.device import pick_device
    device = pick_device()[0]            # CUDA, else Apple's GPU, else the CPU (one rule for every model)
    torch.hub.set_dir(str(MODELS_DIR))   # nothing may be written outside the tool folder
    if not CHECKPOINT.exists():
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.load_state_dict_from_url(CHECKPOINT_URL, model_dir=str(CHECKPOINT.parent),
                                           map_location="cpu")
    state = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    # init_weights=False: the constructor would otherwise download ImageNet
    # ConvNeXt weights that the checkpoint overwrites anyway (strict load)
    model = Net(seqlen=WINDOW, init_weights=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    _model, _device = model, device
    return model, device


class AllTrackerStream:
    """One query frame, N query points, frames pushed in order.

    Coordinates are in the pixel frame of the frames you push (any size; they
    are padded internally to multiples of 64). Call `start(frame0, xy)` with
    the query frame, then `push(frame)` for every following frame; whenever a
    window completes, `push` returns (first_frame_index, tracks (L, N, 2),
    vis (L, N), conf (L, N)) covering the last <=16 frames — the caller
    overwrites rows exactly as with CoTracker3. `flush()` finishes a tail.
    """

    def __init__(self, iters: int = 4):
        import torch
        self.torch = torch
        self.model, self.device = get_alltracker()
        self.iters = iters
        self._buf: list = []          # padded, normalised frames since the last window start
        self._n_fed = 0               # frames fed (query frame = 0)
        self._padder = None
        self._fmap_anchor = None
        self._flows8 = None
        self._visconfs8 = None
        self._fmaps2 = None
        self._xy = None               # (N, 2) float32 query points
        self._hw = None
        self._first = True

    # ------------------------------------------------------------ helpers
    def _prep(self, frame_rgb: np.ndarray):
        torch = self.torch
        img = torch.from_numpy(np.ascontiguousarray(frame_rgb)).to(self.device).float()
        img = img.permute(2, 0, 1)[None]                       # 1,3,H,W in [0,255]
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1, 3, 1, 1)
        img = (img / 255.0 - mean) / std
        if self._padder is None:
            from nets.blocks import InputPadder
            self._padder = InputPadder(img.shape)
        return self._padder.pad(img)[0]

    def _fmaps(self, imgs):
        """imgs: (T,3,Hp,Wp) -> (1,T,C,H8,W8)"""
        T = imgs.shape[0]
        C = self.model.dim if self.model.no_split else self.model.dim * 2
        H8, W8 = imgs.shape[-2] // 8, imgs.shape[-1] // 8
        # The vendored get_fmaps does `images_.cuda()` whenever is_training is
        # False - the ONLY thing that flag changes for a window of <= 64 frames
        # (the other branch is a chunked loop for longer inputs). On a CPU or
        # an Apple GPU that call raises, so those devices pass True: same
        # arithmetic, tensors stay where they are. CUDA keeps the original call.
        not_cuda = self.device != "cuda"
        return self.model.get_fmaps(imgs, 1, T, None, not_cuda).reshape(1, T, C, H8, W8)

    def _sample(self, maps):
        """maps: (S, C, H, W) at frame res -> (S, N, C) bilinear at the query points."""
        torch = self.torch
        S, C, H, W = maps.shape
        xy = torch.from_numpy(self._xy).to(maps.device)
        gx = xy[:, 0] / max(W - 1, 1) * 2 - 1
        gy = xy[:, 1] / max(H - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).reshape(1, 1, -1, 2).expand(S, 1, -1, 2)
        out = torch.nn.functional.grid_sample(maps, grid, mode="bilinear", padding_mode="border",
                                              align_corners=True)   # S,C,1,N
        return out[:, :, 0, :].permute(0, 2, 1)

    # ---------------------------------------------------------------- api
    def start(self, frame0_rgb: np.ndarray, query_xy: np.ndarray) -> None:
        torch = self.torch
        self._xy = np.asarray(query_xy, np.float32).reshape(-1, 2)
        self._hw = frame0_rgb.shape[:2]
        with torch.no_grad():
            img = self._prep(frame0_rgb)
            self._fmap_anchor = self._fmaps(img)[:, 0]          # 1,C,H8,W8
        self._buf = [img]
        self._n_fed = 1
        self._first = True

    def _run_window(self, imgs, start_idx: int):
        """Run one 16-frame window over `imgs` (S,3,Hp,Wp); return rows for it."""
        torch = self.torch
        S = WINDOW
        H8, W8 = imgs.shape[-2] // 8, imgs.shape[-1] // 8
        with torch.no_grad():
            if self._first:
                self._flows8 = torch.zeros((1, S, 2, H8, W8), device=self.device)
                self._visconfs8 = torch.zeros((1, S, 2, H8, W8), device=self.device)
                self._fmaps2 = self._fmaps(imgs)
                self._first = False
            else:
                self._flows8 = torch.cat([self._flows8[:, STRIDE:STRIDE + S // 2],
                                          self._flows8[:, STRIDE + S // 2 - 1:STRIDE + S // 2].repeat(1, S // 2, 1, 1, 1)], dim=1)
                self._visconfs8 = torch.cat([self._visconfs8[:, STRIDE:STRIDE + S // 2],
                                             self._visconfs8[:, STRIDE + S // 2 - 1:STRIDE + S // 2].repeat(1, S // 2, 1, 1, 1)], dim=1)
                self._fmaps2 = torch.cat([self._fmaps2[:, STRIDE:STRIDE + S // 2],
                                          self._fmaps(imgs[S // 2:])], dim=1)
            flows8 = self._flows8.reshape(S, 2, H8, W8).detach()
            visconfs8 = self._visconfs8.reshape(S, 2, H8, W8).detach()
            flow_preds, visconf_preds, flows8, visconfs8, _ = self.model.forward_window(
                self._fmap_anchor, self._fmaps2, visconfs8, iters=self.iters, flowfeat=None,
                flows8=flows8, is_training=False)
            self._flows8 = flows8.reshape(1, S, 2, H8, W8)
            self._visconfs8 = visconfs8.reshape(1, S, 2, H8, W8)
            flow = self._padder.unpad(flow_preds[-1]).reshape(S, 2, *self._hw)
            vc = self._padder.unpad(torch.sigmoid(visconf_preds[-1])).reshape(S, 2, *self._hw)
            # sample at the query points: track = query + flow(query)
            d = self._sample(flow)                                   # S,N,2
            v = self._sample(vc)                                     # S,N,2 (vis, conf)
            tracks = (torch.from_numpy(self._xy).to(d.device)[None] + d).cpu().numpy()
            vis = v[:, :, 0].cpu().numpy()
            conf = v[:, :, 1].cpu().numpy()
        return start_idx, tracks.astype(np.float32), vis.astype(np.float32), conf.astype(np.float32)

    def push(self, frame_rgb: np.ndarray):
        """Feed the next frame. Returns a window result or None."""
        torch = self.torch
        with torch.no_grad():
            self._buf.append(self._prep(frame_rgb))
        self._n_fed += 1
        if self._first and len(self._buf) == WINDOW:
            imgs = torch.cat(self._buf, dim=0)
            out = self._run_window(imgs, 0)
            self._buf = self._buf[STRIDE:]          # keep the overlapping half
            return out
        if not self._first and len(self._buf) == WINDOW:
            imgs = torch.cat(self._buf, dim=0)
            out = self._run_window(imgs, self._n_fed - WINDOW)
            self._buf = self._buf[STRIDE:]
            return out
        return None

    def flush(self):
        """Finish a tail shorter than a full window by repeating the last frame
        (reference behaviour). Returns the rows for the real frames or None."""
        torch = self.torch
        n_real = len(self._buf)
        if n_real == 0 or (not self._first and n_real <= WINDOW // 2):
            return None      # nothing new since the last window
        if self._first and n_real < 2:
            return None
        pad = WINDOW - n_real
        imgs = torch.cat(self._buf + [self._buf[-1]] * pad, dim=0)
        start = 0 if self._first else self._n_fed - n_real
        idx, tr, vis, conf = self._run_window(imgs, start)
        return idx, tr[:n_real], vis[:n_real], conf[:n_real]
