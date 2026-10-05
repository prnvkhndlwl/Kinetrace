"""(I260) The segment must not turn into ANOTHER animal (owner report 2026-10-03: on a clip of
several bats, SAM 3 lost the clicked bat behind a railing / at the picture edge and, under the
16-frame animal-lost rule, picked up another bat each time -- the run went on as if nothing
happened).

  [1] `segmenter.SegmentIdentity` on summaries: a gap and a return far from where the animal was
      heading = another object (the first frame of the gap is where it ends); a return near the
      prediction, a slow / still animal behind an occluder, a one-frame jump within the limit, a
      clicked frame = the same animal; a one-frame jump of many body lengths = another object.
  [2] the worker (segment only, a stand-in for SAM, no GPU): the switch stops the run at the first
      frame of the gap, with Auto-pause on AND off, reason "switched", and no silhouette of the
      other object is emitted; a return at the predicted place runs to the end; a click on the
      other object makes it the animal (no stop).

CPU, no weights.  .venv\\Scripts\\python.exe tests\\verify_segment_identity.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402

app = QCoreApplication.instance() or QCoreApplication([])

from kinetrace import segmenter  # noqa: E402
from kinetrace import tracker as trk  # noqa: E402
from kinetrace.segmenter import SegmentIdentity, summarize_mask  # noqa: E402
from kinetrace.video_source import FrameCache  # noqa: E402

FAILS: list[str] = []
OUT = os.path.join(HERE, "out", "segment_identity")
os.makedirs(OUT, exist_ok=True)
W, H, N = 320, 240, 60


def check(label, cond, detail=""):
    cond = bool(cond)
    print(("  ok    " if cond else "  FAIL  ") + label + ("" if cond or not detail else f"   [{detail}]"), flush=True)
    if not cond:
        FAILS.append(label)


def blob(cx, cy, r=6, w=W, h=H):
    m = np.zeros((h, w), bool)
    cv2.circle(m.view(np.uint8), (int(round(cx)), int(round(cy))), r, 1, -1)
    return m


def summ_at(c):
    if c is None:
        return summarize_mask(np.zeros((H, W), bool), 1.0, -2.0)
    return summarize_mask(blob(*c), 1.0, 5.0)


def replay(path, prompted=()):
    """SegmentIdentity over {frame: centre or None}; the first answer that is not None."""
    g = SegmentIdentity()
    for f in sorted(path):
        end = g.check(f, summ_at(path[f]), prompted=f in prompted)
        if end is not None:
            return end, f
    return None


# ------------------------------------------------------------------- [1] the rule
print("[1] SegmentIdentity on summaries")
fly = {f: (40 + 4 * f, 120) for f in range(0, 40)}                 # 4 px / frame to the right, 13 px across
gap_far = {**fly, **{f: None for f in range(20, 30)}, 30: (40, 40), 31: (44, 40)}
r = replay({f: c for f, c in gap_far.items() if f <= 31})
check("a return far from where it was heading = another object, ending at the gap's first frame",
      r == (20, 30), f"{r}")
gap_near = {**fly, **{f: None for f in range(20, 30)}}
r = replay({f: c for f, c in gap_near.items()})
check("a return where its speed put it (behind an occluder for 10 frames) = the same animal", r is None, f"{r}")
still = {f: (150, 120) for f in range(0, 40)}
still.update({f: None for f in range(10, 30)})
still[30] = (160, 118)
r = replay({f: c for f, c in still.items() if f <= 30})
check("a still animal back 10 px from where it hid = the same animal", r is None, f"{r}")
jump = {f: (40 + 4 * f, 120) for f in range(0, 15)}
jump[15] = (40 + 4 * 15 + 60, 120)                                # 60 px off: < 4 x (body 18 + speed 4)
r = replay(jump)
check("a one-frame jump within 4 x (body + speed) = the same animal", r is None, f"{r}")
jump[15] = (300, 220)                                             # far across the picture in one frame
r = replay(jump)
check("a one-frame jump of many body lengths = another object, ending at that frame", r == (15, 15), f"{r}")
r = replay({f: c for f, c in gap_far.items() if f <= 31}, prompted={30})
check("a frame the user clicked is trusted (the history starts again)", r is None, f"{r}")
r = replay({0: (100, 100)})
check("the first frame alone = nothing to judge", r is None)


# ------------------------------------------------------------------- [2] the worker
print("[2] the worker, segment only, a stand-in for SAM")
VIDEO = os.path.join(OUT, "plain.mp4")
if not os.path.exists(VIDEO):
    wr = cv2.VideoWriter(VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for f in range(N):
        im = np.full((H, W, 3), 60, np.uint8)
        cv2.putText(im, str(f), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        wr.write(im)
    wr.release()


class Stand:
    """SAM stand-in: the centre of the object on each frame from `path` (None = nothing found)."""

    def __init__(self, path):
        self.path = path

    def new_session(self, start, size):
        path = self.path

        class S:
            def step(self, native, idx, prompts):
                c = path.get(idx)
                m = np.zeros((H, W), bool) if c is None else blob(*c)
                return segmenter.FrameMasks(idx, [1], m[None], np.array([8.0 if c else -2.0], np.float32),
                                            (W, H), (W, H))
        return S()


def run(path, autopause=True, clicks=None):
    saved = trk.get_segmenter
    trk.get_segmenter = lambda be, **k: Stand(path)
    try:
        an = trk.AnimalSpec(clicks or {0: [(path[0][0], path[0][1], 1)]}, backend="sam2.1-base-plus")
        w = trk.TrackingWorker(VIDEO, 0, np.zeros((0, 2), np.float32), [], FrameCache(1 << 26), N,
                               specs=[], autopause=autopause, animal=an)
        got = {"masks": {}, "auto": None, "end": None}
        w.masks_ready.connect(lambda ss: [got["masks"].__setitem__(s["frame"], s) for s in ss])
        w.autopaused.connect(lambda f, p: got.__setitem__("auto", (f, p)))
        w.finished_ok.connect(lambda f, p: got.__setitem__("end", (f, p)))
        w.error.connect(lambda e: got.__setitem__("err", e))
        w.run()
        app.processEvents()
        got["reason"] = w._autopause_reason
        return got
    finally:
        trk.get_segmenter = saved


def present_from(got, f0):
    return [f for f, s in got["masks"].items() if f >= f0 and s["area"] > 0]


# the clicked animal flies right, is hidden at 20-29, and SAM takes a look-alike far away from 30 on
other = {**{f: (40 + 4 * f, 120) for f in range(20)}, **{f: None for f in range(20, 30)},
         **{f: (60 + 3 * (f - 30), 40) for f in range(30, N)}}
for ap in (True, False):
    g = run(other, autopause=ap)
    tag = "on" if ap else "off"
    check(f"Auto-pause {tag}: the run stops at the gap's first frame (20), the segment's own stop",
          g["auto"] == (20, -1) and g["reason"] == "switched", f"{g['auto']} {g['reason']!r} {g.get('err', '')[-300:]}")
    check(f"Auto-pause {tag}: no silhouette of the other object is emitted", not present_from(g, 20),
          f"{present_from(g, 20)[:5]}")
    check(f"Auto-pause {tag}: the frames before the switch keep theirs", len(present_from(g, 0)) == 20,
          f"{len(present_from(g, 0))}")

same = {**{f: (40 + 4 * f, 120) for f in range(N)}, **{f: None for f in range(20, 30)}}
g = run(same)
check("a return where it was heading runs to the end (no stop)", g["auto"] is None and g["end"][0] == N - 1,
      f"{g['auto']} {g['end']}")
check("... and keeps every silhouette", len(present_from(g, 0)) == N - 10, f"{len(present_from(g, 0))}")

clicked = {0: [(40, 120, 1)], 30: [(60, 40, 1)]}
g = run(other, clicks=clicked)
check("a click on the other object at frame 30 makes it the animal: no stop",
      g["auto"] is None and g["end"][0] == N - 1, f"{g['auto']} {g['end']} {g['reason']!r}")

print("\nverify_segment_identity: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
