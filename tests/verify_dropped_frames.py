"""Frame N is the N-th picture reading forward - also on a file whose camera dropped frames (I382-I384).

Every picture of the generated clips carries its own number. For each clip with uneven timestamps
(dropped frames, variable rate, the same in Matroska) EVERY frame is asked for through the app's own
doors - `seek_capture` on a fresh capture, one capture jumping about, `VideoSource.get_frame`,
`FrameReader` - right after `probe_video` (what the app has the moment a video is open, before any
background check) and again after `check_seeks`; the picture must be the one reading forward gives.
The frame count must be the number of pictures. Clips with even timestamps keep the plain seek.

No GPU, no window. Run: .venv\\Scripts\\python.exe tests\\verify_dropped_frames.py   (about a minute)
"""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import numpy as np  # noqa: E402

import _synth_frames as sf  # noqa: E402
from kinetrace import video_source as vs  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out", "dropped_frames")
FAILED = []


def check(ok, what, detail=""):
    print(("OK   " if ok else "FAIL ") + what + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAILED.append(what)


def wrong_cells(got: dict, ref: list) -> list:
    return sorted((f, ref[f], g) for f, g in got.items() if g != ref[f])


def via_fresh(path, frames, ref):
    got = {}
    for f in frames:
        cap = vs.open_capture(path)
        ok = vs.seek_capture(cap, path, f)
        good, bgr = cap.read() if ok else (False, None)
        got[f] = sf.barcode(bgr) if good else None
        cap.release()
    return wrong_cells(got, ref)


def via_one_capture(path, frames, ref):
    cap = vs.open_capture(path)
    got = {}
    for f in frames:
        ok = vs.seek_capture(cap, path, f)
        good, bgr = cap.read() if ok else (False, None)
        got[f] = sf.barcode(bgr) if good else None
    cap.release()
    return wrong_cells(got, ref)


def via_source(path, frames, ref):
    src = vs.VideoSource(path, vs.FrameCache(8 * 1024 ** 2))
    got = {f: sf.barcode(src.get_frame(f)) for f in frames}
    src.close()
    return wrong_cells(got, ref)


def via_reader(path, frames, ref):
    got = {}
    for f in frames:
        r = vs.FrameReader(path, f)
        got[f] = sf.barcode(r.read(f))
        r.release()
    return wrong_cells(got, ref)


def all_doors(name, path, ref, when):
    n = len(ref)
    every = list(range(n))
    w = via_fresh(path, every, ref)
    check(not w, f"{name} {when}: every frame through a fresh capture is the forward picture ({n} frames)",
          f"{len(w)} wrong, first (asked, forward, got) {w[:4]}")
    mixed = every[:]
    random.Random(7).shuffle(mixed)
    if name in sf.LONG:                      # 1500 frames: every frame once more, on one capture, is enough
        mixed = mixed[:400]
    w = via_one_capture(path, mixed, ref)
    check(not w, f"{name} {when}: every frame on one capture jumping about", f"{len(w)} wrong, first {w[:4]}")
    w = via_source(path, mixed, ref)
    check(not w, f"{name} {when}: every frame through VideoSource.get_frame", f"{len(w)} wrong, first {w[:4]}")
    some = sorted(set(every[::7]) | set(range(max(0, n - 6), n)) | set(range(140, min(n, 160))))
    w = via_reader(path, some, ref)
    check(not w, f"{name} {when}: FrameReader ({len(some)} frames)", f"{len(w)} wrong, first {w[:4]}")
    cap = vs.open_capture(path)
    ok = vs.seek_capture(cap, path, n)
    past = cap.read()[0] if ok else False
    cap.release()
    check(not past, f"{name} {when}: nothing is read past the last frame")


clips = sf.make_clips(OUT)
for name, path in clips.items():
    uneven = sf.KINDS[name][1]
    ref = sf.forward(path)
    n = len(ref)
    print(f"\n== {name}: {sf.KINDS[name][2]} ({n} pictures reading forward)", flush=True)
    vs.set_seek_plan(path, None)
    info = vs.probe_video(path)
    check(info.n_frames == n, f"{name}: the frame count is the number of pictures", f"{info.n_frames} vs {n}")
    plan = vs.seek_plan(path)
    if uneven:
        check(plan is not None and not plan.exact and plan.pts_map is not None,
              f"{name}: uneven timestamps are known the moment the video is open (a plan, with a map)",
              repr(plan)[:160])
        check(bool(plan is not None and plan.note), f"{name}: ... with a sentence for the user")
        if plan is not None and name.startswith("vfr"):
            check("variable frame rate" in plan.note and "skip" not in plan.note,
                  f"{name}: a variable rate is not called dropped frames", plan.note[:120])
        if plan is not None and name.startswith(("dropped", "gap", "mkv")):
            check("dropped" in plan.note and plan.gaps in (1, 3), f"{name}: the sentence says how many frames "
                  "the timestamps skip", f"gaps {plan.gaps}: {plan.note[:100]}")
        all_doors(name, path, ref, "at open")
        checked = vs.check_seeks(path, info.n_frames)
        check(not checked.exact and checked.pts_map is not None,
              f"{name}: the background check never calls an uneven file exact", repr(checked)[:160])
        vs.set_seek_plan(path, checked)
        all_doors(name, path, ref, "after the check")
        t = checked.pts_map.pts
        check(bool(np.all(np.diff(t) > 0)), f"{name}: the map's times are strictly increasing")
    else:
        check(plan is None, f"{name}: even timestamps cost nothing at open (no plan yet)", repr(plan)[:120])
        w = via_fresh(path, list(range(n)), ref)
        check(not w, f"{name}: every frame through a fresh capture (plain seeks)", f"{len(w)} wrong {w[:4]}")
        checked = vs.check_seeks(path, info.n_frames)
        check(checked.exact, f"{name}: the background check finds its seeks exact", checked.note[:160])
    vs.set_seek_plan(path, None)

# the small rules, on their own
# jitter around an even rate is even; a dropped frame is not, wherever it is
even = np.cumsum([0] + [20, 21] * 150) * (1000.0 / 600.0)
check(not vs.uneven_times(even), "29.97 fps written as 20, 21, 20 ticks is an even rate")
for at in (3, 150, 296):
    t = np.arange(300) * (1000.0 / 30.0)
    t[at:] += 1000.0 / 30.0
    check(vs.uneven_times(t), f"one frame's worth of time missing at frame {at} of 300 is uneven")
check(not vs.uneven_times(np.round(np.arange(600) * (1000.0 / 240.0))), "240 fps in whole milliseconds is even")
m = vs.PtsMap(np.array([0.0, 33.3, 66.7, 66.9, 100.0]), 0, 0.0, 5.0)
check(m.index(33.3) == 1 and m.index(100.0) == 4, "PtsMap names a picture by its time")
check(m.index(66.8) is None and m.index(66.7) is None,
      "a time two entries could claim names nothing (the reader lands earlier instead of guessing)")
check(m.index(50.0) is None, "a time that is none of the file's names nothing")

if FAILED:
    print(f"\nDROPPED FRAMES FAILED ({len(FAILED)}): " + "; ".join(FAILED[:6]))
    sys.exit(1)
print("\nDROPPED FRAMES PASSED")
