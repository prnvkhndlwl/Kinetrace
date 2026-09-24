"""REPL-style verification of video_source.py and session.py (no torch needed)."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "tests", "out"), exist_ok=True)
import numpy as np

sys.path.insert(0, ROOT)
from kinetrace.video_source import VideoSource, FrameCache, probe_video
from kinetrace.session import TrackingSession

VID = os.path.join(ROOT, r"test600.mp4")

# ---- probe ----
info = probe_video(VID)
print("probe:", info)
assert info.n_frames == 600 and info.width == 640 and info.height == 480
assert not info.vfr_suspected, "synthetic CFR video flagged as VFR"
assert info.header_frames == 600 and info.header_overcount == 0, "a clean file has no phantom frames"

# ---- VideoSource ----
cache = FrameCache(max_bytes=50 * 1024 * 1024)
src = VideoSource(VID, cache)
i0, f0 = src.read_next()
assert i0 == 0 and f0.shape == (480, 640, 3)
# sequential
for expect in range(1, 10):
    i, _ = src.read_next()
    assert i == expect
# far seek
f500 = src.get_frame(500)
assert f500 is not None and f500.shape == (480, 640, 3)
# frame identity: seek back to 500 must give identical pixels (cache) and
# a fresh decode of 500 after cache clear must match too (seek accuracy)
cache.clear()
f500b = src.get_frame(500)
assert np.array_equal(f500, f500b), "seek to 500 not frame-accurate after cache clear"
# near-forward fast path
f505 = src.get_frame(505)
assert f505 is not None
# cache hit
t = time.perf_counter()
_ = src.get_frame(505)
assert time.perf_counter() - t < 0.01
# EOF
assert src.get_frame(600) is None, "expected None past EOF"
# backward seek
f100 = src.get_frame(100)
assert f100 is not None
# nearest
n_idx, _ = cache.nearest(103)
assert abs(n_idx - 103) <= 10
src.close()
print("video_source OK")

# ---- cache budget ----
small = FrameCache(max_bytes=5 * f0.nbytes)
for k in range(20):
    small.put(k, f0.copy())
assert small.get(19) is not None and small.get(0) is None
assert small._bytes <= small.max_bytes
print("cache budget OK")

# ---- session ----
s = TrackingSession(VID, 600, 30.0, 640, 480)
p0 = s.add_point(0, 100.5, 200.25)
p1 = s.add_point(0, 300.0, 100.0)
p2 = s.add_point(5, 50.0, 50.0)
assert s.n_points == 3 and s.seedable_at(0) == [0, 1] and s.seedable_at(5) == [2]
assert np.isnan(s.positions_at(3)).all()

# write a window for points 0 and 1 starting at frame 0
L = 16
win = np.stack([np.stack([np.linspace(100, 115, L), np.linspace(200, 215, L)], -1),
                np.stack([np.linspace(300, 315, L), np.linspace(100, 115, L)], -1)], 1).astype(np.float32)
vis = np.ones((L, 2), bool); vis[3, 1] = False
s.write_segment(0, win, vis, [0, 1])
assert s.tracked[15, 0] and s.tracked[15, 1] and not s.tracked[15, 2]
assert not s.visibility[3, 1]
assert not s.manual[0, 0], "model write must clear manual flag"
# clipped write at the end
s.write_segment(595, win, vis, [0, 1])
assert s.tracked[599, 0] and s.tracks.shape[0] == 600

# manual correction + snapshot/restore
snap = s.snapshot()
s.set_position(10, 0, 999.0, 999.0)
assert s.manual[10, 0] and s.tracks[10, 0, 0] == 999.0
s.restore(snap)
assert not s.manual[10, 0] and abs(s.tracks[10, 0, 0] - win[10, 0, 0]) < 1e-4

# remove point
s.remove_point(1)
assert s.n_points == 2 and s.tracks.shape[1] == 2

# project-file round-trip
tmp = os.path.join(ROOT, "tests", "out", "sess.kinetrace")
s.current_frame = 123
s.save(tmp)
assert not s.dirty
s2 = TrackingSession.load(tmp)
assert s2.n_points == 2 and s2.current_frame == 123 and s2.n_frames == 600
assert np.allclose(s2.tracks, s.tracks, equal_nan=True)
assert (s2.tracked == s.tracked).all()
assert s2.points[0].name == s.points[0].name and s2.points[0].color == s.points[0].color
# new point after load continues the name counter without collision
pid = s2.add_point(0, 1, 1)
assert s2.points[pid].name not in [p.name for p in s2.points[:-1]]
print("session core OK")

# ---- exports ----
scratch = os.path.join(ROOT, "tests", "out")
s.export_csv(os.path.join(scratch, "out.csv"))
s.export_tsv_sparse(os.path.join(scratch, "out.tsv"))
s.export_mat(os.path.join(scratch, "out.mat"))
head = open(os.path.join(scratch, "out.csv")).read().splitlines()
assert head[0] == "frame,P1_x,P1_y,P1_visible,P3_x,P3_y,P3_visible", head[0]
assert head[1].startswith("0,100.000,200.000,1,")
assert head[4].endswith(",,"), "untracked P3 cell should be blank: " + head[4]
tsv = open(os.path.join(scratch, "out.tsv")).read().splitlines()
assert tsv[0] == "frame\tpoint\tx\ty\tvisible"
assert len(tsv) - 1 == int(s.tracked.sum()), "sparse TSV row count mismatch"
from scipy.io import loadmat
m = loadmat(os.path.join(scratch, "out.mat"))
assert m["tracks"].shape == (600, 2, 2) and abs(float(m["fps"].squeeze()) - 30.0) < 1e-9
assert np.allclose(m["tracks"][0, 0], [100.0, 200.0])
print("exports OK")

# ---- export speed at 40k x 10 ----
big = TrackingSession("x.mp4", 40000, 30.0, 3840, 2160)
for i in range(10):
    big.add_point(0, i * 10.0, i * 5.0)
big.write_segment(0, np.random.rand(40000, 10, 2).astype(np.float32) * 1000,
                  np.ones((40000, 10), bool), list(range(10)))
t = time.perf_counter(); big.export_csv(os.path.join(scratch, "big.csv")); t_csv = time.perf_counter() - t
t = time.perf_counter(); big.export_tsv_sparse(os.path.join(scratch, "big.tsv")); t_tsv = time.perf_counter() - t
t = time.perf_counter(); big.export_mat(os.path.join(scratch, "big.mat")); t_mat = time.perf_counter() - t
t = time.perf_counter(); big.save(os.path.join(scratch, "big.kinetrace")); t_proj = time.perf_counter() - t
print(f"40k x 10 export: csv {t_csv:.2f}s, tsv {t_tsv:.2f}s, mat {t_mat:.2f}s, project {t_proj:.2f}s")
# KINETRACE_PERF_SCALE relaxes the budgets on slow shared machines (the CI
# workflow sets it: GitHub's shared runners missed the 2 s export budget on
# Ubuntu 22.04 in the first cross-OS run); 1 on a workstation
SLOW = float(os.environ.get("KINETRACE_PERF_SCALE", "1"))
assert t_csv < 2 * SLOW and t_mat < 2 * SLOW and t_proj < 2 * SLOW, \
    f"export too slow: csv {t_csv:.2f} s, mat {t_mat:.2f} s, project {t_proj:.2f} s (budget 2 s x {SLOW:g})"

# ---- confidence, groups, events, unique names, ui_state survive a save ----

s3 = TrackingSession(VID, 600, 30.0, 640, 480)
p = s3.add_point(0, 10.0, 10.0)
g = s3.add_point(0, 100.0, 100.0, kind="group", radius=40.0)
assert s3.points[g].kind == "group" and s3.points[g].radius == 40.0
assert s3.points[g].name.startswith("G"), s3.points[g].name
assert s3.confidence[0, p] == 1.0, "manual placement must have confidence 1"
v0 = s3.data_version
s3.set_position(1, p, 11.0, 11.0)
assert s3.data_version > v0, "mutations must bump data_version"

# write_segment with per-cell confidence; OOB rows blank conf too
L = 16
win = np.tile(np.array([[50.0, 50.0], [100.0, 100.0]], np.float32), (L, 1, 1))
conf = np.linspace(0.1, 1.0, L, dtype=np.float32)[:, None].repeat(2, 1)
s3.write_segment(20, win, np.ones((L, 2), bool), [p, g], conf)
assert np.allclose(s3.confidence[20:36, p], conf[:, 0], atol=1e-6)
win2 = win.copy(); win2[5, 0] = (10_000.0, 10_000.0)
s3.write_segment(100, win2, np.ones((L, 2), bool), [p, g], conf)
assert s3.confidence[105, p] == 0.0 and not s3.tracked[105, p]
# legacy call without conf still works -> conf 1.0 where tracked
s3.write_segment(200, win, np.ones((L, 2), bool), [p, g])
assert (s3.confidence[200:216, [p, g]] == 1.0).all()

# rename collision -> auto-suffix, never overwrites
s3.points[p].name = "ball"
assert s3.rename_point(g, "ball") == "ball (2)"
assert s3.rename_point(g, "ball") == "ball (2)", "re-resolving must be stable"
assert s3.rename_point(p, "ball") == "ball", "renaming to own name is free"
p3 = s3.add_point(0, 1.0, 1.0)
assert s3.rename_point(p3, "ball") == "ball (3)"
s3.remove_point(p3)

# events: reversed range swaps, out-of-range clamps, blank name auto-fills
i = s3.add_event("swing", 400, 200)
assert s3.events[i].start == 200 and s3.events[i].end == 400
j = s3.add_event("  ", -50, 9999)
assert s3.events[j].start == 0 and s3.events[j].end == 599 and s3.events[j].name
s3.update_event(i, name="swing2", start=500)  # start beyond end -> swap
assert s3.events[i].name == "swing2" and (s3.events[i].start, s3.events[i].end) == (400, 500)

# undo snapshot round-trips confidence and point metadata
snap = s3.snapshot()
s3.set_position(50, p, 5.0, 5.0)
s3.points[g].radius = 99.0
s3.restore(snap)
assert not s3.manual[50, p] and s3.points[g].radius == 40.0

# full save/load round-trip incl. v2 fields and ui_state
s3.ui_state.update(selected=1, zoom=2.5, follow=False)
v2p = os.path.join(scratch, "sessv2.kinetrace")
s3.save(v2p)
r = TrackingSession.load(v2p)
assert r.points[g].kind == "group" and r.points[g].radius == 40.0
assert len(r.events) == 2 and r.events[i].name == "swing2"
assert r.ui_state["zoom"] == 2.5 and r.ui_state["selected"] == 1 and r.ui_state["follow"] is False
assert np.allclose(r.confidence, s3.confidence, atol=1e-6)
# event counter continues after load (no duplicate default names)
k = r.add_event("", 10, 20)
assert r.events[k].name not in [e.name for e in r.events[:-1]]

# .mat carries confidence + events; sidecar CSV; empty-events .mat also valid
s3.export_mat(os.path.join(scratch, "v2.mat"))
m2 = loadmat(os.path.join(scratch, "v2.mat"))
assert m2["confidence"].shape == (600, 2)
assert [str(n[0]) for n in m2["event_names"].squeeze()] == ["swing2", s3.events[1].name]
assert m2["event_start"].squeeze().tolist() == [400.0, 0.0]
assert str(m2["point_kind"].squeeze()[1][0]) == "group"
s3.export_events_csv(os.path.join(scratch, "v2_events.csv"))
ev_lines = open(os.path.join(scratch, "v2_events.csv")).read().splitlines()
assert ev_lines[0].startswith("name,start_frame,end_frame") and ev_lines[1].startswith("swing2,400,500")
s_noev = TrackingSession(VID, 10, 30.0, 640, 480)
s_noev.add_point(0, 1.0, 1.0)
s_noev.export_mat(os.path.join(scratch, "noev.mat"))   # must not crash with zero events
m3 = loadmat(os.path.join(scratch, "noev.mat"))
assert m3["event_start"].size == 0
print("schema v2 OK")

# ---- frame rate: high-speed headers believed, a timebase is not a rate (I37) ----
import threading
import cv2
from kinetrace import video_source as _vs


def _write_clip(path, fps, n=40):
    vw = cv2.VideoWriter(path, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (64, 48))
    assert vw.isOpened(), path
    for k in range(n):
        img = np.full((48, 64, 3), (k * 5) % 255, np.uint8)
        cv2.putText(img, str(k), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        vw.write(img)
    vw.release()
    return path


for rate in (1200, 2000, 5000):
    _i = probe_video(_write_clip(os.path.join(scratch, f"hs_{rate}.mp4"), rate))
    assert abs(_i.fps - rate) < 1e-6 and _i.fps_source == "header" and _i.fps_note == "", (rate, _i)
assert info.fps_source == "header" and info.fps_note == "", info


class _FakeCap:
    """A real capture whose header rate / timestamps are overridden."""
    def __init__(self, cap, fps=None, msec=None):
        self._cap, self._fps, self._msec = cap, fps, msec
        self._k = -1

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS and self._fps is not None:
            return self._fps
        if prop == cv2.CAP_PROP_POS_MSEC and self._msec is not None:
            return self._msec(self._k)
        return self._cap.get(prop)

    def grab(self):
        self._k += 1
        return self._cap.grab()

    def __getattr__(self, name):
        return getattr(self._cap, name)


def _probe_with(fps=None, msec=None, path=VID):
    real = _vs.open_capture
    _vs.open_capture = lambda p: _FakeCap(real(p), fps, msec)
    try:
        return probe_video(path)
    finally:
        _vs.open_capture = real


for bad in (0.0, float("nan"), 1000.0, 90000.0, 1.2e6):
    _i = _probe_with(fps=bad)      # test600 is 30 fps; its timestamps say so
    assert abs(_i.fps - 30.0) < 1e-3 and _i.fps_source == "timestamps", (bad, _i)
    assert "30 fps" in _i.fps_note and "timestamps" in _i.fps_note, _i.fps_note
_i = _probe_with(fps=0.0, msec=lambda k: 0.0)            # no header, no clock: an announced guess
assert _i.fps == 30.0 and _i.fps_source == "assumed" and "ASSUMING" in _i.fps_note, _i
print("frame rate OK: 1200 / 2000 / 5000 fps headers kept; 0 / NaN / timebase headers measured "
      "from the timestamps; nothing at all = assumed and said")

# ---- VFR sniff: a whole-millisecond clock is not a variable rate (I42) ----
for rate in (400, 600, 960):
    _i = probe_video(_write_clip(os.path.join(scratch, f"cfr_{rate}.mkv"), rate))
    assert not _i.vfr_suspected, f"constant {rate} fps MKV flagged as variable frame rate"
    assert abs(_i.fps - rate) < 1e-6 and _i.fps_source == "header", (rate, _i)
# a genuinely variable clock in whole ms is still caught: 30 fps with dropped frames
_gaps = np.cumsum([33 if k % 7 else 67 for k in range(80)]).astype(float)
_i = _probe_with(msec=lambda k: float(_gaps[k]) if 0 <= k < len(_gaps) else 0.0)
assert _i.vfr_suspected, "real VFR in a millisecond clock must still be flagged"
_q = lambda k: float(round(k * 1000.0 / 400.0))          # CFR 400 fps quantised to 1 ms
_i = _probe_with(fps=400.0, msec=_q)
assert not _i.vfr_suspected and _i.fps == 400.0 and _i.fps_source == "header", _i
print("VFR sniff OK: 400 / 600 / 960 fps MKV not flagged, a real variable clock still is")

# ---- ReadAhead: an exception is not the end of the video (I40), stop() really stops (I39) ----
from kinetrace.video_source import ReadAhead


class _RaisingSrc:
    def __init__(self):
        self._pos = 0

    def read_next(self):
        if self._pos == 5:
            raise MemoryError("could not allocate a 4K frame")
        self._pos += 1
        return self._pos - 1, np.zeros((2, 2, 3), np.uint8)


_ra = ReadAhead(_RaisingSrc())
_got, _raised = [], None
try:
    while True:
        _nxt = _ra.read_next()
        if _nxt is None:
            break
        _got.append(_nxt[0])
except MemoryError as e:
    _raised = e
finally:
    _ra.stop()
assert _got == [0, 1, 2, 3, 4] and isinstance(_raised, MemoryError), (_got, _raised)


class _SlowSrc:
    def __init__(self):
        self._pos = 0
        self.inside = threading.Event()

    def read_next(self):
        self.inside.set()
        time.sleep(2.6)            # one stalled read (a network share), longer than the old 2 s join
        self._pos += 1
        self.inside.clear()
        return self._pos - 1, np.zeros((2, 2, 3), np.uint8)


_slow = _SlowSrc()
_ra = ReadAhead(_slow)
assert _slow.inside.wait(5)
_t0 = time.time()
_ra.stop()
assert not _ra._thread.is_alive() and not _slow.inside.is_set(), \
    f"stop() returned after {time.time() - _t0:.2f} s with the producer still reading the source"
print(f"ReadAhead OK: a decode error reaches the consumer; stop() waited {time.time() - _t0:.1f} s "
      "for the stalled read instead of handing the source back mid-read")

print("ALL CORE CHECKS PASSED")
