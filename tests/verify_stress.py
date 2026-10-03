"""Stress battery: hostile edge cases for the session model, timeline math,
and gesture/event bookkeeping. No GPU, no tracking — runs in seconds."""
import json
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)

from kinetrace.session import TrackingSession, PointMeta  # noqa: E402

# ---- 1. two-frame video session: everything clamps, nothing crashes ----
s = TrackingSession("x.mp4", 2, 30.0, 64, 48)
p = s.add_point(0, 10.0, 10.0)
s.add_event("e", -5, 100)
assert (s.events[0].start, s.events[0].end) == (0, 1)
s.write_segment(1, np.full((16, 1, 2), 5.0, np.float32), np.ones((16, 1), bool), [p])
assert s.tracked[1, p] and s.tracks.shape == (2, 1, 2)
s.update_event(0, start=1, end=0)   # inverted update swaps
assert s.events[0].start <= s.events[0].end
print("2-frame session OK")

# ---- 2. rename collision storm (100 points fighting for one name) ----
s2 = TrackingSession("x.mp4", 10, 30.0, 640, 480)
for i in range(100):
    s2.add_point(0, float(i), 1.0)
applied = [s2.rename_point(i, "ball") for i in range(100)]
assert len(set(applied)) == 100, "collision storm produced duplicate names"
assert applied[0] == "ball" and applied[99] == "ball (100)"
# sanitizer-aware: "a,b" vs "a_b" must not merge into one CSV column
s2.rename_point(0, "a,b")
assert s2.rename_point(1, "a_b") != "a_b"
csvp = os.path.join(SCRATCH, "storm.csv")
s2.export_csv(csvp)
header = open(csvp).readline().strip().split(",")
xcols = [c for c in header if c.endswith("_x")]
assert len(set(xcols)) == len(xcols) == 100, "export columns must stay unique"
print("rename storm OK")

# ---- 3. weird names survive round-trip and exports ----
s3 = TrackingSession("x.mp4", 5, 30.0, 640, 480)
weird = ['точка "1"', "日本語,名前", "tab\there", "  spaces  ", "a" * 300]
for i, nm in enumerate(weird):
    pid = s3.add_point(0, float(i), 1.0)
    s3.rename_point(pid, nm)
s3.add_event("ev,\twith\nbad chars", 1, 3)
pth = os.path.join(SCRATCH, "weird.kinetrace")
s3.save(pth)
r3 = TrackingSession.load(pth)
assert [p.name for p in r3.points] == [p.name for p in s3.points]
r3.export_csv(os.path.join(SCRATCH, "weird.csv"))
r3.export_tsv_sparse(os.path.join(SCRATCH, "weird.tsv"))
r3.export_mat(os.path.join(SCRATCH, "weird.mat"))
r3.export_events_csv(os.path.join(SCRATCH, "weird_ev.csv"))
print("weird names OK")

# ---- 3b. no name reaches a CSV as a spreadsheet formula (M7, CSV injection) ----
from kinetrace.project import Project  # noqa: E402
from kinetrace import projectfile  # noqa: E402

sf = TrackingSession("x.mp4", 5, 30.0, 640, 480)
pa = sf.add_point(0, 1.0, 1.0, name="=HYPERLINK(\"http://x\",\"y\")")
pb = sf.add_point(0, 2.0, 1.0)
sf.rename_point(pb, "+cmd|' /C calc'!A0")
pc = sf.add_point(0, 3.0, 1.0, name="@SUM(1)")
pd_ = sf.add_point(0, 4.0, 1.0, name="-2+3")
sf.add_point(0, 5.0, 1.0, name="\t=x")
sf.add_event("=evil()", 0, 1)
sf.add_event("walk", 1, 2)
sf.update_event(1, name="@run")
sf.apply_skeleton({"name": "t", "landmarks": ["=snout", "tail"], "bones": [["=snout", "tail"]], "head": "=snout"})
names = [p.name for p in sf.points] + [e.name for e in sf.events]
assert not any(n.strip()[:1] in "=+-@" for n in names), names
assert "snout" in names and sf.skeleton["head"] == "snout" and ["snout", "tail"] in sf.skeleton["bones"], sf.skeleton
assert [p.name for p in sf.points][:2] == ['HYPERLINK("http://x","y")', "cmd|' /C calc'!A0"], names
pf_path = os.path.join(SCRATCH, "formula.kinetrace")
projectfile.save(Project([sf]), pf_path)
pcsv = os.path.join(pf_path, "cameras", "cam1", "points.csv")
text = open(pcsv, encoding="utf-8").read()
open(pcsv, "w", encoding="utf-8", newline="").write(text.replace("\ntail,", "\n=tail,", 1))
back = projectfile.load(pf_path)
assert "tail" in [p.name for p in back.sessions[0].points] and not any(
    p.name.startswith("=") for p in back.sessions[0].points), [p.name for p in back.sessions[0].points]
pr = Project([sf])
assert pr.rename_landmark("tail", "=tail2") == "tail2"
sf.export_csv(os.path.join(SCRATCH, "formula.csv"))
sf.export_events_csv(os.path.join(SCRATCH, "formula_ev.csv"))
for f in ("formula.csv", "formula_ev.csv"):
    for cell in open(os.path.join(SCRATCH, f), encoding="utf-8").read().replace("\n", ",").split(","):
        assert not cell.strip().strip('"')[:1] in ("=", "+", "@"), (f, cell)
print("formula-safe names OK")

# ---- 4. remove points/events in every order; snapshot isolation ----
s4 = TrackingSession("x.mp4", 20, 30.0, 640, 480)
for i in range(5):
    s4.add_point(0, float(i * 10), 5.0)
for i in range(4):
    s4.add_event(f"e{i}", i, i + 5)
snap = s4.snapshot()
s4.remove_point(0)
s4.remove_point(3)
s4.remove_event(0)
assert s4.n_points == 3 and len(s4.events) == 3
s4.restore(snap)
assert s4.n_points == 5, "restore must bring points back"
assert len(s4.events) == 3, "undo must NOT touch events (tracking-run scope only)"
snap.points[0].name = "mutated"
assert s4.points[0].name != "mutated", "snapshot must not alias live metas"
print("remove/restore isolation OK")

# ---- 5. unknown members load; a newer format and odd view state are handled ----
# (both forms: the single file, a zip, and the project folder, I145)
import shutil  # noqa: E402
import zipfile  # noqa: E402
from kinetrace import projectfile  # noqa: E402
from kinetrace.project import Project  # noqa: E402
s5 = TrackingSession("x.mp4", 8, 30.0, 100, 100)
s5.add_point(0, 1.0, 1.0)
for form in ("single file", "folder"):
    p99 = os.path.join(SCRATCH, "future.kinetrace" if form == "single file" else "future-folder.kinetrace")
    if os.path.isdir(p99):
        shutil.rmtree(p99)
    elif os.path.exists(p99):
        os.remove(p99)
    projectfile.save(Project([s5]), p99, single_file=form == "single file")
    if form == "single file":
        with zipfile.ZipFile(p99) as z:
            members = {n: z.read(n) for n in z.namelist()}
    else:
        members = {}
        for d_, _dirs, fs in os.walk(p99):
            if any(part.startswith(".") for part in os.path.relpath(d_, p99).split(os.sep) if part != "."):
                continue
            for f_ in fs:
                full = os.path.join(d_, f_)
                members[os.path.relpath(full, p99).replace(os.sep, "/")] = open(full, "rb").read()

    def rezip(changes, form=form, p99=p99, members=members):
        if form == "single file":
            with zipfile.ZipFile(p99, "w") as z:
                for n, b in {**members, **changes}.items():
                    z.writestr(n, b)
            return
        for n, b in {**members, **changes}.items():
            full = os.path.join(p99, *n.split("/"))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as fh:
                fh.write(b)


    rezip({"mystery_folder/extra.csv": b"a,b\n1,2\n"})           # a member this version does not know
    assert TrackingSession.load(p99).n_points == 1
    meta = json.loads(members["kinetrace.json"])
    rezip({"kinetrace.json": json.dumps(dict(meta, format_version=99)).encode()})
    try:
        TrackingSession.load(p99)
        raise AssertionError("a newer format must be refused")
    except projectfile.ProjectFileError as e:
        assert "newer" in str(e), e
    # hand-edited state / view files with odd values fall back to defaults instead of refusing
    view = next(n for n in members if n.endswith("/view.json"))
    for bad in (b'"just a string"', b"[1,2,3]", b'{"tools": "x"}', b'{"zoom": "big", "current_frame": "a"}'):
        rezip({"state.json": bad, view: bad})
        r = TrackingSession.load(p99)
        assert r.n_points == 1 and isinstance(r.ui_state, dict) and r.current_frame == 0
    rezip({"state.json": b"{invalid json"})
    try:
        TrackingSession.load(p99)
        raise AssertionError("unreadable JSON must be refused with its file name")
    except projectfile.ProjectFileError as e:
        assert "state.json" in str(e), e
    print(f"unknown members / newer format / odd view state OK ({form})")

# ---- 6. tracker geometry helpers under hostile inputs (no model needed) ----
from kinetrace.tracker import sample_members, fit_group  # noqa: E402
m = sample_members(np.array([5000.0, -50.0], np.float32), 400.0, 640, 480)
assert (m[:, 0] >= 1).all() and (m[:, 0] <= 638).all() \
    and (m[:, 1] >= 1).all() and (m[:, 1] <= 478).all(), "hostile centers must clamp"
c, v, cf = fit_group(m, np.array([5000.0, -50.0], np.float32), m.copy(),
                     np.full(13, 0.9, np.float32), np.ones(13, bool), 400.0, (640, 480))
assert np.isfinite(c).all()
zero_r = sample_members(np.array([320.0, 240.0], np.float32), 0.0, 640, 480)
assert np.allclose(zero_r, (320, 240)), "zero radius = all members at center"
c0, _, cf0 = fit_group(zero_r, np.array([320.0, 240.0], np.float32),
                       zero_r + 3.0, np.full(13, 0.9, np.float32),
                       np.ones(13, bool), 0.0, (640, 480))
assert np.allclose(c0, (323, 243), atol=0.5), "degenerate group must fall back to median"
print("hostile geometry OK")

# ---- 7. timeline widget with pathological sessions ----
from PySide6.QtWidgets import QApplication  # noqa: E402
app = QApplication([])
from kinetrace.timeline import TimelinePanel  # noqa: E402
tl = TimelinePanel()
tl.resize(600, 80)
tl.set_session(None)
tl.repaint()                                  # no session
s7 = TrackingSession("x.mp4", 1, 30.0, 10, 10)  # 1-frame video, 0 points
tl.set_session(s7)
tl.repaint()
assert tl._frame_at(1e9) == 0 and tl._frame_at(-1e9) == 0
for i in range(25):                            # more lanes than fit -> scroll path
    s7.add_point(0, 1.0, 1.0)
tl.set_session(s7)
tl.repaint()
s7.add_event("z", 0, 0)                        # zero-length event ribbon
tl.repaint()
print("timeline pathological OK")

print("STRESS PASSED")
