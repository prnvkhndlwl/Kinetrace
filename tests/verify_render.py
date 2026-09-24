"""Overlay video render: `draw_overlay` on synthetic state (markers, names,
bones, silhouette, trails, hidden cells skipped, frame counter, events,
notes, scaling) and `OverlayRenderer` end to end on test600.mp4 (file
written, playable, right length and size, cancel deletes the partial file).
Then the File menu path through the app (offscreen)."""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

VID = os.path.join(ROOT, "test600.mp4")
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
for leftover in (VID + ".cotracker.npz",):
    if os.path.exists(leftover):
        os.remove(leftover)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

app = QApplication([])
from kinetrace.session import TrackingSession, MaskTrack
from kinetrace.render import OverlayOptions, OverlayRenderer, draw_overlay, open_writer

# ---- pure drawing on a synthetic session --------------------------------------
W, H, T = 640, 480, 50
s = TrackingSession("synthetic.mp4", T, 30.0, W, H)
a = s.add_point(0, 100.0, 100.0, name="head")
b = s.add_point(0, 200.0, 150.0, name="tail")
g = s.add_point(0, 400.0, 300.0, kind="group", radius=30.0, shape="rect",
                outline=[[370, 280], [430, 280], [430, 320], [370, 320]])
tw = np.zeros((T, 3, 2), np.float32)
tw[:, 0] = (100, 100)
tw[:, 0, 0] += np.arange(T) * 2
tw[:, 1] = (200, 150)
tw[:, 2] = (400, 300)
s.write_segment(0, tw, np.ones((T, 3), bool), [0, 1, 2], np.full((T, 3), 0.9, np.float32))
s.visibility[20, 1] = False
s.set_occluded(20, 0, True)
s.skeleton = {"name": "t", "landmarks": ["head", "tail"], "bones": [["head", "tail"]], "head": "head"}
s.add_event("run", 10, 30)
s.set_note(20, "note here", "PK")
s.ensure_animal()
bitmap = np.zeros((H, W), bool)
bitmap[200:260, 300:360] = True
s.masks.set(20, bitmap, 9.0)
assert s.masks.has(20)
blank = np.zeros((H, W, 3), np.uint8)
opts = OverlayOptions(0, T - 1, 1.0, trails=10)
img = draw_overlay(blank.copy(), 20, s, opts, s.bones())
assert img.shape == (H, W, 3)
# head is hidden at frame 20: nothing drawn at its position; tail drawn hollow; group outline drawn
hx, hy = tw[20, 0].astype(int)
assert img[hy, hx].sum() == 0, "hidden cell must not be drawn"
assert img[150, 200 + 7].sum() > 0 or img[150 - 7, 200].sum() > 0, "hollow tail ring"
assert img[280, 400].sum() > 0, "rect region outline"
assert img[10:40, 5:200].sum() > 0, "frame counter drawn"
assert img[H - 25:H, 5:300].sum() > 0, "note drawn"
img2 = draw_overlay(blank.copy(), 5, s, opts, s.bones())
assert img2[100, 100 + 10].sum() > 0, "head marker at frame 5"
assert img2[100:150, 100:200].sum() > img[100:150, 100:200].sum() * 0 + 0
half = draw_overlay(blank.copy(), 5, s, OverlayOptions(0, T - 1, 0.5), s.bones())
assert half.shape == (H // 2, W // 2, 3)
assert half[50, 55].sum() > 0 or half[50, 50].sum() > 0, "scaled marker position"
no = draw_overlay(blank.copy(), 5, s, OverlayOptions(0, T - 1, 1.0, markers=False, names=False,
                                                     bones=False, trails=0, frame_number=False,
                                                     events=False, notes=False, mask=False), [])
assert no.sum() == 0, "everything off draws nothing"
print("draw_overlay OK")

# ---- writer: codec fallback ---------------------------------------------------
vw, codec = open_writer(os.path.join(SCRATCH, "codec_probe.mp4"), 30.0, (64, 48))
vw.release()
assert codec in ("avc1", "mp4v"), codec
print(f"writer codec: {codec}")

# ---- renderer thread on the real clip -----------------------------------------
from kinetrace.app import MainWindow, READY

win = MainWindow()
win.show()


def pump(cond, timeout, what):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.005)
        if time.time() - t0 > timeout:
            raise TimeoutError(what)


win._open_video(VID)
pump(lambda: win.state == READY, 20, "open")
sess = win.session
sess.add_point(0, 320.0, 240.0)
sess.add_point(0, 100.0, 100.0)
L = 120
tw = np.zeros((L, 2, 2), np.float32)
tw[:, 0] = (320, 240)
tw[:, 1, 0] = 100 + np.arange(L)
tw[:, 1, 1] = 100
sess.write_segment(0, tw, np.ones((L, 2), bool), [0, 1], np.full((L, 2), 0.9, np.float32))
sess.add_event("bit", 40, 60)
out = os.path.join(SCRATCH, "overlay_test.mp4")
if os.path.exists(out):
    os.remove(out)
done = {"ok": None, "err": None, "prog": []}
r = OverlayRenderer(sess, VID, out, OverlayOptions(30, 89, 0.5, trails=20), sess.bones())
r.progress.connect(lambda d, t: done["prog"].append((d, t)))
r.finished_ok.connect(lambda p, c: done.update(ok=(p, c)))
r.error.connect(lambda m: done.update(err=m))
r.start()
pump(lambda: done["ok"] is not None or done["err"] is not None, 120, "render")
r.wait(5000)
assert done["err"] is None, done["err"]
assert os.path.exists(out) and os.path.getsize(out) > 10_000, os.path.getsize(out)
cap = cv2.VideoCapture(out)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ok, frame = cap.read()
cap.release()
assert ok and n == 60, n
assert (w, h) == (sess.width // 2, sess.height // 2), (w, h)
assert done["prog"] and done["prog"][-1] == (60, 60), done["prog"][-3:]
print(f"rendered {n} frames at {w}x{h} ({done['ok'][1]})")

# cancel leaves no file behind
out2 = os.path.join(SCRATCH, "overlay_cancel.mp4")
d2 = {"ok": None, "err": None}
r2 = OverlayRenderer(sess, VID, out2, OverlayOptions(0, 599, 1.0), [])
r2.finished_ok.connect(lambda p, c: d2.update(ok=(p, c)))
r2.error.connect(lambda m: d2.update(err=m))
r2.progress.connect(lambda d, t: r2.request_cancel() if d >= 10 else None)
r2.start()
pump(lambda: d2["ok"] is not None or d2["err"] is not None, 120, "cancel")
r2.wait(5000)
assert d2["err"] == "cancelled" and not os.path.exists(out2), (d2, os.path.exists(out2))
print("cancel OK")

# ---- edits while it renders (I38): the renderer draws from a frozen copy ----------
import copy as _copy                                    # noqa: E402

before = _copy.deepcopy(sess)
out3 = os.path.join(SCRATCH, "overlay_edit.mp4")
d3 = {"ok": None, "err": None}
r3 = OverlayRenderer(sess, VID, out3, OverlayOptions(30, 89, 0.5, trails=20), sess.bones())
r3.finished_ok.connect(lambda p, c: d3.update(ok=(p, c)))
r3.error.connect(lambda m: d3.update(err=m))
snap = sess.snapshot()
r3.start()
edits = 0
t0 = time.time()
while d3["ok"] is None and d3["err"] is None:
    # what a user can do meanwhile: add a point (N), delete it, Ctrl+Z
    k = sess.add_point(0, 50.0 + edits % 7, 60.0)
    sess.write_segment(0, np.full((L, 1, 2), 55.0, np.float32), np.ones((L, 1), bool), [k])
    sess.remove_point(k)
    if edits % 3 == 0:
        sess.restore(snap)
    edits += 1
    app.processEvents()
    time.sleep(0.002)
    if time.time() - t0 > 120:
        raise TimeoutError("render during edits")
r3.wait(5000)
sess.restore(snap)
assert d3["err"] is None, f"render died while the session was edited ({edits} edits): {d3['err']}"
assert edits > 3, f"the render finished before the edits could race it ({edits} edits)"
assert r3.session is not sess and r3.session.n_points == before.n_points
assert np.array_equal(r3.session.tracks, before.tracks, equal_nan=True)
ref = draw_overlay(np.zeros((480, 640, 3), np.uint8), 60, before, r3.opts, before.bones())
got = draw_overlay(np.zeros((480, 640, 3), np.uint8), 60, r3.session, r3.opts, r3.bones)
assert np.array_equal(ref, got), "the frozen copy must draw exactly what the session drew at the start"
cap = cv2.VideoCapture(out3)
n3 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
assert n3 == 60 and r3.frames_written == 60 and r3.note == "", (n3, r3.frames_written, r3.note)
print(f"edits during render OK: {edits} add / delete / undo edits, render complete and unaffected")

# ---- a range past the frames that decode (I41) ------------------------------------
long_s = TrackingSession(VID, 650, 30.0, 640, 480)      # a project saved with a header's over-count
long_s.add_point(0, 100.0, 100.0)
long_s.add_event("inside", 590, 640)
long_s.add_event("phantom", 620, 630)
out4 = os.path.join(SCRATCH, "overlay_tail.mp4")
d4 = {"ok": None, "err": None}
r4 = OverlayRenderer(long_s, VID, out4, OverlayOptions(560, 649, 0.5), [])
r4.finished_ok.connect(lambda p, c: d4.update(ok=(p, c)))
r4.error.connect(lambda m: d4.update(err=m))
r4.start()
pump(lambda: d4["ok"] is not None or d4["err"] is not None, 60, "tail render")
r4.wait(5000)
cap = cv2.VideoCapture(out4)
n4 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
assert d4["err"] is None and r4.frames_written == 40 == n4, (d4, r4.frames_written, n4)
assert "40 of 90" in r4.note and "frame 599" in r4.note, r4.note
out5 = os.path.join(SCRATCH, "overlay_phantom.mp4")
d5 = {"ok": None, "err": None}
r5 = OverlayRenderer(long_s, VID, out5, OverlayOptions(610, 649, 0.5), [])
r5.finished_ok.connect(lambda p, c: d5.update(ok=(p, c)))
r5.error.connect(lambda m: d5.update(err=m))
r5.start()
pump(lambda: d5["ok"] is not None or d5["err"] is not None, 60, "phantom render")
r5.wait(5000)
assert d5["ok"] is None and "nothing was written" in (d5["err"] or ""), d5
assert not os.path.exists(out5), "an empty overlay must not be left behind"
from kinetrace.render import OverlayDialog          # noqa: E402

dlg = OverlayDialog(None, long_s, out4, 300, (590, 640), n_frames=600)
assert "0–599" in dlg.r_all.text() and dlg.f1.maximum() == 599, (dlg.r_all.text(), dlg.f1.maximum())
assert "(590–599)" in dlg.r_sel.text(), dlg.r_sel.text()
assert dlg.ev_box.count() == 1 and dlg.ev_box.itemData(0) == (590, 599), dlg.ev_box.count()
assert dlg._range() == (590, 599)
dlg.r_all.setChecked(True)
assert dlg._range() == (0, 599)
dlg.deleteLater()
print("short video OK: 40 of 90 frames reported, an all-phantom range refused, the dialog "
      "offers only frames that decode")

# ---- through the app: the File menu action exists and is gated ---------------
assert win.act_overlay.isEnabled(), "Export Overlay Video should be enabled with a video open"

# the dialog's "Skeleton bones" tick decides, even with bones hidden on the canvas (I44)
import kinetrace.render as _render                  # noqa: E402
from PySide6.QtWidgets import QDialog                   # noqa: E402

sess.skeleton = {"name": "t", "landmarks": [p.name for p in sess.points],
                 "bones": [[sess.points[0].name, sess.points[1].name]], "head": sess.points[0].name}
assert sess.bones() == [(0, 1)]
win.act_show_bones.setChecked(False)
_real_dlg = _render.OverlayDialog
out6 = os.path.join(SCRATCH, "overlay_app.mp4")


def _auto_dialog(*a, **k):
    dlg = _real_dlg(*a, **k)
    dlg.r_custom.setChecked(True)
    dlg.f0.setValue(30)
    dlg.f1.setValue(40)
    dlg.path.setText(out6)
    assert dlg.c_bones.isEnabled() and dlg.c_bones.isChecked()
    dlg.exec = lambda: (dlg._accept(), QDialog.Accepted)[1]
    return dlg


_render.OverlayDialog = _auto_dialog
try:
    win._export_overlay()
finally:
    _render.OverlayDialog = _real_dlg
r6 = win._overlay
assert r6 is not None, "the overlay render did not start"
assert r6.bones == [(0, 1)], f"bones ticked in the dialog but not drawn: {r6.bones}"
pump(lambda: win._overlay is None, 60, "app render")
r6.wait(5000)
assert os.path.exists(out6)
win.act_show_bones.setChecked(True)
print("bones tick OK: honoured with View -> bones off")
win._dev_probe.wait(30000)
win.close()
app.processEvents()
for leftover in (VID + ".cotracker.npz",):
    if os.path.exists(leftover):
        os.remove(leftover)
print("verify_render PASSED")
