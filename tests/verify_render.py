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
from _clean import forget_recovery  # noqa: E402
forget_recovery(VID)
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

# ---- slow motion (G194): the playback arithmetic and the speed label ------------
from kinetrace.render import playback_plan, speed_words  # noqa: E402

assert playback_plan(30.0, 30.0, 4.0) == (1, 4, 30.0), playback_plan(30.0, 30.0, 4.0)
assert playback_plan(30.0, 30.0, 1.0) == (1, 1, 30.0), playback_plan(30.0, 30.0, 1.0)
assert playback_plan(500.0, 30.0, 2.0) == (5, 1, 50.0), playback_plan(500.0, 30.0, 2.0)
st, rp, fo = playback_plan(500.0, 30.0, 500.0 / 30.0)
assert (st, rp) == (1, 1) and abs(fo - 30.0) < 1e-9, (st, rp, fo)
st, rp, fo = playback_plan(30.0, 30.0, 20.0)             # 1.5 source frames a second: each shown 16 times
assert (st, rp) == (1, 16) and abs(fo - 24.0) < 1e-9, (st, rp, fo)
st, rp, fo = playback_plan(23.976, 23.976, 2.0)        # film at 2x: doubled, not tripled (review nit)
assert (st, rp) == (1, 2) and abs(fo - 23.976) < 1e-9, (st, rp, fo)
from kinetrace.render import slowdown_sentence  # noqa: E402
assert "lasts 1000 s" in slowdown_sentence(30.0, 30.0, 1000.0, 30), slowdown_sentence(30.0, 30.0, 1000.0, 30)
assert speed_words(4.0) == "4x slower", speed_words(4.0)
assert speed_words(500.0 / 30.0) == "16.7x slower", speed_words(500.0 / 30.0)
assert speed_words(0.8) == "1.25x faster", speed_words(0.8)
assert speed_words(1.0) == "" and speed_words(1.002) == "" and speed_words(None) == ""
lab_opts = OverlayOptions(0, T - 1, 1.0, trails=10)
plain = draw_overlay(blank.copy(), 20, s, lab_opts, s.bones())
assert np.array_equal(plain, img), "the default drawing must not change"
for kw in ({"slowdown": None}, {"slowdown": 1.0}):
    assert np.array_equal(draw_overlay(blank.copy(), 20, s, lab_opts, s.bones(), **kw), plain), kw
off = OverlayOptions(0, T - 1, 1.0, trails=10, speed_label=False)
assert np.array_equal(draw_overlay(blank.copy(), 20, s, off, s.bones(), slowdown=4.0), plain), \
    "the Speed tick off draws no label"
lab = draw_overlay(blank.copy(), 20, s, lab_opts, s.bones(), slowdown=4.0)
ys, xs = np.nonzero((lab != plain).any(axis=2))
assert len(ys), "the speed label was not drawn"
assert ys.max() < 120 and xs.max() < W // 2, ("the label belongs in the top-left text block",
                                               ys.max(), xs.max())
print("slow-motion arithmetic + label OK")

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


# ---- slow motion end to end (G194) --------------------------------------------------
def render_wait(session, path, o, what):
    if os.path.exists(path):
        os.remove(path)
    d = {"ok": None, "err": None}
    job = OverlayRenderer(session, VID, path, o, [])
    job.finished_ok.connect(lambda p, c: d.update(ok=(p, c)))
    job.error.connect(lambda m: d.update(err=m))
    job.start()
    pump(lambda: d["ok"] is not None or d["err"] is not None, 120, what)
    job.wait(5000)
    return job, d


def probe(path):
    cap = cv2.VideoCapture(path)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    for _ in range(min(n, 8)):
        ok, fr = cap.read()
        if ok:
            frames.append(fr.astype(np.int16))
    cap.release()
    return n, fps, frames


out_slow = os.path.join(SCRATCH, "overlay_slow4.mp4")
rs, ds = render_wait(sess, out_slow, OverlayOptions(30, 89, 0.5, trails=20, slowdown=4.0), "4x render")
assert ds["err"] is None, ds["err"]
n_s, fps_s, fr_s = probe(out_slow)
assert n_s == 240 and abs(fps_s - 30.0) < 0.01, ("60 frames 4x slower at 30 fps = 240 frames, 8 s", n_s, fps_s)
assert rs.frames_written == 60 and rs.note == "" and rs.slowdown == 4.0, (rs.frames_written, rs.note)
# frames 0-3 repeat source frame 30, frame 4 is source frame 31 (its frame counter changed); the
# encoder's noise between repeats stays far below 40 grey levels (measured: max 21), a new frame
# changes hundreds of pixels by more than that


def changed(i):
    return int((np.abs(fr_s[i + 1] - fr_s[i]).max(axis=2) > 40).sum())


assert [changed(i) for i in range(7)][:3] == [0, 0, 0] and changed(3) > 50 and changed(4) == 0, \
    ("each source frame must be shown 4 times", [changed(i) for i in range(7)])
print(f"4x slower OK: {n_s} frames at {fps_s:g} fps, each source frame 4 times")

hs = TrackingSession(VID, 600, 300.0, 640, 480)          # a 300 fps camera saved as a 30 fps file
hs.file_fps = 30.0
rh, dh = render_wait(hs, os.path.join(SCRATCH, "overlay_hs2.mp4"),
                     OverlayOptions(30, 89, 0.5, slowdown=2.0), "high-speed 2x render")
assert dh["err"] is None, dh["err"]
n_h, fps_h, _ = probe(rh.out_path)
assert n_h == 20 == rh.frames_written and abs(fps_h - 50.0) < 0.01, ("every 3rd frame at 50 fps", n_h, fps_h)
rd, dd = render_wait(hs, os.path.join(SCRATCH, "overlay_hs_file.mp4"), OverlayOptions(30, 89, 0.5),
                     "high-speed default render")
n_d, fps_d, _ = probe(rd.out_path)
assert dd["err"] is None and n_d == 60 and abs(fps_d - 30.0) < 0.01, (dd, n_d, fps_d)
assert abs(rd.slowdown - 10.0) < 1e-9, ("the file itself plays 10x slower than real time", rd.slowdown)
rn, dn = render_wait(sess, os.path.join(SCRATCH, "overlay_default.mp4"), OverlayOptions(30, 89, 0.5),
                     "default render")
n_n, fps_n, _ = probe(rn.out_path)
assert dn["err"] is None and n_n == 60 and abs(fps_n - 30.0) < 0.01 and rn.slowdown == 1.0, (n_n, fps_n)
print("high-speed OK: 2x slower keeps every 3rd frame at 50 fps; the default plays as the file (10x slower)")

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
# frames skipped (a 90 fps camera at real time: every 2nd frame) and the video ends at 599
long_hs = TrackingSession(VID, 650, 90.0, 640, 480)
long_hs.file_fps = 30.0
r7, d7 = render_wait(long_hs, os.path.join(SCRATCH, "overlay_tail_step.mp4"),
                     OverlayOptions(560, 649, 0.5, slowdown=1.0), "tail render with skipped frames")
assert d7["err"] is None and r7.frames_written == 20, (d7, r7.frames_written)
assert "20 of 45" in r7.note and "frame 598" in r7.note and "Frame 600" in r7.note, r7.note
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

# ---- the Speed row, with the keyboard and mouse (G194) -------------------------------
from PySide6.QtCore import Qt                           # noqa: E402
from PySide6.QtTest import QTest                        # noqa: E402
from PySide6.QtWidgets import QDialogButtonBox          # noqa: E402

dlg = OverlayDialog(None, sess, os.path.join(SCRATCH, "overlay_dlg.mp4"), 150)
dlg.show()
app.processEvents()
QTest.mouseClick(dlg.r_custom, Qt.LeftButton)
dlg.f0.setFocus()
dlg.f0.selectAll()
QTest.keyClicks(dlg.f0, "0")
dlg.f1.setFocus()
dlg.f1.selectAll()
QTest.keyClicks(dlg.f1, "299")
assert dlg._range() == (0, 299), dlg._range()
assert "Real time" in dlg.speed.currentText() and not dlg.speed_x.isEnabled(), dlg.speed.currentText()
dlg.speed.setFocus()
QTest.keyClicks(dlg.speed, "4")                         # the combo's own type-to-select
assert dlg.speed.currentText() == "4x slower", dlg.speed.currentText()
assert "40.0 s" in dlg.speed_info.text() and "4x slower" in dlg.speed_info.text(), dlg.speed_info.text()
QTest.qWait(QApplication.keyboardInputInterval() + 100)     # a new type-to-select search
QTest.keyClicks(dlg.speed, "O")
assert dlg.speed.currentText().startswith("Other") and dlg.speed_x.isEnabled(), dlg.speed.currentText()
dlg.speed_x.setFocus()
dlg.speed_x.selectAll()
QTest.keyClicks(dlg.speed_x, "3.3")
assert "33.0 s" in dlg.speed_info.text() and "3.3x slower" in dlg.speed_info.text(), dlg.speed_info.text()
QTest.mouseClick(dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok), Qt.LeftButton)
assert dlg.result_options is not None and abs(dlg.result_options.slowdown - 3.3) < 1e-9, dlg.result_options
assert dlg.result_options.speed_label, "the Speed tick starts on"
dlg.deleteLater()
dlg2 = OverlayDialog(None, hs, os.path.join(SCRATCH, "overlay_dlg2.mp4"), 150)
assert dlg2.speed.currentText() == "As the file plays (10x slower)", dlg2.speed.currentText()
assert dlg2.speed.findText("Real time") >= 0, "a high-speed clip can be sped up to real time"
dlg2._accept()
assert dlg2.result_options.slowdown is None, "the default plays as the file, untouched"
dlg2.deleteLater()
print("Speed row OK: type-to-select 4x, Other 3.3x typed, the line gives the length; high-speed default")

# ---- through the app: the File menu action exists and is gated ---------------
assert win.act_overlay.isEnabled(), "Export Overlay Video should be enabled with a video open"

# the dialog's "Skeleton bones" tick decides, even with bones hidden on the canvas (I44)
import kinetrace.render as _render                  # noqa: E402
from PySide6.QtWidgets import QDialog                   # noqa: E402

# (G153, G157) a bone joins two points of one animal
k = sess.add_segment()
sess.move_points([0, 1], k)
assert sess.connect_bone(0, 1) and sess.bones() == [(0, 1)], sess.bones()
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
    dlg.speed.setCurrentIndex(dlg.speed.findText("4x slower"))
    dlg.exec = lambda: (dlg._accept(), QDialog.Accepted)[1]
    return dlg


toasts = []
_real_toast = win.toast.show_message
win.toast.show_message = lambda text, *a, **k: (toasts.append(text), _real_toast(text, *a, **k))
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
win.toast.show_message = _real_toast
assert os.path.exists(out6)
n6, _, _ = probe(out6)
assert n6 == 44, ("frames 30-40 at 4x slower = 44 frames", n6)
assert any("4x slower" in t for t in toasts), toasts
win.act_show_bones.setChecked(True)
print("bones tick OK: honoured with View -> bones off")
win._dev_probe.wait(30000)
win.close()
app.processEvents()
forget_recovery(VID)

# ---- the cameras together (G195) ---------------------------------------------------------
from kinetrace.render import tile_layout                 # noqa: E402
from kinetrace.viewgrid import grid_shape                # noqa: E402

assert [grid_shape(n) for n in (1, 2, 3, 4, 5)] == [(1, 1), (2, 1), (2, 2), (2, 2), (3, 2)], \
    [grid_shape(n) for n in (1, 2, 3, 4, 5)]
canvas, rects = tile_layout([(640, 480), (640, 480)], 0, 2, 1, 1.0)
assert canvas == (1280, 480) and rects == [(0, 0, 640, 480, 1.0), (640, 0, 640, 480, 1.0)], (canvas, rects)
canvas, rects = tile_layout([(3840, 2160)] * 4, 0, 2, 2, 1.0)
assert canvas[0] <= 3840 and canvas[1] <= 2160 and canvas[0] % 2 == 0 and canvas[1] % 2 == 0, canvas
assert canvas == (3840, 2160) and rects[3][:2] == (1920, 1080), (canvas, rects)
canvas, rects = tile_layout([(640, 480), (320, 240)], 0, 2, 1, 1.0)
assert rects[1] == (640, 0, 640, 480, 2.0), rects
canvas, rects = tile_layout([(640, 480), (640, 240)], 0, 2, 1, 1.0)     # wider: black bars above / below
assert rects[1] == (640, 120, 640, 240, 1.0), rects
canvas, rects = tile_layout([(640, 480), (640, 480)], 0, 2, 1, 0.5)
assert canvas == (640, 240) and rects[1] == (320, 0, 320, 240, 0.5), (canvas, rects)
print("grid_shape + tile_layout OK")

VW, VH, VN = 320, 240, 80
BITS, SQ = 7, 12                                        # frame index as 7 black / white squares of 12 px
BX, BY = VW - BITS * SQ - 8, VH - SQ - 8                # a corner the overlay never draws on


def frame_clip(name, n=VN, shade=60):
    """Frame f carries its own index in binary, a row of black / white squares (survives the encoders;
    a grey level per frame did not: measured errors of 2-4 levels)."""
    path = os.path.join(SCRATCH, name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (VW, VH))
    for f in range(n):
        im = np.full((VH, VW, 3), shade, np.uint8)
        for b in range(BITS):
            im[BY:BY + SQ, BX + b * SQ:BX + (b + 1) * SQ] = 235 if (f >> b) & 1 else 15
        vw.write(im)
    vw.release()
    return path


def read_all(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return out, fps


def block_frame(img, rect):
    """The source frame a tile shows (its squares read back), or None for a dark (not recording) tile:
    the clip's grey background (60) is gone there."""
    x0, y0, w, h, sc = rect

    def at(x, y):
        cx, cy = int(x0 + x * sc), int(y0 + y * sc)
        return float(img[cy - 2:cy + 3, cx - 2:cx + 3].mean())

    if at(BX - 20, BY + SQ / 2) < 30:
        return None
    return sum(1 << b for b in range(BITS) if at(BX + (b + 0.5) * SQ, BY + SQ / 2) > 125)


CLIPS = [frame_clip(f"views_{c}.mp4") for c in "abc"]
for c in CLIPS:
    forget_recovery(c)
win2 = MainWindow()
win2.show()
win2._open_video(CLIPS[0])
pump(lambda: win2.state == READY and win2.project is not None and not win2._loading, 20, "open camera a")
for c in CLIPS[1:]:
    assert win2._add_view(c), c
pump(lambda: win2.project.n_views == 3 and not win2._loading and win2.state == READY, 30, "three cameras")
p = win2.project
p.set_offset(1, 10.0)                                   # camera b starts 10 frames earlier
assert p.set_fps(2, 15.0)                               # camera c really ran at half camera a's rate
OUT_V = os.path.join(SCRATCH, "overlay_views.mp4")


def export_views(together, f1=39, speed=None, check_dialog=None):
    """File -> Export -> Overlay Video with the real dialog, its Views choice clicked."""
    seen = {}

    def _dlg(*a, **k):
        dlg = _real_dlg(*a, **k)
        dlg.show()
        app.processEvents()
        if check_dialog:
            check_dialog(dlg)
        QTest.mouseClick(dlg.r_together if together else dlg.r_one, Qt.LeftButton)
        QTest.mouseClick(dlg.r_custom, Qt.LeftButton)
        dlg.f0.setValue(0)
        dlg.f1.setValue(f1)
        if speed:
            dlg.speed.setCurrentIndex(dlg.speed.findText(speed))
        seen["default"] = dlg.path.text()
        dlg.path.setText(OUT_V)
        dlg.exec = lambda: (dlg._accept(), QDialog.Accepted)[1]
        return dlg

    if os.path.exists(OUT_V):
        os.remove(OUT_V)
    msgs = []
    real = win2.toast.show_message
    win2.toast.show_message = lambda text, *a, **k: (msgs.append(text), real(text, *a, **k))
    _render.OverlayDialog = _dlg
    try:
        win2._export_overlay()
    finally:
        _render.OverlayDialog = _real_dlg
    job = win2._overlay
    assert job is not None, "the render did not start"
    pump(lambda: win2._overlay is None, 120, "views render")
    job.wait(5000)
    win2.toast.show_message = real
    return job, msgs, seen


def check3(dlg):
    assert dlg.r_together.isEnabled() and "(3)" in dlg.r_together.text(), dlg.r_together.text()
    assert dlg.r_one.isChecked(), "This camera stays the starting choice"


job, msgs, seen = export_views(True, check_dialog=check3)
assert seen["default"].endswith("_overlay_cameras.mp4"), seen["default"]
frames_v, fps_v = read_all(OUT_V)
assert len(frames_v) == 40 and abs(fps_v - 30.0) < 0.01, (len(frames_v), fps_v)
assert frames_v[0].shape[:2] == (480, 640), ("three 320x240 cameras in a 2x2 grid", frames_v[0].shape)
assert [t[0] for t in job.tiles] == [0, 1, 2], job.tiles
for f, img in enumerate(frames_v):
    for view, *rect in job.tiles:
        want = f if view == 0 else p.map_frame(0, view, f)
        got = block_frame(img, rect)
        assert got == want, (f"output frame {f}: camera {view} shows frame {got}, map_frame says {want}")
assert not any("could not" in m for m in msgs), msgs
print("cameras together OK: 2x2 grid, every tile on map_frame's frame (offset +10, half-rate camera)")

p.set_offset(1, -20.0)                                  # camera b starts 20 frames LATER
job, msgs, _ = export_views(True)
frames_v, _ = read_all(OUT_V)
rect_b = job.tiles[1][1:]
for f in (0, 10, 19, 20, 30):
    got = block_frame(frames_v[f], rect_b)
    assert got == (None if f < 20 else f - 20), (f, got)
x0, y0, w, h, _ = rect_b
assert frames_v[5][y0:y0 + h, x0:x0 + w].max() > 150, "a not-recording tile says so in words"
print("not recording OK: camera b's tile is dark with a sentence until it starts")

job_plain, _, _ = export_views(True, f1=9)
plain, _ = read_all(OUT_V)
job_slow, msgs, _ = export_views(True, f1=9, speed="4x slower")
slow, _ = read_all(OUT_V)
assert len(slow) == 40 and any("4x slower" in m for m in msgs), (len(slow), msgs)
diff = (np.abs(slow[0].astype(np.int16) - plain[0].astype(np.int16)).max(axis=2) > 40)
ys, xs = np.nonzero(diff)
x0, y0, w, h, _ = job_slow.tiles[0][1:]
assert len(ys) and xs.max() < x0 + w and ys.max() < y0 + h, "the speed label belongs to the working tile only"
print("speed label OK: once, in the working camera's tile")

def check2(dlg):
    assert "(2)" in dlg.r_together.text(), dlg.r_together.text()


win2.grid.set_hidden(2, True)                            # camera c's eye off
job, _, _ = export_views(True, check_dialog=check2)
frames_v, _ = read_all(OUT_V)
assert [t[0] for t in job.tiles] == [0, 1] and frames_v[0].shape[:2] == (240, 640), (job.tiles, frames_v[0].shape)
win2.grid.set_hidden(2, False)
job, _, seen = export_views(False)
frames_v, fps_v = read_all(OUT_V)
assert len(frames_v) == 40 and frames_v[0].shape[:2] == (VH, VW) and abs(fps_v - 30.0) < 0.01, \
    (len(frames_v), frames_v[0].shape, fps_v)
assert not seen["default"].endswith("_overlay_cameras.mp4"), seen["default"]
print("hidden camera left out; 'This camera' writes one camera as before")
win2._dev_probe.wait(30000)
win2.close()
app.processEvents()
for c in CLIPS:
    forget_recovery(c)
print("verify_render PASSED")
