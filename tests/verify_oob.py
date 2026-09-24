"""Out-of-frame behavior + continue-same-point workflow.

Builds a video where one dot exits the right edge mid-way and one stays inside;
verifies exported coordinates go blank after exit, and that the click-reuse
rule continues the same point without creating a new one."""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
VID = os.path.join(SCRATCH, "exit_test.mp4")

# ---- build: dot A exits right edge ~frame 150, re-enters ~frame 300; dot B stays ----
W, H, T = 640, 480, 450
rng = np.random.default_rng(1)
bg = rng.integers(30, 90, size=(H, W, 3), dtype=np.uint8)
bg = cv2.GaussianBlur(bg, (0, 0), 1.5)
vw = cv2.VideoWriter(VID, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
ax, ay = [], []
for t in range(T):
    x = 320 + t * 2.4            # exits (x>656) around t=150, off till re-entry below
    if t >= 300:
        x = 600 - (t - 300) * 2  # re-enters moving left
    y = 240 + 40 * np.sin(t / 40)
    ax.append(x); ay.append(y)
    frame = bg.copy()
    cv2.circle(frame, (round(x), round(y)), 9, (60, 60, 230), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (round(150 + 30 * np.sin(t / 50)), round(300)), 9, (60, 230, 60), -1,
               lineType=cv2.LINE_AA)
    vw.write(frame)
vw.release()
gtx = np.array(ax); gty = np.array(ay)

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
app = QApplication([])
from cotracker_app.app import MainWindow, READY

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
s = win.session

win._on_add(gtx[0], gty[0])                 # P1: the dot that will exit
win._on_add(150.0 + 0.0, 300.0)             # P2: stays in frame
# Adding a point selects it, and the run scope follows the panel selection —
# clear it so this run covers BOTH points (see CLAUDE.md).
win.point_list.clearSelection()
win.selected = None
assert win._run_scope()[0] is None, "this run must cover every point"
# P1 leaving the frame IS a real confidence collapse, so auto-pause would
# legitimately stop the run at ~frame 136 and truncate P2 with it. This suite is
# about out-of-frame data hygiene and click-continue; verify_conf_autopause owns
# the pausing. The BLANKING of a lost point is independent of the toggle and is
# what frac_blank below measures.
win.btn_autopause.setChecked(False)
win._toggle_tracking()
pump(lambda: win.state != READY, 60, "start")
pump(lambda: win.state == READY, 300, "finish")

# ---- what did the model do around/after exit? ----
p1_tracked = s.tracked[:, 0]
exit_zone = np.arange(170, 290)             # dot fully out of frame here
frac_blank = 1 - p1_tracked[exit_zone].mean()
print(f"P1 blank fraction while out of frame (170-290): {frac_blank:.2%}")
print(f"P1 tracked count 0-150: {p1_tracked[:150].sum()}/150, "
      f"in exit zone: {p1_tracked[exit_zone].sum()}/{len(exit_zone)}")
assert p1_tracked[:130].all(), "P1 should be tracked while its center is in frame (t<133)"
# Two mechanisms blank a vanished dot, and this measures their combination:
#   1. write_segment blanks any prediction that lands OUTSIDE the frame (the
#      deterministic invariant, unit-tested in verify_core);
#   2. the worker blanks a point whose confidence has collapsed AND which the
#      model reports as not visible, sustained for CONF_PAUSE_RUN frames — the
#      visibility-based fallback, which catches the case (1) cannot: the model
#      parks its guess INSIDE the frame, so bounds alone never fire.
# Before (2) existed this measured 0% here; it leaves headroom anyway because
# exactly where the model parks a lost dot varies run to run.
assert frac_blank > 0.5, ("model predictions stayed in-bounds while the dot was gone — "
                          "OOB masking insufficient, needs visibility-based fallback")
# P2 unaffected throughout
assert s.tracked[:, 1].all(), "in-frame point must remain fully tracked"

# ---- did the model re-acquire P1 after re-entry on its own? (it may) ----
back = np.arange(310, 440)
reacq = s.tracked[back, 0]
if reacq.mean() > 0.5:
    got = s.tracks[back, 0, 0][reacq]
    err_re = np.abs(got - gtx[back][reacq])
    print(f"model re-acquired P1 after re-entry ({reacq.mean():.0%} of frames, "
          f"mean |x err| {err_re.mean():.2f} px)")
else:
    print(f"model did not re-acquire P1 after re-entry ({reacq.mean():.0%} tracked)")

# ---- export: blank cells while out of frame ----
csvp = os.path.join(SCRATCH, "exit.csv")
s.export_csv(csvp)
lines = open(csvp).read().splitlines()
blank_frames = [t for t in range(170, 290) if not s.tracked[t, 0]]
t_blank = blank_frames[0]
row = lines[1 + t_blank].split(",")
assert row[1] == "" and row[2] == "", f"frame {t_blank} P1 cells should be blank: {row[:4]}"
print(f"export blanks OK (checked frame {t_blank})")

# ---- continue-same-point: click while the selected point has no data here ----
t_gap = blank_frames[len(blank_frames) // 2]
win._goto(t_gap)
assert not s.tracked[t_gap, 0]
win.selected = 0
n_before = s.n_points
win._on_add(500.0, 250.0)                    # plain click
assert s.n_points == n_before, "click must CONTINUE P1, not add a new point"
assert s.tracked[t_gap, 0] and s.manual[t_gap, 0]
print(f"click-continue OK at frame {t_gap} (no new point created)")

# undo that placement, then do the real user flow at re-entry: correct + re-track
win.session.tracks[t_gap, 0] = np.nan
win.session.tracked[t_gap, 0] = False
win.session.manual[t_gap, 0] = False
win._goto(320)
win._on_place(0, float(gtx[320]), float(gty[320]))
win._toggle_tracking()                       # re-track from 320 (P2 seeds too)
pump(lambda: win.state != READY, 60, "restart")
pump(lambda: win.state == READY, 300, "refinish")
err_tail = np.abs(s.tracks[330:440, 0, 0] - gtx[330:440])
print(f"P1 after continuation: mean |x err| {err_tail.mean():.2f} px")
assert np.isfinite(s.tracks[330:440, 0, 0]).all() and err_tail.mean() < 6.0

# ---- Esc then click must ADD a new point ----
win._deselect()
assert win.selected is None
win._goto(10)
win._on_add(400.0, 100.0)
assert s.n_points == n_before + 1, "after Esc, click must create a new point"
print("Esc â†’ new-point behavior OK")

win.close()
app.processEvents()
os.remove(VID)
print("OOB + CONTINUE-POINT PASSED")

