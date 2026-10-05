"""(G144-G146) The GoPro workflow, on self-made GoPro-style files (tests/_synth_gopro.py; no footage):

  [1] gpmf: the flag (is_gopro / read: None for other footage and for a broken file), the settings, the
      timecode, dropped frames, a knock (the frame, the tilt it left), stabilisation said as a problem,
      settings that differ between cameras, the timecode sync prior; a copy without the header box still
      counts as GoPro footage but has no lens model.
  [2] the lens: GoPro's polynomial as Kinetrace's fisheye profile (focal, field of view, the whole border
      usable); "GoPro lens + your boards" recovers a unit's focal length and centre from boards confined
      to the middle of the picture; an unsettled centre keeps GoPro's and says so; the curvature is worded
      as the lens's, not an error (G144).
  [3] the window (offscreen, real clicks): the flag set at open, a GoPro problem said once in a notice, other
      footage gets nothing; 3D -> GoPro Cameras… table, its lens button attaches GoPro's lens, a Moved cell
      goes to that camera and frame; Sync's search starts from the timecode; the lens wizard shows the GoPro
      choices only for GoPro footage and "Use GoPro's lens without boards" gives the nominal profile.

.venv\\Scripts\\python.exe tests\\verify_gopro.py
"""
import os
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
OUT = os.path.join(HERE, "out", "gopro")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from _synth_gopro import H12_POLY, H12_ZFOV, H12_ZMPL, make_gopro_video  # noqa: E402
from kinetrace import gpmf, lens  # noqa: E402

FAILS = []


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(what)


def plain_clip(path, n=60):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    rng = np.random.RandomState(3)
    base = rng.randint(0, 255, (280, 320 + 4 * n, 3)).astype(np.uint8)
    for f in range(n):
        vw.write(np.ascontiguousarray(base[20:260, 4 * f:4 * f + 320]))
    vw.release()
    return path


# ------------------------------------------------------------------ [1] gpmf
print("[1] the GoPro metadata")
GA = make_gopro_video(os.path.join(OUT, "camA.mp4"), seconds=6, sensors=dict(bump_at=3.5, dropped_at=1),
                      timecode_frame=1_000_000)
GB = make_gopro_video(os.path.join(OUT, "camB.mp4"), seconds=6, header=dict(hs="HIGH"),
                      sensors=dict(shutter=1 / 480), timecode_frame=1_000_030)
NOHEAD = make_gopro_video(os.path.join(OUT, "copy.mp4"), seconds=2, with_header=False)
PLAIN = plain_clip(os.path.join(OUT, "plain.mp4"))
JUNK = os.path.join(OUT, "junk.mp4")
open(JUNK, "wb").write(b"\0\0\0\x10moov" + os.urandom(200))
check(gpmf.is_gopro(GA) and gpmf.is_gopro(NOHEAD) and not gpmf.is_gopro(PLAIN) and not gpmf.is_gopro(JUNK),
      "the flag: GoPro files yes (also a copy without the header), other footage and a broken file no")
check(gpmf.read(PLAIN) is None and gpmf.read_safe(JUNK) is None, "read: None for other footage, read_safe never raises")
a = gpmf.read(GA)
check(a.model == "HERO12 Black" and a.lens_mode == "Wide" and (a.width, a.height) == (320, 240)
      and abs(a.fps - 30.0) < 1e-6 and a.stabilisation == "off" and not a.stabilised and a.has_lens,
      "the header: model, lens mode, picture, frame rate, stabilisation, lens model", a.label)
check(a.timecode == (1_000_000, 30.0) and a.dropped_frames == 1 and abs(a.shutter_s - 1 / 1920) < 1e-7
      and a.iso == 1600, "the timecode, a dropped frame, shutter and ISO", (a.timecode, a.dropped_frames))
tl = a.tilt()
check(abs(tl[0] - 55.4) < 1.0 and abs(tl[1] - 10.9) < 1.0, "the tilt from the gravity sensor (55 down, roll 11)", tl)
mv = a.moves
check(len(mv) == 1 and mv[0]["frame"] == 105 and mv[0]["to_end"] and abs(mv[0]["tilt_deg"] - 1.3) < 0.15
      and a.jolts and a.jolts[0]["frame"] == 105, "the knock at 3.5 s = frame 105, 1.3 degrees from then on",
      (mv, a.jolts))
pa = gpmf.problems(a, "camA")
check(any("dropped frame" in s for s in pa) and any("moved at frame 105" in s for s in pa) and len(pa) == 2,
      "problems: the dropped frame and the move, in words", pa)
b = gpmf.read(GB)
pb = gpmf.problems(b, "camB")
check(b.stabilised and any("stabilisation was ON" in s for s in pb), "stabilisation on is a problem, said", pb)
rig = gpmf.rig_problems({"camA": a, "camB": b})
check(any("shutter speeds differ" in s for s in rig), "the rig: different shutter speeds are said", rig)
check(gpmf.timecode_prior([a, b], [30.0, 30.0]) == [0.0, -30.0],
      "the timecode prior: camB started 1 s (30 frames) after camA")
n = gpmf.read(NOHEAD)
check(n is not None and not n.header_found and n.sensors_found and not n.has_lens and gpmf.lens_profile(n) is None,
      "a copy without the header: GoPro footage with sensors but no lens model")

# ------------------------------------------------------------------ [2] the lens
print("[2] the lens")
h12 = gpmf.GoProInfo("h12", model="HERO12 Black", width=2704, height=1520, fps=239.76, lens_mode="Wide",
                     poly=list(H12_POLY), zmpl=H12_ZMPL, zfov_deg=H12_ZFOV)
base = gpmf.lens_profile(h12)
chk = base.border_check()
check(abs(base.K[0, 0] - 1299.6) < 1.0 and base.report["gopro_poly_fit_px"] < 0.5 and chk["valid_frac"] == 1.0
      and abs(chk["fov_diag_deg"] - H12_ZFOV) < 1.0,
      "GoPro's HERO12 2.7K Wide as a fisheye profile: f 1299.6 px, the whole border, its field of view",
      (base.K[0, 0], base.report["gopro_poly_fit_px"], chk))
check("lens curvature" in base.summary() and "bends the edges" not in base.summary(),
      "G144: the curvature is worded as the lens's, not an error", base.summary())

# a unit 1.2 % longer than nominal, centre off by (+8, -6), boards only in the middle of the picture
rng = np.random.default_rng(4)
K_true = base.K_square().copy()
K_true[0, 0] = K_true[1, 1] = base.K[0, 0] * 1.012
K_true[0, 2] += 8.0
K_true[1, 2] -= 6.0
D = np.asarray(base.dist, float).reshape(4, 1)
obj = lens._object_points((9, 6), 25.0)
views = []
while len(views) < 60:
    rv = rng.normal(0, 0.5, 3)
    tv = np.array([rng.uniform(-200, 200), rng.uniform(-120, 120), rng.uniform(450, 900)])
    pr, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rv, tv, K_true, D)
    pr = pr.reshape(-1, 2) + rng.normal(0, 0.15, (len(obj), 2))
    r = np.hypot(pr[:, 0] - 1351.5, pr[:, 1] - 759.5) / np.hypot(1351.5, 759.5)
    if r.max() < 0.6 and (pr[:, 0] > 0).all():
        views.append(pr.astype(np.float32))
idx, err, why = lens.auto_select(views, (9, 6), 25.0, (2704, 1520), "gopro", base=base)
prof = lens.calibrate_lens([views[i] for i in idx], (9, 6), 25.0, (2704, 1520), "gopro",
                           max_views=max(len(idx), lens.MAX_VIEWS), base=base)
df = prof.K[0, 0] / K_true[0, 0] - 1
dc = (prof.K[0, 2] - K_true[0, 2], prof.K[1, 2] - K_true[1, 2])
check(abs(df) < 0.003 and max(abs(c) for c in dc) < 3.0 and prof.report["centre_fixed"] == [False, False],
      "GoPro lens + boards in the middle 60 %: the unit's focal length (0.3 %) and centre (3 px) recovered",
      (df, dc, prof.report["centre_fixed"], prof.report.get("centre_split")))
check(prof.border_check()["valid_frac"] == 1.0 and any("GoPro's own model" in s for s in prof.report["verdict_reasons"]),
      "... valid over the whole border, and the report says where the curve comes from")
saved = lens.CENTRE_SPLIT_FRAC, lens.CENTRE_SPLIT_MIN_PX
lens.CENTRE_SPLIT_FRAC = lens.CENTRE_SPLIT_MIN_PX = -1.0     # every split counts as unsettled: the fallback runs
try:
    fixed = lens.calibrate_lens([views[i] for i in idx], (9, 6), 25.0, (2704, 1520), "gopro",
                                max_views=max(len(idx), lens.MAX_VIEWS), base=base)
finally:
    lens.CENTRE_SPLIT_FRAC, lens.CENTRE_SPLIT_MIN_PX = saved
check(fixed.report["centre_fixed"] == [True, True] and abs(fixed.K[0, 2] - base.K[0, 2]) < 1e-6
      and any("could not be pinned down" in s for s in fixed.report["verdict_reasons"]),
      "an unsettled centre keeps GoPro's, and the report says so")

# ------------------------------------------------------------------ [3] the window
print("[3] through the window")
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
app = QApplication.instance() or QApplication([])
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.goprodialog import GoProDialog  # noqa: E402


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def settle(w):
    for _ in range(300):
        pump(0.05)
        if w.state == READY and w.project is not None and not w._loading:
            break
    pump(0.3)


def notices(w):
    return " ".join(str(n[0]) for n in getattr(w.toast, "_notices", []))


w = MainWindow()
w.resize(1400, 900)
w.show()
app.setActiveWindow(w)
w._open_video(PLAIN)
settle(w)
check(w._views[0].info.gopro is None and not w.act_gopro.isEnabled() and "GoPro" not in notices(w),
      "other footage: no flag, no GoPro entry, no GoPro notice")
w._open_video(GA)
settle(w)
pump(0.3)
check(w._views[0].info.gopro is not None and w._views[0].info.gopro.model == "HERO12 Black",
      "GoPro footage: the flag is set when the video opens")
t = notices(w)
check("moved at frame 105" in t and "dropped frame" in t, "the notice names the move and the dropped frame", t[:300])
check(w._add_view(GB), "a second GoPro camera")
pump(0.4)
t = notices(w)
check("stabilisation was ON" in t and "shutter speeds differ" in t and t.count("moved at frame 105") == 1,
      "the second camera's stabilisation and the rig's shutters are said, the first camera's move not again", t[:400])
w._apply_state()
check(w.act_gopro.isEnabled(), "3D -> GoPro Cameras… is enabled")

DRV = {}


def drive(steps):
    def tick():
        dlg = QApplication.activeModalWidget()
        if dlg is None or dlg.__class__.__name__ != steps.__name__.split("_")[0]:
            QTimer.singleShot(30, tick)
            return
        try:
            steps(dlg)
        except Exception as e:      # noqa: BLE001
            DRV.setdefault("err", []).append(repr(e))
            dlg.reject()
    QTimer.singleShot(30, tick)


def GoProDialog_steps(dlg):
    DRV["rows"] = dlg.table.rowCount()
    DRV["footage"] = [dlg.table.item(r, 1).text() for r in range(dlg.table.rowCount())]
    QTest.mouseClick(dlg.btn_lens, Qt.LeftButton)
    pump(0.1)
    DRV["lens_cells"] = [dlg.table.item(r, 7).text() for r in range(dlg.table.rowCount())]
    rect = dlg.table.visualItemRect(dlg.table.item(0, 6))
    QTest.mouseClick(dlg.table.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())


drive(GoProDialog_steps)
w.act_gopro.trigger()
pump(0.5)
check(DRV.get("rows") == 2 and all("HERO12 Black" in f for f in DRV.get("footage", [])), "the table: one row per camera",
      DRV)
p = w.project
check(all(p.lenses[v] is not None and p.lenses[v].report.get("gopro_nominal") for v in range(2))
      and all("GoPro's lens model" in c for c in DRV.get("lens_cells", [])),
      "the lens button gave both cameras GoPro's lens model", DRV.get("lens_cells"))
check(p.active == 0 and w.current == 105, "a click on camA's Moved cell went to camA, frame 105", (p.active, w.current))
check(not DRV.get("err"), "the dialog driver ran without errors", DRV.get("err"))

from kinetrace.syncdialog import SyncDialog  # noqa: E402
sd = SyncDialog(w, p, [rt.info.path for rt in w._views], 0, gopro=[rt.info.gopro for rt in w._views])
check(sd.prior == [0.0, -30.0] and "GoPro timecode" in sd.prior_note.text(),
      "Sync starts its search from the timecode (camB 1 s later)", (sd.prior, sd.prior_note.text()))
sd.deleteLater()

from kinetrace.lenswizard import LensWizard  # noqa: E402
wiz = LensWizard(w, p, GA, 0, OUT)
wiz.page_video.initializePage()
pv = wiz.page_video
check(not pv.r_gopro.isHidden() and pv.r_gopro.isChecked() and pv.model() == "gopro" and not pv.btn_gopro.isHidden(),
      "the lens wizard on GoPro footage: the GoPro choice shown and chosen")
QTest.mouseClick(pv.btn_gopro, Qt.LeftButton)
pump(0.1)
check(wiz.result_profile is not None and wiz.result_profile.report.get("gopro_nominal") and pv.isComplete(),
      "'Use GoPro's lens without boards' gives the nominal profile, ready to attach")
pv.path.setText(PLAIN)
pv._detect_gopro(PLAIN)
check(pv.r_gopro.isHidden() and pv.btn_gopro.isHidden() and pv.model() == "auto",
      "other footage: no GoPro choices in the lens wizard")
wiz.reject()
wiz.deleteLater()

w.project.dirty = False
w.close()
pump(0.3)
w._dev_probe.wait(10000)
print("\nverify_gopro: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
