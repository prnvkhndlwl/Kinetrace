"""The 3D picture of a wand calibration (G185, G186; offscreen, no GPU).

  A. field of view: every border ray of a camera with a fisheye lens model, projected back through
     the camera, lands on its border pixel; a camera without a lens correction is flagged "pinhole";
  B. coverage: the number of cameras that see a point equals an independent pinhole computation
     (K, R, t; in front and inside the picture) on 2000 random points; the covered-volume sentences
     carry a verdict;
  C. a wand calibration with 3 clicks moved 30 px: those wand ends are marked "rejected", the others not;
  D. the wizard: the user's click on "Show in 3D" opens the 3D window on this run; the Wand calibration
     tick changes the picture; hovering a wand end tells its frame;
  E. the app: 3D View -> Wand calibration (the user's click on the tick), a click on a wand end moves the
     timeline to that end's frame; "Covered by >= 3" changes the picture once the volume is computed.

Run: .venv\\Scripts\\python.exe tests\\verify_calib3d.py
"""
import os
import sys
import time
import traceback

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

import verify_wand_subset as ws          # the synthetic 5-camera wand rig, the wizard helpers
from kinetrace import calib
from kinetrace import wand as wand_mod
from kinetrace import wanddata as wd
from kinetrace.calib import Calibration, CameraCalibration, NoUndistort, OpenCVUndistort, dlt_from_camera

pump = ws.pump


def rig_cameras():
    """ws.CAMS as (K, R, t) too, for an independent count."""
    out = []
    for k in range(ws.N_CAM):
        a = 0.3 + k * 2 * np.pi / ws.N_CAM
        pos = np.array([2.8 * np.cos(a), 2.8 * np.sin(a), 1.2 + 0.35 * (k % 2)])
        R, t = ws.look_at(pos, np.array([0.0, 0.0, 1.0]))
        K = np.array([[ws.FOCAL[k], 0, (ws.W - 1) / 2], [0, ws.FOCAL[k], (ws.H - 1) / 2], [0, 0, 1]])
        out.append((K, R, t))
    return out


def sec_a():
    from kinetrace import calibview as cv3
    K, R, t = rig_cameras()[0]
    dist = np.array([0.05, -0.02, 0.01, -0.005])
    cam = CameraCalibration(dlt_from_camera(K, R, t), ws.W, ws.H, OpenCVUndistort(K, dist, True), pixel_origin=0.0)
    cal = Calibration([cam, ws.CAMS[1]], "m")
    probe = calib.working_probe(cal)
    f = cv3.fov(cam, 3.0, probe, n=24)
    assert not f.pinhole and len(f.border) >= 4 * 20, (f.pinhole, len(f.border))
    back = cam.project(f.border)
    err = np.linalg.norm(back - f.raw, axis=1)
    assert np.nanmax(err) < 0.05, np.nanmax(err)
    # every border point lies on the flat far plane 3 m in front of the camera (as Blender draws it)
    axis = R[2] / np.linalg.norm(R[2])
    d = (f.border - cam.center()) @ axis
    assert np.allclose(d, 3.0, atol=1e-6), (d.min(), d.max())
    assert len(f.rays) == 4
    f2 = cv3.fov(ws.CAMS[1], 3.0, probe, n=24)
    assert f2.pinhole
    print(f"A. field of view: {len(f.border)} border rays land on their pixels (max {np.nanmax(err):.4f} px), "
          "a camera without a lens correction is drawn as a pinhole OK")


def sec_b():
    from kinetrace import calibview as cv3
    cal = Calibration([CameraCalibration(c.coefs.copy(), ws.W, ws.H, NoUndistort(), pixel_origin=0.0)
                       for c in ws.CAMS], "m")
    rng = np.random.default_rng(1)
    X = rng.uniform([-2.5, -2.5, -0.5], [2.5, 2.5, 2.5], (2000, 3))
    got = cv3.seen_by(cal, X)
    want = np.zeros(len(X), int)
    for K, R, t in rig_cameras():
        xc = X @ R.T + t
        z = xc[:, 2]
        uv = (xc @ K.T)[:, :2] / z[:, None]
        inside = (z > 0) & (uv[:, 0] >= -0.5) & (uv[:, 0] <= ws.W - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= ws.H - 0.5)
        want += inside
    assert np.array_equal(got, want), (np.abs(got - want).sum(), np.bincount(want))
    cov = cv3.coverage(cal, np.array([-1.0, -1.0, 0.3]), np.array([1.0, 1.0, 1.7]), n_max=40)
    pts = rng.uniform([-0.6, -0.6, 0.6], [0.6, 0.6, 1.4], (300, 3))
    p2, p3 = cov.pct(pts, 2), cov.pct(pts, 3)
    assert 0 <= p3 <= p2 <= 100, (p2, p3)
    lines = cov.sentences(pts)
    assert any("2+" in ln and ("good" in ln or "ok" in ln or "poor" in ln) for ln in lines), lines
    assert 2 in cov.surface(2)[1].shape or len(cov.surface(2)[1]) > 0
    print(f"B. coverage: camera counts equal the pinhole truth on 2000 points (counts {np.bincount(want).tolist()}); "
          f"{lines[0]} OK")


def sec_c():
    from kinetrace import calibview as cv3
    P = ws.build(no_wand=(), tag="c3d")
    uv0, fr0 = wd.collect_wand(P, "wand A", "wand B", max_frames=400)
    picks = [int(fr0[k]) for k in (60, 180, 300)]                        # three sampled reference frames
    bad = [(1, int(P.local_index(1, picks[0])), 0), (3, int(P.local_index(3, picks[1])), 1),
           (4, int(P.local_index(4, picks[2])), 0)]                       # (camera, local frame, end) moved 30 px
    for c, f, e in bad:
        s = P.sessions[c]
        j = s.pid_by_name("wand A" if e == 0 else "wand B")
        s.tracks[f, j] += np.float32(30.0)
    uv, frames = wd.collect_wand(P, "wand A", "wand B", max_frames=400)      # as the wizard samples
    focal = [float(f) for f in ws.FOCAL]
    res = wand_mod.calibrate_wand(uv, ws.WAND_L, [(ws.W, ws.H)] * ws.N_CAM, focal=focal, frame_ids=frames,
                                  names=list(P.names))
    pic = cv3.from_wand_result(res, frames, list(P.names))
    bad_ref = {(int(P.reference_index(c, f)), e) for c, f, e in bad}
    bad_ref &= {(int(fr), e) for fr in frames for e in (0, 1)}            # the moved clicks this sample holds
    assert bad_ref, "the sample holds none of the moved clicks"
    rej = {(int(fr), int(e)) for fr, e, st in zip(pic.end_frame, pic.end_which, pic.end_state) if st == 3}
    assert bad_ref <= rej, (sorted(bad_ref), sorted(rej)[:10])
    assert len(rej) <= len(bad_ref) + 3, sorted(rej)
    assert len(pic.cams) == ws.N_CAM and all(c.name for c in pic.cams)
    print(f"C. the 3 moved clicks are marked rejected ({sorted(bad_ref)}), {len(rej) - len(bad_ref)} other(s) OK")


def sec_d():
    P = ws.build(tag="d3d")
    wz = ws.calibrate_through_wizard(P)
    try:
        pg = wz.page_run
        assert pg.btn_3d.isEnabled()
        QTest.mouseClick(pg.btn_3d, Qt.LeftButton)          # the user's click: Show in 3D
        pump(0.3)
        v = pg.view3d
        assert v is not None and v.isVisible() and v.chk_calib.isChecked()
        v._render()
        with_calib = v.last_image.copy()
        QTest.mouseClick(v.chk_calib, Qt.LeftButton, pos=QPoint(4, v.chk_calib.height() // 2))
        pump(0.1)
        v._render()
        assert not np.array_equal(with_calib, v.last_image), "the Wand calibration tick changes nothing"
        QTest.mouseClick(v.chk_calib, Qt.LeftButton, pos=QPoint(4, v.chk_calib.height() // 2))
        pump(0.1)
        v._render()
        pts = v.end_screen_points()
        k = int(np.nanargmin(np.abs(pts[:, 0] - v.canvas.width() / 2) + np.abs(pts[:, 1] - v.canvas.height() / 2)))
        # the mouse moving over that end (sent to the picture itself: offscreen, overlapping windows swallow
        # QTest's cursor moves - .claude/rules/tests.md)
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QMouseEvent
        hp = QPointF(float(round(pts[k, 0])), float(round(pts[k, 1])))
        ws.app.sendEvent(v.canvas, QMouseEvent(QEvent.MouseMove, hp, v.canvas.mapToGlobal(hp), Qt.NoButton,
                                               Qt.NoButton, Qt.NoModifier))
        pump(0.1)
        tip = v.canvas.toolTip()
        assert "frame" in tip and str(int(v.scene.calib.end_frame[k])) in tip, tip
        print(f"D. wizard -> Show in 3D: the window opens on this run, the tick redraws, the hover says '{tip}' OK")
    finally:
        ws.close_wizard(wz)


def sec_e():
    import cv2
    from PySide6.QtWidgets import QDialog, QMessageBox
    from _clean import forget_recovery
    from kinetrace.app import READY, MainWindow
    P = ws.build(tag="e3d")
    paths = [s.video_path for s in P.sessions]
    img = np.full((ws.H, ws.W, 3), 30, np.uint8)
    for pth, s in zip(paths, P.sessions):
        vw = cv2.VideoWriter(pth, cv2.VideoWriter_fourcc(*"mp4v"), ws.FPS, (ws.W, ws.H))
        for _ in range(s.n_frames):
            vw.write(img)
        vw.release()
        forget_recovery(pth)
    proj_path = os.path.join(ws.SCRATCH, "calib3d_gui.kinetrace")
    P.save(proj_path)
    real_exec = ws.cw.WandWizard.exec
    win = MainWindow()
    win.show()
    try:
        win._open_project_from_path(proj_path)
        for _ in range(300):
            pump(0.05)
            if win.state == READY and win.project is not None and win.project.n_views == ws.N_CAM:
                break

        def drive(self):
            self.show()
            self.restart()
            pump(0.1)
            ws.goto(self, self.page_wand)
            self.page_wand.length.setValue(ws.WAND_L)
            ws.goto(self, self.page_cams)
            ws.known_focal(self)
            ws.goto(self, self.page_frame)
            self.page_frame.r_none.setChecked(True)
            ws.goto(self, self.page_run)
            ws.run_and_wait(self)
            assert self.result_calibration is not None, self.page_run.report.toPlainText()[:300]
            self.accept()
            return QDialog.Accepted

        ws.cw.WandWizard.exec = drive
        win._wand_wizard()
        pump(0.2)
        win.act_view3d.trigger()                             # 3D -> Show 3D View
        pump(0.3)
        v = win.view3d
        assert v is not None and v.isVisible()
        QTest.mouseClick(v.chk_calib, Qt.LeftButton, pos=QPoint(4, v.chk_calib.height() // 2))
        for _ in range(100):
            pump(0.05)
            if v.scene.calib is not None:
                break
        assert v.scene.calib is not None, "the Wand calibration layer has no picture"
        v._render()
        pts = v.end_screen_points()
        p = win.project
        target = None
        for k in np.argsort(np.abs(pts[:, 0] - v.canvas.width() / 2)):
            if not np.isfinite(pts[k]).all():
                continue
            x, y = int(round(pts[k, 0])), int(round(pts[k, 1]))
            hit = v._pick(x, y)                       # the end a click there lands on (dots overlap)
            if hit is None:
                continue
            loc = int(p.local_index(p.active, int(v.scene.calib.end_frame[hit])))
            if loc != win.current and 0 <= loc < win.session.n_frames:
                target = (x, y, loc)
                break
        assert target is not None
        x, y, loc = target
        QTest.mouseClick(v.canvas, Qt.LeftButton, pos=QPoint(x, y))
        pump(0.3)
        assert win.current == loc, (win.current, loc)
        # Covered by >= 3: the picture changes once the volume is computed
        QTest.mouseClick(v.chk_cover, Qt.LeftButton, pos=QPoint(4, v.chk_cover.height() // 2))
        for _ in range(400):
            pump(0.05)
            if v.scene.calib.cover:
                break
        assert v.scene.calib.cover, "the covered volume was never computed"
        v._render()
        two = v.last_image.copy()
        v.cmb_cover.setCurrentIndex(v.cmb_cover.findData(3))
        for _ in range(200):
            pump(0.05)
            if 3 in v.scene.calib.cover:
                break
        v._render()
        assert not np.array_equal(two, v.last_image), "Covered by >= 3 draws the same as >= 2"
        assert "2+" in v.info.text() or "cameras see" in v.info.text(), v.info.text()
        # (owner 2026-10-12) the Cameras menu: two cameras unticked -> their fields of view go, the covered
        # space is counted from the three others
        v.chk_fov.setChecked(True)
        pump(0.1)
        v._render()
        before = v.last_image.copy()
        every = [int(a.data()) for a in v.cam_actions]     # the calibrated cameras (4: cam3 saw no wand)
        v.cam_actions[0].trigger()                        # Cameras ▾ -> the first camera (the menu entry's click)
        v.cam_actions[2].trigger()                        # Cameras ▾ -> the third
        left = set(every) - {every[0], every[2]}
        for _ in range(200):
            pump(0.05)
            if v.scene.calib.coverage is not None and v.scene.calib.coverage.n_cams == len(left):
                break
        assert v.scene.calib_views == left, (v.scene.calib_views, left)
        assert v.scene.calib.coverage.n_cams == len(left), v.scene.calib.coverage.n_cams
        assert [v.cmb_cover.itemData(i) for i in range(v.cmb_cover.count())] == list(range(2, len(left) + 1))
        v._render()
        assert not np.array_equal(before, v.last_image)
        # after saving and reopening, nothing of the wand run is kept: the picture is rebuilt from the two
        # wand landmarks (remembered as display state) with the project's calibration
        assert win._save_project(), "Save failed"
        pump(0.5)
        win._open_project_from_path(proj_path)
        for _ in range(300):
            pump(0.05)
            if win.state == READY and win.project is not None and win.project is not p:
                break
        p2 = win.project
        assert p2 is not p and p2.calibration is not None and p2.calibration.views() == [0, 1, 3, 4]
        assert p2.sessions[0].ui_state.get("wand_ends") == ["wand A", "wand B"], p2.sessions[0].ui_state.get("wand_ends")
        if not win.act_view3d.isChecked():
            win.act_view3d.trigger()
        pump(0.3)
        v = win.view3d
        if not v.chk_calib.isChecked():
            QTest.mouseClick(v.chk_calib, Qt.LeftButton, pos=QPoint(4, v.chk_calib.height() // 2))
        for _ in range(200):
            pump(0.05)
            if v.scene.calib is not None:
                break
        assert v.scene.calib is not None and v.scene.calib.source == "tracks", v.scene.calib
        assert v.scene.calib.end_names == ("wand A", "wand B") and len(v.scene.calib.finite_ends()) > 200
        print(f"E. app: 3D View -> Wand calibration, a click on a wand end moves to frame {loc}; covered by >= 3 "
              f"redraws; two cameras unticked: recounted from the others; reopened: rebuilt from the tracks; '{v.info.text()[:60]}' OK")
    finally:
        ws.cw.WandWizard.exec = real_exec
        if hasattr(win, "_dev_probe"):
            win._dev_probe.wait(60000)
        if win.project is not None:
            win.project.dirty = False
        win.close()
        pump(0.2)
        win.deleteLater()
        pump(0.1)
        for pth in paths:
            forget_recovery(pth)


SECTIONS = [sec_a, sec_b, sec_c, sec_d, sec_e]

if __name__ == "__main__":
    failed = []
    for fn in SECTIONS:
        try:
            fn()
        except Exception as e:      # noqa: BLE001
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {str(e)[:300]}".encode("ascii", "replace").decode())
            traceback.print_exc(limit=2)
    if failed:
        print("FAILED: " + ", ".join(failed))
        sys.exit(1)
    print("PASSED")
