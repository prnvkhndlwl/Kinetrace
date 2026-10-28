"""One pixel-centre rule (sweep PR 4; dev/work/2026-10-27-pixel-centre-rule, I357-I361).

Pixel i's CENTRE is at coordinate i (OpenCV), so the picture's outer edges are at -0.5 and W - 0.5 and a picture
shrunk by s maps p to (p + 0.5) * s - 0.5 (`pixels.centre_scale`), not p * s. Each check below FAILS on main
60a50ed for the stated reason:
  I357  a SAM click on a 4K camera reaches SAM's 1024 x 576 picture 1.4 native px down-right of where it was made
  I358  the board corner editor stores a corner dropped on a painted pixel centre half a pixel off (snap unable to
        help) and draws the handles half a pixel off; the board search seeds its full-size refinement with c / scale
  I359  body joints drawn at a reduced size sit ~2.2 native px down-right of the footage spot
  I360  a reduced-size overlay export draws markers / silhouettes / trails / region outlines 0.25-0.4 px off
  I361  three "inside the picture" rules half a pixel apart; a click or import in pixel 0's outer half was moved
        to 0.0; the tracker blanked x = -0.3 (on pixel 0) and kept x = W - 0.3 (past the last pixel)
  I362  (fails on PR 4 f1b0712) the lens board tiles draw corners / axes ~0.44 tile px down-right; the result
        page's map cuts positions off instead of rounding (dots up to 0.58 map px off)
Full-size drawing (scale 1) is pinned by md5s recorded on main (I362's on PR 4): it must not change.

Offscreen; no GPU. Run: .venv\\Scripts\\python.exe tests\\verify_pixel_centre.py
"""
import hashlib
import os
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
OUT = os.path.join(HERE, "out", "pixel_centre")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"
os.environ["KINETRACE_SETTINGS"] = os.path.join(OUT, "settings.ini")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
app = QApplication.instance() or QApplication([])

from kinetrace import body, bodyview, boardreview, lens, render, retrack, trackio  # noqa: E402
from kinetrace.pixels import centre_scale, centre_unscale  # noqa: E402
from kinetrace.segmenter import SegSession  # noqa: E402
from kinetrace.session import TrackingSession, in_frame  # noqa: E402

FAILS: list = []

# md5 of the full-size (scale 1) drawings, recorded on main 60a50ed: scale 1 must stay pixel-identical
MD5_POSE_NATIVE = "35b0b33e285b6199407a17a321878250"
MD5_OVERLAY_NATIVE = "3e90e1baa2763202e76b672fc12913bf"
# (I362) the full-size lens board drawing, recorded on PR 4 (pixel-centre-rule f1b0712)
MD5_BOARD_NATIVE = "00bf15487e9882f985ee3a9d76bccbdb"
MD5_SEEN_NATIVE = "11d4751944287a0ee2ccd2831f3034b6"


def check(label, cond, detail=None):
    if callable(cond):
        try:
            cond = cond()
        except Exception as e:  # noqa: BLE001 - the old code raises where the new one answers
            cond, detail = False, f"raised {e!r}"
    cond = bool(cond)
    line = ("  ok    " if cond else "  FAIL  ") + label + ("" if cond or detail is None else f"   [{detail}]")
    print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    if not cond:
        FAILS.append(label)
    return cond


def pump(sec=0.1):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.003)


def wait(cond, timeout):
    t0 = time.time()
    while True:
        try:
            if cond():
                return True
        except Exception:  # noqa: BLE001
            pass
        if time.time() - t0 > timeout:
            return False
        pump(0.02)


def md5(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def qimage_rgb(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_RGB888)
    a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())
    return a[:, :img.width() * 3].reshape(img.height(), img.width(), 3).astype(np.float64)


def centroid(weight: np.ndarray, cx: float, cy: float, half: int) -> np.ndarray:
    """Weighted centroid (array index units) of `weight` in a (2 half + 1) window around (cx, cy)."""
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    win = weight[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1]
    yy, xx = np.mgrid[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1]
    tot = win.sum()
    return np.array([(win * xx).sum() / tot, (win * yy).sum() / tot])


# ---------------------------------------------------------------- I357 SAM prompt
def section_sam():
    print("I357 SAM prompts (4K -> 1024 x 576)")
    ses = SegSession.__new__(SegSession)                    # the prompt mapping, no model needed
    ses.native_size, ses.work = (3840, 2160), (1024, 576)
    got = np.array(ses._to_work((1000.0, 500.0)))
    want = np.array([centre_unscale(1000.0, 3840 / 1024), centre_unscale(500.0, 2160 / 576)])
    check("a click at native (1000, 500) reaches SAM at (266.300, 132.967), the pixel-centre rule",
          np.allclose(got, want, atol=1e-9), f"got {got.round(3).tolist()} (main: 266.667, 133.333)")
    back = np.array([centre_scale(got[0], 3840 / 1024), centre_scale(got[1], 2160 / 576)])
    check("native -> SAM -> outline rule returns the click (1e-9)", np.allclose(back, (1000.0, 500.0), atol=1e-9),
          back.round(4).tolist())


# ---------------------------------------------------------------- I358 board corner editor + board search
def section_board():
    print("I358 board corner editor (zoom 4, the snap unable to help)")
    iw, ih, s = 64, 48, 4.0
    bgr = np.zeros((ih, iw, 3), np.uint8)
    bgr[20, 30] = 255                                        # OpenCV pixel (x = 30, y = 20)
    corners = np.array([[10.0, 10.0], [20.0, 10.0], [10.0, 30.0], [40.0, 30.0]])
    # pattern (1, 4): a one-corner "first row", so no first-row line is drawn over the handles
    dlg = boardreview.CornerEditor(None, bgr, corners, (1, 4), 1.0, None, 0)
    view = dlg.view
    view.setFixedSize(int(iw * s), int(ih * s))
    dlg.show()
    pump(0.2)
    check("(setup) the editor shows the picture at zoom 4 with no border", view._fit() == (s, 0.0, 0.0),
          view._fit())
    before = qimage_rgb(view.grab().toImage())
    # where the picture's pixel (30, 20) is painted: brightness centroid, widget coordinates (index + 0.5)
    c = centroid(before.sum(axis=2), (30 + 0.5) * s, (20 + 0.5) * s, 12) + 0.5
    check("(setup) the painted centre of pixel (30, 20) is at widget (122, 82)", np.allclose(c, (122, 82), atol=0.05),
          c.round(3).tolist())
    # corner 1 (picture (20, 10)) is drawn as a green ring: its centre must be pixel (20, 10)'s painted centre
    g = before
    green = ((g[..., 1] > 120) & (g[..., 0] < 0.6 * g[..., 1]) & (g[..., 2] < 0.8 * g[..., 1])).astype(np.float64)
    ring = centroid(green, (20 + 0.5) * s, (10 + 0.5) * s, 10) + 0.5
    check("an unmoved corner's handle is drawn on its own pixel's centre (82, 42)",
          np.allclose(ring, (82, 42), atol=0.5), f"ring centre {ring.round(2).tolist()} (main: 80, 40)")
    real_snap = cv2.cornerSubPix

    def no_snap(*a, **k):
        raise cv2.error("the snap finds no corner here")

    cv2.cornerSubPix = no_snap
    try:
        start = view._to_widget(dlg.corners[0])
        p0 = QPointF(round(float(start[0])), round(float(start[1])))
        drop = QPointF(float(round(c[0])), float(round(c[1])))
        QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, p0.toPoint())
        pump(0.02)
        mv = QMouseEvent(QEvent.MouseMove, drop, view.mapToGlobal(drop), Qt.NoButton, Qt.LeftButton,
                         Qt.NoModifier)
        QApplication.sendEvent(view, mv)                     # the drag, button held
        QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, drop.toPoint())
        pump(0.05)
    finally:
        cv2.cornerSubPix = real_snap
    check("(setup) the drag moved corner 0", dlg.moved())
    check("a corner dropped on the painted centre of pixel (30, 20) is stored as (30.0, 20.0)",
          np.allclose(dlg.corners[0], (30.0, 20.0), atol=0.02),
          f"stored {np.round(dlg.corners[0], 3).tolist()} (main: 30.5, 20.5)")
    dlg.close()
    dlg.deleteLater()
    pump(0.05)

    print("I358 board search seed (detection downscaled by 2)")
    W, H, sq = 960, 720, 60
    img = np.full((H, W), 200, np.uint8)
    for r in range(7):                                       # 10 x 7 squares = 9 x 6 inner corners
        for q in range(10):
            if (r + q) % 2 == 0:
                x0, y0 = 170 + q * sq, 140 + r * sq
                img[y0:y0 + sq, x0:x0 + sq] = 30
    # a little sub-pixel shift + blur, so the corners are not on whole pixels
    M = np.float32([[1, 0, 0.37], [0, 1, 0.21]])
    img = cv2.GaussianBlur(cv2.warpAffine(img, M, (W, H), borderValue=200), (0, 0), 1.0)
    vid = os.path.join(OUT, "board.mp4")
    vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for _ in range(3):
        vw.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    vw.release()
    found, seeds = [], []
    real_detect = lens.detect_board_info

    def detect(gray, pattern, *a, **k):
        info = real_detect(gray, pattern, *a, **k)
        if info is not None:
            found.append(np.asarray(info[0], np.float64).reshape(-1, 2).copy())
        return info

    def snap(image, pts, *a, **k):
        if image.shape[:2] == (H, W):
            seeds.append(np.asarray(pts, np.float64).reshape(-1, 2).copy())
        return real_snap(image, pts, *a, **k)

    lens.detect_board_info, cv2.cornerSubPix = detect, snap
    try:
        lens.scan_video(vid, (9, 6), max_candidates=3, downscale_max=480)
    finally:
        lens.detect_board_info, cv2.cornerSubPix = real_detect, real_snap
    check("(setup) the board is found and refined at full size", found and len(seeds) == len(found),
          (len(found), len(seeds)))
    if found and len(seeds) == len(found):
        d = max(float(np.abs(sd - centre_unscale(c_, 0.5)).max()) for c_, sd in zip(found, seeds))
        check("the full-size refinement starts from the pixel-centre rule (c + 0.5) / scale - 0.5", d < 1e-3,
              f"max seed offset {d:.3f} px (main: c / scale, 0.500)")


# ---------------------------------------------------------------- I359 body view
def section_body():
    print("I359 body joints at s = 0.2 on a 4K frame")
    S = 0.2
    rng = np.random.default_rng(1)
    bgr = np.zeros((2160, 3840, 3), np.uint8)
    errs = []
    for _ in range(40):
        x, y = rng.uniform(500, 3300), rng.uniform(500, 1600)
        bt = body.BodyTrack(1, "coco17", 1)
        j = np.full((17, 2), np.nan)
        j[0] = (x, y)
        bt.set_person(0, 0, joints2d=j, conf=np.ones(17), score=1.0)
        img = bodyview.draw_pose(bgr, bt, 0, bodyview.PoseDrawOptions(bones=False, angle_labels=False), scale=S)
        sx, sy = img.shape[1] / 3840, img.shape[0] / 2160
        want = np.array([centre_scale(x, sx), centre_scale(y, sy)])
        errs.append(centroid(img.astype(np.float64).sum(axis=2), want[0], want[1], 8) - want)
    m = np.mean(errs, axis=0)
    check("the drawn joints sit on the resized footage spot (mean within 0.1 px)", np.all(np.abs(m) < 0.1),
          f"mean offset ({m[0]:+.3f}, {m[1]:+.3f}) output px (main: +0.447, +0.407)")
    # full size: unchanged, pixel for pixel
    rng = np.random.default_rng(5)
    frame = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    bt = body.BodyTrack(1, "coco17", 1)
    bt.set_person(0, 0, joints2d=np.column_stack([rng.uniform(40, 600, 17), rng.uniform(40, 440, 17)]),
                  conf=np.linspace(0.3, 1.0, 17), score=1.0, bbox=np.array([30.3, 25.7, 610.2, 455.9]))
    native = bodyview.draw_pose(frame, bt, 0, bodyview.PoseDrawOptions(box=True), scale=1.0)
    h = md5(native)
    print(f"  (scale 1 md5 {h})")
    check("a full-size body drawing is pixel-identical to main's", h == MD5_POSE_NATIVE, h)


# ---------------------------------------------------------------- I360 overlay export
def _overlay_session(W, H, pts):
    n = len(pts)
    s = TrackingSession("synthetic.mp4", 2, 30.0, W, H)
    for i, (x, y) in enumerate(pts):
        s.add_point(0, float(x), float(y), name=f"p{i}")
    tw = np.zeros((2, n, 2), np.float32)
    tw[:] = pts.astype(np.float32)
    s.write_segment(0, tw, np.ones((2, n), bool), list(range(n)), np.full((2, n), 0.9, np.float32))
    return render.freeze_session(s)


def _marker_offset(sc, pts, W, H):
    fz = _overlay_session(W, H, pts)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dark = np.zeros((H, W), np.float32)
    for x, y in pts:
        x0, x1, y0, y1 = int(x) - 25, int(x) + 26, int(y) - 25, int(y) + 26
        dark[y0:y1, x0:x1] += np.exp(-((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2) / (2 * 6.0 ** 2))
    base = (255 - np.clip(dark, 0, 1) * 255).astype(np.uint8)
    shrunk = cv2.resize(cv2.cvtColor(base, cv2.COLOR_GRAY2BGR), None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    opts = render.OverlayOptions(0, 1, sc, markers=True, names=False, bones=False, mask=False, trails=0,
                                 frame_number=False, events=False, notes=False, marker_px=2, speed_label=False)
    drawn = render.draw_overlay(np.full((H, W, 3), 255, np.uint8), 0, fz, opts, None)
    blobs = 255 - cv2.cvtColor(shrunk, cv2.COLOR_BGR2GRAY).astype(np.float64)
    marks = 255 - cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY).astype(np.float64)
    half = max(4, int(round(20 * sc)))
    d = []
    for x, y in pts:
        ex, ey = centre_scale(x, sc), centre_scale(y, sc)
        d.append(centroid(marks, ex, ey, half) - centroid(blobs, ex, ey, half))
    return np.mean(d, axis=0)


class _Recorder:
    """render's cv2 with polylines recorded (the positions every outline is drawn at)."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return getattr(cv2, name)

    def polylines(self, img, pts, closed, color, *a, **k):
        self.calls.append((bool(closed), tuple(int(v) for v in color),
                           [np.asarray(p).reshape(-1, 2).copy() for p in pts]))
        return cv2.polylines(img, pts, closed, color, *a, **k)


def _scene(W=400, H=300):
    """Frame 2 of a small session: a moving point (trail), a rectangle region, a silhouette, a bone."""
    s = TrackingSession("synthetic.mp4", 3, 30.0, W, H)
    s.add_point(0, 100.3, 120.6, name="a")
    out = np.array([[-20.0, -10.0], [20.0, -10.0], [20.0, 10.0], [-20.0, 10.0]]) + (250.4, 150.7)
    s.add_point(0, 250.4, 150.7, kind="group", radius=15.0, name="r", shape="rectangle", outline=out)
    tw = np.array([[[100.3, 120.6], [250.4, 150.7]], [[110.7, 125.2], [252.1, 151.3]],
                   [[121.2, 130.9], [253.6, 152.2]]], np.float32)
    s.write_segment(0, tw, np.ones((3, 2), bool), [0, 1], np.full((3, 2), 0.9, np.float32))
    s.points[0].color, s.points[1].color = (0, 255, 0), (0, 0, 255)
    fz = render.freeze_session(s)
    sil = np.array([[60.4, 200.2], [180.7, 205.9], [170.1, 270.3], [70.8, 262.6]])
    fz.segs = [(SimpleNamespace(color=(255, 0, 0)), SimpleNamespace(contours={2: [sil]}))]
    return fz, sil, tw


def section_render():
    print("I360 overlay export at a reduced size")
    rng = np.random.RandomState(7)
    g = np.arange(100, 900, 100)
    pts = np.array([(x + rng.uniform(10, 90), y + rng.uniform(10, 90)) for x in g for y in g])
    for sc, main in ((0.5, "+0.29, +0.26"), (0.25, "+0.40, +0.42")):
        m = _marker_offset(sc, pts, 1000, 1000)
        check(f"scale {sc}: the markers sit on the resized feature (mean within 0.1 px)", np.all(np.abs(m) < 0.1),
              f"mean marker - feature ({m[0]:+.3f}, {m[1]:+.3f}) px (main: {main})")
    sc = 0.25
    fz, sil, tw = _scene()
    rec = _Recorder()
    real = render.cv2
    render.cv2 = rec
    try:
        opts = render.OverlayOptions(2, 2, sc, markers=True, names=False, bones=False, mask=True, trails=5,
                                     frame_number=False, events=False, notes=False, marker_px=3,
                                     speed_label=False)
        render.draw_overlay(np.full((300, 400, 3), 128, np.uint8), 2, fz, opts, None, mask_color=(0, 0, 255))
    finally:
        render.cv2 = real

    def drawn(closed, colour):
        return [c[2] for c in rec.calls if c[0] == closed and c[1] == colour]

    def want(p):
        return np.round(centre_scale(np.asarray(p, np.float64), sc)).astype(np.int32)

    silh = drawn(True, (0, 0, 255))
    check("the silhouette outline is drawn at the pixel-centre positions",
          len(silh) == 1 and np.array_equal(silh[0][0], want(sil)), silh[:1])
    trail = drawn(False, (0, 255, 0))
    check("the trail is drawn at the pixel-centre positions",
          len(trail) == 1 and np.array_equal(trail[0][0], want(tw[:, 0])), trail[:1])
    region = drawn(True, (255, 0, 0))
    meta = fz.points[1]
    check("the region outline is drawn at the pixel-centre positions",
          len(region) == 1 and np.array_equal(region[0][0], want(meta.outline_at(tw[2, 1]))), region[:1])
    # full size: unchanged, pixel for pixel
    fz, _, _ = _scene()
    rng = np.random.RandomState(11)
    frame = (rng.rand(300, 400, 3) * 255).astype(np.uint8)
    opts = render.OverlayOptions(2, 2, 1.0, markers=True, names=True, bones=True, mask=True, trails=5,
                                 frame_number=True, events=True, notes=True, marker_px=4, speed_label=False)
    native = render.draw_overlay(frame, 2, fz, opts, [(0, 1)])
    h = md5(native)
    print(f"  (scale 1 md5 {h})")
    check("a full-size overlay export frame is pixel-identical to main's", h == MD5_OVERLAY_NATIVE, h)


# ---------------------------------------------------------------- I362 lens board tiles + result map
class _ArrowRecorder:
    """lens's cv2 with arrowedLine recorded (where the board's X / Y / Z arrows start and end)."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return getattr(cv2, name)

    def arrowedLine(self, img, o, q, *a, **k):
        self.calls.append((tuple(o), tuple(q)))
        return cv2.arrowedLine(img, o, q, *a, **k)


def _dot_offsets(draw, s, n, seed):
    """`n` random 4K corners, each drawn alone by `draw(p)`: (mean, worst) offset of the drawn dot's centroid
    from `centre_scale(p, s)`. The dot is weighted by green minus red, which leaves out corner 0's ring
    (red-blue) and the dark map background."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        p = rng.uniform([200.0, 200.0], [3640.0, 1960.0])
        img = draw(p).astype(np.float64)
        want = centre_scale(p, s)
        d.append(centroid(np.clip(img[:, :, 1] - img[:, :, 2], 0, None), want[0], want[1], 4) - want)
    d = np.asarray(d)
    return d.mean(axis=0), np.abs(d).max()


def section_lens_thumbs():
    print("I362 lens board tiles (480 px wide) and the result page's map (640 px wide) of a 4K video")
    s = 480 / 3840
    tile = np.zeros((270, 480, 3), np.uint8)
    m, _ = _dot_offsets(lambda p: lens.draw_board_review(tile, p[None], (1, 1), prof=None, scale=s, axes=False),
                        s, 200, 3)
    check("a checkerboard tile's corner dots sit on the shrunk corner (mean within 0.1 tile px)",
          np.all(np.abs(m) < 0.1), f"mean dot - corner ({m[0]:+.3f}, {m[1]:+.3f}) tile px (PR 4: +0.462, +0.424)")
    # a ChArUco view that missed corner 0: the NaN row sends it through _draw_seen_board
    m, _ = _dot_offsets(lambda p: lens.draw_board_review(tile, np.array([[np.nan, np.nan], p]), (2, 1), prof=None,
                                                         scale=s, axes=False), s, 200, 4)
    check("a ChArUco tile's corner dots sit on the shrunk corner (mean within 0.1 tile px)",
          np.all(np.abs(m) < 0.1), f"mean dot - corner ({m[0]:+.3f}, {m[1]:+.3f}) tile px (PR 4: +0.449, +0.490)")
    # the board's axes: video points whose p * s and centre_scale round to different tile pixels
    a, b = np.array([1004.6, 508.6]), np.array([1100.6, 604.6])
    want = (tuple(int(v) for v in np.round(centre_scale(a, s))), tuple(int(v) for v in np.round(centre_scale(b, s))))
    real_axes, real_cv2 = lens.board_axes, lens.cv2
    lens.board_axes = lambda *a_, **k_: [("X", a, b)]
    for label, corners, pattern in (("checkerboard", np.array([[2000.3, 1000.7]]), (1, 1)),
                                    ("ChArUco", np.array([[np.nan, np.nan], [2000.3, 1000.7]]), (2, 1))):
        rec = _ArrowRecorder()
        lens.cv2 = rec
        try:
            lens.draw_board_review(tile, corners, pattern, prof=object(), scale=s, axes=True)
        finally:
            lens.cv2 = real_cv2
        check(f"a {label} tile's axis arrow is drawn at the pixel-centre positions", rec.calls == [want],
              f"drawn {rec.calls}, wanted {want} (PR 4: (126, 64) -> (138, 76))")
    lens.board_axes = real_axes
    # the result page's "where the board went" map
    sc = 640 / 3840
    m, worst = _dot_offsets(lambda p: lens.coverage_image(SimpleNamespace(size=(3840, 2160), corners=[p[None]]),
                                                          [0]), sc, 200, 5)
    check("the result map's dots sit on the shrunk corner (mean within 0.1 map px)", np.all(np.abs(m) < 0.1),
          f"mean dot - corner ({m[0]:+.3f}, {m[1]:+.3f}) map px")
    check("every result-map dot is on the map pixel nearest the corner (within 0.5 map px)", worst <= 0.5 + 1e-9,
          f"worst {worst:.3f} map px (PR 4: 0.582, a cut-off instead of rounding)")
    # full size: unchanged, pixel for pixel
    rng = np.random.default_rng(9)
    frame = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    grid = np.array([(80.0 + 90 * i, 90.0 + 90 * j) for j in range(4) for i in range(6)])
    corners = grid + rng.uniform(-3, 3, grid.shape)
    seen = corners.copy()
    seen[[3, 10]] = np.nan
    for label, c, md5_main in (("checkerboard", corners, MD5_BOARD_NATIVE), ("ChArUco", seen, MD5_SEEN_NATIVE)):
        h = md5(lens.draw_board_review(frame, c, (6, 4), prof=None, scale=1.0, axes=False))
        print(f"  ({label} scale 1 md5 {h})")
        check(f"a full-size {label} board drawing is pixel-identical to PR 4's", h == md5_main, h)


# ---------------------------------------------------------------- I361 one in-picture rule
def section_rule():
    print("I361 one in-picture rule: the pixel edges [-0.5, W - 0.5) (W = 1920, H = 1080)")
    W, H = 1920, 1080
    cases = {(-0.3, 500.0): True, (-0.5, 500.0): True, (1919.4, 500.0): True, (1919.5, 500.0): False,
             (1919.7, 500.0): False, (-0.6, 500.0): False, (500.0, -0.3): True, (500.0, 1079.7): False}
    got = {k: bool(in_frame(np.array(k), W, H)) for k in cases}
    check("session.in_frame: x = -0.3 / -0.5 inside (pixel 0), x = 1919.5 / 1919.7 outside (past the last pixel)",
          got == cases, {k: v for k, v in got.items() if v != cases[k]})
    check("retrack.outside_by is 0 on the last pixel (x = 1919.25)",
          retrack.outside_by((1919.25, 500.0), W, H) == 0.0, retrack.outside_by((1919.25, 500.0), W, H))
    check("retrack.outside_by measures from the pixel edge (x = -0.6 -> 0.1 px)",
          abs(retrack.outside_by((-0.6, 500.0), W, H) - 0.1) < 1e-9, retrack.outside_by((-0.6, 500.0), W, H))

    # the tracker's write path keeps a point on pixel 0 and blanks one past the last pixel
    s = TrackingSession("x.mp4", 3, 30.0, W, H)
    s.add_point(0, 100.0, 100.0)
    s.add_point(0, 1800.0, 100.0)
    tw = np.array([[[100.0, 100.0], [1800.0, 100.0]], [[-0.3, 100.0], [1919.7, 100.0]],
                   [[-0.6, 100.0], [1919.4, 100.0]]], np.float32)
    s.write_segment(0, tw, np.ones((3, 2), bool), [0, 1], np.full((3, 2), 0.9, np.float32))
    check("a tracked x = -0.3 (on pixel 0) is kept", bool(s.tracked[1, 0]) and float(s.tracks[1, 0, 0]) == np.float32(-0.3),
          (bool(s.tracked[1, 0]), float(s.tracks[1, 0, 0])))
    check("a tracked x = 1919.7 (past the last pixel) is blank", not s.tracked[1, 1] and np.isnan(s.tracks[1, 1]).all(),
          (bool(s.tracked[1, 1]), s.tracks[1, 1].tolist()))
    check("(edges) x = -0.6 blank, x = 1919.4 kept", not s.tracked[2, 0] and bool(s.tracked[2, 1]),
          (bool(s.tracked[2, 0]), bool(s.tracked[2, 1])))

    # an import keeps the file's position on pixel 0 and drops one past the last pixel
    s = TrackingSession("x.mp4", 4, 30.0, W, H)
    xy = np.array([[[-0.3, 100.0]], [[1919.7, 100.0]], [[100.0, -0.4]], [[1919.4, 1079.4]]], np.float32)[None]
    res = trackio.apply(s, trackio.Imported("dlc", ["p"], xy, np.ones(xy.shape[:3], np.float32)))
    check("an imported x = -0.3 is stored at -0.3 (not moved to 0.0)",
          bool(s.tracked[0, 0]) and float(s.tracks[0, 0, 0]) == np.float32(-0.3), s.tracks[0, 0].tolist())
    check("an imported y = -0.4 is stored at -0.4", bool(s.tracked[2, 0]) and float(s.tracks[2, 0, 1]) == np.float32(-0.4),
          s.tracks[2, 0].tolist())
    check("an imported x = 1919.7 is outside (dropped), 1919.4 / 1079.4 kept",
          not s.tracked[1, 0] and bool(s.tracked[3, 0]) and res["outside"] == 1, (s.tracked[:4, 0].tolist(), res))

    # Snap / re-track: a crossing on the last pixel is kept where it is; one just outside goes to the border pixel
    sess = [TrackingSession("a.mp4", 3, 30.0, W, H), TrackingSession("b.mp4", 3, 30.0, W, H)]
    for x in sess:
        x.add_point(0, 500.0, 500.0, name="p")
    proj = SimpleNamespace(calibration=SimpleNamespace(camera=lambda v: object()), n_views=2, sessions=sess,
                           probe_at=lambda v, f: np.zeros(3))
    real = retrack.epipolar_polyline, retrack.intersect_polylines
    out = {}
    try:
        retrack.epipolar_polyline = lambda *a, **k: np.array([[0.0, 0.0], [10.0, 10.0]])
        for tx in (1919.25, -0.3, 1921.0, -1.5):
            retrack.intersect_polylines = lambda lines, near, info=None, tx=tx: np.array([tx, 500.0])
            r = retrack.ray_target(proj, 1, "p", 1, observations=[(0, object(), (1.0, 1.0))])
            out[tx] = None if r is None else float(r[0][0])
    finally:
        retrack.epipolar_polyline, retrack.intersect_polylines = real
    check("Snap keeps a crossing on the last / first pixel (x = 1919.25 / -0.3) where it is",
          out.get(1919.25) == 1919.25 and out.get(-0.3) == -0.3, out)
    check("Snap puts a crossing just outside on the border pixel's centre (1921 -> 1919, -1.5 -> 0)",
          out.get(1921.0) == 1919.0 and out.get(-1.5) == 0.0, out)


def section_point_models():
    print("I361 the Moving spot and Ball marker take the same edges (a click on pixel 0's outer half)")
    from kinetrace import spots
    from kinetrace.balls import BallTracker
    W, H = 100, 100
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = 40 + 200 * np.exp(-((xx - 0.0) ** 2 + (yy - 50.0) ** 2) / (2 * 1.6 ** 2))      # a spot on pixel 0
    rgb = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    check("a spot clicked at x = -0.3 is measured (the small-spot hint, the point test)",
          spots.measure_spot(rgb, (-0.3, 50.0)) is not None)
    check("(edges) x = -0.6 and x = 99.5 are outside the picture",
          spots.measure_spot(rgb, (-0.6, 50.0)) is None and spots.measure_spot(rgb, (99.5, 50.0)) is None)
    tr = spots.SpotTracker(spots.SpotSettings("bright", 6.0, 0.5, 1.6), (-0.3, 50.0), size=(W, H))
    tr.start(0, rgb)
    fix = tr.step(1, rgb)
    check("a Moving spot started at x = -0.3 keeps tracking (not 'left the picture' at once)",
          fix is not None and tr.stopped is None, tr.stopped)
    check("a ball clicked at x = -0.3 is inside a crop at the picture's edge (it gets its SAM prompt)",
          BallTracker._inside((0, 0, 200, 200), (-0.3, 50.0)) and BallTracker._inside((0, 0, 200, 200), (50.0, -0.4)))
    check("(unchanged) away from the picture's edge a crop is [x0, x1)",
          not BallTracker._inside((10, 10, 210, 210), (9.7, 50.0)) and BallTracker._inside((10, 10, 210, 210), (10.0, 50.0)))


def section_click():
    print("I361 a click on the outer half of pixel 0 stays where it was made")
    from _clean import forget_recovery
    from kinetrace.app import READY, MainWindow
    from kinetrace.bodyview import BodyRunDialog
    from kinetrace.pointtest import PointModelTest
    BodyRunDialog.exec = lambda self: QDialog.Rejected
    PointModelTest.exec = lambda self: QDialog.Rejected
    from _stubs import stub_track_choice
    stub_track_choice()
    W, H, N = 320, 240, 10
    rng = np.random.RandomState(3)
    bg = cv2.GaussianBlur((rng.rand(H, W) * 90 + 40).astype(np.float32), (0, 0), 3).astype(np.uint8)
    vid = os.path.join(OUT, "edge.mp4")
    vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for _ in range(N):
        vw.write(cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR))
    vw.release()
    forget_recovery(vid)
    win = MainWindow()
    win.resize(1300, 850)
    win.show()
    win._open_video(vid)
    wait(lambda: win.state == READY and win.project is not None and not win._loading, 60)
    pump(0.3)
    win._goto(0, force=True)
    wait(lambda: win.cache.get(0) is not None, 10)
    win.act_mode_manual.trigger()                # Track > Manual: no model runs on a click
    pump(0.05)
    cv = win.canvas
    cv.fit()
    cv.scale(8.0, 8.0)                           # zoomed far in on the left edge
    cv.centerOn(QPointF(0.0, 120.0))
    pump(0.1)
    vp = cv.mapFromScene(QPointF(-0.3, 120.0))
    sp = cv.mapToScene(vp)
    check("(setup) a viewport pixel lands on pixel 0's outer half", -0.5 <= sp.x() < 0.0, sp.x())
    s = win.session
    QTest.keyClick(win, Qt.Key_N)                # arm: the next click adds a point
    pump(0.05)
    QTest.mouseClick(cv.viewport(), Qt.LeftButton, Qt.NoModifier, vp)
    pump(0.1)
    pid = s.n_points - 1
    check("(setup) N + the click added a point on frame 0", s.n_points == 1 and bool(s.tracked[0, pid]), s.n_points)
    if s.n_points == 1:
        x = float(s.tracks[0, pid, 0])
        check("the click is stored where it was made (x < 0, on pixel 0)", x == np.float32(sp.x()) and x < 0,
              f"stored {x:.3f}, clicked {sp.x():.3f} (main: moved to 0.0)")
        check("a Track can start from it (session.hand_clicks)", 0 in s.hand_clicks(pid, 0, N), s.hand_clicks(pid, 0, N))
    if win.project is not None:
        win.project.dirty = False
    win.close()
    pump(0.3)


for section in (section_sam, section_board, section_body, section_render, section_lens_thumbs, section_rule,
                section_point_models, section_click):
    try:
        section()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        check(f"({section.__name__} ran to its end)", False, repr(e))
for w_ in QApplication.topLevelWidgets():
    probe = getattr(w_, "_dev_probe", None)
    if probe is not None:
        probe.wait(10000)
print("\nverify_pixel_centre: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
