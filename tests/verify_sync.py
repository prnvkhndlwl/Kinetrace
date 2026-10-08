"""Whole-frame camera sync from motion (`sync.py`) and from sound
(`audiosync.py`), on synthetic videos with a known offset: two cuts of the same
moving-dot scene starting at different frames recover the shift exactly; a
camera at twice the frame rate recovers it to a frame; two clips that never
see the same motion get verdict "none"; claps in noise muxed onto the clips
with ffmpeg (one microphone much noisier, with a mains hum) recover the shift
to a frame with the filter, a silent clip says so; then the app's dialog with
both methods on a two-camera project (offscreen, no GPU)."""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(errors="replace")

import cv2
import numpy as np

from _clean import forget_recovery  # noqa: E402
from kinetrace import sync

OUT = os.path.join(ROOT, "tests", "out")
os.makedirs(OUT, exist_ok=True)
W, H = 320, 240
rng = np.random.RandomState(11)
# the "world": a dot that pauses and darts (bursts of motion every camera sees)
N_WORLD = 700
pos = np.zeros((N_WORLD, 2))
p = np.array([160.0, 120.0])
# still / moving in blocks of RANDOM length (a periodic schedule would let a wrong
# offset alias onto the period and look plausible, which real footage does not do)
moving, t = False, 0
while t < N_WORLD:
    n = int(rng.randint(15, 70))
    for k in range(t, min(N_WORLD, t + n)):
        if moving:
            p = p + rng.normal(0, 6.0, 2)
            p = np.clip(p, 20, [W - 20, H - 20])
        pos[k] = p
    moving = not moving
    t += n


def write_clip(path, start, n, rate=1, view_shift=(0, 0), tint=0):
    """Frames world[start + k / rate]: a camera that started `start` world frames
    in; rate 2 = two frames per world frame (duplicated), a different viewpoint
    shift and tint so the pictures are not identical."""
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0 * rate, (W, H))
    for k in range(n):
        tw = start + k / rate
        i = int(np.floor(tw))
        if i >= N_WORLD:
            break
        img = np.full((H, W, 3), 40 + tint, np.uint8)
        x, y = pos[i] + np.array(view_shift)
        cv2.circle(img, (int(x), int(y)), 9, (230, 230, 230), -1)
        cv2.rectangle(img, (5, 5), (60, 30), (90, 90, 90), -1)
        vw.write(img)
    vw.release()
    return path


A = write_clip(os.path.join(OUT, "sync_a.mp4"), 0, 600)
B = write_clip(os.path.join(OUT, "sync_b.mp4"), 37, 560, view_shift=(15, -10), tint=20)     # starts 37 frames later
C = write_clip(os.path.join(OUT, "sync_c.mp4"), 20, 900, rate=2, view_shift=(-20, 5))       # 60 fps camera
D = write_clip(os.path.join(OUT, "sync_d.mp4"), 640, 50)                                    # never overlaps

# ---- signals ----------------------------------------------------------------------
t0 = time.time()
sa = sync.motion_signal(A, 0, 600)
print(f"motion signal: 600 frames in {time.time() - t0:.2f}s, "
      f"{int((sa > np.nanmedian(sa) + 1e-6).sum())} moving frames")
assert len(sa) == 600 and np.isfinite(sa).all()

# ---- B: same rate, 37 frames later --------------------------------------------------
# B's frame k shows world frame 37 + k; A's frame t shows world t. local_B = t + offset -> offset = -37
res = sync.estimate_offsets_from_motion([A, B], [1.0, 1.0], (100, 400), search=100)
r = res[0]
print(f"B: offset {r.offset:+.1f} (truth -37), lag {r.lag_ref}, verdict {r.result.verdict}, "
      f"score {r.result.score:.2f}, margin {r.result.margin:.2f}")
assert r.result.verdict == "clear", r.result.why
assert abs(r.offset + 37) <= 0.5, r.offset

# ---- C: twice the frame rate, 20 world frames later -----------------------------------
# C's frame k shows world 20 + k/2 -> local_C = 2 t - 40 -> offset -40
res = sync.estimate_offsets_from_motion([A, C], [1.0, 2.0], (100, 400), search=100)
r = res[0]
print(f"C (rate 2): offset {r.offset:+.1f} (truth -40), verdict {r.result.verdict}, score {r.result.score:.2f}")
assert r.result.verdict in ("clear", "weak"), r.result.why
assert abs(r.offset + 40) <= 2.0, r.offset

# ---- D: no shared motion -> verdict none, offset not to be trusted -----------------------
res = sync.estimate_offsets_from_motion([A, D], [1.0, 1.0], (100, 400), search=60)
print(f"D (no overlap): verdict {res[0].result.verdict}: {res[0].result.why[:90]}")
assert res[0].result.verdict == "none"

# ---- a prior from the recording clocks in the file names (GoPro style) ---------------------
pri = sync.offsets_from_filenames([r"E:\x\CAM1_20250101_120004_GX010001.MP4",
                                   r"E:\x\CAM7_20250101_120006_GX010002.MP4",
                                   r"E:\x\CAM2_20250101_120000_GX010003.MP4",
                                   r"D:\no_stamp_here.mp4"], 239.76)
assert pri[0] == 0.0 and pri[3] is None
assert abs(pri[1] - (-2 * 239.76)) < 1e-6, pri      # started 2 s later -> its frame is 480 lower at t=0
assert abs(pri[2] - (+4 * 239.76)) < 1e-6, pri      # started 4 s earlier
# a prior far from zero with a narrow search still finds the offset (B, prior -37 +- 10)
res = sync.estimate_offsets_from_motion([A, B], [1.0, 1.0], (100, 400), search=10, prior=[0.0, -37.0])
assert res[0].result.verdict == "clear" and abs(res[0].offset + 37) <= 0.5, (res[0].offset, res[0].result.verdict)
# ... and a wrong prior with a search that does not reach the truth says so instead of guessing
res = sync.estimate_offsets_from_motion([A, B], [1.0, 1.0], (100, 400), search=10, prior=[0.0, +200.0])
assert res[0].result.verdict == "none", res[0].result.verdict
print("file-name clock prior OK (and a search that misses the truth says none)")

# ---- audio: claps in noise, muxed onto the clips with ffmpeg, a known shift ------------
from kinetrace import audiosync   # noqa: E402

have_ffmpeg, note = audiosync.ffmpeg_status()
print("ffmpeg:", note[:70])
assert have_ffmpeg, "imageio-ffmpeg is part of the install: " + note
import subprocess, wave, struct   # noqa: E402

SR = 8000
dur_a, dur_b = 600 / 30.0, 560 / 30.0            # the clips' lengths at 30 fps
world_len = 700 / 30.0
arng = np.random.RandomState(5)
# the world's sound: a few claps (sharp bursts) + a voice-like low hum + white noise
world = 0.02 * arng.normal(size=int(world_len * SR))
tt = np.arange(len(world)) / SR
world += 0.05 * np.sin(2 * np.pi * 140 * tt) * (1 + 0.3 * np.sin(2 * np.pi * 0.7 * tt))
for tc in (2.3, 5.1, 9.7, 12.4, 15.8, 19.2):
    k = int(tc * SR)
    burst = arng.normal(size=400) * np.exp(-np.arange(400) / 60.0)
    world[k:k + 400] += 1.5 * burst


def wav_of(start_s, dur_s, path, snr_noise=0.08, hum_hz=None):
    """Camera microphone: the world from start_s, plus its OWN noise (fan / wind) - the
    filter and the whitening have to cope with a different noise floor per camera."""
    k0 = int(start_s * SR)
    n = int(dur_s * SR)
    x = world[k0:k0 + n].copy()
    x += snr_noise * arng.normal(size=n)
    if hum_hz:
        x += 0.15 * np.sin(2 * np.pi * hum_hz * np.arange(n) / SR)      # a loud mains-like hum
    x = np.clip(x / (np.abs(x).max() + 1e-9) * 0.8, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(struct.pack("<%dh" % n, *(x * 32767).astype(np.int16)))
    return path


def mux(video, wav_path, out):
    exe = audiosync.find_ffmpeg()
    r = subprocess.run([exe, "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-i", wav_path,
                        "-c:v", "copy", "-c:a", "aac", "-shortest", out], capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    return out


A2 = mux(A, wav_of(0.0, dur_a, os.path.join(OUT, "sync_a.wav"), 0.05),
         os.path.join(OUT, "sync_a_audio.mp4"))
# B starts 37 frames = 1.2333 s later, with a much noisier microphone and a 60 Hz hum
B2 = mux(B, wav_of(37 / 30.0, dur_b, os.path.join(OUT, "sync_b.wav"), 0.25, hum_hz=60.0),
         os.path.join(OUT, "sync_b_audio.mp4"))
assert audiosync.has_audio(A2) and audiosync.has_audio(B2) and not audiosync.has_audio(A)
sig = audiosync.audio_signal(A2, 0.0, 5.0)
assert abs(len(sig) - 5 * SR) < SR // 10, len(sig)
res = audiosync.estimate_offsets_from_audio([A2, B2], [30.0, 30.0], (3.0, 18.0), 2.5)
r = res[0]
print(f"audio B: offset {r.offset:+.2f} frames (truth -37), started {r.lag_s:+.4f} s later (truth 1.2333), "
      f"verdict {r.result.verdict}, score {r.result.score:.2f}, blocks {r.result.n_blocks}")
assert r.result.verdict == "clear", r.result.why
assert abs(r.offset + 37) <= 0.5, r.offset
# the filter option matters on the humming microphone: without the band-pass and without
# whitening, plain correlation of the raw tracks is worse or fails
res_raw = audiosync.estimate_offsets_from_audio([A2, B2], [30.0, 30.0], (3.0, 18.0), 2.5, band=None, whiten=False)
print(f"audio B raw (no filter, no whitening): offset {res_raw[0].offset:+.2f}, verdict {res_raw[0].result.verdict}, "
      f"score {res_raw[0].result.score:.2f}")
assert res_raw[0].result.score <= r.result.score + 1e-9 or abs(res_raw[0].offset + 37) > 0.5
# a clip without a sound track says so instead of guessing
res_silent = audiosync.estimate_offsets_from_audio([A2, D], [30.0, 30.0], (3.0, 18.0), 2.5)
assert res_silent[0].has_audio is False and res_silent[0].result.verdict == "none"
# a prior far from the truth with a search that misses it: none, not a wrong number
res_bad = audiosync.estimate_offsets_from_audio([A2, B2], [30.0, 30.0], (3.0, 18.0), 0.3, prior=[0.0, +200.0])
print(f"audio B with a wrong prior: verdict {res_bad[0].result.verdict}")
assert res_bad[0].result.verdict != "clear" or abs(res_bad[0].offset + 37) <= 0.5
# (I11) a camera whose recording ends before the stretch has a sound track: say it did not
# record the stretch, not "no sound track"; and a silent REFERENCE is named as the reference
res_eof = audiosync.estimate_offsets_from_audio([A2, B2], [30.0, 30.0], (19.0, 20.0), 0.3)
print(f"audio B past its end: has_audio {res_eof[0].has_audio}, out_of_reach {res_eof[0].out_of_reach}: "
      f"{res_eof[0].result.why[:70]}")
assert res_eof[0].has_audio is True and res_eof[0].out_of_reach and res_eof[0].result.verdict == "none"
assert "does not reach" in res_eof[0].result.why and "no sound track" not in res_eof[0].result.why
res_ref = audiosync.estimate_offsets_from_audio([D, A2, B2], [30.0, 30.0, 30.0], (3.0, 18.0), 2.5)
assert all(r_.ref_silent and r_.has_audio for r_ in res_ref), [(r_.ref_silent, r_.has_audio) for r_ in res_ref]
assert "reference camera" in res_ref[0].result.why and "no sound track" in res_ref[0].result.why
print("a camera past its end and a silent reference are told apart from a silent camera (I11) OK")

# (I10) a camera that started EARLIER than the prior must be found as well as one that started
# later, with the shared sounds only at the END of the stretch (synthetic sound tracks)
_real_signal = audiosync.audio_signal
_world = np.zeros(int(120 * SR))
_crng = np.random.RandomState(8)
for _tc in (55.0, 58.0):
    _k = int(_tc * SR)
    _world[_k:_k + 240] += _crng.normal(size=240) * np.exp(-np.arange(240) / 32.0)
_starts = {"ref": 10.0, "late": 16.0, "early": 4.0}          # world second each camera started at
_tracks = {nm: _world + 0.01 * _crng.normal(size=len(_world)) for nm in _starts}


def _fake_signal(path, t0, duration, sr=SR, should_cancel=None):
    a = int(round((_starts[path] + max(0.0, float(t0))) * sr))
    b = min(len(_tracks[path]), a + int(round(float(duration) * sr)))
    return _tracks[path][max(0, a):b].copy() if b > a else np.zeros(0)


audiosync.audio_signal = _fake_signal
try:
    # reference stretch 40-50 s = world 50-60 s: both claps in its last half; search 8 s, no prior
    res_ab = audiosync.estimate_offsets_from_audio(["ref", "late", "early"], [240.0] * 3, (40.0, 50.0), 8.0,
                                                   prior=[0.0, 0.0, 0.0])
finally:
    audiosync.audio_signal = _real_signal
for r_, truth in zip(res_ab, (-6.0 * 240, +6.0 * 240)):
    print(f"  camera started {'later' if truth < 0 else 'earlier'} by 6 s: offset {r_.offset:+.1f} (truth "
          f"{truth:+.0f}), verdict {r_.result.verdict}, score {r_.result.score:.2f}")
    assert r_.result.verdict == "clear" and abs(r_.offset - truth) <= 1.0, (r_.offset, truth, r_.result.why)
print("sound sync finds cameras that started earlier as well as later (I10) OK")
print("audio sync core OK")

# ---- the app: 3D -> Sync Cameras on a two-camera project -------------------------------
from PySide6.QtWidgets import QApplication, QMessageBox, QDialog
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
app = QApplication([])
from kinetrace.app import MainWindow, READY
from kinetrace.syncdialog import SyncDialog

win = MainWindow()


def _close_window_at_exit():
    # a failed assertion must stay a failure: without this the window's threads were still running at
    # exit and Qt aborted the process (Mac report 2026-10-08)
    try:
        win.close()
        QApplication.processEvents()
        win._dev_probe.wait(30000)
    except Exception:      # noqa: BLE001 - best effort while exiting
        pass


import atexit  # noqa: E402
atexit.register(_close_window_at_exit)
win.show()


def pump(sec):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def wait(cond, timeout, what):
    t = time.time()
    while not cond():
        pump(0.05)
        if time.time() - t > timeout:
            raise TimeoutError(what)


win._open_video(A2)
wait(lambda: win.state == READY, 30, "open")
assert win._add_view(B2)
pump(0.3)
assert win.act_sync.isEnabled(), "the sync action must be available with two cameras"
win._goto(250)
dlg = SyncDialog(win, win.project, [rt.info.path for rt in win._views], win.current)
dlg.show()
assert dlg.r_audio.isEnabled() and dlg.r_audio.isChecked() and dlg.filter_row.isEnabled()
# the motion method first
dlg.r_motion.setChecked(True)
assert not dlg.filter_row.isEnabled()
dlg.span.setValue(150)
dlg.search.setValue(100)
dlg._run()
wait(lambda: dlg.results is not None, 120, "sync run")
pump(0.2)
row = dlg.results[0]
print(f"dialog (motion): {dlg.status.text()[:120]}")
# Evidence for the cross-OS CI: the first run on GitHub's macOS machine found
# nothing here although the same core call on the cv2-written clips (above) was
# clear - so say how OpenCV sees the ffmpeg-remuxed clips on this machine, and
# what the core says on them directly, before asserting.
try:
    import imageio_ffmpeg
    print(f"  ffmpeg {imageio_ffmpeg.get_ffmpeg_version()}")
except Exception as _e:        # noqa: BLE001
    print(f"  ffmpeg version unknown ({_e})")
for _name, _path in (("A", A), ("A2", A2), ("B", B), ("B2", B2)):
    _cap = cv2.VideoCapture(_path)
    _n, _fps = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)), _cap.get(cv2.CAP_PROP_FPS)
    _cap.set(cv2.CAP_PROP_POS_FRAMES, 175)
    _ok, _fr = _cap.read()
    _cap.release()
    print(f"  {_name}: {_n} frames @ {_fps:.3f} fps (cv2 backend {cv2.VideoCapture(_path).getBackendName()}), "
          f"frame 175 {'decodes, mean ' + format(float(_fr.mean()), '.1f') if _ok else 'does NOT decode'}")
_direct = sync.estimate_offsets_from_motion([A2, B2], [1.0, 1.0], (175, 325), 100)
print(f"  core on the remuxed clips: offset {_direct[0].offset:+.1f}, verdict {_direct[0].result.verdict}, "
      f"score {_direct[0].result.score:.2f}: {_direct[0].result.why[:100]}")
print(f"  dialog row: offset {row.offset:+.1f}, verdict {row.result.verdict}, score {row.result.score:.2f}, "
      f"margin {row.result.margin:.2f}")
assert row.result.verdict == "clear" and abs(row.offset + 37) <= 0.5
assert "clear" in dlg.table.item(0, 3).text().lower()
# then the sound method, with the filter
dlg.r_audio.setChecked(True)
dlg.span.setValue(450)
dlg.search.setValue(75)
dlg._run()
wait(lambda: dlg.results is not None, 120, "audio sync run")
pump(0.2)
row = dlg.results[0]
print(f"dialog (sound): {dlg.status.text()[:160]}")
assert row.result.verdict == "clear" and abs(row.offset + 37) <= 0.5, (row.offset, row.result.verdict)
assert "travel" in dlg.status.text(), "the sound method must state its acoustic blind spot"
dlg._apply()
assert abs(win.project.offsets[1] + 37) <= 0.5, win.project.offsets
assert win.project.offsets[0] == 0.0
print("sync dialog through the app (motion and sound) OK")
# (I11) the dialog names the silent reference, and a camera that did not record the stretch
dlg2 = SyncDialog(win, win.project, [rt.info.path for rt in win._views], win.current)
dlg2._done(res_ref[:1])
assert win.project.name(0) in dlg2.status.text() and "reference" in dlg2.status.text(), dlg2.status.text()
assert not dlg2.btn_apply.isEnabled()
dlg2._done(res_eof)
assert win.project.name(1) in dlg2.status.text() and "did not record" in dlg2.status.text(), dlg2.status.text()
dlg2.close()
# (I12) the stretch is centred on the instant ON SCREEN: with the second camera active, the
# playhead is in ITS numbering and must be mapped onto the reference's (offset -37 now)
win._set_active_view(1)
pump(0.3)
win._goto(200)
pump(0.3)
assert win.project.active == 1 and win.current == 200, (win.project.active, win.current)
dlg3 = SyncDialog(win, win.project, [rt.info.path for rt in win._views], win.current)
want = int(round(win.project.reference_time(1, 200)))
print(f"dialog opened on {win.project.name(1)} frame 200: centred on reference frame {dlg3.centre.value()} "
      f"(the same instant: {want})")
assert dlg3.centre.value() == want == 237, (dlg3.centre.value(), want)
dlg3.close()
win._set_active_view(0)
pump(0.3)
print("the sync dialog names the silent file (I11) and centres on the instant on screen (I12) OK")
win.close()
pump(0.3)
forget_recovery(A, B, A2, B2)
print("verify_sync PASSED")
