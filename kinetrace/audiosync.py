"""Whole-frame camera sync from the SOUND tracks (claps, voices, the wand
knocking a tripod): the usual way hand-started cameras are lined up.

The audio tracks of all cameras are extracted with ffmpeg and
cross-correlated to get the offsets. Field audio is usually full of background
noise, so a filter option is offered.

How it works
* `ffmpeg` (the static binary that `imageio-ffmpeg` installs inside the
  venv, or one on PATH / in $KINETRACE_FFMPEG) decodes the audio track of a
  stretch of each video to mono 8 kHz floats through a pipe - a few seconds
  for a whole GoPro file, no temporary files.
* A band-pass FILTER (default 300-3800 Hz) removes rumble, wind and hum below
  and hiss above the band where claps and voices live; both edges are user
  choices.
* The two signals are compared with a block-averaged GCC-PHAT
  ("whitened" cross-correlation): the recording is cut into blocks, each
  block's cross-power spectrum is normalised to unit magnitude (only the
  PHASE - i.e. the timing - survives, so a loud fan does not out-vote a
  clap), and the blocks' correlograms are summed, each weighted by how
  clearly it peaks. Plain cross-correlation is offered too, for quiet rooms.
  Measured on a real 8-camera rig (microphones metres apart in a noisy
  hall): plain correlation found nothing (peak/next ratio 1.01); block
  GCC-PHAT placed one camera at +631.7 frames against +633 from the motion
  sync and an independent sync table.
* The verdict is the ratio of the best peak to the next-best one at least
  10 ms away (clear >= 1.5, weak >= 1.2, else none) plus a plain sentence.

What it cannot do, said to the user: sound travels ~343 m/s, so a camera
6 m farther from the clap hears it 17 ms later = 4 frames at 240 fps. The
audio offset is therefore exact to within (camera spacing / 343 m/s); the
motion sync and, after tracking, 3D -> Estimate Sub-frame Offsets have no such
bias and refine it. Pure numpy/scipy + a subprocess; no Qt.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

import numpy as np

from kinetrace.sync import AlignResult, CameraSync

SAMPLE_RATE = 8000          # Hz: enough for timing (0.125 ms per sample, 1/33 of a 240 fps frame)
BAND_DEFAULT = (300.0, 3800.0)
BLOCK_S = 4.0               # GCC-PHAT block length; the lag range must fit inside a block
FFMPEG_TIMEOUT_S = 600.0    # one read of a sound track (a stretch, or a whole recording over a network share)
SILENT = 1e-9               # a block / a track whose largest sample is below this holds no sound
SPEED_OF_SOUND = 343.0      # m/s, for the caveat


# ------------------------------------------------------------------ ffmpeg

def find_ffmpeg() -> str | None:
    """The ffmpeg executable: $KINETRACE_FFMPEG, then the one imageio-ffmpeg
    installed inside the venv (self-contained folder), then PATH."""
    env = os.environ.get("KINETRACE_FFMPEG")
    if env and os.path.exists(env):
        return env
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:       # noqa: BLE001 - not installed, or its binary missing
        pass
    return shutil.which("ffmpeg")


def ffmpeg_status() -> tuple[bool, str]:
    exe = find_ffmpeg()
    if exe:
        return True, f"ffmpeg found: {exe}"
    py = ".venv\\Scripts\\python.exe" if os.name == "nt" else ".venv/bin/python"
    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    return False, ("No ffmpeg: audio sync needs it to read the sound tracks. Install it inside the "
                   f"folder with  {py} -m pip install imageio-ffmpeg  (31 MB), "
                   f"or put an {exe_name} on PATH / in the KINETRACE_FFMPEG environment variable.")


def has_audio(path: str, should_cancel=None) -> bool | None:
    """True / False from ffmpeg's stream listing; None when it cannot be told: ffmpeg is
    missing, or ffmpeg cannot open the file at all (missing, damaged, not a video) -- that
    is "could not be read", not "has no sound track" (G97)."""
    exe = find_ffmpeg()
    if not exe:
        return None
    try:
        rc, _out, err = _run([exe, "-hide_banner", "-i", str(path)], should_cancel, 120.0)
    except (OSError, InterruptedError):
        return None
    text = err.decode("utf-8", "replace")
    if "Input #" not in text:           # ffmpeg prints this line for every file it managed to open
        return None
    return any("Audio:" in line for line in text.splitlines())


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _run(cmd: list[str], should_cancel=None, timeout: float | None = FFMPEG_TIMEOUT_S) -> tuple[int, bytes, bytes]:
    """Run ffmpeg, collecting (returncode, stdout, stderr), polled against `should_cancel`: a
    cancel kills the process and raises InterruptedError, `timeout` seconds raise OSError (a
    plain subprocess.run could neither be stopped nor give up -- G96)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                            creationflags=_no_window())
    box: dict = {}

    def work():
        try:
            box["out"] = proc.communicate()
        except Exception as e:      # noqa: BLE001 - reported below
            box["err"] = e

    th = threading.Thread(target=work, daemon=True)
    th.start()
    t0 = time.monotonic()
    while th.is_alive():
        th.join(0.1)
        if not th.is_alive():
            break
        if should_cancel is not None and should_cancel():
            proc.kill()
            th.join(10)
            raise InterruptedError("cancelled")
        if timeout is not None and time.monotonic() - t0 > timeout:
            proc.kill()
            th.join(10)
            raise OSError(f"ffmpeg did not finish within {timeout:.0f} s (a very slow drive or network share?)")
    if "err" in box:
        raise OSError(f"ffmpeg failed: {box['err']}")
    out, err = box["out"]
    return proc.returncode, out, err


def audio_signal(path: str, t0: float, duration: float, sr: int = SAMPLE_RATE, should_cancel=None) -> np.ndarray:
    """Mono float64 samples of `path` from `t0` seconds for `duration` seconds,
    decoded by ffmpeg through a pipe (no temp file). Empty array when the file
    has no audio track; OSError when ffmpeg is missing, fails, or the file cannot
    be read; InterruptedError when `should_cancel()` turned true (ffmpeg is killed)."""
    exe = find_ffmpeg()
    if not exe:
        raise OSError(ffmpeg_status()[1])
    t0 = max(0.0, float(t0))
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-nostdin",
           "-ss", f"{t0:.3f}", "-t", f"{max(0.0, float(duration)):.3f}", "-i", str(path),
           "-vn", "-ac", "1", "-ar", str(int(sr)), "-f", "f32le", "-"]
    rc, out, err = _run(cmd, should_cancel)
    if rc != 0 and not out:
        lines = err.decode("utf-8", "replace").strip().splitlines()
        msg = lines[-1] if lines else f"ffmpeg exit code {rc}"
        # ffmpeg 7 reports a video without a sound track as "Error opening output
        # files: Invalid argument" (older builds: "does not contain any stream")
        if has_audio(path, should_cancel) is False:
            return np.zeros(0)
        raise OSError(f"ffmpeg could not read the audio of {os.path.basename(str(path))}: {msg}")
    return np.frombuffer(out, np.float32).astype(np.float64)


# ------------------------------------------------------------------ filtering

def bandpass(x: np.ndarray, lo: float, hi: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Zero-phase Butterworth band-pass (4th order). lo <= 0 = no high-pass,
    hi >= sr/2 = no low-pass. The filter option for noisy recordings."""
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(x, np.float64)
    if len(x) < 64:
        return x.copy()
    nyq = 0.5 * sr
    lo = float(max(0.0, lo))
    hi = float(min(hi, nyq * 0.98))
    if lo <= 0 and hi >= nyq * 0.98:
        return x.copy()
    if lo <= 0:
        sos = butter(4, hi, btype="low", fs=sr, output="sos")
    elif hi >= nyq * 0.98:
        sos = butter(4, lo, btype="high", fs=sr, output="sos")
    else:
        if hi <= lo * 1.2:
            hi = min(nyq * 0.98, lo * 1.5)
        sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


# ------------------------------------------------------------------ alignment

# one result type for both sync methods (R21): `sync.AlignResult` / `sync.CameraSync`; the names the
# sound method used to have stay as aliases
AudioAlign = AlignResult
AudioCameraSync = CameraSync


def _correlogram(a: np.ndarray, b: np.ndarray, max_lag: int, whiten: bool,
                 block: int, hop: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Block-averaged cross-correlation of a and b (same start time, same sr).
    c[k] = sum_n a[n + k] b[n]: a positive k means b[n] lines up with a[n + k],
    i.e. b is AHEAD of a by k samples (align_audio negates it into a delay).
    Blocks are weighted by their own peak sharpness so a block holding a clap
    counts more than one holding fan noise."""
    n = min(len(a), len(b))
    block = int(min(block, n))
    if block < 4 * max_lag + 64:
        block = min(n, 4 * max_lag + 64)
    nfft = 1 << int(2 * block - 1).bit_length()
    lags = np.arange(-max_lag, max_lag + 1)
    acc = np.zeros(len(lags))
    win = np.hanning(block)
    n_blocks = 0
    wsum = 0.0
    starts = list(range(0, n - block + 1, max(1, hop))) or [0]
    for s0 in starts:
        sa = a[s0:s0 + block]
        sb = b[s0:s0 + block]
        if len(sa) < block or len(sb) < block:
            break
        # a block counts only when BOTH tracks hold sound: a digitally silent one has nothing to
        # time, and whitening it made a "block" of zeros that read as "no offset stands out" (G97)
        if np.max(np.abs(sa)) < SILENT or np.max(np.abs(sb)) < SILENT:
            continue
        fa = np.fft.rfft(sa * win, nfft)
        fb = np.fft.rfft(sb * win, nfft)
        G = fa * np.conj(fb)
        if whiten:
            G = G / (np.abs(G) + 1e-12)
        else:
            na, nb = np.linalg.norm(sa), np.linalg.norm(sb)
            if na < 1e-12 or nb < 1e-12:
                continue
            G = G / (na * nb)
        c = np.fft.irfft(G, nfft)
        c = np.concatenate([c[-max_lag:], c[:max_lag + 1]]) if max_lag > 0 else c[:1]
        spread = float(np.median(np.abs(c))) + 1e-12
        w = float(np.max(c)) / spread          # sharpness: a transient makes a spike, noise does not
        w = float(np.clip(w, 0.0, 50.0))
        acc += w * c
        wsum += w
        n_blocks += 1
    if wsum > 0:
        acc /= wsum
    return lags, acc, n_blocks


def align_audio(ref: np.ndarray, other: np.ndarray, max_lag_s: float, sr: int = SAMPLE_RATE,
                whiten: bool = True, block_s: float = BLOCK_S) -> AudioAlign:
    """Lag (seconds) by which `other` is delayed relative to `ref` (both start
    at the same nominal instant), searched over +-max_lag_s, with a verdict."""
    ref = np.asarray(ref, np.float64)
    other = np.asarray(other, np.float64)
    n = min(len(ref), len(other))
    if n < sr:      # under a second: nothing to compare
        return AlignResult(0.0, float("nan"), float("nan"), "none",
                           "The two sound tracks do not overlap for even a second - check the search range.")
    max_lag = int(round(max_lag_s * sr))
    block = int(round(block_s * sr))
    lags, curve, n_blocks = _correlogram(ref[:n], other[:n], max_lag, whiten, block, block // 2)
    if n_blocks == 0 or not np.isfinite(curve).any():
        return AlignResult(0.0, float("nan"), float("nan"), "none", "One of the sound tracks is silent.")
    k = int(np.argmax(curve))
    peak = float(curve[k])
    far = np.abs(lags - lags[k]) > int(0.010 * sr)
    rival = float(np.max(curve[far])) if far.any() else 0.0
    score = peak / rival if rival > 1e-12 else float("inf")
    # the correlogram's positive lag = `other` AHEAD; report the DELAY of `other`
    # (measured on two cameras of a real rig: the peak sits at +0.635 s and the
    # second started 2.635 s after the first with a 2 s clock prior, so
    # delay = -peak lag)
    lag_s = -float(lags[k]) / sr
    if score >= 1.5 and peak > 0:
        verdict = "clear"
        why = (f"The sound tracks line up at one offset (the best match is {score:.1f}x the next-best): "
               f"a clear alignment from {n_blocks} blocks of {block_s:.0f} s.")
    elif score >= 1.2 and peak > 0:
        verdict = "weak"
        why = (f"A likely offset, but not a sharp one (best match only {score:.2f}x the next-best). "
               "Background noise or few shared sounds; try a stretch with claps, or the whole recording, "
               "and check it by eye.")
    else:
        verdict = "none"
        why = (f"No offset stands out (best match {score:.2f}x the next-best). The cameras probably did not "
               "hear the same sounds in this stretch (too far apart, or wind on one microphone), or the true "
               "offset lies outside the search range.")
    margin = peak - rival if rival > 1e-12 else peak
    return AlignResult(lag_s, float(score), float(margin), verdict, why, n_blocks, peak)


def acoustic_caveat(fps: float, spacing_m: float = 5.0) -> str:
    frames = spacing_m / SPEED_OF_SOUND * fps
    return (f"Sound needs {1000 * spacing_m / SPEED_OF_SOUND:.0f} ms to travel {spacing_m:.0f} m, so a camera "
            f"that much farther from the clap hears it about {frames:.1f} frames late at {fps:.0f} fps. The "
            "audio offset is exact only to that; the Motion method in the same dialog has no such bias, and "
            "3D -> Estimate Sub-frame Offsets (after tracking) refines it to a fraction of a frame.")


def estimate_offsets_from_audio(paths: list[str], fps: list[float], ref_range_s: tuple[float, float],
                                search_s: float, reference: int = 0, prior: list[float | None] | None = None,
                                band: tuple[float, float] | None = BAND_DEFAULT, whiten: bool = True,
                                progress=None, should_cancel=None) -> list[AudioCameraSync]:
    """For every other camera, the whole-frame offset that lines its sound up
    with the reference camera's over reference seconds `ref_range_s`, searching
    +-`search_s` seconds around a `prior` offset per camera (Kinetrace frames of
    that camera; `sync.offsets_from_filenames` gives one from the recording
    clocks). Returns Kinetrace offsets: local_i = rate_i * t + offset_i, so a
    camera that started D seconds later has offset -D * fps_i."""
    t0, t1 = float(ref_range_s[0]), float(ref_range_s[1])
    dur = max(1.0, t1 - t0)
    n_views = len(paths)
    fps = [float(f) for f in fps]
    if not find_ffmpeg():
        raise OSError(ffmpeg_status()[1])           # missing ffmpeg is the caller's to explain, not a row's
    if progress:
        progress(0.0)

    def signal(path, a, b):
        """(samples, None) or (None, why) when this file cannot be read at all (G97)."""
        try:
            return audio_signal(path, a, b, should_cancel=should_cancel), None
        except InterruptedError:                    # cancelled: the callers check should_cancel next
            return None, None
        except OSError as e:
            return None, str(e)

    ref, ref_unreadable = signal(paths[reference], t0, dur)
    if ref is None:
        ref = np.zeros(0)
    if band is not None and len(ref):
        ref = bandpass(ref, band[0], band[1])
    ref_why = ""
    if should_cancel is not None and should_cancel():
        return []
    if ref_unreadable:
        ref_why = (f"The reference camera (the first view) could not be read ({ref_unreadable}): check that its "
                   "video file is there and opens, or make another camera the first view.")
    elif len(ref) == 0:
        # (I11) say WHICH file is silent: this used to tell the user that every OTHER
        # camera had no sound track and send them all to the Motion method
        ref_why = ("The reference camera (the first view) has no sound track: make a camera with sound the first "
                   "view, or use the Motion method." if has_audio(paths[reference]) is False else
                   "The reference camera's sound does not reach this stretch (its recording is shorter): choose "
                   "a stretch inside it.")
    elif np.max(np.abs(ref)) < SILENT:
        # (G97) a track that exists but holds no sound (a muted microphone)
        ref_why = ("The reference camera's (the first view's) sound track is silent in this stretch: no sound was "
                   "recorded. Make a camera with sound the first view, or use the Motion method.")
    out: list[CameraSync] = []
    done = 1
    for i, path in enumerate(paths):
        if i == reference:
            continue
        if should_cancel is not None and should_cancel():
            break
        pri_frames = float(prior[i]) if (prior is not None and i < len(prior) and prior[i] is not None) else 0.0
        # camera i started D0 = -prior/fps_i seconds after the reference: its own
        # time for the reference stretch is t - D0
        d0 = -pri_frames / fps[i]
        g0 = t0 - d0 - search_s
        pad_front = max(0.0, -g0)                 # the file starts after the stretch begins
        want_s = dur + 2 * search_s - pad_front
        unreadable = None
        if ref_why:
            sig = np.zeros(0)
        else:
            sig, unreadable = signal(path, max(0.0, g0), want_s)
            if should_cancel is not None and should_cancel():
                break
            if sig is None:
                sig = np.zeros(0)
        silent = bool(len(sig)) and not ref_why and float(np.max(np.abs(sig))) < SILENT
        if len(sig) == 0 or ref_why or silent:
            if ref_why:
                row = CameraSync(i, pri_frames, AlignResult(0.0, float("nan"), float("nan"), "none", ref_why),
                                 ref_silent=True)
            elif unreadable:
                row = CameraSync(i, pri_frames, AlignResult(
                    0.0, float("nan"), float("nan"), "none",
                    f"This video could not be read ({unreadable}): check that the file is there and opens."),
                    has_audio=None)
            elif silent:
                row = CameraSync(i, pri_frames, AlignResult(
                    0.0, float("nan"), float("nan"), "none",
                    "This video's sound track is silent in this stretch (no sound was recorded: a muted "
                    "microphone?): use the Motion method for it."))
            elif has_audio(path) is False:
                row = CameraSync(i, pri_frames, AlignResult(0.0, float("nan"), float("nan"), "none",
                                                            "This video has no sound track."), has_audio=False)
            else:
                # (I11) ffmpeg returns nothing for a window past the end of a file that
                # HAS sound: the camera had stopped (or started much later than assumed)
                row = CameraSync(i, pri_frames, AlignResult(
                    0.0, float("nan"), float("nan"), "none",
                    f"This camera's recording does not reach the stretch searched (from {max(0.0, g0):.1f} s of "
                    "its own time): it had stopped, or it started much later than the offset assumed. Choose a "
                    "stretch every camera recorded, or widen the search."), out_of_reach=True)
            out.append(row)
            done += 1
            if progress:
                progress(done / n_views)
            continue
        if band is not None:
            sig = bandpass(sig, band[0], band[1])
        # put the two on a common time base: pad the reference so both start at g0 (camera-i time)
        sr = SAMPLE_RATE
        lead = int(round((search_s - pad_front) * sr))
        ref_al = np.concatenate([np.zeros(max(0, lead)), ref]) if lead >= 0 else ref[-lead:]
        # (I10) and pad it at the END to the other camera's length: align_audio compares
        # min(len) samples, which cut the other camera's last `search` seconds - a camera
        # that started EARLIER than the prior (a positive lag) lost the partners of the
        # sounds at the end of the stretch (6 s early with an 8 s search: found nothing)
        if len(ref_al) < len(sig):
            ref_al = np.concatenate([ref_al, np.zeros(len(sig) - len(ref_al))])
        res = align_audio(ref_al, sig, max_lag_s=search_s, sr=sr, whiten=whiten)
        if res.verdict != "clear" and len(sig) < (want_s - 1.0) * sr:
            res.why += (f" (This camera's sound covers only {len(sig) / sr:.1f} of the {want_s:.1f} s searched: "
                        "its recording ends inside the stretch.)")
        # `sig` delayed by lag means camera i shows the sound lag seconds later in its
        # own numbering than the (prior-shifted) reference: it started lag seconds EARLIER
        # than the prior said -> D = d0 - lag seconds after the reference
        d = d0 - res.lag
        offset = -d * fps[i]
        out.append(CameraSync(i, float(offset), res, lag_s=float(d)))
        done += 1
        if progress:
            progress(done / n_views)
    return out
