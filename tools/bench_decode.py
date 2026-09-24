"""Decode-backend benchmark: find the fastest safe decode path for YOUR footage.

Synthetic clips do not predict real-camera decode behavior (bitrate, GOP
length, and codec dominate), so measure on a file you actually work with:

    .venv\\Scripts\\python.exe tools\\bench_decode.py "D:\\path\\to\\real4k.mp4"

It reports, per backend, the two access patterns the app actually uses:
  * sequential decode  -> the ceiling for TRACKING runs
  * random seek        -> what SCRUBBING feels like
and verifies each backend agrees with the software reference on frame count
and frame identity (a backend that shifts indices is unusable regardless of
speed).

Apply the winner with the COTRACKER_DECODE environment variable
(auto | msmf | hw | ffmpeg), e.g. in PowerShell:

    $env:COTRACKER_DECODE = "msmf"; .\\run.bat

Leave it unset to keep the default (software FFmpeg).
"""
from __future__ import annotations

import sys

# Windows consoles default to cp1252; keep the arrows/checkmarks from crashing the report.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")
import time
from pathlib import Path

import cv2
import numpy as np

SEQ_FRAMES = 90
SEEKS = 12


def backends() -> list[tuple[str, str, callable]]:
    """(env value, label, opener) — mirrors video_source.open_capture."""
    out = [
        ("auto", "auto / software FFmpeg (default)",
         lambda p: cv2.VideoCapture(p)),
        ("ffmpeg", "FFmpeg, forced software",
         lambda p: cv2.VideoCapture(p, cv2.CAP_FFMPEG,
                                    [cv2.CAP_PROP_HW_ACCELERATION,
                                     cv2.VIDEO_ACCELERATION_NONE])),
        ("hw", "FFmpeg + hardware (D3D11VA)",
         lambda p: cv2.VideoCapture(p, cv2.CAP_FFMPEG,
                                    [cv2.CAP_PROP_HW_ACCELERATION,
                                     cv2.VIDEO_ACCELERATION_ANY])),
    ]
    if hasattr(cv2, "CAP_MSMF"):
        out.append(("msmf", "Media Foundation (hardware)",
                    lambda p: cv2.VideoCapture(p, cv2.CAP_MSMF)))
    if hasattr(cv2, "CAP_INTEL_MFX"):
        out.append(("(mfx)", "Intel Media SDK / Quick Sync",
                    lambda p: cv2.VideoCapture(p, cv2.CAP_INTEL_MFX)))
    return out


def measure(opener, path: str, ref: dict | None):
    cap = opener(path)
    if cap is None or not cap.isOpened():
        return None
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        hw = cap.get(cv2.CAP_PROP_HW_ACCELERATION)
        ok, first = cap.read()
        if not ok:
            return None
        h, w = first.shape[:2]

        t0 = time.perf_counter()
        k = 0
        while k < SEQ_FRAMES:
            ok, _ = cap.read()
            if not ok:
                break
            k += 1
        seq = k / max(time.perf_counter() - t0, 1e-9)

        hi = max(n - 5, 10)
        targets = np.random.default_rng(0).integers(5, hi, SEEKS)
        samples, t0 = {}, time.perf_counter()
        for t in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
            ok, fr = cap.read()
            if ok and int(t) not in samples:
                samples[int(t)] = fr
        seek_ms = (time.perf_counter() - t0) / len(targets) * 1000

        identity = "reference"
        if ref is not None:
            diffs, missing = [], 0
            for i, a in ref["samples"].items():
                b = samples.get(i)
                if b is None or b.shape != a.shape:
                    missing += 1
                    continue
                diffs.append(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
            if missing or not diffs:
                identity = "MISMATCH (frames missing)"
            elif n != ref["n"]:
                identity = f"MISMATCH (count {n} vs {ref['n']})"
            elif max(diffs) < 3.0:
                identity = f"ok (mean |d| {max(diffs):.2f})"
            else:
                identity = f"DIFFERENT PIXELS (mean |d| {max(diffs):.1f})"
        return {"n": n, "hw": hw, "wh": (w, h), "seq": seq,
                "seek_ms": seek_ms, "samples": samples, "identity": identity}
    finally:
        cap.release()


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    if not Path(path).exists():
        sys.exit(f"no such file: {path}")

    print(f"OpenCV {cv2.__version__}\nfile: {path}\n")
    ref, rows = None, []
    for env, label, opener in backends():
        try:
            r = measure(opener, path, ref)
        except Exception as e:  # noqa: BLE001 — a failing backend is a result
            print(f"  {label:34s}: EXCEPTION {type(e).__name__}: {e}")
            continue
        if r is None:
            print(f"  {label:34s}: unavailable for this file/codec")
            continue
        if ref is None:
            ref = r
            print(f"{r['wh'][0]}x{r['wh'][1]}, {r['n']} frames\n")
            print(f"  {'backend':34s} {'seq fps':>9s} {'seek ms':>9s}  "
                  f"{'hw':>3s}  frame identity")
            print("  " + "-" * 78)
        rows.append((env, label, r))
        print(f"  {label:34s} {r['seq']:9.1f} {r['seek_ms']:9.1f} "
              f"{r['hw']:3.0f}  {r['identity']}")

    safe = [(e, l, r) for e, l, r in rows
            if "MISMATCH" not in r["identity"] and "DIFFERENT" not in r["identity"]]
    if not safe:
        return
    best_seq = max(safe, key=lambda x: x[2]["seq"])
    best_seek = min(safe, key=lambda x: x[2]["seek_ms"])
    print("\n  fastest sequential (tracking runs): "
          f"{best_seq[1]}  ({best_seq[2]['seq']:.1f} fps)")
    print("  fastest random seek (scrubbing)   : "
          f"{best_seek[1]}  ({best_seek[2]['seek_ms']:.1f} ms)")
    if best_seq[0] == best_seek[0]:
        print(f"\n  → set COTRACKER_DECODE={best_seq[0]}"
              if best_seq[0] != "auto" else
              "\n  → the default is already the best on this file; change nothing.")
    else:
        print("\n  → the two patterns disagree. Scrubbing responsiveness is what "
              "you feel most,\n    so prefer the seek winner unless you mostly run "
              "long unattended tracks.")
    print("\n  NOTE: backends differ slightly in YUV->RGB conversion, so pick ONE "
          "and keep it\n  for a project; re-run tests\\verify_4k.py after switching "
          "to confirm accuracy.")


if __name__ == "__main__":
    main()
