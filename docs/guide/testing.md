<sub>Kinetrace guide — [Home](../../README.md) · [Install & run](install.md) · [Workflow: track an animal](workflow.md) · [Segments, silhouettes & skeletons](segments.md) · [Several cameras & 3D](cameras-3d.md) · [Human bodies](bodies.md) · [Export formats](exports.md) · [Projects & autosave](projects.md) · [Keyboard & mouse](shortcuts.md) · [Accuracy](accuracy.md) · [Performance (4K)](performance.md) · [Limitations](limitations.md) · **Testing** · [Folder map & design](folder-map.md) · [User manual](../MANUAL.md) · [Licences](../../THIRD_PARTY_LICENSES.md)</sub>

# Testing

`make_test_video.py` generates synthetic videos with known ground truth and
validates exports against it:

```
python make_test_video.py demo.mp4 --frames 600 --size 1920x1080 --dots 4
python make_test_video.py demo.mp4 --check exported_tracks.csv
```

For a first look, make the demo clip with
`python make_test_video.py test600.mp4 --seed 0` (inside `.venv`), open it,
place the four dots (press **N**, click a dot; once per dot), press Track, and
watch.

The verification suites live in `tests/` (run them with the environment's
Python: `.venv\Scripts\python.exe` on Windows, `.venv/bin/python` on Linux and
macOS — `tests/verify_<name>.py`; each prints `PASSED` and exits non-zero on
failure). `tests/run_suites.py --cpu|--gpu` runs a whole group and generates
the synthetic test videos it needs first. Ball markers have
`verify_balls.py` and the automatic re-track `verify_retrack.py` (both GPU);
`verify_sweep_fixes.py` pins the fixes of the 2026-09-22 release sweep —
camera timing and exports, undo and keyframes, the multi-camera, canvas and
timeline behaviour (offscreen, no GPU, part of `--cpu`). Two audit tools are
not suites: `tests\audit_sweep.py` drives every enabled menu entry, hotkey and
context-menu entry in every state offscreen and must end with 0 exceptions;
`tests/audit_gui_real.py` drives the app with real mouse clicks and key
presses on the real desktop, screenshots every window, dialog, wizard
page and menu into `tests\out\gui_real\`, and checks that nothing raises,
menus show their tooltips, hotkeys work wherever the focus is, the view never
moves on its own, a pause lands within a second and nothing is clipped at
1366×768 (`--no-gpu` skips the segment and tracking states).
The 3D layer has two of its own: `verify_3d.py` (cameras, triangulation,
undistortion, sub-frame sync, visual hull — synthetic ground truth, no GPU)
and `verify_3d_gui.py` (the 3D menu end to end, offscreen). The body layer has
`verify_body.py` (joint angles against a synthetic walker **built from** known
angles, storage, exports, the backend registry, the drawing — no GPU, no
weights) and `verify_body_gui.py` (the real pose model through the app, from
the run dialog to the exported video).

---

[← Limitations](limitations.md) &nbsp;|&nbsp; [Contents](../../README.md#contents) &nbsp;|&nbsp; [Folder map & design →](folder-map.md)
