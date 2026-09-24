<sub>Kinetrace guide — [Home](../../README.md) · [Install & run](install.md) · [Workflow: track an animal](workflow.md) · [Segments, silhouettes & skeletons](segments.md) · [Several cameras & 3D](cameras-3d.md) · [Human bodies](bodies.md) · [Export formats](exports.md) · [Projects & autosave](projects.md) · [Keyboard & mouse](shortcuts.md) · [Accuracy](accuracy.md) · [Performance (4K)](performance.md) · [Limitations](limitations.md) · [Testing](testing.md) · **Folder map & design** · [User manual](../MANUAL.md) · [Licences](../../THIRD_PARTY_LICENSES.md)</sub>

# Folder map

```
run.bat / run.sh        launchers (Windows / Linux + macOS; create .venv on first run)
install.py              installs PyTorch for this machine, then requirements.txt
requirements.txt        the other dependencies
cotracker_app/          the application (calib.py / hull.py / view3d.py = the 3D layer)
docs/                   the user manual (also in the app, F1) and the developer handbook
tests/                  verification suites and audit tools (tests/out/ is scratch)
tools/                  developer tools (a decode benchmark)
make_test_video.py      synthetic test videos with ground truth
LICENSES/, THIRD_PARTY_LICENSES.md   licences of everything Kinetrace uses
created on your computer (never in the repository):
  .venv/                all Python dependencies
  models/               the point, segmentation and body models
  skeletons/            your own skeleton templates (JSON)
```


## Appearance

The app uses a quiet, dark, content-first design: the video is the hero,
controls stay out of the way, and one blue accent consistently means
"active / selected / primary" — the filled **Track** button, checked tools,
selections, and the timeline's frame-window highlight are all the same blue.

---

[← Testing](testing.md) &nbsp;|&nbsp; [Contents](../../README.md#contents) &nbsp;|&nbsp; [Home →](../../README.md)
