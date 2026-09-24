<sub>Kinetrace guide — [Home](../../README.md) · [Install & run](install.md) · [Workflow: track an animal](workflow.md) · [Segments, silhouettes & skeletons](segments.md) · [Several cameras & 3D](cameras-3d.md) · [Human bodies](bodies.md) · [Export formats](exports.md) · [Projects & autosave](projects.md) · [Keyboard & mouse](shortcuts.md) · **Accuracy** · [Performance (4K)](performance.md) · [Limitations](limitations.md) · [Testing](testing.md) · [Folder map & design](folder-map.md) · [User manual](../MANUAL.md) · [Licences](../../THIRD_PARTY_LICENSES.md)</sub>

# Accuracy

Learned point trackers see the video at reduced resolution (AllTracker at up
to 1024 px, CoTracker3 at ~512×384 internally), so their raw localization
error grows with resolution: CoTracker3 measured **≈3.7 px mean on 4K**
synthetic ground truth. This tool therefore runs a
second stage at native resolution — a gated Lucas-Kanade refinement seeded
from the model's own prediction, chained from your exact seed click and
re-anchored to the model every frame (it can never drift more than a few
pixels from the model's track). Measured result on 4K ground truth:
**≈0.7 px mean / 1.8 px max**. At ≤720p the model is already at native scale
and refinement automatically stays out of the way. ROI zoom attacks the same
root cause from the other side: inside a crop, small objects reach the model
at several times the detail, which is what makes tracking robust to
background changes around them.

Tips for best precision and robustness:
- Click points on visually distinct features when you can — or better, drag a
  circle snugly around the object so it tracks as a region.
- The seed frame keeps your exact clicked coordinates in the export —
  corrections are treated as ground truth.
- Track a few extra "sacrificial" points on the same rigid object: the point
  model tracks all points jointly, so they support each other. Ignore the extras at
  export (or just use a region, which does this internally).
- Trust the timeline reds: they mark exactly where the model struggled.

---

[← Keyboard & mouse](shortcuts.md) &nbsp;|&nbsp; [Contents](../../README.md#contents) &nbsp;|&nbsp; [Performance (4K) →](performance.md)
