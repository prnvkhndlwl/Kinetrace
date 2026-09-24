<sub>Kinetrace guide — [Home](../../README.md) · [Install & run](install.md) · [Workflow: track an animal](workflow.md) · [Segments, silhouettes & skeletons](segments.md) · [Several cameras & 3D](cameras-3d.md) · [Human bodies](bodies.md) · [Export formats](exports.md) · [Projects & autosave](projects.md) · [Keyboard & mouse](shortcuts.md) · [Accuracy](accuracy.md) · **Performance (4K)** · [Limitations](limitations.md) · [Testing](testing.md) · [Folder map & design](folder-map.md) · [User manual](../MANUAL.md) · [Licences](../../THIRD_PARTY_LICENSES.md)</sub>

# Performance tuning (4K scrubbing)

Because decoding is the bottleneck, a faster GPU does little; these help:

- **Frame cache** — sized automatically to a quarter of your RAM (max 24 GB),
  so a whole working region of 4K frames stays in memory and re-scrubbing a
  stretch you are correcting is instant. Override with the `COTRACKER_CACHE_GB`
  environment variable if you want more or less.
- **Decode backend** — hardware decode roughly doubles *sequential* decode
  (long tracking runs) but is not automatically better for seeking. Measure
  your own footage, then apply the winner:

  ```
  .venv\Scripts\python.exe tools\bench_decode.py "D:\path\to\your4k.mp4"
  ```

  Apply with `COTRACKER_DECODE` = `auto` (default) | `msmf` | `hw` | `ffmpeg`.
  Pick one and keep it for a project: backends agree on frame numbering but
  differ marginally in color conversion.

---

[← Accuracy](accuracy.md) &nbsp;|&nbsp; [Contents](../../README.md#contents) &nbsp;|&nbsp; [Limitations →](limitations.md)
