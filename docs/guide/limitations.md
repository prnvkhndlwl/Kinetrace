<sub>Kinetrace guide — [Home](../../README.md) · [Install & run](install.md) · [Workflow: track an animal](workflow.md) · [Segments, silhouettes & skeletons](segments.md) · [Several cameras & 3D](cameras-3d.md) · [Human bodies](bodies.md) · [Export formats](exports.md) · [Projects & autosave](projects.md) · [Keyboard & mouse](shortcuts.md) · [Accuracy](accuracy.md) · [Performance (4K)](performance.md) · **Limitations** · [Testing](testing.md) · [Folder map & design](folder-map.md) · [User manual](../MANUAL.md) · [Licences](../../THIRD_PARTY_LICENSES.md)</sub>

# Limitations

- **Forward-only tracking** (both point models run online, forward in time): to fix the
  past, scrub back, correct, and re-track forward from there.
- **Regions** can be circles, rectangles or polygons (Add ▾); their members are
  sampled inside the outline. The animal silhouette covers the deformable case.
- **One animal per project.** Several animals with identities are planned (the
  segmentation sessions already support it; the UI does not yet).
- **Variable-frame-rate video** (some phone/screen recordings): frame numbers
  become ambiguous. The app detects and warns; re-encode first:
  `ffmpeg -i in.mp4 -vsync cfr -r 30 -c:v libx264 -crf 18 out.mp4`
- **A file that states no usable frame rate**: the app measures it from the
  frames' timestamps (or, failing that, assumes 30 fps) and says so when the
  video opens. Check that number, because every time, speed and camera sync
  uses it. High-speed rates up to 100 000 fps are read from the file.
- **A damaged frame** stops tracking there ("Tracking stopped: frame N of the
  video could not be decoded") and keeps everything before it; scrubbing onto
  it shows a blank frame and says so (Shift+C retries).
- Tracking speed is usually limited by video *decoding*, not the GPU
  (measured ≈40 fps at 1080p, ≈12–15 fps at 4K with refinement). A full 40k-frame
  4K pass is roughly an hour — pause/resume and autosave make that practical.

---

[← Performance (4K)](performance.md) &nbsp;|&nbsp; [Contents](../../README.md#contents) &nbsp;|&nbsp; [Testing →](testing.md)
