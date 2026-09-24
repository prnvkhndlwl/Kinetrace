<sub>Kinetrace guide — [Home](../../README.md) · [Install & run](install.md) · [Workflow: track an animal](workflow.md) · [Segments, silhouettes & skeletons](segments.md) · [Several cameras & 3D](cameras-3d.md) · [Human bodies](bodies.md) · [Export formats](exports.md) · **Projects & autosave** · [Keyboard & mouse](shortcuts.md) · [Accuracy](accuracy.md) · [Performance (4K)](performance.md) · [Limitations](limitations.md) · [Testing](testing.md) · [Folder map & design](folder-map.md) · [User manual](../MANUAL.md) · [Licences](../../THIRD_PARTY_LICENSES.md)</sub>

# Projects, autosave, crash safety

- **File → Save Project** writes a `.cotrk` file (tracks + points + the frame
  you were on). **Open Project** puts you back exactly where you stopped —
  even on another machine, it will ask you to locate the video if the path
  moved.
- The session **autosaves every 30 s** (also during tracking, and on every
  pause/export/exit) to `<video>.cotracker.npz` next to the video, or to your
  project file once you have one. If the app or machine dies mid-run, you
  lose at most 30 seconds of work — reopening the video offers to resume.
- An autosave that cannot be resumed (unreadable, or saved for a video with a
  different number of frames) is never overwritten: it is set aside as
  `<video>.cotracker.<date-time>.bak.npz` and the app says so.
- A Save that fails says so and writes nothing. If the project file cannot be
  written, autosave falls back to `<video>.cotracker.npz` next to the video and
  tells you to use **File → Save Project As…**.

---

[← Export formats](exports.md) &nbsp;|&nbsp; [Contents](../../README.md#contents) &nbsp;|&nbsp; [Keyboard & mouse →](shortcuts.md)
