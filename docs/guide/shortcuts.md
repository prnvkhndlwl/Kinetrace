<sub>Kinetrace guide — [Home](../../README.md) · [Install & run](install.md) · [Workflow: track an animal](workflow.md) · [Segments, silhouettes & skeletons](segments.md) · [Several cameras & 3D](cameras-3d.md) · [Human bodies](bodies.md) · [Export formats](exports.md) · [Projects & autosave](projects.md) · **Keyboard & mouse** · [Accuracy](accuracy.md) · [Performance (4K)](performance.md) · [Limitations](limitations.md) · [Testing](testing.md) · [Folder map & design](folder-map.md) · [User manual](../MANUAL.md) · [Licences](../../THIRD_PARTY_LICENSES.md)</sub>

# Shortcuts

| Key | Action |
|---|---|
| N | Add a point: arms the crosshair — next click places it (drag = a region; Add ▾ picks circle / rectangle / polygon) |
| click (not armed) | Annotate by hand: place the point selected in the list on this frame, replacing the tracker's position (◆ on the timeline) |
| Shift+< / Shift+> | First / last frame the selected point has data on; with nothing selected, the segment's first / last silhouette |
| , / . | Previous / next hand-placed frame of the selected point |
| right-click a point | Rename, appearance lock, data source, hidden-here, *May leave the segment*, delete — plus go to its first / last / first hand-placed / last hand-placed / first doubtful frame; clear its position on this frame, in the selected window, or its whole track; fill the gaps between its hand placements with a smooth curve (or replace everything between them); and, with a calibration, snap it to the other cameras' rays |
| J / Shift+J | Next / previous low-confidence (red) stretch |
| Shift+X | Mark the selected point hidden on this frame: kept, not exported, not used for 3D (again to unmark; timeline selection → right-click marks a window) |
| Shift+N | Note on this frame (green ▲ on the timeline; Edit → Annotator Name records who) |
| O / L | Onion skin (ghosts of the previous / next frame) / loupe (magnifier under the cursor) |
| View → Trails / Display filter | Fading trails with optional upcoming path; contrast / brighten / frame-difference view (display only) |
| S | Segment tool: click the animal (Shift+click = not the animal, drag = box); S again or Esc when done |
| SEGMENT row (right panel) | The tracked segment as its own row: checkbox shows/hides the silhouette, double-click renames, right-click jumps to its first/last silhouette, clears this frame / the selected window / all of them (clicks kept), or removes it |
| Body (toolbar toggle) | Keep tracked points on the segment: a point a few pixels outside the silhouette is nudged back onto it; a landmark that really leaves it **stops the run at that frame** (its track ends on the frame before). Right-click a point → *May leave the segment (free point)* to exempt it |
| Segment ▾ | Pick the segmentation model (SAM 3 / SAM 2.1), with each entry's weight status; last entry opens Settings |
| Ctrl+, | Settings: segmentation model (SAM 3 / SAM 2.1), Hugging Face token, mask opacity — also the last entry of the **Segment ▾** dropdown |
| T | Start tracking / pause (semi-automatic mode: one step) |
| Space | Play / pause preview — pauses tracking while a run is live |
| X | Pause tracking |
| F / B | Frame forward / back (in semi-automatic mode, F *tracks* one frame forward) |
| Shift+F / Shift+B | Jump forward / back by the *step* size (toolbar spinner) |
| H | Pan tool — left-drag moves the view (middle-drag always pans) |
| + / − (or =) | Zoom the video in / out around the pointer |
| Shift+ + / Shift+ − | Zoom the timeline's time axis (also Ctrl+wheel over it) |
| ← / → (Shift: ±step) | Step frames |
| Home / End | First / last frame |
| E | Mark event start / end at the current frame |
| R | Reset view (fit video to window) |
| Shift+C | Clear the frame cache and re-decode the current frame (failsafe for a stale/garbled picture) |
| Shift+drag (timeline) | Select a frame window **and the lanes under the drag** — segment lane, point lanes, or both |
| Delete | Clear the timeline selection (the lanes it covers decide: silhouettes, those points' tracks, or both) — or, with no selection, delete the selected point(s) |
| Esc | In this order: leave the segment tool, cancel point placement (or a half-drawn region), cancel a half-marked event, clear the timeline selection, deselect the point |
| Ctrl+Z | Undo the last tracking run or the last edit (a click, drag, Ctrl+click, Delete, Shift+X, a timeline clear) |
| Ctrl+2 | Show only the working camera (hides the others and stops them decoding) |
| Ctrl+1 | Show / hide the Segment & Points panel |
| Ctrl+3 / Ctrl+4 / Ctrl+5 | Reconstruct 3D landmarks / carve the volume at this frame / show the 3D view |
| Ctrl+6 | Body: the side-by-side view |
| Enter / double-click | Close a polygon region |
| F1 | The user manual (**Help → Keyboard & Mouse Reference** lists every key) |
| Ctrl+O / Ctrl+Shift+O | Open video / project |
| Ctrl+S / Ctrl+Shift+S | Save project / save as |
| Ctrl+E | Export tracks |
| File → Export Overlay Video… | An MP4 with markers, names, skeleton, silhouette, trails, frame counter, events and notes drawn on it |

---

[← Projects & autosave](projects.md) &nbsp;|&nbsp; [Contents](../../README.md#contents) &nbsp;|&nbsp; [Accuracy →](accuracy.md)
