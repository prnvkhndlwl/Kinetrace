<sub>Kinetrace guide — [Home](../../README.md) · [Install & run](install.md) · [Workflow: track an animal](workflow.md) · **Segments, silhouettes & skeletons** · [Several cameras & 3D](cameras-3d.md) · [Human bodies](bodies.md) · [Export formats](exports.md) · [Projects & autosave](projects.md) · [Keyboard & mouse](shortcuts.md) · [Accuracy](accuracy.md) · [Performance (4K)](performance.md) · [Limitations](limitations.md) · [Testing](testing.md) · [Folder map & design](folder-map.md) · [User manual](../MANUAL.md) · [Licences](../../THIRD_PARTY_LICENSES.md)</sub>

# Segments, silhouettes and skeletons

Point trackers hold textured features (an eye, a snout, a marker) but lose thin
undulating tails and dark textureless bodies — measured on real 4K footage.
The app therefore tracks the **animal as a whole** too:

- **One click defines the animal.** SAM 3 (or SAM 2.1) segments it on that frame
  and, during tracking, on every frame — following it through occlusion with a
  memory of what it looks like. The crop that the point tracker works in follows
  the silhouette, and extra hidden support points sampled inside the silhouette
  anchor the joint tracking to the body.
- **Silhouette landmarks** are geometry, not appearance: the body midline
  (anchored at your head landmark), the tail tip (end of the midline), points at
  25/50/75 % of body length, the centroid, and extremities (the farthest
  protrusions on each side — feet, wing or patagium tips). They are ordinary
  points in the export. Left and right are named **as seen from above** (a
  dorsal view): filmed from below, the animal's left and right swap, so rename
  those landmarks (or swap their rules). Exports made before 2026-09-22 from
  dorsal footage had them the wrong way round.
- **Skeletons** give every landmark a fixed name so exports line up across
  videos and, later, across cameras. Built-in templates cover lizards, generic
  quadrupeds, gliding mammals, flying lizards, undulating bodies and birds/bats;
  *Custom skeleton…* saves yours as JSON in `skeletons/`. Right-click any point
  → **Data source** to switch it between appearance tracking and a silhouette
  rule (it asks before erasing the point's existing track; Ctrl+Z brings it
  back).
- **Points stay on the animal.** With the **Body** toggle on (default), a
  tracked point the model puts just outside the silhouette (within about 3 % of
  the silhouette's diagonal, at least 8 px) is nudged back onto it — that is
  outline jitter. A landmark that goes further has **left** the animal: the run
  **stops at that frame**, its track ends on the frame before, and the app
  takes you there. Pulling it back onto the nearest silhouette pixel would put
  it on some other spot of the body and carry on as if nothing had happened.
  Click it where it really is and press Track. Exempt a point that lives
  elsewhere (a ground marker) with right-click → *May leave the segment (free
  point)*.
- **Corrections are clicks.** Pause, go to the frame, press **S**, click (or
  Shift+click), press **Track**: the correction applies from that frame on. A
  landmark that leaves the silhouette stops the run (above); one that sits off
  the body at low confidence turns red on the timeline; and with Auto-pause on
  the run also stops when a point stays unreliable, or when the animal itself
  is lost or leaves the frame.
- **Segment ▾** picks the segmentation model right on the button — the same way
  **Track ▾** picks the point model — and each entry says whether its weights are
  ready, need downloading (with the size), or are gated behind a token. SAM 3 is
  the default when its weights are in `models\sam3` (best masks, ~13 fps at 4K);
  SAM 2.1 base+ is the fast fallback (~27 fps). Robustness was chosen over speed
  by design. **Settings** (Ctrl+,) holds the same choice plus the Hugging Face
  token and the mask opacity; the last menu entry opens it.

A step-by-step guide for new users is in [docs/MANUAL.md](../../docs/MANUAL.md)
(also in the app: **F1**).

---

[← Workflow: track an animal](workflow.md) &nbsp;|&nbsp; [Contents](../../README.md#contents) &nbsp;|&nbsp; [Several cameras & 3D →](cameras-3d.md)
