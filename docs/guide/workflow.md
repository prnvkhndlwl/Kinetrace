<sub>Kinetrace guide — [Home](../../README.md) · [Install & run](install.md) · **Workflow: track an animal** · [Segments, silhouettes & skeletons](segments.md) · [Several cameras & 3D](cameras-3d.md) · [Human bodies](bodies.md) · [Export formats](exports.md) · [Projects & autosave](projects.md) · [Keyboard & mouse](shortcuts.md) · [Accuracy](accuracy.md) · [Performance (4K)](performance.md) · [Limitations](limitations.md) · [Testing](testing.md) · [Folder map & design](folder-map.md) · [User manual](../MANUAL.md) · [Licences](../../THIRD_PARTY_LICENSES.md)</sub>

# Workflow

1. **Open a video** (Ctrl+O) — any common format/codec via OpenCV/FFmpeg.
2. Press **N** (or the **Add** button in the control bar under the timeline; on
   a narrow window its buttons show only their icons — hover one for its name)
   — the cursor becomes a crosshair —
   then **click** the thing you want to track. One press, one placement:
   stray clicks never add or move anything. Points are listed in the right
   panel (double-click to rename, checkbox to show/hide, right-click for
   rename/appearance-lock/delete; Ctrl/Shift+click selects several).
   **Or, while armed, drag a circle** around an object instead: the region is
   tracked as a cloud of internal points and exported as **one** point — its
   robustly fitted center. Regions survive rotation, partial occlusion, and background
   changes far better than a single point (draw the circle to match the
   object, not larger — the tracked points are sampled *inside* it).
   **Balls** — wand balls, reflective markers, a dropped ball — get **Add ▾ →
   Ball marker (SAM circle)** instead: click the ball, and on every frame SAM
   outlines it, a circle is fitted and its centre is the point (balls about
   5 px across still work). Balls tracked in one run share one 960-px window:
   if two are more than about 740 px apart the run stops and says to track
   each ball on its own (select it alone in the POINTS list, then Track).
   **Segment the animal (optional):** press **S** (or the **Segment** button) and **click the animal**.
   Nothing waits for this step — points alone track fine, and for an animal
   only a few pixels across (a small fish or a bird seen from far away) **skip it**:
   there is no silhouette at that size and the model grabs something bigger.
   Place a point with **N** and Track instead; the ROI zoom gives the tracker a
   close-up around it. When you do segment, the
   silhouette appears within a second (the first click loads the model). If it
   grabbed too much, **Shift+click** the wrong part; if it missed a part, click
   it; or **drag a box** around the animal. Right-click a click marker to remove
   it. The **▾ arrow on the Segment button** picks which segmentation model does
   this (SAM 3 / SAM 2.1), each entry showing whether its weights are ready,
   need downloading, or are gated — the same choice as in Settings.
   The silhouette is always the whole animal: that is what the segmentation
   model answers with, and it is the reliable way to use it. For a single body
   part — one wing, one leg — track named points on it instead.
   Then choose **Skeleton ▾** in the right panel: a named landmark set for
   your study animal (lizard/iguana, quadruped, gliding mammal, flying lizard,
   undulating body, bird/bat, or your own). Landmarks marked with a dotted square
   are *derived from the silhouette* (tail tip, midline %, feet, wing tips) and
   need no clicking; the others you place: select one, press **N**, click it.
   Place the head landmark first — it anchors the midline.
3. Press **Track ▶** (or **T**). Tracking runs forward *frame by frame, visibly* —
   the progress bar shows position, the status bar shows speed and ETA, and
   the **timeline panel** fills in live. (**Space** only plays the video; during
   a run it pauses.)
   The Track button's dropdown also picks the **point model**: **AllTracker**
   (the default; the installer fetches it) or **CoTracker3**. AllTracker tracks every point from the frame you
   started on with a high-resolution dense correlation, so it does not
   accumulate drift along a uniform body — measured on a real lizard clip it keeps
   head and hip on the animal for 270 frames where CoTracker3 slides off, and it
   halves the error on a small textured feature. CoTracker3 is about twice as
   fast and slightly sharper on high-contrast markers (0.74 px at 4K after
   refinement). Both get the same silhouette constraint and sub-pixel refinement.
   The Track button's dropdown offers two modes:
   - **Automatic** (default): run to the end of the video.
   - **Semi-automatic**: each press of **F** tracks *one* frame forward and
     pauses. Check the result, fix anything, press **F** again — every step
     re-seeds from what you see, so you supervise the track frame by frame.
     (Where no point has data, F just moves forward as usual; B always just
     steps back.)
4. **Pause any time**: press **X**, Space, or the Track button (it reads
   **Pause ■ (X)** during a run). It stops within a fraction of a second. With
   **Auto-pause** on (default), the app also stops *itself* when it loses a
   point (low confidence for 16 frames in a row): it jumps back to the first
   unreliable frame, selects the point, and **cuts that point's track there** —
   what the run wrote for it from that frame on is cleared, so its data ends
   where it stopped being trustworthy. Ordinary occlusion does not trigger it.
   With a segment and the **Body** toggle on, a landmark that **leaves the
   silhouette** stops the run at that frame as well (Auto-pause on or off): its
   track ends on the frame before. Click it where it really is and press Track
   again.
5. **Review & correct**: scrub by **clicking or dragging anywhere on the
   timeline panel** — it is the scrubber, and each point's lane shows exactly
   which frames are tracked (red tint = the model was unsure there). Drag the
   **splitter above the panel** to make it taller (more point lanes; the
   wheel scrolls any that still overflow). **Shift + +/− (or Ctrl+wheel over
   the panel) zoom the time axis** around the cursor, Premiere-style: the
   ruler relabels itself at sensible 1/2/5-step intervals as you zoom, and a
   **scroll strip appears along the panel's bottom edge** showing where the
   view window sits in the whole video — click or drag it (or middle-drag
   the lanes) to move around; the view also follows the playhead. The zoom
   buttons next to the play controls do the same (the third shows the whole
   video again). Hotkeys work wherever the keyboard focus is, except while you
   type a name. The full reference lives in **Help → Keyboard & Mouse Reference**. Use **F/B** keys (frame forward/back; hold Shift to
   jump by the *step* size in the control panel), arrow keys, or the frame
   box (type a number + Enter).
   Zoom with the mouse wheel or **+/−** (anchored under the pointer); pan
   with middle-drag or the **✋ Pan** tool (**H**) for left-drag panning.
   The view never moves on its own unless you switch **⌖ Follow** on (it is
   off by default): then it keeps tracked points in sight while zoomed in,
   following the *selected* point (holding still across gaps in its track),
   or — with nothing selected — re-framing **all** points as they move. **R**
   always fits the whole frame. If a
   marker hides the exact spot it sits on (a ball close to the camera, or
   zoomed way in), shrink the **marker px** size in the control panel —
   markers keep a fixed on-screen size by design, but you pick that size.
   Fix any point on any frame:
   - **drag** the point marker, or
   - select it, then **Ctrl+click** where it should be, or
   - select it in the POINTS list and simply **click** the video (no N): it is
     placed there on this frame by hand (◆ on the timeline) — the way to
     digitize frames the tracker gets wrong.
   Each of these is one undo step (Ctrl+Z). A landmark derived from the
   silhouette cannot be placed by hand: switch it first with right-click →
   **Data source → Track by appearance**.
6. Press **Track** again: the points **selected in the right panel** re-seed from
   their positions on the current frame (your corrections included) and re-track
   forward, overwriting later frames. Select all, or nothing, to track every point;
   Ctrl/Shift+click to pick several. `Ctrl+Z` (**Edit → Undo Last Run / Edit**) undoes the whole last tracking run — or the last hand edit — if needed.
   **Bulk cleanup:** to wipe a bad stretch, **Shift+drag on the timeline**
   across the frames *and* the lanes you want gone, then press **Delete**.
   The drag is a marquee in both directions, so it picks the target for you:
   drag along the **segment lane** to drop just the **silhouettes** in that
   stretch, along one or more **point lanes** to drop just **their** tracked
   data, or across both to drop both in a single undoable step. The band
   labels what Delete will clear, and right-clicking inside it offers the same
   choices explicitly. Dragging in the **ruler** (or right-click an event →
   *Select this window*) selects the frames without naming lanes: Delete then
   falls back to the points selected in the right panel, asking first if none
   are. Either way the points themselves stay — only that stretch of their
   data goes. Delete with points (and no frame window) selected removes those
   points entirely. All of it is undoable with Ctrl+Z.
7. **Mark events** as you go: press **E** at a frame of interest, scrub, press
   **E** again — the name dialog offers your **existing event types in a
   dropdown**, so tagging another occurrence of "swing" is one click (type a
   new name to create a new type). All occurrences of a type share one color
   on the timeline, and the Events menu groups them ("swing (3×)" opens the
   list of its occurrences). Click a ribbon to select its window and jump
   there; right-click to rename, retime, or delete. Events are saved in the
   project and included in exports.
8. **Export** (Ctrl+E) when happy.

---

[← Install & run](install.md) &nbsp;|&nbsp; [Contents](../../README.md#contents) &nbsp;|&nbsp; [Segments, silhouettes & skeletons →](segments.md)
