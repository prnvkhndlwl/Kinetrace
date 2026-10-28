"""Which test suites to run for a change: changed file -> suites (`tests/run_suites.py --affected`).

One table, in the order a reader looks things up. A module that is not listed falls back to FALLBACK and is
reported, and `verify_suite_map` fails until it is added - so a new module cannot slip past its tests.
Then still run both groups (`--cpu`, `--gpu` with an NVIDIA GPU) once before a pull request.
"""
import fnmatch
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GUI = ["verify_gui", "verify_oob", "verify_keys_follow", "verify_timeline_events", "verify_semiauto_pan", "verify_stress",
       "verify_annotate", "verify_ui_focus", "verify_display", "verify_segment_panel", "verify_point_menu",
       "verify_sweep_fixes", "verify_review_app1", "verify_review_app2", "verify_review_app3", "verify_review_leftovers",
       "verify_review_ui", "verify_camera_views", "verify_tour"]
SAVE = ["verify_projectfile", "verify_recovery", "verify_loading", "verify_open_replace", "verify_review_persist",
        "verify_save_during_run", "verify_save_safety"]
INSTALL = ["verify_portable", "verify_install_paths", "verify_update", "verify_downloads", "verify_crashlog",
           "verify_welcome", "verify_winapp", "verify_review_infra"]
FALLBACK = ["verify_core", "verify_stress", "verify_sweep_fixes"]

K = "kinetrace/"
MAP = [
    # the tracking engine
    ([K + "tracker.py"], ["verify_tracker", "verify_groups", "verify_conf_autopause", "verify_roi", "verify_4k",
                          "verify_track_all", "verify_track_selection", "verify_review_tracking", "verify_review_i185",
                          "verify_long", "verify_oob",
                          # (sweep area 3) they drive TrackingWorker too; verify_fingerprint [6b] its decode probe
                          "verify_balls", "verify_spots", "verify_segments", "verify_segment_identity",
                          "verify_fingerprint",
                          # (I293) the derived-landmark rules on a bare worker
                          "verify_silhouette_loops", "verify_loop_track"]),
    # (I335-I339) ROI restarts with a hidden point / a turned region, lost-point blanking, the re-anchor, the
    # first pause (a run's start and end in the window, the region's pose on a resumed Track)
    ([K + f for f in ("tracker.py", "app.py", "session.py", "alltracker_backend.py")], ["verify_tracker_restarts"]),
    # (R37, R39) pixel-centre arithmetic: every user's suites
    ([K + "pixels.py"], ["verify_tracker", "verify_review_tracking", "verify_4k", "verify_roi", "verify_balls",
                         "verify_segmenter", "verify_animal", "verify_animal_gui", "verify_segments_sam",
                         "verify_segment_identity", "verify_onbody_rules", "verify_fingerprint",
                         "verify_decode_identity"]),
    ([K + "alltracker_backend.py"], ["verify_alltracker", "verify_alltracker_mps", "verify_portable", "verify_tracker"]),
    ([K + "spots.py", K + "pointtest.py", K + "pointedit.py"], ["verify_spots", "verify_point_tools"]),
    ([K + "balls.py"], ["verify_balls"]),
    ([K + "segmenter.py", K + "silhouette.py"], ["verify_segmenter", "verify_animal", "verify_animal_gui",
                                                 "verify_segments_sam", "verify_segment_identity",
                                                 "verify_onbody_rules", "verify_silhouette_loops",
                                                 "verify_loop_track"]),
    ([K + "skeletons.py"], ["verify_animal", "verify_layers", "verify_shared_landmarks"]),
    # (I287) the one Hugging Face token rule: the suites that check which token reaches a model load
    ([K + "segmenter.py", K + "downloads.py", K + "bodypose.py"], ["verify_review_tracking", "verify_review_leftovers",
                                                                   "verify_review_body"]),
    # the data model, the window
    ([K + "session.py"], ["verify_core", "verify_gui", "verify_oob", "verify_stress", "verify_layers", "verify_segments",
                          "verify_segment_targets", "verify_projectfile", "verify_annotate", "verify_review_data",
                          "verify_shared_landmarks", "verify_interop"]),
    ([K + f for f in ("app.py", "canvas.py", "timeline.py", "widgets.py", "viewgrid.py", "camerapanel.py", "theme.py",
                      "icons.py")], GUI),
    # (I320, I316, I319, G217, G218) hand-placed frames are the truth: every writer of a point's cells, the runs
    # that keep and re-seed from them, the imports that mark them, the Track pop-up
    ([K + f for f in ("session.py", "tracker.py", "spots.py", "app.py", "projectfile.py", "trackio.py",
                      "dltdvproject.py", "dltdvimport.py", "canvas.py", "timeline.py", "trackchoice.py")],
     ["verify_hand_truth", "verify_hand_truth_models"]),
    # (I316, I318, PR B) a re-seed session per click for the point models: the run driver, the frames it reads
    # from the cache, the app's plan of the sessions and their writes
    ([K + f for f in ("tracker.py", "video_source.py", "session.py", "app.py", "spots.py")], ["verify_hand_restart"]),
    # (D9 choice 3, G217) Only fill the gaps: the gaps' plan, the pop-up's third choice, the runs over the gaps
    ([K + f for f in ("session.py", "app.py", "trackchoice.py", "tracker.py", "video_source.py")], ["verify_gap_fill"]),
    # (G216, G219) a tracker per camera and "Hand only": the Tracker menu, the passes of a press over every
    # camera's trackers, what a press leaves alone (pop-up counts, Fresh erase, 3D re-track, hints, DLC scorer)
    ([K + f for f in ("app.py", "session.py", "projectfile.py", "project.py", "layers.py", "retrack.py",
                      "tracker.py", "trackchoice.py", "pointtest.py")],
     ["verify_tracker_per_camera", "verify_tracker_per_camera_models"]),
    # (G215) Manual mode: no tracker, a click lands where it was made and moves the playhead on
    ([K + f for f in ("app.py", "canvas.py", "session.py")], ["verify_manual_mode"]),
    # (G217) the Track pop-up replaced the "Re-track from here?" question (G101's check)
    ([K + "trackchoice.py"], ["verify_review_app2"]),
    # (I303) the wand wizard's accept path and its one Ctrl+Z live in app.py
    ([K + "app.py"], ["verify_wand_subset", "verify_wand_sync_gui"]),
    # (I305) the automatic 3D result follows every 2D gesture; the saved result's format
    ([K + f for f in ("app.py", "projectfile.py", "session.py")], ["verify_auto3d"]),
    # (I309, G206, G207) the 3D window during a run, why a point has no 3D, the camera rays
    ([K + f for f in ("app.py", "view3d.py", "calib.py")], ["verify_live3d"]),
    ([K + "threads.py"], ["verify_sweep_fixes", "verify_review_infra", "verify_review_tracking", "verify_review_ui",
                           "verify_review_app3", "verify_downloads"]),
    ([K + "tour.py", K + "demo.py"], ["verify_tour", "verify_ui_focus", "verify_keys_follow"]),
    ([K + "layers.py"], ["verify_layers", "verify_segment_panel", "verify_segment_targets", "verify_animal_gui"]),
    ([K + "project.py", K + "folderimport.py", K + "cameraorder.py"],
     ["verify_multicam", "verify_shared_landmarks", "verify_multiview", "verify_camera_views", "verify_projectfile",
      "verify_3d_gui", "verify_open_replace"]),
    # (G205) the recent-projects list: its logic, the File menu entry and the start-screen panel
    ([K + "recent.py", K + "app.py", K + "canvas.py"], ["verify_recent"]),
    ([K + "cameraoverview.py"], ["verify_camera_overview", "verify_camera_views"]),
    ([K + f for f in ("projectfile.py", "recovery.py", "csvpool.py", "autoexport.py")], SAVE),
    # (I326-I334) the open / autosave / end-of-run / close paths, the app's own cuts, the shortcut's icon
    ([K + f for f in ("app.py", "session.py", "winapp.py", "macapp.py")], ["verify_save_safety"]),
    ([K + "paths.py"], SAVE),            # (R43) every save / autosave replaces through paths.replace_retry
    # video
    ([K + "video_source.py", K + "framecheck.py", K + "fingerprint.py"],
     ["verify_core", "verify_scrub", "verify_fingerprint", "verify_fingerprint_gui", "verify_decode_identity",
      "verify_long"]),
    # cameras, calibration, 3D, body
    ([K + f for f in ("calib.py", "hull.py", "view3d.py", "retrack.py", "kinematics.py", "calibview.py")],
     ["verify_3d", "verify_3d_gui", "verify_retrack", "verify_review_3d", "verify_multicam",
      "verify_wand_subset", "verify_calib3d", "verify_auto3d"]),
    # (I302) estimate_offsets' verdict and its robust search; (G200) the picture the wand wizard opens
    ([K + "calib.py"], ["verify_wand_sync", "verify_review_app3"]),
    ([K + f for f in ("view3d.py", "calibview.py")], ["verify_wand_sync_gui"]),
    ([K + f for f in ("calibio.py", "trackio.py", "convert.py")], ["verify_interop", "verify_review_data",
                                                                  "verify_wand_subset"]),
    # (I310, G211) File > Import > DLTdv Project: the reader, the click-shift check, the dialogs, the app wiring,
    # and the calibration / track writers it goes through
    ([K + f for f in ("dltdvproject.py", "clickshift.py", "dltdvimport.py")],
     ["verify_dltdv_project", "verify_interop", "verify_3d"]),
    ([K + f for f in ("app.py", "calib.py", "calibio.py", "trackio.py", "project.py")], ["verify_dltdv_project"]),
    ([K + f for f in ("wand.py", "wanddata.py", "calibwizard.py", "wandsync.py")],
     ["verify_wand", "verify_wand_gui", "verify_review_calib", "verify_wand_subset", "verify_calib3d",
      # (I271, I300-I304, G199, G200) camera timing from the wand
      "verify_wand_sync", "verify_wand_sync_gui"]),
    ([K + f for f in ("lens.py", "lenswizard.py", "lensattach.py", "boardreview.py")],
     ["verify_lens", "verify_lens_gui", "verify_lens_export", "verify_lens_turn", "verify_lens_attach",
      "verify_review_calib", "verify_camera_overview"]),
    # (I321-I323) ChArUco boards: the board and its frame rules, the lens fit from them, the camera solve
    ([K + "charuco.py"], ["verify_charuco", "verify_charuco_gui", "verify_lens", "verify_lens_gui", "verify_wand",
                          "verify_wand_gui"]),
    ([K + f for f in ("lens.py", "calibwizard.py", "wand.py", "calibio.py")], ["verify_charuco"]),
    # (G220, G221) the board's boxes, the two wizards and the menu entries that open them
    ([K + f for f in ("charucoui.py", "lenswizard.py", "boardreview.py", "calibwizard.py", "app.py")],
     ["verify_charuco_gui"]),
    ([K + "charucowizard.py"], ["verify_charuco", "verify_charuco_gui"]),
    ([K + "gpmf.py"], ["verify_gopro", "verify_lens_turn"]),
    ([K + f for f in ("sync.py", "audiosync.py", "syncdialog.py")], ["verify_sync"]),
    ([K + f for f in ("body.py", "bodypose.py", "bodyview.py")], ["verify_body", "verify_body_gui", "verify_review_body"]),
    ([K + "render.py"], ["verify_render"]),
    # (I357-I361) one pixel-centre rule: SAM prompts, the board editor / search, body + overlay drawing, the
    # in-picture rule (session, retrack, canvas clicks, imports)
    ([K + f for f in ("segmenter.py", "boardreview.py", "lens.py", "bodyview.py", "render.py", "session.py",
                      "retrack.py", "canvas.py", "trackio.py", "pixels.py", "app.py", "calibview.py")],
     ["verify_pixel_centre"]),
    # install, update, device, errors
    ([K + f for f in ("device.py", "downloads.py", "paths.py", "update.py", "updatedialog.py", "crashlog.py",
                      "errors.py", "welcome.py", "macapp.py", "winapp.py", "winlaunch.py", "appicon.py",
                      "__init__.py", "__main__.py")]
     + ["install.py", "run.bat", "run.sh", "update.bat", "update.sh", "Update.command", "Kinetrace.command",
        "uninstall.sh", "requirements.txt"], INSTALL),
    # docs the app renders, the test machinery
    (["docs/MANUAL.md"], ["verify_timeline_events"]),
    (["make_test_video.py"], ["verify_core", "verify_4k"]),
    (["tests/run_suites.py", "tests/suite_map.py"], ["verify_suite_map"]),
]
# Added on top of MAP, never instead of it (a module listed only here would still be reported as unmapped):
# any module's messages and the manual may name a menu entry ("3D → Reconstruct 3D Landmarks"), and
# verify_menu_paths checks every such path against the live menu bar (G188)
ALSO = [([K + "*.py", "docs/MANUAL.md"], ["verify_menu_paths"])]


def _helper_users(helper):
    """The suites that import a tests/_helper.py module, directly or through other helpers."""
    tests = os.path.join(ROOT, "tests")
    src = {f[:-3]: open(os.path.join(tests, f), encoding="utf-8", errors="replace").read()
           for f in sorted(os.listdir(tests)) if f.endswith(".py")}
    todo, seen = [os.path.splitext(os.path.basename(helper))[0]], set()
    while todo:                                   # helpers that import this helper count as it
        stem = todo.pop()
        seen.add(stem)
        pat = re.compile(rf"\b(import|from)\s+{re.escape(stem)}\b")
        todo += [m for m, text in src.items() if m.startswith("_") and m not in seen and pat.search(text)]
    pats = [re.compile(rf"\b(import|from)\s+{re.escape(s)}\b") for s in seen]
    return [m for m, text in src.items() if m.startswith("verify_") and any(p.search(text) for p in pats)]


def suites_for(paths):
    """(suites in a stable order, unmapped code paths) for a list of repo-relative changed paths."""
    chosen, unmapped = [], []

    def add(names):
        for n in names:
            if n not in chosen:
                chosen.append(n)

    for p in paths:
        p = p.replace("\\", "/")
        hit = [suites for pats, suites in MAP if any(fnmatch.fnmatch(p, g) for g in pats)]
        for suites in hit:
            add(suites)
        also = [suites for pats, suites in ALSO if any(fnmatch.fnmatch(p, g) for g in pats)]
        if hit:
            for suites in also:
                add(suites)
            continue
        base = os.path.basename(p)
        if p.startswith("tests/") and base.startswith("verify_") and base.endswith(".py"):
            add([base[:-3]])
        elif p.startswith("tests/_") and base.endswith(".py"):
            add(_helper_users(p))
        elif p.endswith(".py") and (p.startswith(K) or "/" not in p):
            unmapped.append(p)
            add(FALLBACK)
        for suites in also:
            add(suites)
    return chosen, unmapped
