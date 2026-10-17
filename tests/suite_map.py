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
        "verify_save_during_run"]
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
    # (I303) the wand wizard's accept path and its one Ctrl+Z live in app.py
    ([K + "app.py"], ["verify_wand_subset", "verify_wand_sync_gui"]),
    # (I305) the automatic 3D result follows every 2D gesture; the saved result's format
    ([K + f for f in ("app.py", "projectfile.py", "session.py")], ["verify_auto3d"]),
    ([K + "threads.py"], ["verify_sweep_fixes", "verify_review_infra", "verify_review_tracking", "verify_review_ui",
                           "verify_review_app3", "verify_downloads"]),
    ([K + "tour.py", K + "demo.py"], ["verify_tour", "verify_ui_focus", "verify_keys_follow"]),
    ([K + "layers.py"], ["verify_layers", "verify_segment_panel", "verify_segment_targets", "verify_animal_gui"]),
    ([K + "project.py", K + "folderimport.py", K + "cameraorder.py"],
     ["verify_multicam", "verify_shared_landmarks", "verify_multiview", "verify_camera_views", "verify_projectfile",
      "verify_3d_gui", "verify_open_replace"]),
    ([K + "cameraoverview.py"], ["verify_camera_overview", "verify_camera_views"]),
    ([K + f for f in ("projectfile.py", "recovery.py", "csvpool.py", "autoexport.py")], SAVE),
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
    ([K + f for f in ("wand.py", "wanddata.py", "calibwizard.py", "wandsync.py")],
     ["verify_wand", "verify_wand_gui", "verify_review_calib", "verify_wand_subset", "verify_calib3d",
      # (I271, I300-I304, G199, G200) camera timing from the wand
      "verify_wand_sync", "verify_wand_sync_gui"]),
    ([K + f for f in ("lens.py", "lenswizard.py", "lensattach.py", "boardreview.py")],
     ["verify_lens", "verify_lens_gui", "verify_lens_export", "verify_lens_turn", "verify_lens_attach",
      "verify_review_calib", "verify_camera_overview"]),
    ([K + "gpmf.py"], ["verify_gopro", "verify_lens_turn"]),
    ([K + f for f in ("sync.py", "audiosync.py", "syncdialog.py")], ["verify_sync"]),
    ([K + f for f in ("body.py", "bodypose.py", "bodyview.py")], ["verify_body", "verify_body_gui", "verify_review_body"]),
    ([K + "render.py"], ["verify_render"]),
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
