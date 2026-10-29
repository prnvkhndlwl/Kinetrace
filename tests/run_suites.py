"""Run a list of verification suites one after another and record, for each,
its summary line (summary_line) and its exit code, in tests/out/suite_runs.txt. ASCII
only (Windows consoles are cp1252). Usage:

    .venv\\Scripts\\python.exe tests\\run_suites.py verify_core verify_3d ...
    .venv\\Scripts\\python.exe tests\\run_suites.py --gpu        (the GPU group)
    .venv\\Scripts\\python.exe tests\\run_suites.py --cpu        (offscreen GUI + core)
    .venv/bin/python tests/run_suites.py --cpu                  (Linux / macOS)
    .venv\\Scripts\\python.exe tests\\run_suites.py --affected [base] (the suites for the files changed since
                                                         base, default main: tests/suite_map.py)
    .venv\\Scripts\\python.exe tests\\run_suites.py --stamp      (the evidence block for a pull request)

Each run's header names the commit it ran on ("HEAD <sha>", "dirty" when tracked files had uncommitted edits),
so --stamp can say which results belong to the code in the pull request.

Every suite keeps its unsaved-work recovery copies in tests/out/recovery
(KINETRACE_RECOVERY_DIR), emptied before each suite, so no later suite meets a
"restore unsaved work?" question.
"""
import os
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(errors="replace")      # a suite's last line may carry an arrow or a dash
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable                           # the venv's interpreter on any platform
OUT = os.path.join(ROOT, "tests", "out")
os.makedirs(OUT, exist_ok=True)
RECOVERY = os.path.join(OUT, "recovery")

GPU = ["audit_sweep", "verify_balls", "verify_animal", "verify_tracker", "verify_groups", "verify_conf_autopause",
       "verify_roi", "verify_4k", "verify_segmenter", "verify_animal_gui", "verify_gui", "verify_oob",
       "verify_keys_follow", "verify_semiauto_pan", "verify_alltracker", "verify_long", "verify_retrack",
       "verify_track_all", "verify_track_selection",
       # the 2026-10-02 code review's regression checks that need the GPU
       "verify_review_tracking", "verify_review_app2", "verify_review_i185",
       # two segments with the real segmentation model in one pass (G149)
       "verify_segments_sam",
       # (I293) a real Track run over a self-contact loop
       "verify_loop_track",
       # (I320, G217) hand-placed frames are the truth with the point models; Fresh run over two passes
       "verify_hand_truth_models",
       # (G216, G219) a tracker per camera (AllTracker / CoTracker3 in one every-camera press) and Hand only
       "verify_tracker_per_camera_models",
       # (I316, I318, PR B) AllTracker / CoTracker3 points start again at the clicks: a session per restart
       "verify_hand_restart",
       # (D9 choice 3, G217, PR B) Only fill the gaps: nothing that exists is touched
       "verify_gap_fill",
       # (I335-I339) restarts with a hidden point / a turned region, blanking, re-anchor, the first pause
       "verify_tracker_restarts"]
# verify_pythonw is in no group on purpose: it must be launched DETACHED (Start-Process,
# no output redirect) or the stderr guard it checks cannot fail; see CLAUDE.md
CPU = ["verify_portable", "verify_core", "verify_projectfile", "verify_recovery", "verify_interop", "verify_shared_landmarks", "verify_multiview", "verify_loading", "verify_stress", "verify_3d", "verify_wand", "verify_lens", "verify_body",
       "verify_onbody_rules", "verify_silhouette_loops", "verify_sync",
       "verify_timeline_events", "verify_menu_paths", "verify_scrub", "verify_multicam", "verify_3d_gui", "verify_ui_focus",
       "verify_annotate", "verify_display", "verify_render", "verify_segment_panel", "verify_point_menu",
       "verify_wand_gui", "verify_wand_subset", "verify_calib3d", "verify_auto3d", "verify_live3d", "verify_lens_gui", "verify_body_gui", "verify_sweep_fixes", "verify_crashlog", "verify_update", "verify_downloads",
       "verify_spots", "verify_alltracker_mps", "verify_install_paths", "verify_winapp",
       # the 2026-10-02 code review's regression checks (each fails on the code before its fixes)
       "verify_review_persist", "verify_review_data", "verify_review_3d", "verify_review_calib",
       "verify_review_body", "verify_review_infra", "verify_review_ui", "verify_review_app1", "verify_review_app3",
       "verify_review_leftovers",
       # owner reports 2026-10-03: SAM taking another animal (I260), the open question (G143)
       "verify_segment_identity", "verify_open_replace",
       # (G205) File -> Open Recent and the start-screen list
       "verify_recent",
       # GoPro footage (G144-G146) and the point tools (G147)
       "verify_gopro", "verify_point_tools", "verify_lens_export",
       # (I269) a lens profile on a camera filmed on its side
       "verify_lens_turn",
       # (G176) one lens file on several cameras
       "verify_lens_attach",
       # (I357-I361) one pixel-centre rule
       "verify_pixel_centre",
       # (I378) the point models' map between the native and the shrunk picture
       "verify_point_model_scale",
       # (I363) a checkerboard view with two corners on one spot is flagged and never auto-picked
       "verify_board_merged",
       # (I368-I377, G226-G231) sweep PR 7: the window's behaviour rows
       "verify_ui_behaviour",
       # (I372) the end-of-run 3D in the background; recovery copies make their 3D again when opened
       "verify_recovery_3d",
       # (I310, G211) File > Import > DLTdv Project
       "verify_dltdv_project",
       # (I321-I323, G220, G221) ChArUco boards: lens and camera calibration, Argus files
       "verify_charuco", "verify_charuco_gui",
       # (G177) the cameras at a glance: lens badges, Cameras Overview
       "verify_camera_overview",
       # (G178, X37) after an install: the closing words, the one-time pop-up, the Mac hand-over
       "verify_welcome",
       # (G183) the interface tour: the first-start offer, the spotlight, the demo, nothing of the user's changed
       "verify_tour",
       # any number of segments (G149) and which segment each action takes (G150, G151)
       "verify_segments", "verify_segment_targets",
       # animal layers, the two zooms (G152-G160)
       "verify_layers",
       # (G168-G172) many cameras on screen and their order; (I265) Save during a run
       "verify_camera_views", "verify_save_during_run",
       # (I326-I334) save / open / recovery safety (the 2026-10-25 sweep's PR 1)
       "verify_save_safety",
       # (I340-I356, G222-G224) calibration, 3D and sync fixes (the 2026-10-25 sweep's PR 3)
       "verify_calib_sync_fixes", "verify_calib_sync_gui",
       # (I266) a video that decodes differently on another computer: the fingerprint, exact seeks
       "verify_fingerprint", "verify_fingerprint_gui", "verify_decode_identity",
       # (I271, I300-I304, G199, G200) camera timing from the wand: refine, checks, pending offsets, the picture
       "verify_wand_sync", "verify_wand_sync_gui",
       # (I320, I316, I317, I319, G217, G218) hand-placed frames are the truth: the rules, imported clicks, Moving
       # spot re-seeds, clicks on silhouette landmarks, the Track pop-up
       "verify_hand_truth",
       # (G216, G219) a tracker per camera and Hand only: the menus, the passes over every camera, a stand-in model
       "verify_tracker_per_camera",
       # (G215) Manual mode: no tracker, clicks frame by frame
       "verify_manual_mode",
       # the test-selection table and the pull-request evidence stamp
       "verify_suite_map"]
# the synthetic test videos are generated on demand (deterministic; not stored in the repo)
TEST_VIDEOS = {
    "test600.mp4": ([], None),
    "test4k.mp4": (["--frames", "300", "--size", "3840x2160", "--dots", "4", "--seed", "7"],
                   {"verify_4k", "verify_roi", "verify_alltracker"}),
    "test5000.mp4": (["--frames", "5000", "--size", "1920x1080", "--dots", "6", "--seed", "3"], {"verify_long"}),
}


def ensure_test_videos(names):
    for video, (args, needed_by) in TEST_VIDEOS.items():
        path = os.path.join(ROOT, video)
        if os.path.exists(path) or (needed_by is not None and not needed_by & set(names)):
            continue
        print(f"generating {video} ...", flush=True)
        subprocess.check_call([PY, os.path.join(ROOT, "make_test_video.py"), path, *args], cwd=ROOT)


def clean():
    shutil.rmtree(RECOVERY, ignore_errors=True)       # unsaved-work copies the last suite left


def summary_line(stdout, stderr):
    """The line that says why: a traceback's last line, else the suite's own last
    printed line. A warning on stderr is not the verdict (torch's pickle-protocol
    warning stood in for verify_downloads' FAIL line on every CI run)."""
    out = [ln for ln in stdout.splitlines() if ln.strip()]
    err = [ln for ln in stderr.splitlines() if ln.strip()]
    lines = err if err and (not out or any(ln.startswith("Traceback") for ln in err)) else out
    return lines[-1][:160] if lines else ""


def git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout.strip() if r.returncode == 0 else ""


def head_stamp():
    """' HEAD <sha>[ dirty]' for the run header ('' outside a git checkout)."""
    sha = git("rev-parse", "--short=10", "HEAD")
    if not sha:
        return ""
    return f" HEAD {sha}" + (" dirty" if git("status", "--porcelain") else "")   # untracked files count too


def affected(base):
    """The suites for the files changed since `base` (committed, staged or not), via tests/suite_map.py."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import suite_map
    if not git("rev-parse", "--verify", "--quiet", base + "^{commit}"):
        if base == "main" and git("rev-parse", "--verify", "--quiet", "origin/main^{commit}"):
            base = "origin/main"                      # a clone without a local main
        else:
            print(f"--affected: '{base}' is not a branch or commit here - nothing was run")
            sys.exit(2)
    merge_base = git("merge-base", base, "HEAD") or base
    changed = sorted(set(git("diff", "--name-only", merge_base).splitlines()) |
                     set(git("ls-files", "--others", "--exclude-standard").splitlines()))
    names, unmapped = suite_map.suites_for(changed)
    print(f"changed since {base}: {len(changed)} file(s) -> {len(names)} suite(s)")
    for p in unmapped:
        print(f"  not in tests/suite_map.py (ran the fallback set; add it): {p}")
    return [n for n in names if os.path.exists(os.path.join(ROOT, "tests", n + ".py"))]


def stamp():
    """Print, for the current commit, the latest result of every suite run on it (or on a commit whose
    difference from it touches no code), grouped as CPU / GPU / other: the pull request's evidence."""
    head = git("rev-parse", "--short=10", "HEAD")
    path = os.path.join(OUT, "suite_runs.txt")
    if not head or not os.path.exists(path):
        print("no commit or no tests/out/suite_runs.txt")
        return 1
    results, run, sha, when, dirty = {}, {}, None, None, False
    for raw in open(path, encoding="utf-8", errors="replace"):
        if raw.startswith("==== run at"):
            results.update(run)                       # the previous run counts
            run = {}
            parts = raw.split()
            when = " ".join(parts[3:5])
            sha = parts[parts.index("HEAD") + 1] if "HEAD" in parts else None
            dirty = "dirty" in parts
            continue
        if raw.startswith("==== changed during the run"):
            run = {}                                  # the code moved under the run: none of it is evidence
            continue
        bits = raw.split()
        if len(bits) >= 4 and bits[1] in ("ok", "BAD") and bits[2] == "exit" and sha and not dirty:
            run[bits[0]] = (sha, when, bits[1] == "ok", raw.strip()[:150])
    results.update(run)
    same = {}

    def is_code(f):
        return (f.startswith(("kinetrace/", "tests/")) or f == "docs/MANUAL.md"
                or f.endswith((".py", ".bat", ".sh", ".command", ".txt", ".yml", ".json")))

    def equivalent(s):           # tested commit == HEAD, or only files that are not code differ since
        if s not in same:
            if s == head:
                same[s] = True
            elif git("cat-file", "-t", s) != "commit":
                same[s] = False                       # a commit this checkout does not have
            else:
                same[s] = not any(is_code(f) for f in git("diff", "--name-only", s, "HEAD").splitlines())
        return same[s]

    print(f"Suites on {head} (run on this machine; results from tests/out/suite_runs.txt):")
    for label, group in (("CPU group", CPU), ("GPU group", GPU)):
        got = [results[n] for n in group if n in results and equivalent(results[n][0])]
        bad = [r[3] for r in got if not r[2]]
        missing = [n for n in group if not (n in results and equivalent(results[n][0]))]
        last = max((r[1] for r in got), default="never")
        print(f"- {label}: {len(got) - len(bad)} of {len(group)} ok on this code (last run {last})"
              + (f"; NOT RUN on this code: {', '.join(missing)}" if missing else ""))
        for b in bad:
            print(f"    BAD {b}")
    other = [(n, r) for n, r in sorted(results.items()) if n not in CPU and n not in GPU and equivalent(r[0])]
    for n, r in other:
        print(f"- {n}: {'ok' if r[2] else 'BAD'} ({r[1]})")
    return 0


def main(argv):
    if "--stamp" in argv:
        return stamp()
    names = []
    args = list(argv)
    while args:
        a = args.pop(0)
        if a == "--gpu":
            names += GPU
        elif a == "--cpu":
            names += CPU
        elif a == "--affected":
            base = "main"
            if args and not args[0].startswith(("-", "verify_", "audit_")):
                base = args.pop(0)                    # e.g. origin/main
            names += [n for n in affected(base) if n not in names]
        else:
            names.append(a)
    if not names:
        print("no suites to run")
        return 0
    ensure_test_videos(names)
    log = open(os.path.join(OUT, "suite_runs.txt"), "a", encoding="utf-8")
    started = head_stamp()
    log.write(f"\n==== run at {time.strftime('%Y-%m-%d %H:%M:%S')}{started}\n")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8", KINETRACE_RECOVERY_DIR=RECOVERY,
               KINETRACE_UPDATE_API="http://127.0.0.1:9/api")   # no suite ever asks GitHub (G37)
    bad = 0
    for name in names:
        path = os.path.join(ROOT, "tests", name + ".py")
        print(f"{name:28s} running ...", flush=True)
        t0 = time.time()
        clean()
        try:
            r = subprocess.run([PY, path], cwd=ROOT, env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=3600)
            last = summary_line(r.stdout, r.stderr)
            code = r.returncode
        except subprocess.TimeoutExpired:
            last, code = "TIMEOUT after 3600 s", -1
        clean()
        ok = code == 0
        bad += 0 if ok else 1
        line = f"{name:28s} {'ok ' if ok else 'BAD'} exit {code:>4}  {time.time() - t0:6.0f}s  {last}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        if ok:
            try:
                os.remove(os.path.join(OUT, f"suite_{name}.log"))   # an older failure's log, now stale
            except OSError:
                pass
        if not ok:
            with open(os.path.join(OUT, f"suite_{name}.log"), "w", encoding="utf-8") as fh:
                fh.write(r.stdout if code != -1 else "")
                fh.write("\n---- stderr ----\n")
                fh.write(r.stderr if code != -1 else "")
    if started and head_stamp() != started:       # files edited / committed while the suites ran
        log.write(f"==== changed during the run:{head_stamp()} (these results are not evidence for any commit)\n")
        print("note: the code changed while the suites ran; --stamp will not count this run")
    print(f"{len(names) - bad} of {len(names)} suites exit 0")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
