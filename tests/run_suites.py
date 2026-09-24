"""Run a list of verification suites one after another and record, for each,
its last printed line and its exit code, in tests/out/suite_runs.txt. ASCII
only (Windows consoles are cp1252). Usage:

    .venv\\Scripts\\python.exe tests\\run_suites.py verify_core verify_3d ...
    .venv\\Scripts\\python.exe tests\\run_suites.py --gpu        (the GPU group)
    .venv\\Scripts\\python.exe tests\\run_suites.py --cpu        (offscreen GUI + core)
    .venv/bin/python tests/run_suites.py --cpu                  (Linux / macOS)

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
       "verify_keys_follow", "verify_semiauto_pan", "verify_alltracker", "verify_long", "verify_retrack"]
# verify_pythonw is in no group on purpose: it must be launched DETACHED (Start-Process,
# no output redirect) or the stderr guard it checks cannot fail; see CLAUDE.md
CPU = ["verify_portable", "verify_core", "verify_projectfile", "verify_recovery", "verify_interop", "verify_stress", "verify_3d", "verify_wand", "verify_lens", "verify_body",
       "verify_onbody_rules", "verify_sync",
       "verify_timeline_events", "verify_scrub", "verify_multicam", "verify_3d_gui", "verify_ui_focus",
       "verify_annotate", "verify_display", "verify_render", "verify_segment_panel", "verify_point_menu",
       "verify_wand_gui", "verify_lens_gui", "verify_body_gui", "verify_sweep_fixes"]
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


def main(argv):
    names = []
    for a in argv:
        if a == "--gpu":
            names += GPU
        elif a == "--cpu":
            names += CPU
        else:
            names.append(a)
    ensure_test_videos(names)
    log = open(os.path.join(OUT, "suite_runs.txt"), "a", encoding="utf-8")
    log.write(f"\n==== run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8", KINETRACE_RECOVERY_DIR=RECOVERY)
    bad = 0
    for name in names:
        path = os.path.join(ROOT, "tests", name + ".py")
        t0 = time.time()
        clean()
        try:
            r = subprocess.run([PY, path], cwd=ROOT, env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=3600)
            lines = [ln for ln in (r.stdout + "\n" + r.stderr).strip().splitlines() if ln.strip()]
            last = lines[-1][:160] if lines else ""
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
        if not ok:
            with open(os.path.join(OUT, f"suite_{name}.log"), "w", encoding="utf-8") as fh:
                fh.write(r.stdout if code != -1 else "")
                fh.write("\n---- stderr ----\n")
                fh.write(r.stderr if code != -1 else "")
    print(f"{len(names) - bad} of {len(names)} suites exit 0")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
