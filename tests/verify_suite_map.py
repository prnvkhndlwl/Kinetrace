"""The test-selection table (tests/suite_map.py) and the evidence stamp of tests/run_suites.py.

- every module in kinetrace/ is in the table (a new module cannot slip past its tests);
- every suite the table names exists, and every suite of the CPU / GPU groups can be selected by it;
- a changed suite runs itself, a changed test helper runs the suites that import it, MANUAL.md runs the
  suite that renders it, an unknown module gets the fallback set and is reported;
- each run's header names its commit, and --stamp counts only results on this code (a dirty run or another
  commit's code is not evidence).
ASCII output only.
"""
import contextlib
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_suites  # noqa: E402
import suite_map  # noqa: E402

fails = 0


def check(cond, what):
    global fails
    print(("ok   " if cond else "FAIL ") + what)
    fails += not cond


# 1. every module is mapped
mods = sorted("kinetrace/" + f for f in os.listdir(os.path.join(ROOT, "kinetrace")) if f.endswith(".py"))
_, unmapped = suite_map.suites_for(mods)
check(not unmapped, f"all {len(mods)} kinetrace modules are in tests/suite_map.py" +
      (f" (missing: {', '.join(unmapped)})" if unmapped else ""))

# 2. every named suite exists; every grouped suite is reachable
named = {n for _, suites in suite_map.MAP + suite_map.ALSO for n in suites} | set(suite_map.FALLBACK)
missing = sorted(n for n in named if not os.path.exists(os.path.join(HERE, n + ".py")))
check(not missing, "every suite the table names exists" + (f" (no file: {missing})" if missing else ""))
unreachable = sorted(n for n in run_suites.CPU + run_suites.GPU if n not in named and n != "audit_sweep")
check(not unreachable, "every CPU / GPU group suite can be selected by a change" +
      (f" (never selected: {unreachable})" if unreachable else ""))

# 3. the selection rules
s, _ = suite_map.suites_for(["tests/verify_lens.py"])
check(s == ["verify_lens"], "a changed suite runs itself")
s, _ = suite_map.suites_for(["tests/_synth.py"])
s2, _ = suite_map.suites_for(["tests/_synth_spots.py"])
check(set(s) == {"verify_animal", "verify_animal_gui", "verify_hand_restart"} and "verify_spots" in s2,
      "a changed test helper runs the suites that import it")
s, _ = suite_map.suites_for(["docs/MANUAL.md"])
check(s == ["verify_timeline_events", "verify_menu_paths"],
      "MANUAL.md runs the suite that renders it and the menu-path check")
s, u = suite_map.suites_for(["kinetrace/brand_new.py"])
check(u == ["kinetrace/brand_new.py"] and s == suite_map.FALLBACK + ["verify_menu_paths"],
      "an unknown module: fallback set (+ the menu-path check), reported")
s, u = suite_map.suites_for(["README.md", "wiki/Home.md"])
check(s == [] and u == [], "text outside the app selects nothing")
s, _ = suite_map.suites_for(["kinetrace/tracker.py", "kinetrace/tracker.py"])
check(len(s) == len(set(s)) and s[0] == "verify_tracker", "no suite twice; the table's order kept")
check(s[-1] == "verify_menu_paths", "every module change also checks the menu paths its messages name")


def git_checks():
    """Sections 4-5 need a git checkout (a ZIP install has none: they are skipped there)."""
    # 4. the run header and the stamp
    hs = run_suites.head_stamp()
    check(hs.startswith(" HEAD ") and len(hs.split()[1]) == 10, f"the run header names the commit ({hs.strip()})")
    head = run_suites.git("rev-parse", "--short=10", "HEAD")
    with tempfile.TemporaryDirectory() as tmp:
        old_out = run_suites.OUT
        run_suites.OUT = tmp
        open(os.path.join(tmp, "suite_runs.txt"), "w", encoding="utf-8").write(
            f"\n==== run at 2026-01-01 10:00:00 HEAD {head}\n"
            "verify_core                  ok  exit    0       5s  ALL CORE CHECKS PASSED\n"
            f"\n==== run at 2026-01-01 11:00:00 HEAD {head} dirty\n"
            "verify_scrub                 BAD exit    1       5s  a dirty run is not evidence\n"
            "\n==== run at 2026-01-01 12:00:00 HEAD deadbeef00\n"
            "verify_lens                  ok  exit    0       5s  another commit's code\n"
            "\n==== run at 2026-01-01 13:00:00\n"
            "verify_wand                  ok  exit    0       5s  an old run without a commit\n"
            f"\n==== run at 2026-01-01 14:00:00 HEAD {head}\n"
            "verify_sync                  ok  exit    0       5s  the code moved under this run\n"
            f"==== changed during the run: HEAD {head} dirty (these results are not evidence for any commit)\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_suites.stamp()
        run_suites.OUT = old_out
    text = buf.getvalue()
    check(f"Suites on {head}" in text, "the stamp names the commit")
    cpu = [ln for ln in text.splitlines() if ln.startswith("- CPU group")]
    check(bool(cpu) and cpu[0].startswith(f"- CPU group: 1 of {len(run_suites.CPU)} ok"),
          "only the clean run on this commit counts (CPU group: 1 ok)")
    check("verify_scrub" not in text.split("NOT RUN", 1)[0] and "BAD" not in text,
          "a dirty run's failure is not reported as evidence")
    check("verify_lens" in text and "verify_wand" in text and "NOT RUN on this code" in text,
          "another commit's and an unstamped run's results show as NOT RUN")
    check("verify_sync" in text.split("NOT RUN on this code", 1)[-1].splitlines()[0],
          "a run during which the code changed is not evidence")

    # 5. a base that does not exist runs nothing and fails, instead of reporting 0 suites as success
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run_suites.affected("no-such-branch-x41")
        check(False, "--affected with an unknown base fails")
    except SystemExit as e:
        check(e.code == 2, "--affected with an unknown base fails (exit 2, nothing run)")


if run_suites.git("rev-parse", "--short=10", "HEAD"):
    git_checks()
else:
    print("skip the run-header / stamp checks: not a git checkout (a ZIP install)")


print("verify_suite_map: PASSED" if not fails else f"verify_suite_map: {fails} FAILED")
sys.exit(1 if fails else 0)
