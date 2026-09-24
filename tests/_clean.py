"""Test helper: forget the unsaved-work recoveries earlier runs left for a
test's videos, so a test never gets an "unsaved work was found" question from
a previous (crashed) run. Offscreen runs keep recoveries in a temp folder, or
in $KINETRACE_RECOVERY_DIR (run_suites.py sets it to tests/out/recovery)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kinetrace import recovery  # noqa: E402


def forget_recovery(*videos) -> None:
    want = {os.path.normcase(os.path.abspath(str(v))) for v in videos}
    for info in recovery.scan():
        if any(os.path.normcase(os.path.abspath(v)) in want for v in info.get("videos", [])):
            recovery.discard(info["project_id"])
