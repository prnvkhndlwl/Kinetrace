"""Run under pythonw.exe (no console): prove the stderr guard fixes the
torch.hub crash, exercising the exact path the user hit (get_model + a short
tracking run). Results go to a file since there is no stdout."""
import os
import sys
import traceback

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "pythonw_result.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def report(text):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(text + "\n")


try:
    open(OUT, "w").close()
    report(f"stdout={sys.stdout!r} stderr={sys.stderr!r}")
    assert sys.stdout is None and sys.stderr is None, "not actually running under pythonw"

    # the same guard main() applies
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)
    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication([])

    import numpy as np
    from cotracker_app.tracker import TrackingWorker, get_model
    from cotracker_app.video_source import FrameCache

    get_model()  # exactly the call that crashed (torch.hub 'Using cache' -> stderr)
    report("get_model OK under pythonw")

    VID = os.path.join(ROOT, r"test600.mp4")
    GT = np.load(VID + ".gt.npz")["gt"]
    done = {}
    w = TrackingWorker(VID, 0, GT[0].astype(np.float32), [0, 1, 2, 3],
                       FrameCache(200 * 1024**2), 64, refine=True)
    w.error.connect(lambda m: done.update(err=m))
    w.finished_ok.connect(lambda last, p: done.update(last=last))
    w.run()
    assert "err" not in done, done.get("err", "")[:500]
    assert done.get("last") == 63, f"unexpected last frame: {done}"
    report("short tracking run OK under pythonw (through frame 63)")
    report("PYTHONW GUARD PASSED")
except Exception:
    report("FAILED:\n" + traceback.format_exc())
