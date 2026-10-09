"""Model downloads (`kinetrace/downloads.py`, I151-I154, G45; CPU group, no
internet: a local HTTP server stands in for Hugging Face / GitHub).

  [1] the pins: every model file and code folder in use here is the pinned
      one (sha256 / code digest), and every Hugging Face model the app loads
      has a pinned commit.
  [2] fetch: progress in bytes, the sha256 checked (a wrong file is deleted,
      never used), a cut-off download resumes from its .part, Cancel stops it
      between chunks, a server that stalls or is not there ends in a sentence
      within the timeout -- never a hang.
  [3] code: the commit's zip unpacked only when its Python files have the
      pinned digest; an archive with an unsafe member (../, absolute) is
      refused and nothing is written outside the target.
  [4] weights: load_weights reads tensors only (a checkpoint that would run
      code on unpickling is refused).
  [5] the app: the Track "Preparing the models" dialog names the model the run
      really uses (AllTracker by default), shows MB and time left, and its
      Cancel stops the run with a sentence, not a traceback. Which models are
      present is set by the test, never read off models/ (CI has none).
  [6] run_suites' one-line summary names the failing check, not a warning
      that happened to be printed last on stderr.

Run: .venv\\Scripts\\python.exe tests\\verify_downloads.py
"""
import hashlib
import http.server
import io
import os
import pickle
import shutil
import socket
import sys
import threading
import time
import zipfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
OUT = Path(ROOT) / "tests" / "out" / "downloads"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)

from kinetrace import downloads as dl  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + str(msg))
    if not cond:
        FAILS.append(str(msg))


# ---------------------------------------------------------------- a stand-in server
BLOB = os.urandom(3_500_000)                          # 3.5 MB: several chunks
SERVED = {"/blob": BLOB}
STALL = {"after": None}                               # bytes after which /stall stops sending


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = SERVED.get(self.path.split("?")[0])
        if self.path.startswith("/stall"):
            body = BLOB
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        data = body[start:]
        if self.path.startswith("/stall"):
            self.wfile.write(data[:STALL["after"]])
            self.wfile.flush()
            time.sleep(6)                             # longer than the test's timeout
            return
        self.wfile.write(data)


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}"
SHA = hashlib.sha256(BLOB).hexdigest()

# ---------------------------------------------------------------- [1] pins
print("[1] every model is pinned, and the copies here are the pinned ones")
for key, f in dl.FILES.items():
    check("/resolve/main/" not in f.url and len(f.sha256) == 64 and f.size > 0, f"{key}: a pinned URL + sha256")
    if f.dest.is_file():
        check(f.dest.stat().st_size == f.size and dl.file_sha256(f.dest) == f.sha256,
              f"{key}: models/{f.dest.relative_to(dl.MODELS_DIR).as_posix()} is the pinned file")
for key, c in dl.CODE.items():
    check(len(c.commit) == 40 and len(c.digest) == 64, f"{key}: code at a pinned commit")
    if dl.code_present(key):
        check(dl.code_digest(c.dest) == c.digest, f"{key}: the code in models/ has the pinned digest")
from kinetrace import bodypose, segmenter  # noqa: E402
# SAM 3 too (Mac install audit P1: it was the one unpinned download); SAM 3D Body loads from
# local folders only
repos = [v[0] for v in segmenter.BACKENDS.values()] + [bodypose.DETECTOR_REPO] + \
        [s.repo for s in bodypose.BACKENDS.values() if s.kind != "sam3d_body"]
unpinned = [r for r in repos if not dl.hf_revision(r)]
check(not unpinned, f"every Hugging Face model the app downloads has a pinned commit ({unpinned})")
# a SAM 3 snapshot fetches only what transformers reads: not Meta's 3.45 GB sam3.pt (6.9 GB -> 3.4 GB)
import fnmatch  # noqa: E402
import huggingface_hub  # noqa: E402
_real_snap, _real_cached, asked = huggingface_hub.snapshot_download, dl.hf_cached, {}
huggingface_hub.snapshot_download = lambda repo, **kw: asked.update(repo=repo, **kw)
dl.hf_cached = lambda repo: False
try:
    dl.hf_snapshot("facebook/sam3", "SAM 3")
finally:
    huggingface_hub.snapshot_download, dl.hf_cached = _real_snap, _real_cached
pats = asked.get("allow_patterns") or ["*"]
fetched = [f for f in ("model.safetensors", "config.json", "processor_config.json", "tokenizer.json",
                       "vocab.json", "merges.txt", "sam3.pt", "LICENSE") if any(fnmatch.fnmatch(f, p) for p in pats)]
check(asked.get("revision") == dl.hf_revision("facebook/sam3") and "sam3.pt" not in fetched
      and "model.safetensors" in fetched and "processor_config.json" in fetched and "merges.txt" in fetched,
      f"SAM 3: pinned commit, sam3.pt left out ({fetched})")
for r in repos:
    if dl.hf_cached(r):
        a = dl.hf_load_args(r)
        check(a.get("local_files_only") is True and a["revision"] == dl.hf_revision(r),
              f"{r}: cached at its pin -> loaded with local_files_only (no Hub request)")

# ---------------------------------------------------------------- [2] fetch
print("[2] fetch: progress, checksum, resume, cancel, stall")
seen = []
dest = OUT / "blob.bin"
dl.fetch(BASE + "/blob", dest, label="a test file", sha256=SHA, progress=lambda d, t: seen.append((d, t)))
check(dest.read_bytes() == BLOB and not dest.with_name("blob.bin.part").exists(), "downloaded, checked, moved into place")
check(len(seen) >= 3 and seen[-1] == (len(BLOB), len(BLOB)) and all(a[0] <= b[0] for a, b in zip(seen, seen[1:])),
      f"progress in bytes, rising to the total ({len(seen)} updates)")
bad = OUT / "bad.bin"
try:
    dl.fetch(BASE + "/blob", bad, label="a test file", sha256="0" * 64)
    check(False, "a wrong checksum must refuse the file")
except dl.DownloadError as e:
    check(not bad.exists() and not bad.with_name("bad.bin.part").exists() and "checksum" in str(e),
          f"a file with the wrong checksum is deleted, not used: '{str(e)[:70]}...'")
part = OUT / "resume.bin.part"
part.write_bytes(BLOB[:1_000_000])
first = []
dl.fetch(BASE + "/blob", OUT / "resume.bin", label="a test file", sha256=SHA, progress=lambda d, t: first.append(d))
check((OUT / "resume.bin").read_bytes() == BLOB and first[0] > 1_000_000, "a cut-off download goes on from its .part")
n = {"k": 0}


def cancel_after_two():
    n["k"] += 1
    return n["k"] > 2


try:
    dl.fetch(BASE + "/blob", OUT / "cancel.bin", label="a test file", sha256=SHA, cancel=cancel_after_two)
    check(False, "cancel must stop the download")
except dl.DownloadCancelled as e:
    check(not (OUT / "cancel.bin").exists() and (OUT / "cancel.bin.part").exists() and "stopped" in str(e),
          "Cancel stops between chunks; what arrived is kept for the next try")
STALL["after"] = 200_000
t0 = time.monotonic()
old_retries = dl.RETRIES
dl.RETRIES = 1
try:
    dl.fetch(BASE + "/stall", OUT / "stall.bin", label="the test model", timeout=1.5)
    check(False, "a stalled server must end the download")
except dl.DownloadError as e:
    took = time.monotonic() - t0
    check(took < 10 and "stalled" in str(e) and "internet" in str(e),
          f"a server that stops sending ends in {took:.1f} s with a sentence: '{str(e)[:80]}...'")
s = socket.socket()
s.bind(("127.0.0.1", 0))
dead = s.getsockname()[1]
s.close()
try:
    dl.fetch(f"http://127.0.0.1:{dead}/x", OUT / "dead.bin", label="the test model", timeout=2)
    check(False, "no server must raise")
except dl.DownloadError as e:
    check("could not be downloaded" in str(e) and "internet connection" in str(e), f"no server: '{str(e)[:90]}'")
try:
    dl.fetch(BASE + "/missing", OUT / "missing.bin", label="the test model", timeout=2)
    check(False, "a 404 must raise")
except dl.DownloadError as e:
    check("404" in str(e), f"a 404 is said: '{str(e)[:70]}'")
dl.RETRIES = old_retries

# ---------------------------------------------------------------- [3] code
print("[3] model code from a commit's zip")
src = {"pkg/__init__.py": b"", "pkg/net.py": b"def f():\r\n    return 1\r\n", "README.md": b"# x"}
ref = OUT / "ref"
for rel, data in src.items():
    (ref / rel).parent.mkdir(parents=True, exist_ok=True)
    (ref / rel).write_bytes(data)
digest = dl.code_digest(ref)


def zip_of(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for rel, data in members.items():
            z.writestr("repo-abc123/" + rel, data.replace(b"\r\n", b"\n"))      # a GitHub zip: LF
    return buf.getvalue()


SERVED["/good.zip"] = zip_of(src)
SERVED["/other.zip"] = zip_of({**src, "pkg/net.py": b"import os; os.system('x')\n"})
evil = io.BytesIO()
with zipfile.ZipFile(evil, "w") as z:
    z.writestr("repo-abc123/pkg/net.py", b"x = 1\n")
    z.writestr("repo-abc123/../../escaped.py", b"x = 2\n")
SERVED["/evil.zip"] = evil.getvalue()
target = OUT / "code_target"
for name, url_path, ok_expected in (("good", "/good.zip", True), ("other", "/other.zip", False),
                                    ("evil", "/evil.zip", False)):
    shutil.rmtree(target, ignore_errors=True)
    dl.CODE["_test"] = dl.Code("the test code", "x/y", "abc123", digest, target, "pkg/net.py")
    real_fetch = dl.fetch
    dl.fetch = lambda url, dest, **kw: real_fetch(BASE + url_path, dest, **kw)
    try:
        dl.ensure_code("_test")
        got = True
    except dl.DownloadError as e:
        got = False
        why = str(e)
    finally:
        dl.fetch = real_fetch
        del dl.CODE["_test"]
    if ok_expected:
        check(got and dl.code_digest(target) == digest and (target / "README.md").is_file(),
              "the commit's zip is unpacked when its Python files have the pinned digest (CRLF / LF alike)")
    else:
        check(not got and not target.exists() and not (OUT.parent / "escaped.py").exists()
              and not any(p.name.endswith(".download") for p in OUT.iterdir()),
              f"{name}: refused, nothing left behind ('{why[:70]}...')")

# ---------------------------------------------------------------- [4] weights
print("[4] weights are read as tensors only")
import torch  # noqa: E402

good = OUT / "w.pth"
torch.save({"model": {"a": torch.arange(4.0)}}, good)
check(torch.equal(dl.load_weights(good)["model"]["a"], torch.arange(4.0)), "a plain state dict loads")


class _Boom:
    def __reduce__(self):
        return (print, ("THIS RAN",))


evil_w = OUT / "evil.pth"
with open(evil_w, "wb") as fh:
    pickle.dump({"model": _Boom()}, fh, protocol=2)    # torch's own protocol: no warning on stderr
try:
    dl.load_weights(evil_w)
    check(False, "a checkpoint that runs code must be refused")
except Exception as e:  # noqa: BLE001 - torch's UnpicklingError
    check("weights_only" in str(e) or "Unpickl" in type(e).__name__, f"a checkpoint that would run code is refused ({type(e).__name__})")

# ---------------------------------------------------------------- [5] the app
print("[5] the Track dialog while a model downloads")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from kinetrace.app import MainWindow  # noqa: E402
from kinetrace.downloads import DownloadCancelled  # noqa: E402

QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
win = MainWindow()


class _W:                     # the attributes _models_needed reads off a worker
    def __init__(self, backend, specs=True, animal=None, balls=()):
        self.point_backend, self.specs, self.animal, self.balls = backend, [1] if specs else [], animal, list(balls)
        self.download_failed = None


real_is_file, real_code_present = Path.is_file, dl.code_present
# which models are "here" is set by the test, never read off models/: a clean checkout (CI) has
# none, and "CoTracker3 present" failed there on every OS
here = set()
Path.is_file = lambda p: (any(p == dl.FILES[k].dest for k in here)
                          or (real_is_file(p) and not any(p == f.dest for f in dl.FILES.values())))
dl.code_present = lambda key: key in here
try:
    parts, down = win._models_needed(_W("alltracker"))
    check(down and parts and "AllTracker" in parts[0] and "66 MB" in parts[0],
          f"AllTracker (the default) missing -> named with its size: '{parts[0] if parts else ''}'")
    parts, down = win._models_needed(_W("cotracker3"))
    check(down and parts and "CoTracker3" in parts[0],
          f"CoTracker3 missing -> named: '{parts[0] if parts else ''}'")
    here.add("cotracker3")
    parts, down = win._models_needed(_W("cotracker3"))
    check(not down and not parts, "CoTracker3 present -> nothing to download")
    parts, down = win._models_needed(_W("alltracker"))
    check(down, "... while AllTracker, not present, is still downloaded")
finally:
    Path.is_file, dl.code_present = real_is_file, real_code_present


class _Ball:
    backend = "sam2.1-base-plus"


real_seg = sys.modules["kinetrace.app"].seg_is_cached
sys.modules["kinetrace.app"].seg_is_cached = lambda b: False
try:
    parts, down = win._models_needed(_W("alltracker", specs=False, balls=[_Ball()]))
    check(down and any("segmentation model" in p for p in parts), "ball markers alone need SAM: said (was not)")
finally:
    sys.modules["kinetrace.app"].seg_is_cached = real_seg
win._model_worker = _W("alltracker")
win._on_model_progress("Downloading the AllTracker point model", 20e6, 66e6)
app.processEvents()
dlg = win._model_dialog
check(dlg is not None and "20 of 66 MB" in dlg.labelText() and dlg.maximum() == 1000 and 290 <= dlg.value() <= 310,
      f"the dialog: '{dlg.labelText().splitlines()[1] if dlg else ''}', bar at {dlg.value() if dlg else '-'} / 1000")
time.sleep(0.3)
win._on_model_progress("Downloading the AllTracker point model", 40e6, 66e6)
check("left" in dlg.labelText().splitlines()[1] and dlg.labelText().count("(first use only)") == 1,
      f"... with the time left: '{dlg.labelText().splitlines()[1]}'")
paused = {"n": 0}


class _Worker:
    def request_pause(self):
        paused["n"] += 1

    def wait(self, *a):
        return True


from kinetrace import app as appmod  # noqa: E402

win.worker = _Worker()
win.state = appmod.TRACKING
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QPushButton  # noqa: E402

dlg.show()
app.processEvents()
btn = dlg.findChild(QPushButton)
check(btn is not None and btn.text() == "Cancel", "the dialog has a Cancel button (it had none)")
QTest.mouseClick(btn, Qt.LeftButton)
app.processEvents()
check(paused["n"] == 1, "Cancel on the dialog stops the run (request_pause)")
# the run's start closes the dialog -- QProgressDialog emits canceled on close, and that
# paused every run that first loaded SAM (found by verify_track_all: every camera stopped at its start)
win._model_dialog = None
win._open_model_dialog("Loading the segmentation model", 0)
win._model_dialog.show()
app.processEvents()
win._render_timer.stop()
win._on_track_started()
app.processEvents()
win._render_timer.stop()
check(paused["n"] == 1 and win._model_dialog is None, "the run's start closes the dialog without pausing the run")
paused["n"] = 0
win._open_model_dialog("x", 0)
QTest.mouseClick(win._model_dialog.findChild(QPushButton), Qt.LeftButton)
app.processEvents()
check(paused["n"] == 1, "Cancel on the dialog stops the run (request_pause)")
win._model_worker.download_failed = DownloadCancelled("The download of the AllTracker point model was stopped.")
msgs = []
real_show = win.toast.show_message
win.toast.show_message = lambda text, *a, **k: msgs.append(text)
boxes = []
QMessageBox.critical = staticmethod(lambda *a, **k: boxes.append(a))
appmod._retire = lambda th: None
win._on_track_error("The download of the AllTracker point model was stopped.")
check(msgs and "stopped" in msgs[-1] and "press Track" in msgs[-1] and not boxes and win._model_dialog is None,
      f"a cancelled download ends with a notice, no error dialog, no traceback: '{msgs[-1][:80] if msgs else ''}'")
win.toast.show_message = real_show
win.worker = None
win.state = appmod.IDLE
win._dev_probe.wait(20000)
win.close()

# ---------------------------------------------------------------- [6] the CI summary
print("[6] the suite runner's one-line summary names the failing check")
sys.path.insert(0, os.path.join(ROOT, "tests"))
from run_suites import summary_line  # noqa: E402

warn = ("...\\torch\\_weights_only_unpickler.py:590: UserWarning: Detected pickle protocol 4 in the checkpoint\n"
        "  return Unpickler(file, encoding=encoding).load()\n")
got = summary_line("[5] ...\n  FAIL  CoTracker3 present\n\nverify_downloads FAILED (1):\n  - CoTracker3 present\n", warn)
check(got == "  - CoTracker3 present", f"a warning on stderr is not the verdict: '{got}' (was torch's warning)")
got = summary_line("[1] ...\n", warn + "Traceback (most recent call last):\n  File \"x.py\", line 1\nValueError: boom\n")
check(got == "ValueError: boom", f"a crash: the exception's line: '{got}'")
check(summary_line("verify_x PASSED\n", "") == "verify_x PASSED", "a pass: the suite's own last line")

print()
if FAILS:
    print(f"verify_downloads FAILED ({len(FAILS)}):")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print("verify_downloads PASSED")
