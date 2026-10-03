"""Regression checks for the 2026-10-02 code review's install / update / downloads /
sync / device findings (I218-I220, I235-I239, G93-G98, I195's downloads half, R21).
Every check was written to FAIL on the code before the fixes: a section that raises
is counted as a failure (so this file also runs against an old tree and says what
broke there). No internet: local HTTP servers stand in for GitHub and the model hosts.

  [I218/I219] update.apply_zip: a half-failed update, a case-only rename
  [I220]      the launchers' Python check runs its imports before the version exit
  [I235]      probe_video refuses a video no frame of which decodes
  [I236/G93]  downloads.fetch: a cut connection resumes, disk errors are said as such
  [G94/R21]   update: connection errors, resume / stall, no traceback from the CLI
  [I237]      the motion sync searches only the requested window
  [I238/G98/R21] install.py: --force really forces; a missing Linux library is named
  [I239/G95/R21] device: a failed GPU run is not saved, the driver is in the key, notes
  [G96/G97/R21] sound sync: cancel kills ffmpeg, silent tracks, unreadable files, one
              result type, the dialog never blocks on close
  [misc]      appicon cache key, stale comments

Run: .venv\\Scripts\\python.exe tests\\verify_review_infra.py
"""
import builtins
import errno
import http.client
import http.server
import importlib.util
import io
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import zipfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="kt_review_"))
FAILS: list = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + str(what))
    if not cond:
        FAILS.append(str(what))


def section(title):
    """Run a function as a section: an exception is a failed check, not an abort."""
    def deco(fn):
        print(title)
        try:
            fn()
        except Exception as e:      # noqa: BLE001
            check(False, f"{title}: raised {type(e).__name__}: {str(e)[:160]}")
            traceback.print_exc(limit=3)
        return fn
    return deco


# ------------------------------------------------------------------ a stand-in server
BLOB = os.urandom(3_000_000)
LOG: list = []                         # (path, Range header or None)
SEEN: dict = {}
CUT_AT = 1_200_000


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _start(self, body):
        rng = self.headers.get("Range")
        start = int(rng[6:].split("-")[0]) if rng and rng.startswith("bytes=") else 0
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        return body[start:], start

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        LOG.append((path, self.headers.get("Range")))
        rng = self.headers.get("Range")
        if path == "/blob":
            data, _ = self._start(BLOB)
            self.wfile.write(data)
        elif path == "/cut":                                  # first answer ends early, a Range request is fine
            data, start = self._start(BLOB)
            self.wfile.write(data if start else data[:CUT_AT])
        elif path == "/alwayscut":                            # every answer ends early
            data, start = self._start(BLOB)
            self.wfile.write(data[:CUT_AT // 2])
        elif path == "/chunked":
            SEEN["chunked"] = SEEN.get("chunked", 0) + 1
            if SEEN["chunked"] > 1:                           # only the first answer dies in the middle
                data, _ = self._start(BLOB)
                self.wfile.write(data)
            else:
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"%x\r\n" % 1_000_000 + BLOB[:1_000_000] + b"\r\n")
                self.wfile.write(b"%x\r\n" % 500_000 + BLOB[1_000_000:1_001_000])   # then the line goes dead
        elif path == "/rst":                                  # a connection reset in the middle of the body
            self.send_response(200)
            self.send_header("Content-Length", str(len(BLOB)))
            self.end_headers()
            self.wfile.write(BLOB[:300_000])
            self.wfile.flush()
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            self.connection.close()
        elif path == "/stall":
            data, _ = self._start(BLOB)
            self.wfile.write(data[:200_000])
            self.wfile.flush()
            time.sleep(8)
        elif path == "/cutapi/releases/latest":               # the release answer ends in the middle
            self.send_response(200)
            self.send_header("Content-Length", "500")
            self.end_headers()
            self.wfile.write(b'{"tag_name": "v9.9.9", "zip')
        else:
            self.send_error(404)


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}"
os.environ["KINETRACE_UPDATE_API"] = BASE + "/cutapi"

from kinetrace import update  # noqa: E402
from kinetrace import downloads as dl  # noqa: E402


def zip_bytes(files: dict, top="owner-Kinetrace-abc1234") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for rel, data in files.items():
            zf.writestr(f"{top}/{rel}", data)
    return buf.getvalue()


def make_install(name: str, with_marker=True, extra_manifest=()) -> Path:
    r = TMP / name
    files = {"kinetrace/__init__.py": 'APP_VERSION = "1.0.0"\n', "install.py": "# installer v1\n",
             "requirements.txt": "numpy\n"}
    for rel, data in files.items():
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data.encode())
    if with_marker:
        (r / ".venv").mkdir(exist_ok=True)
        (r / ".venv" / "kinetrace-install.json").write_text("{}")
    (r / update.MANIFEST).write_text(json.dumps({"version": "1.0.0", "files": list(files) + list(extra_manifest)}),
                                     encoding="utf-8")
    return r


# =================================================================== I218
@section("[I218] a half-failed ZIP update keeps the installer from being skipped")
def _i218():
    inst = make_install("half")
    z = TMP / "half.zip"
    z.write_bytes(zip_bytes({"install.py": "# installer v2\n", "requirements.txt": "numpy\nnewpackage\n",
                             "kinetrace/__init__.py": 'APP_VERSION = "1.1.0"\n',
                             "kinetrace/zz_locked.py": "x = 1\n"}))
    real = update._replace

    def locked(src, dst, tries=10):
        if Path(dst).name == "zz_locked.py":
            raise PermissionError(13, "the file is in use")
        return real(src, dst, tries)

    update._replace = locked
    try:
        try:
            update.apply_zip(inst, z, "1.1.0")
            check(False, "the locked file must fail the update")
        except PermissionError:
            pass
    finally:
        update._replace = real
    check((inst / "install.py").read_text() == "# installer v2\n", "install.py was replaced before the failure")
    check(not (inst / ".venv" / "kinetrace-install.json").exists(),
          "the install marker is gone after the half-failed update (the next start runs install.py)")
    res = update.apply_zip(inst, z, "1.1.0")
    check(res.reinstall, "the retry still reports that the installer must run again")
    check((inst / "kinetrace/zz_locked.py").exists() and not (inst / update.STAGING).exists(), "the retry finishes the job")
    res2 = update.apply_zip(inst, z, "1.1.0")
    check(not res2.reinstall and res2.changed == [], "applying it a third time changes nothing and asks for nothing")


# =================================================================== I219
@section("[I219] a release that renames a file by letter case only keeps the file")
def _i219():
    for same_content in (True, False):
        inst = make_install("case_%d" % same_content, extra_manifest=["kinetrace/Foo.py"])
        (inst / "kinetrace/Foo.py").write_text("x = 1\n")
        z = TMP / ("case_%d.zip" % same_content)
        z.write_bytes(zip_bytes({"install.py": "# installer v1\n", "requirements.txt": "numpy\n",
                                 "kinetrace/__init__.py": 'APP_VERSION = "1.0.0"\n',
                                 "kinetrace/foo.py": "x = 1\n" if same_content else "x = 2\n"}))
        res = update.apply_zip(inst, z, "1.0.1")
        names = [p.name for p in (inst / "kinetrace").iterdir()]
        check(any(n.lower() == "foo.py" for n in names) and (inst / "kinetrace/foo.py").exists(),
              f"foo.py is still there after the case-only rename (same content: {same_content}; {names})")
        if not same_content:
            check((inst / "kinetrace/foo.py").read_text() == "x = 2\n", "and holds the new content")
        check("kinetrace/Foo.py" not in res.removed or (inst / "kinetrace/foo.py").exists(), "no removal took it away")


# =================================================================== I220
@section("[I220] the launchers' Python check runs its imports before the version exit")
def _i220():
    bat = (ROOT / "run.bat").read_text(encoding="utf-8", errors="replace")
    sh = (ROOT / "run.sh").read_text(encoding="utf-8")
    pc_bat = re.search(r'^set "PYCHECK=(.*)"\s*$', bat, re.M).group(1)
    pc_sh = re.search(r"^PYCHECK='(.*)'\s*$", sh, re.M).group(1)
    check(pc_bat == pc_sh, "run.bat and run.sh check the same thing")
    for name, pc in (("run.bat", pc_bat), ("run.sh", pc_sh)):
        good = subprocess.run([sys.executable, "-c", pc], capture_output=True)
        check(good.returncode == 0, f"{name}: a good Python (venv + ensurepip + 3.10-3.14) passes")
        broken = pc.replace("venv, ensurepip", "no_such_module_kinetrace, ensurepip")
        r = subprocess.run([sys.executable, "-c", broken], capture_output=True)
        check(r.returncode != 0, f"{name}: a Python that cannot import venv is rejected (the string exits {r.returncode})")
        wrongver = pc.replace("(3, 10) <=", "(9, 10) <=")
        check(subprocess.run([sys.executable, "-c", wrongver], capture_output=True).returncode != 0,
              f"{name}: a Python outside 3.10-3.14 is rejected")
    launch = [ln for ln in bat.splitlines() if "-m kinetrace %*" in ln]
    check(len(launch) == 1 and launch[0].rstrip().endswith("|| pause & exit /b"), "run.bat still launches on one line (I142)")
    check('[ "$rc" = 3 ]' in sh, "run.sh words the missing-system-library exit (G98)")
    r = subprocess.run(["bash", "-n", str(ROOT / "run.sh")], capture_output=True, text=True) if shutil.which("bash") else None
    check(r is None or r.returncode == 0, "bash -n run.sh")


# =================================================================== I235
@section("[I235] a video no frame of which decodes is refused")
def _i235():
    import cv2
    from kinetrace import audiosync
    from kinetrace.video_source import probe_video
    ffmpeg = audiosync.find_ffmpeg()
    if not ffmpeg:
        print("  (no ffmpeg here - skipped)")
        return
    src = str(TMP / "plain.mp4")
    vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    for k in range(40):
        img = np.full((120, 160, 3), 30 + k, np.uint8)
        cv2.circle(img, (20 + k, 60), 8, (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    good = str(TMP / "good264.mp4")
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p", good],
                       capture_output=True)
    if r.returncode != 0:
        print("  (this ffmpeg cannot encode H.264 - skipped)")
        return
    info = probe_video(good)
    check(info.n_frames == 40, f"a good H.264 clip probes ({info.n_frames} frames)")
    data = bytearray(Path(good).read_bytes())
    i = data.find(b"mdat")
    size = int.from_bytes(data[i - 4:i], "big")
    for k in range(i + 4, min(len(data), i - 4 + size)):
        data[k] = 0                                          # the headers stay, every picture is zeros
    bad = TMP / "zero264.mp4"
    bad.write_bytes(bytes(data))
    try:
        got = probe_video(str(bad))
        check(False, f"a video with no decodable frame must be refused (it was accepted: {got.n_frames} frames)")
    except ValueError as e:
        check("No frame of this video could be decoded" in str(e) and "ffmpeg" in str(e),
              f"refused in words, with the re-encode hint: {str(e)[:60]!r}")


# =================================================================== I236 / G93
@section("[I236] a download cut short resumes instead of counting as complete")
def _i236():
    import hashlib
    sha = hashlib.sha256(BLOB).hexdigest()
    for path in ("/cut", "/chunked"):
        LOG.clear()
        dest = TMP / f"dl{path.replace('/', '_')}.bin"
        try:
            dl.fetch(BASE + path, dest, label="a test file")
            ok = dest.is_file() and dest.read_bytes() == BLOB
            check(ok, f"{path}: the whole file arrives ({dest.stat().st_size if dest.is_file() else 0} of {len(BLOB)} bytes)")
            if path == "/cut":
                check(any(r for _, r in LOG), f"{path}: the rest was asked for with a Range request")
            else:
                check(len(LOG) >= 2, f"{path}: an IncompleteRead is retried, not raised ({len(LOG)} requests)")
        except Exception as e:      # noqa: BLE001
            check(False, f"{path}: fetch raised {type(e).__name__}: {str(e)[:80]}")
    dest = TMP / "dl_cut_sha.bin"
    try:
        dl.fetch(BASE + "/cut", dest, label="a test file", sha256=sha)
        check(dest.read_bytes() == BLOB, "with a checksum: a cut is resumed, not reported as a wrong file")
    except dl.DownloadError as e:
        check(False, f"with a checksum a cut ended as an error: {str(e)[:80]}")
    t0 = time.time()
    try:
        dl.fetch(BASE + "/alwayscut", TMP / "dl_always.bin", label="a test file")
        check(False, "a server that always cuts must end in an error")
    except dl.DownloadError as e:
        check("cut off" in str(e) and not (TMP / "dl_always.bin").exists(),
              f"always cut: said in words and not left as a finished file ('{str(e)[:90]}')")
    check(time.time() - t0 < 30, "and it gives up in time")


@section("[G93] a full disk / a folder that cannot be written is not retried as a network error")
def _g93():
    class _Full:
        def __init__(self, real):
            self.real = real

        def write(self, b):
            raise OSError(errno.ENOSPC, "No space left on device")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.real.close()

    LOG.clear()
    dl.open = lambda p, mode="r", *a, **k: _Full(builtins.open(p, mode))
    try:
        dl.fetch(BASE + "/blob", TMP / "full" / "f.bin", label="the test model")
        check(False, "a full disk must end the download")
    except dl.DownloadError as e:
        msg = str(e)
        check("could not be saved" in msg and "full" in msg and "internet connection" not in msg,
              f"a full disk is said as a disk problem: '{msg[:100]}'")
    finally:
        del dl.open
    check(len([1 for p, _ in LOG if p == "/blob"]) == 1, f"and the server was asked once, not retried ({len(LOG)} requests)")
    # a .part that cannot be opened (a folder in its place)
    LOG.clear()
    d = TMP / "ro"
    (d / "g.bin.part").mkdir(parents=True)
    try:
        dl.fetch(BASE + "/blob", d / "g.bin", label="the test model")
        check(False, "an unwritable .part must end the download")
    except dl.DownloadError as e:
        check("could not be saved" in str(e) and "internet connection" not in str(e), f"said as a folder problem: '{str(e)[:90]}'")
    check(len(LOG) == 1, f"the unwritable folder was not retried ({len(LOG)} requests)")
    # the Hugging Face path: a full disk during snapshot_download
    import huggingface_hub
    real = huggingface_hub.snapshot_download
    seen = {}

    def boom(repo, **kw):
        seen.update(kw)
        raise OSError(errno.ENOSPC, "No space left on device")

    huggingface_hub.snapshot_download = boom
    try:
        dl.hf_snapshot("nobody/none", "the test model")
        check(False, "a full disk must end the snapshot")
    except dl.DownloadError as e:
        check("could not be saved" in str(e) and "full" in str(e), f"hf_snapshot says a full drive: '{str(e)[:90]}'")
    finally:
        huggingface_hub.snapshot_download = real
    os.environ["HF_HOME"] = str(TMP / "somebody_elses_cache")
    try:
        check(dl.hf_cache_dir() == str(dl.MODELS_DIR / "hf" / "hub"), "I195: the cache folder is models/hf/hub whatever HF_HOME says")
        check(seen.get("cache_dir") == str(dl.MODELS_DIR / "hf" / "hub"), "I195: the snapshot is fetched into it explicitly")
        check(dl.hf_cached("facebook/sam2.1-hiera-base-plus") == dl.hf_cached("facebook/sam2.1-hiera-base-plus"),
              "I195: the cached test looks in that folder only")
    finally:
        os.environ.pop("HF_HOME", None)


# =================================================================== G94 / R21 update
@section("[G94] update: connection problems are said as such; the CLI prints a sentence")
def _g94():
    inst = make_install("net")
    real_rel = update.Release(version="9.9.9", tag="v9.9.9", name="x", notes="", zip_url=BASE + "/rst", page_url=BASE)
    for path, needle in (("/rst", None), ("/alwayscut", "cut off")):
        try:
            update.apply(update.Release(**{**real_rel.__dict__, "zip_url": BASE + path}), inst)
            check(False, f"{path}: a broken download must raise")
        except update.UpdateError as e:
            msg = str(e)
            check("could not be written" not in msg and "could not be downloaded" in msg and "Close any program" not in msg,
                  f"{path}: a connection error is not 'could not be written': '{msg[:110]}'")
            if needle:
                check(needle in msg, f"{path}: names what happened ({needle})")
        except Exception as e:      # noqa: BLE001
            check(False, f"{path}: raised {type(e).__name__} instead of an UpdateError")
    # the byte count is checked against Content-Length: a cut zip is not handed to the unzipper
    check(not (inst / update.STAGING).exists(), "the staging folder is cleaned up after a failed download")
    # a write failure keeps its own sentence
    real_apply = update.apply_zip
    update.apply_zip = lambda *a, **k: (_ for _ in ()).throw(PermissionError(13, "locked"))
    try:
        update.apply(update.Release(**{**real_rel.__dict__, "zip_url": BASE + "/blob"}), inst)
        check(False, "a failing write must raise")
    except update.UpdateError as e:
        check("could not be written" in str(e), "the WRITING phase still says 'could not be written'")
    finally:
        update.apply_zip = real_apply
    # the release answer itself cut off
    try:
        update.check("1.0.0", timeout=5)
        check(False, "a cut release answer must raise")
    except update.UpdateError as e:
        check("cut" in str(e) or "could not be read" in str(e), f"a cut release answer is an UpdateError: '{str(e)[:80]}'")
    except Exception as e:      # noqa: BLE001
        check(False, f"a cut release answer raised {type(e).__name__}")
    # python -m kinetrace.update: a sentence, never a traceback
    real_check, real_apply2, real_root = update.check, update.apply, update.ROOT
    update.check = lambda *a, **k: (real_rel, True)
    update.apply = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    update.ROOT = inst
    out = io.StringIO()
    old_out = sys.stdout
    sys.stdout = out
    try:
        code = update.main([])
    except Exception as e:      # noqa: BLE001
        sys.stdout = old_out
        check(False, f"main() let {type(e).__name__} escape (a traceback in update.bat)")
    else:
        sys.stdout = old_out
        check(code == 1 and "boom" in out.getvalue() and "Traceback" not in out.getvalue(),
              "main() prints a sentence and returns 1")
    finally:
        sys.stdout = old_out
        update.check, update.apply, update.ROOT = real_check, real_apply2, real_root


@section("[R21] update.download uses the shared fetch: resume and a stall timeout")
def _r21_download():
    dest = TMP / "upd" / "v.zip"
    dest.parent.mkdir(parents=True)
    Path(str(dest) + ".part").write_bytes(BLOB[:700_000])
    LOG.clear()
    update.download(BASE + "/blob", dest, timeout=10)
    check(dest.read_bytes() == BLOB, "a kept .part is completed, the file is whole")
    check(any(r and r.startswith("bytes=700000-") for _, r in LOG), "resumed with a Range request from the kept .part")
    t0 = time.time()
    try:
        update.download(BASE + "/stall", TMP / "upd" / "s.zip", timeout=1.5)
        check(False, "a stalled server must end in an error")
    except update.UpdateError as e:
        check("stalled" in str(e), f"a stalled download ends in an UpdateError sentence: '{str(e)[:80]}'")
    except Exception as e:      # noqa: BLE001
        check(False, f"a stall raised {type(e).__name__} instead of an UpdateError")
    check(time.time() - t0 < 20, f"in time ({time.time() - t0:.1f} s)")


# =================================================================== I237
@section("[I237] the motion sync searches exactly the window it was given")
def _i237():
    import cv2
    from kinetrace import sync
    W, H, N = 320, 240, 700
    rng = np.random.RandomState(11)
    pos = np.zeros((N, 2))
    p = np.array([160.0, 120.0])
    moving, t = False, 0
    while t < N:
        n = int(rng.randint(15, 70))
        for k in range(t, min(N, t + n)):
            if moving:
                p = np.clip(p + rng.normal(0, 6.0, 2), 20, [W - 20, H - 20])
            pos[k] = p
        moving = not moving
        t += n

    def clip(path, start, n, shift=(0, 0), tint=0):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
        for k in range(n):
            img = np.full((H, W, 3), 40 + tint, np.uint8)
            x, y = pos[start + k] + np.array(shift)
            cv2.circle(img, (int(x), int(y)), 9, (230, 230, 230), -1)
            cv2.rectangle(img, (5, 5), (60, 30), (90, 90, 90), -1)
            vw.write(img)
        vw.release()
        return path

    A = clip(str(TMP / "m_a.mp4"), 0, 600)
    B = clip(str(TMP / "m_b.mp4"), 37, 560, (15, -10), 20)                # starts 37 frames later: offset -37
    r = sync.estimate_offsets_from_motion([A, B], [1.0, 1.0], (100, 400), search=100)[0]
    check(r.result.verdict == "clear" and abs(r.offset + 37) <= 0.5, f"inside the window: -37 found ({r.offset:+.1f}, {r.result.verdict})")
    r = sync.estimate_offsets_from_motion([A, B], [1.0, 1.0], (100, 400), search=20)[0]
    check(not (r.result.verdict == "clear" and abs(r.offset + 37) <= 0.5),
          f"the true offset (-37) lies OUTSIDE +-20: it must not be reported clear ({r.offset:+.1f}, {r.result.verdict})")
    check(abs(r.offset) <= 20 + 3, f"the offset reported stays within the requested window ({r.offset:+.1f})")


# =================================================================== I238 / G98 / R21 install
@section("[I238/G98] install.py: --force forces, a missing Linux library is named")
def _install():
    spec = importlib.util.spec_from_file_location("kinstall_review", str(ROOT / "install.py"))
    inst = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inst)
    calls: list = []
    saved = {k: getattr(inst, k) for k in ("pip", "fetch_alltracker", "imports_ok", "write_marker", "hardware_report",
                                           "torch_matches", "say")}
    inst.pip = lambda *a: calls.append(a)
    inst.fetch_alltracker = lambda: None
    inst.write_marker = lambda c: None
    inst.hardware_report = lambda: None
    inst.say = lambda m: None
    inst.imports_ok = lambda: (True, "")
    inst.torch_matches = lambda: False
    os.environ["KINETRACE_TORCH"] = "cpu"
    try:
        inst.main(force=True)
        torch_call = [c for c in calls if any(str(x).startswith("torch==") for x in c)]
        req_call = [c for c in calls if "-r" in c]
        check(torch_call and "--force-reinstall" in torch_call[0], f"--force: PyTorch is installed with --force-reinstall ({torch_call})")
        check(req_call and "--upgrade" in req_call[0], f"--force: the requirements with --upgrade ({req_call})")
        calls.clear()
        inst.main(force=False)
        torch_call = [c for c in calls if any(str(x).startswith("torch==") for x in c)]
        check(torch_call and "--force-reinstall" not in torch_call[0], "without --force: no forced reinstall")
        # G98: an import that fails for a missing system library
        calls.clear()
        said = []
        inst.say = said.append
        inst.imports_ok = lambda: (False, "ImportError: libEGL.so.1: cannot open shared object file: No such file or directory")
        rc = inst.main(force=False)
        text = "\n".join(said)
        check(rc == 3 and not calls, f"a missing system library: no reinstall, exit 3 (exit {rc}, {len(calls)} pip calls)")
        check("sudo apt-get install -y" in text and "libegl1" in text and ".venv" not in text,
              "the apt-get line is printed and deleting .venv is not advised")
        check(inst.system_libs_hint("ModuleNotFoundError: No module named 'torch'") is None, "other import errors get no apt line")
    finally:
        os.environ.pop("KINETRACE_TORCH", None)
        for k, v in saved.items():
            setattr(inst, k, v)
    from kinetrace import device
    src = (ROOT / "install.py").read_text(encoding="utf-8")
    check(inst.nvidia_driver is device.nvidia_driver and inst.DRIVER_MIN == device.DRIVER_MIN and "def nvidia_driver" not in src,
          "R21: install.py uses device.nvidia_driver / DRIVER_MIN, not copies")
    check(inst.ALLTRACKER_SHA == dl.CODE["alltracker"].commit and 'ALLTRACKER_SHA = "' not in src,
          "R21: the AllTracker pin is derived from downloads.CODE")
    check("277 KB" not in src, "stale size comment gone")


# =================================================================== I239 / G95 / R21 device
@section("[I239/G95] device: a failed GPU benchmark is not saved; notes; memo; driver in the key")
def _device():
    from kinetrace import device as dv
    fake = types.SimpleNamespace(
        __version__="9.9.9", version=types.SimpleNamespace(cuda="13.0"),
        cuda=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: "Fake GPU",
                                   get_device_properties=lambda i: types.SimpleNamespace(total_memory=8 * 1024 ** 3)),
        backends=types.SimpleNamespace())
    bench_calls = []
    state = {"gpu_fails": True, "gpu_ms": 2.0}

    def fake_bench(device, torch=None):
        bench_calls.append(device)
        if device == "cuda":
            if state["gpu_fails"]:
                raise RuntimeError("out of memory: the card is busy")
            return state["gpu_ms"]
        return 10.0

    saved = (dv.BENCH_FILE, dv.benchmark, getattr(dv, "_cache", None), dict(os.environ))
    saved_drv = list(getattr(dv, "_driver_text", []))
    dv.BENCH_FILE = str(TMP / "bench.json")
    dv.benchmark = fake_bench
    os.environ.pop("KINETRACE_DEVICE", None)
    real_torch = sys.modules.get("torch")
    try:
        if hasattr(dv, "_driver_text"):
            dv._driver_text[:] = ["555.55"]
        if hasattr(dv, "_memo"):
            dv._memo.clear()
        c = dv.compare(fake)
        check(c["gpu_ms"] is None and "busy" in c["error"], "a GPU that cannot run the benchmark is reported with the reason")
        check(not os.path.exists(dv.BENCH_FILE), "and that failure is NOT saved to models/device_benchmark.json (I239)")
        n = len(bench_calls)
        dv.compare(fake)
        check(len(bench_calls) == n, "within one run the answer is remembered (R21: compare is memoised)")
        # a new session: the GPU works now
        state["gpu_fails"] = False
        dv._memo.clear()
        c2 = dv.compare(fake)
        check(c2["gpu_ms"] == 2.0 and os.path.exists(dv.BENCH_FILE), "the next session tries the GPU again and saves the good result")
        check(dv.compare(fake)["gpu_ms"] == 2.0, "...and keeps it")
        # a failure saved by the old code does not pin the CPU either
        with open(dv.BENCH_FILE, "w", encoding="utf-8") as fh:
            json.dump({"key": dv._bench_key(fake), "cpu_ms": 10.0, "gpu": "cuda", "gpu_ms": None, "speedup": None,
                       "error": "cuda: RuntimeError: old"}, fh)
        dv._memo.clear()
        check(dv.compare(fake)["gpu_ms"] == 2.0, "an old saved GPU failure is ignored and measured again")
        # the driver is part of the key
        dv._driver_text[:] = ["555.55"]
        k1 = dv._bench_key(fake)
        dv._driver_text[:] = ["600.10"]
        k2 = dv._bench_key(fake)
        check(k1 != k2 and "555.55" in k1, "the NVIDIA driver version is part of the cache key (I239)")
        dv._driver_text[:] = ["555.55"]
    finally:
        dv.BENCH_FILE, dv.benchmark, dv._cache = saved[0], saved[1], saved[2]
        if hasattr(dv, "_driver_text"):
            dv._driver_text[:] = saved_drv
        if real_torch is not None:
            sys.modules["torch"] = real_torch
        else:
            sys.modules.pop("torch", None)
        os.environ.update(saved[3])
        if hasattr(dv, "_memo"):
            dv._memo.clear()


@section("[G95] System Check: no 'No GPU was found' when a GPU exists and the CPU is used")
def _g95():
    from kinetrace import device as dv
    fake = types.SimpleNamespace(
        __version__="9.9.9", version=types.SimpleNamespace(cuda="13.0"),
        cuda=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: "Fake GPU",
                                   get_device_properties=lambda i: types.SimpleNamespace(total_memory=8 * 1024 ** 3)),
        backends=types.SimpleNamespace())
    saved = (dv.BENCH_FILE, dv.benchmark, dv._cache, dict(os.environ), list(getattr(dv, "_driver_text", [])))
    real_torch = sys.modules.get("torch")
    dv.BENCH_FILE = str(TMP / "bench_g95.json")
    dv.benchmark = lambda device, torch=None: 25.0 if device == "cuda" else 10.0
    os.environ.pop("KINETRACE_DEVICE", None)
    try:
        if hasattr(dv, "_driver_text"):
            dv._driver_text[:] = ["555.55"]
        if hasattr(dv, "_memo"):
            dv._memo.clear()
        sys.modules["torch"] = fake
        dv._cache = None
        p = dv.probe(refresh=True)
        check(p["device"] == "cpu" and any("CPU measured faster" in n for n in p["notes"]),
              f"probe: the CPU is chosen, with the reason: {p['notes']}")
        check(not any("No GPU was found" in n for n in p["notes"]), "G95: and no 'No GPU was found' beside it")
    finally:
        dv.BENCH_FILE, dv.benchmark, dv._cache = saved[0], saved[1], saved[2]
        if real_torch is not None:
            sys.modules["torch"] = real_torch
        else:
            sys.modules.pop("torch", None)
        os.environ.update(saved[3])
        if hasattr(dv, "_driver_text"):
            dv._driver_text[:] = saved[4]
        if hasattr(dv, "_memo"):
            dv._memo.clear()


# =================================================================== G96 / G97 / R21 sound sync
@section("[G96] sound sync: a cancel stops ffmpeg; the dialog never waits for it")
def _g96():
    from kinetrace import audiosync
    ffmpeg = audiosync.find_ffmpeg()
    if not ffmpeg:
        print("  (no ffmpeg here - skipped)")
        return
    # a real ffmpeg held to real time (-re): a minute of sound that takes a minute to read
    slow = [ffmpeg, "-hide_banner", "-loglevel", "error", "-re", "-f", "lavfi", "-i", "sine=duration=60",
            "-f", "null", "-"]
    flag = {"stop": False}
    box = {}

    def work():
        t0 = time.time()
        try:
            audiosync._run(slow, lambda: flag["stop"])
            box["r"] = "returned"
        except InterruptedError:
            box["r"] = "cancelled"
        except Exception as e:      # noqa: BLE001
            box["r"] = type(e).__name__
        box["dt"] = time.time() - t0

    th = threading.Thread(target=work)
    th.start()
    time.sleep(1.0)
    flag["stop"] = True
    th.join(10)
    check(not th.is_alive() and box.get("r") == "cancelled" and box.get("dt", 99) < 4,
          f"Cancel kills a running ffmpeg within a moment ({box})")
    t0 = time.time()
    try:
        audiosync._run(slow, None, timeout=1.0)
        check(False, "a read that never ends must time out")
    except OSError as e:
        check("did not finish" in str(e) and time.time() - t0 < 6, f"a hung read times out in words ('{str(e)[:70]}')")
    # audio_signal hands the cancel flag through
    seen_cancel = []
    real_run = audiosync._run

    def spy(cmd, should_cancel=None, timeout=None):
        seen_cancel.append(should_cancel)
        raise InterruptedError("cancelled")

    audiosync._run = spy
    try:
        mark = lambda: True  # noqa: E731
        try:
            audiosync.audio_signal("x.mp4", 0, 5, should_cancel=mark)
        except InterruptedError:
            pass
        check(seen_cancel == [mark], "audio_signal passes its cancel flag to the process runner")
    finally:
        audiosync._run = real_run


@section("[G96] closing the sync dialog while it reads does not block the window")
def _g96_dialog():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from kinetrace import app as appmod
    from kinetrace import syncdialog

    class _Slow(syncdialog._SyncThread):
        def run(self):
            while not self._cancel:
                time.sleep(0.01)
            time.sleep(1.5)                      # a reader that takes a while to stop

    dlg = syncdialog.SyncDialog(None, None, [], 0)
    th = _Slow([], [], (0, 1), 1)
    dlg._thread = th
    th.start()
    time.sleep(0.2)
    t0 = time.perf_counter()
    dlg.done(0)
    dt = time.perf_counter() - t0
    check(dt < 0.6, f"closing the dialog does not wait for the reader ({dt:.2f} s)")
    check(th.isRunning() and th in appmod._ORPHANS, "the still-running thread is kept referenced (closeEvent waits for it)")
    th.wait(15000)
    for _ in range(50):
        app.processEvents()
        time.sleep(0.01)
    check(th not in appmod._ORPHANS, "and let go once it has finished")
    dlg.deleteLater()


@section("[G97/R21] sound sync: silent tracks, unreadable files, one result type")
def _g97():
    from kinetrace import audiosync, sync
    SR = audiosync.SAMPLE_RATE
    rng = np.random.RandomState(3)
    ref = 0.02 * rng.normal(size=10 * SR)
    for tc in (2.0, 4.5, 7.0):
        k = int(tc * SR)
        ref[k:k + 400] += rng.normal(size=400) * np.exp(-np.arange(400) / 60.0)
    r = audiosync.align_audio(ref, np.zeros_like(ref), 1.0)
    check(r.verdict == "none" and "silent" in r.why, f"a digitally silent track: 'One of the sound tracks is silent' ('{r.why[:60]}')")
    sigs = {"ref.mp4": ref, "silent.mp4": np.zeros_like(ref), "zeros_ref.mp4": np.zeros_like(ref)}

    def fake_signal(path, t0, duration, sr=SR, should_cancel=None):
        if path == "bad.mp4":
            raise OSError("ffmpeg could not read the audio of bad.mp4: Invalid data found when processing input")
        return sigs[path].copy()

    real = audiosync.audio_signal
    audiosync.audio_signal = fake_signal
    try:
        rows = audiosync.estimate_offsets_from_audio(["ref.mp4", "silent.mp4"], [30.0, 30.0], (0.0, 9.0), 1.0)
        check(rows[0].result.verdict == "none" and "silent" in rows[0].result.why and rows[0].has_audio is True,
              f"a silent camera is named silent, not 'no offset stands out': '{rows[0].result.why[:70]}'")
        rows = audiosync.estimate_offsets_from_audio(["zeros_ref.mp4", "ref.mp4"], [30.0, 30.0], (0.0, 9.0), 1.0)
        check(rows[0].ref_silent and "reference" in rows[0].result.why and "silent" in rows[0].result.why,
              f"a silent REFERENCE is named: '{rows[0].result.why[:80]}'")
        rows = audiosync.estimate_offsets_from_audio(["ref.mp4", "bad.mp4"], [30.0, 30.0], (0.0, 9.0), 1.0)
        check(rows[0].has_audio is None and "could not be read" in rows[0].result.why,
              f"an unreadable file is 'could not be read', not 'no sound track': '{rows[0].result.why[:80]}'")
    finally:
        audiosync.audio_signal = real
    # has_audio: None when ffmpeg cannot open the file at all
    junk = TMP / "junk.mp4"
    junk.write_text("this is not a video")
    check(audiosync.has_audio(str(junk)) is None and audiosync.has_audio(str(TMP / "missing.mp4")) is None,
          "has_audio is None (could not be read) for a damaged or missing file")
    import cv2
    p = str(TMP / "silentvid.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for k in range(10):
        vw.write(np.full((48, 64, 3), 20 * k, np.uint8))
    vw.release()
    check(audiosync.has_audio(p) is False, "a readable video without a sound track is still False")
    # one result type
    check(audiosync.AudioCameraSync is sync.CameraSync and audiosync.AudioAlign is sync.AlignResult,
          "R21: both methods produce sync.CameraSync / sync.AlignResult")
    fields = set(sync.AlignResult.__dataclass_fields__)
    check(not ({"lags", "curve", "lags_s"} & fields), f"R21: the unread lag / curve arrays are gone ({sorted(fields)})")
    # the dialog says 'could not be read'
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from kinetrace import syncdialog

    class _S:
        fps = 30.0
        n_frames = 1000

    class _P:
        sessions = [_S(), _S()]
        n_views = 2
        active = 0
        rates = [1.0, 1.0]
        offsets = [0.0, 0.0]

        def name(self, i):
            return f"cam{i + 1}"

    dlg = syncdialog.SyncDialog(None, _P(), ["a.mp4", "b.mp4"], 0)
    row = sync.CameraSync(1, 0.0, sync.AlignResult(0.0, float("nan"), float("nan"), "none", "could not be read (x)"),
                          has_audio=None)
    dlg._done([row])
    txt = dlg.status.text()
    check("could not be read" in txt and "no sound track" not in txt, f"the dialog says so: '{txt[:90]}'")
    check(not hasattr(syncdialog, "VERDICT_COLORS"), "R21: the unused VERDICT_COLORS is gone")
    dlg.deleteLater()


# =================================================================== misc
@section("[R21] icon cache key, stale comments")
def _misc():
    from kinetrace import appicon
    f = appicon.cache_file(64)
    check(appicon.STYLE in f.name and f.name.endswith("_64.png"), f"the icon cache name carries a hash of the colours ({f.name})")
    check(appicon.style_hash([(1, 2, 3)]) != appicon.style_hash([(1, 2, 4)]), "another colour gives another cache name")
    sync_src = (ROOT / "kinetrace" / "sync.py").read_text(encoding="utf-8")
    check("(64 px wide)" not in sync_src, "sync.py no longer says 64 px (the thumbnail is 160 px wide)")
    vs_src = (ROOT / "kinetrace" / "video_source.py").read_text(encoding="utf-8")
    check("Reading the choice once" not in vs_src, "video_source: the 'read once' comment is corrected")


srv.shutdown()
shutil.rmtree(TMP, ignore_errors=True)
print()
if FAILS:
    print(f"{len(FAILS)} check(s) FAILED:")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print("VERIFY_REVIEW_INFRA PASSED")
