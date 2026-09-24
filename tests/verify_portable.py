"""Portability: what must hold on any operating system and with or without a
GPU (the 2026-09-24 audit, docs/AUDIT.md "X" findings). No GPU, no network.

  [1] device.py: one rule for the device (forced with KINETRACE_DEVICE), the
      hardware probe's contract, the report's wording, the RAM-aware AllTracker
      working size, SAM dtype per device.
  [2] SAM 3D Body is offered only on CUDA: backend_status says 'needs-gpu' on a
      CPU / Apple-GPU machine, make_estimator refuses, preferred_backend falls
      back to the 2D model, and the Body dialog greys the entry out (offscreen).
  [3] `python -m kinetrace --check` runs as a subprocess with no window.
  [4] install.py: the torch choice per platform / override, the marker path,
      the fetch is skipped when AllTracker is there; run.sh parses; run.bat and
      run.sh and the CI workflow reference the same marker file.
  [5] AllTracker on the CPU (the vendored `.cuda()` call): a real 24-frame run
      with KINETRACE_DEVICE=cpu tracks within 3 px of ground truth. Runs only
      when the checkpoint is already in models/ (no download here).

Run: .venv\\Scripts\\python.exe tests\\verify_portable.py   (.venv/bin/python on Linux / macOS)
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
OUT = os.path.join(ROOT, "tests", "out")
os.makedirs(OUT, exist_ok=True)

import numpy as np  # noqa: E402

from kinetrace import device as dv  # noqa: E402


def ok(msg):
    print("  ok  " + msg)


# ---------------------------------------------------------------- [1] device
print("[1] device module")
os.environ.pop("KINETRACE_DEVICE", None)
os.environ.pop("KINETRACE_ALLTRACKER_MAX_DIM", None)
assert dv.forced() is None
os.environ["KINETRACE_DEVICE"] = "CPU"
assert dv.forced() == "cpu"
os.environ["KINETRACE_DEVICE"] = "tpu"
assert dv.forced() is None, "an unknown name is ignored, not obeyed"
os.environ["KINETRACE_DEVICE"] = "cpu"
d, label = dv.pick_device()
assert d == "cpu" and label.startswith("cpu"), (d, label)
ok(f"forced cpu: {label}")

p = dv.probe(refresh=True)
for k in ("os", "machine", "python", "torch", "cuda", "mps", "device", "label", "gpu", "vram_gb",
          "ram_gb", "driver", "forced", "features", "notes"):
    assert k in p, k
assert p["device"] == "cpu" and p["forced"] == "cpu"
assert p["torch"], "torch imports in the test environment"
feats = p["features"]
assert any("SAM 3D Body" in k for k in feats)
s3 = [v for k, v in feats.items() if "SAM 3D Body" in k][0]
assert s3[0] == "off" and "NVIDIA" in s3[1], s3
pt = [v for k, v in feats.items() if "Point tracking" in k][0]
assert pt[0] == "slow" and "CPU" in pt[1], pt
assert all(v[0] in ("ok", "slow", "off") for v in feats.values())
assert dv.cached_device() == "cpu"
text = dv.describe(p)
assert text.isascii(), "the report is printed to cp1252 consoles"
for must in ("Kinetrace system check", "Computer:", "PyTorch:", "Models run on:", "[OFF ]", "[SLOW]", "KINETRACE_DEVICE=cpu"):
    assert must in text, must
assert "slower" in dv.short_status(p) and "not available" in dv.short_status(p)
ok("probe + report on a forced-CPU machine")

# the RAM-aware AllTracker working size (memory ~ side^4; 18.4 GB at 1024 measured)
GB = 1024 ** 3
assert dv.alltracker_max_dim("cuda", 4 * GB) == 1024, "CUDA keeps the verified size whatever the RAM"
assert dv.alltracker_max_dim("cpu", 128 * GB) == 1024
assert dv.alltracker_max_dim("cpu", 32 * GB) == 896
assert dv.alltracker_max_dim("cpu", 16 * GB) == 768
assert dv.alltracker_max_dim("cpu", 8 * GB) == 512
assert dv.alltracker_max_dim("mps", 8 * GB) == 512
assert dv.alltracker_max_dim("cpu", 0) == 768, "unknown RAM: a middle choice"
for ram in (4, 8, 16, 24, 32, 48, 64, 128):
    dim = dv.alltracker_max_dim("cpu", ram * GB)
    assert dv.ALLTRACKER_PEAK_GB_AT_1024 * (dim / 1024) ** 4 + 2.0 <= ram / 2.0 or dim == 512, (ram, dim)
os.environ["KINETRACE_ALLTRACKER_MAX_DIM"] = "640"
assert dv.alltracker_max_dim("cpu", 128 * GB) == 640 and dv.alltracker_max_dim("cuda", 1) == 640
os.environ["KINETRACE_ALLTRACKER_MAX_DIM"] = "12"      # nonsense: ignored
assert dv.alltracker_max_dim("cpu", 16 * GB) == 768
os.environ.pop("KINETRACE_ALLTRACKER_MAX_DIM")
ok("AllTracker working size follows the RAM off CUDA")

import torch  # noqa: E402
assert dv.sam_dtype(torch, "cuda") is torch.bfloat16
assert dv.sam_dtype(torch, "cpu") is torch.float32 and dv.sam_dtype(torch, "mps") is torch.float32
assert dv.total_ram_bytes() > 0, "RAM is known on this OS"
ok("SAM dtype per device; RAM probe works here")

# a probe whose torch is broken still yields a report
real_pick = dv.pick_device
try:
    def boom():
        raise RuntimeError("no torch")
    dv.pick_device = boom
    import builtins
    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch":
            raise ImportError("simulated: torch missing")
        return real_import(name, *a, **k)
    builtins.__import__ = no_torch
    try:
        pb = dv.probe(refresh=True)
    finally:
        builtins.__import__ = real_import
    assert pb["torch"] is None and "simulated" in pb["torch_error"]
    assert "NOT WORKING" in dv.describe(pb) and any("reinstall" in n for n in pb["notes"])
finally:
    dv.pick_device = real_pick
    dv.probe(refresh=True)
ok("a broken PyTorch is reported, not raised")

# the GPU is used only when it measures faster than the CPU (cached per machine)
ms = dv.benchmark("cpu", torch)
assert 0.05 < ms < 60000, ms
os.environ.pop("KINETRACE_DEVICE", None)
real_file = dv.BENCH_FILE
tmp_bench = os.path.join(OUT, "_device_benchmark_test.json")
try:
    dv.BENCH_FILE = tmp_bench
    if os.path.exists(tmp_bench):
        os.remove(tmp_bench)
    key = dv._bench_key(torch)
    has_gpu = torch.cuda.is_available() or dv._mps_available(torch)
    if has_gpu:
        gpu = "cuda" if torch.cuda.is_available() else "mps"
        # a cache saying the GPU is SLOWER: the CPU is chosen and the label says so
        dv._write_bench({"key": key, "cpu_ms": 10.0, "gpu": gpu, "gpu_ms": 25.0, "speedup": 0.4, "error": ""})
        d, label = dv.pick_device()
        assert d == "cpu" and "faster" in label and "25 ms" in label, (d, label)
        p2 = dv.probe(refresh=True)
        assert p2["kind"] == "cpu" and any("CPU measured faster" in n for n in p2["notes"]), p2["notes"]
        assert "Measured:" in dv.describe(p2)
        # a cache saying the GPU could not run the workload: the CPU, with the reason
        dv._write_bench({"key": key, "cpu_ms": 10.0, "gpu": gpu, "gpu_ms": None, "speedup": None,
                         "error": f"{gpu}: RuntimeError: simulated"})
        d, label = dv.pick_device()
        assert d == "cpu" and "could not run" in label and "simulated" in label, (d, label)
        # a cache saying the GPU is faster: the GPU, with the ratio in the label
        dv._write_bench({"key": key, "cpu_ms": 100.0, "gpu": gpu, "gpu_ms": 5.0, "speedup": 20.0, "error": ""})
        d, label = dv.pick_device()
        assert d == gpu and "20x faster" in label, (d, label)
        p3 = dv.probe(refresh=True)
        assert p3["kind"] == "gpu" and p3["device"] == gpu
        # a stale cache (another key) is ignored and measured afresh
        dv._write_bench({"key": "other machine", "cpu_ms": 100.0, "gpu": gpu, "gpu_ms": 5.0, "speedup": 20.0,
                         "error": ""})
        c = dv.compare(torch)
        assert c["key"] == key and c["cpu_ms"], c
        assert dv._read_bench(key) == c, "the fresh measurement is cached"
        if c["gpu_ms"] is None:
            # a GPU torch reports but that cannot run the workload (GitHub's macOS
            # runners are virtual machines: Metal is "available" and every kernel
            # fails) - the CPU is used and the reason is kept, never an exception
            assert c["error"], c
            d, label = dv.pick_device()
            assert d == "cpu" and "could not run" in label, (d, label)
            p4 = dv.probe(refresh=True)
            assert p4["kind"] == "cpu" and any("could not run the models" in n for n in p4["notes"]), p4["notes"]
            ok(f"GPU reported but unusable here ({c['error'][:60]}...): the CPU, with the reason")
        else:
            assert c["speedup"] is not None and c["speedup"] > 0, c
            ok(f"GPU used only when measured faster: real ratio here {c['speedup']:.1f}x")
    else:
        c = dv.compare(torch)
        assert c["gpu"] is None and c["cpu_ms"] and c["gpu_ms"] is None
        d, label = dv.pick_device()
        assert d == "cpu" and label.startswith("cpu: no GPU"), label
        ok("no GPU here: the CPU, measured and said")
finally:
    dv.BENCH_FILE = real_file
    if os.path.exists(tmp_bench):
        os.remove(tmp_bench)
    os.environ["KINETRACE_DEVICE"] = "cpu"
    dv.probe(refresh=True)

# ------------------------------------------------------- [2] SAM 3D Body gate
print("[2] SAM 3D Body only on CUDA")
from kinetrace import bodypose  # noqa: E402

key = "sam-3d-body-dinov3"
for dev in ("cpu", "mps"):
    st, why = bodypose.backend_status(key, dev)
    assert st == "needs-gpu", (dev, st, why)
    assert "NVIDIA" in why and ("CPU" in why or "Apple" in why), why
st_cuda, _ = bodypose.backend_status(key, "cuda")
assert st_cuda != "needs-gpu", st_cuda
st_cached, _ = bodypose.backend_status(key)            # None -> the cached probe (cpu here)
assert st_cached == "needs-gpu", st_cached
st_vit, _ = bodypose.backend_status("vitpose-base", "cpu")
assert st_vit in ("ready", "download"), st_vit
assert bodypose.preferred_backend() == "vitpose-base", "no 3D backend is preferred without CUDA"
try:
    bodypose.make_estimator(key, device="cpu")
    raise AssertionError("make_estimator must refuse SAM 3D Body on the CPU")
except RuntimeError as e:
    assert "NVIDIA" in str(e), e
ok("backend_status / preferred_backend / make_estimator")

from PySide6.QtWidgets import QApplication  # noqa: E402
app = QApplication.instance() or QApplication([])
from kinetrace.bodyview import BodyRunDialog  # noqa: E402

dlg = BodyRunDialog(None, 100, 5, None, False)
model = dlg.cmb_backend.model()
rows = {dlg.cmb_backend.itemData(i): i for i in range(dlg.cmb_backend.count())}
for k, spec in bodypose.BACKENDS.items():
    item = model.item(rows[k])
    if spec.kind == "sam3d_body":
        assert not item.isEnabled(), k
        assert "NVIDIA" in dlg.cmb_backend.itemText(rows[k]), dlg.cmb_backend.itemText(rows[k])
    else:
        assert item.isEnabled(), k
assert dlg.cmb_backend.currentData() == "vitpose-base"
dlg.deleteLater()
ok("Body dialog greys SAM 3D Body out and explains why")

# ---------------------------------------------------------- [3] --check CLI
print("[3] python -m kinetrace --check")
env = dict(os.environ, KINETRACE_DEVICE="cpu", PYTHONIOENCODING="utf-8")
r = subprocess.run([sys.executable, "-m", "kinetrace", "--check"], cwd=ROOT, capture_output=True,
                   text=True, encoding="utf-8", errors="replace", env=env, timeout=600)
assert r.returncode == 0, r.stderr[-2000:]
assert "Kinetrace system check" in r.stdout and "[OFF ] Body: SAM 3D Body" in r.stdout, r.stdout
ok("prints the report and exits 0")

# ------------------------------------------------------------ [4] installer
print("[4] install.py / launchers / CI agree")
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("kinstall", os.path.join(ROOT, "install.py"))
inst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inst)
saved = os.environ.pop("KINETRACE_TORCH", None)
try:
    for forced in ("cuda", "cpu", "mps"):
        os.environ["KINETRACE_TORCH"] = forced
        assert inst.torch_choice() == forced
    os.environ.pop("KINETRACE_TORCH")
    choice = inst.torch_choice()
    if sys.platform == "darwin":
        assert choice == "mps"
    else:
        assert choice == ("cuda" if inst.has_nvidia() else "cpu")
finally:
    if saved is not None:
        os.environ["KINETRACE_TORCH"] = saved
assert os.path.basename(inst.MARKER) == "kinetrace-install.json"
assert os.path.dirname(inst.MARKER) == sys.prefix
assert "PySide6" in inst.IMPORTS and "torch" in inst.IMPORTS and "cv2" in inst.IMPORTS
ok_imp, why = inst.imports_ok()
assert ok_imp, why
bat = open(os.path.join(ROOT, "run.bat"), encoding="utf-8", errors="replace").read()
sh = open(os.path.join(ROOT, "run.sh"), encoding="utf-8").read()
ci = open(os.path.join(ROOT, ".github", "workflows", "install-check.yml"), encoding="utf-8").read()
cmd = open(os.path.join(ROOT, "Kinetrace.command"), encoding="utf-8").read()
for txt, name in ((bat, "run.bat"), (sh, "run.sh"), (ci, "install-check.yml")):
    assert "kinetrace-install.json" in txt, name + " must check the install marker"
    assert "--check" in txt, name
assert "nuget.org/api/v2/package/python/3.12.10" in bat, "Windows bootstrap: the pinned NuGet CPython"
for txt, name in ((bat, "run.bat"), (sh, "run.sh")):
    assert "KINETRACE_BOOTSTRAP_PYTHON" in txt, name + " must honour the forced-bootstrap switch (CI uses it)"
assert "KINETRACE_BOOTSTRAP_PYTHON" in ci
assert re.search(r"^\s*TAG=\d{8}\s*$", sh, re.M) and re.search(r"^\s*VER=3\.12\.\d+\s*$", sh, re.M) \
    and "python-build-standalone/releases/download/$TAG/cpython-$VER+$TAG-$TRIPLE-install_only_stripped.tar.gz" in sh, \
    "Unix bootstrap: a pinned python-build-standalone release"
for triple in ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu", "aarch64-apple-darwin"):
    assert triple in sh, triple
assert "xcode-select" in sh, "the macOS python3 stub must not be poked without the developer tools"
assert "libxcb-cursor0" in sh and "apt-get" in sh, "Ubuntu's Qt libraries"
assert "exec ./run.sh" in cmd
assert "windows-2022" in ci and "macos-14" in ci and "ubuntu-22.04" in ci
if os.name != "nt":
    r = subprocess.run(["bash", "-n", os.path.join(ROOT, "run.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
ok("torch choice, marker, bootstrap URLs, launchers and CI consistent")

# ---------------------------------------------------- [5] AllTracker on CPU
print("[5] AllTracker on the CPU")
from kinetrace import alltracker_backend as atb  # noqa: E402

if not (atb.available() and atb.is_cached()):
    print("  skip: AllTracker's code or checkpoint is not in models/ (no download in this suite)")
else:
    from kinetrace.tracker import PointSpec, TrackingWorker  # noqa: E402
    from kinetrace.video_source import FrameCache  # noqa: E402
    VID = os.path.join(ROOT, "test600.mp4")
    if not os.path.exists(VID + ".gt.npz"):
        subprocess.run([sys.executable, os.path.join(ROOT, "make_test_video.py"), VID, "--seed", "0"], check=True)
    GT = np.load(VID + ".gt.npz")["gt"]
    n = 24
    tracks = np.full((n, GT.shape[1], 2), np.nan, np.float32)
    ev = {"error": None, "finished": None}
    specs = [PointSpec(i, np.asarray(s, np.float32).copy()) for i, s in enumerate(GT[0])]
    w = TrackingWorker(VID, 0, None, None, FrameCache(128 * 1024 ** 2), n, refine=True, specs=specs,
                       roi=True, autopause=False, point_backend="alltracker")
    w.chunk_ready.connect(lambda w0, tr, vi, cf, m, f: tracks.__setitem__(slice(w0, w0 + tr.shape[0]), tr))
    w.finished_ok.connect(lambda last, p: ev.update(finished=(last, p)))
    w.error.connect(lambda m: ev.update(error=m))
    w.run()
    assert ev["error"] is None, ev["error"]
    assert ev["finished"] is not None and ev["finished"][0] == n - 1, ev
    _, dev = atb.get_alltracker()
    assert dev == "cpu", dev
    err = np.linalg.norm(tracks - GT[:n], axis=-1)
    assert np.isfinite(err).all(), "every frame tracked"
    assert np.nanmean(err) < 3.0 and np.nanmax(err) < 6.0, (np.nanmean(err), np.nanmax(err))
    ok(f"24 frames tracked on the CPU, mean {np.nanmean(err):.2f} px vs GT (max {np.nanmax(err):.2f})")

print("verify_portable PASSED")
