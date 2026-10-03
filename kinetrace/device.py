"""Which processor runs the models, and what that means for the features.

ONE place decides the torch device (NVIDIA CUDA GPU, Apple GPU through Metal /
MPS, or the CPU) and every model loader asks it — tracker.py, the AllTracker
backend, the segmenter and the body layer used to decide on their own, and two
of them never looked for an Apple GPU. `KINETRACE_DEVICE=cuda | mps | cpu`
forces the choice (to run the CPU path on a GPU machine, or when a GPU
misbehaves).

`probe()` is the machine's hardware summary in plain words: what was found,
what runs on it, what is slower and what is switched off — the status bar's
device label, Help → System Check and `python -m kinetrace --check` all read
it. No Qt here, and torch is imported lazily (the app starts before torch is
loaded; `probe()` reports a torch that cannot be imported instead of raising).

The features and their hardware needs (the one hard requirement is SAM 3D
Body, whose upstream code moves tensors with `.cuda()`):

    point tracking (AllTracker / CoTracker3)   any device; several times slower on the CPU
    silhouettes (SAM 2.1 / SAM 3)              any device; SAM 3 much slower without a GPU
    body: ViTPose (2D joints)                  any device
    body: SAM 3D Body (3D joints, mesh)        NVIDIA GPU only
    everything else (3D, sync, exports)        no model, no GPU involved
"""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys

FORCE_ENV = "KINETRACE_DEVICE"
DRIVER_MIN = 580            # NVIDIA driver branch the CUDA 13 PyTorch build needs

_cache: dict | None = None


def forced() -> str | None:
    """The device named in KINETRACE_DEVICE, or None."""
    v = os.environ.get(FORCE_ENV, "").strip().lower()
    return v if v in ("cuda", "mps", "cpu") else None


def _mps_available(torch) -> bool:
    mps = getattr(torch.backends, "mps", None)
    try:
        return bool(mps is not None and mps.is_available())
    except Exception:       # noqa: BLE001 - a broken Metal stack reads as "no GPU"
        return False


BENCH_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "device_benchmark.json")
BENCH_MIN_SPEEDUP = 1.0     # the GPU is used when it is at least this much faster than the CPU


_driver_text: list = []         # [version text or ""], read once: nvidia-smi is a process start


def _driver_for_key() -> str:
    if not _driver_text:
        try:
            d = nvidia_driver()
        except Exception:       # noqa: BLE001 - a key part, never a reason to fail
            d = None
        _driver_text.append(d[0] if d else "")
    return _driver_text[0]


def _bench_key(torch) -> str:
    """What the cached measurement is valid for: this PyTorch, this GPU and its
    driver (a driver update can fix - or break - a GPU, I239), this CPU."""
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else ("mps" if _mps_available(torch) else "none")
    drv = _driver_for_key() if cuda else ""
    return f"{torch.__version__}|{gpu}|{drv}|{platform.machine()}|{os.cpu_count()}|{platform.processor()}"


def benchmark(device: str, torch=None) -> float:
    """Milliseconds for one pass of a workload shaped like the models' inner
    loops: a 3x3 convolution over a 512x512 feature map (32 channels, the
    feature encoders) and a 2048x2048 matrix product (attention / correlation)
    - about 20 GFLOP, heavy enough that throughput, not launch overhead, is
    what is measured (a tiny pass would make a fast CPU beat an Apple GPU on
    overhead alone). Median of 5 timed passes after 3 warm-ups; the GPU is
    synchronised. Raises when the device cannot run it (a broken driver reads
    as "no GPU"). Measured: 14 ms / pass on a 24-core workstation CPU, 0.6 ms
    on an RTX PRO 5000 (25x); the whole comparison takes about a second."""
    import time
    if torch is None:
        import torch
    dev = torch.device(device)
    x = torch.randn(1, 32, 512, 512, device=dev)
    w = torch.randn(32, 32, 3, 3, device=dev)
    a = torch.randn(2048, 2048, device=dev)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    def one():
        y = torch.nn.functional.conv2d(x, w, padding=1)
        z = a @ a
        return y, z

    with torch.no_grad():
        for _ in range(3):
            one()
        sync()
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            one()
            sync()
            times.append(time.perf_counter() - t0)
    times.sort()
    return float(times[len(times) // 2] * 1000.0)


def _read_bench(key: str) -> dict | None:
    try:
        import json
        with open(BENCH_FILE, encoding="utf-8") as fh:
            d = json.load(fh)
        if not (isinstance(d, dict) and d.get("key") == key):
            return None
        if d.get("gpu") and d.get("gpu_ms") is None:
            return None         # a failed GPU run saved by an older version: try the GPU again (I239)
        return d
    except Exception:       # noqa: BLE001 - no cache, or an unreadable one: measure again
        return None


def _write_bench(d: dict) -> None:
    try:
        import json
        os.makedirs(os.path.dirname(BENCH_FILE), exist_ok=True)
        with open(BENCH_FILE, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
    except OSError:
        pass                # a read-only install measures again next time


_memo: dict = {}                # (file, key, file mtime) -> the measurement: once per process


def _bench_mtime() -> tuple | None:
    """What identifies the cache file's current content (time and size: a coarse file system
    clock must not hide a rewrite)."""
    try:
        st = os.stat(BENCH_FILE)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def compare(torch=None) -> dict:
    """The GPU-versus-CPU measurement for this machine, cached in BENCH_FILE:
    {key, cpu_ms, gpu, gpu_ms (None when the GPU could not run it), speedup,
    error}. The first call on a machine takes a few seconds (the CPU pass on
    a slow laptop); afterwards it is a file read, and within one run of the app
    a memo. A GPU that could NOT run the workload is not written to the file
    (a busy card or a half-loaded driver at one start must not keep the CPU in
    use for good): it is remembered for this run only and tried again at the
    next start (I239)."""
    if torch is None:
        import torch
    key = _bench_key(torch)
    sig = (BENCH_FILE, key, _bench_mtime())
    if sig in _memo:
        return _memo[sig]
    cached = _read_bench(key)
    if cached is not None:
        _memo.clear()
        _memo[sig] = cached
        return cached
    gpu = "cuda" if torch.cuda.is_available() else ("mps" if _mps_available(torch) else None)
    d = {"key": key, "cpu_ms": None, "gpu": gpu, "gpu_ms": None, "speedup": None, "error": ""}
    try:
        d["cpu_ms"] = benchmark("cpu", torch)
    except Exception as e:      # noqa: BLE001 - the CPU cannot fail this; record it if it does
        d["error"] = f"cpu: {type(e).__name__}: {e}"
        _memo.clear()
        _memo[sig] = d
        return d
    if gpu is not None:
        try:
            d["gpu_ms"] = benchmark(gpu, torch)
            d["speedup"] = d["cpu_ms"] / max(d["gpu_ms"], 1e-6)
        except Exception as e:  # noqa: BLE001 - a GPU that cannot run the workload is not used
            d["error"] = f"{gpu}: {type(e).__name__}: {e}"
    if gpu is None or d["gpu_ms"] is not None:
        _write_bench(d)
    _memo.clear()
    _memo[sig if gpu is None or d["gpu_ms"] is None else (BENCH_FILE, key, _bench_mtime())] = d
    return d


def pick_device() -> tuple[str, str]:
    """(torch device string, one-line label for the status bar).

    Order: a forced choice, else the GPU (CUDA, else Apple's) WHEN IT IS
    FASTER than the CPU on this machine (`compare()`, measured once and
    cached), else the CPU. The label starts with the device name (the manual
    tells the user that "cpu" at the start means the CPU is doing the work)."""
    import torch
    want = forced()
    cuda = torch.cuda.is_available()
    mps = _mps_available(torch)
    gpu = "cuda" if cuda else ("mps" if mps else None)
    gpu_name = torch.cuda.get_device_name(0) if cuda else ("Apple GPU (Metal)" if mps else "")
    if want is not None:
        if want == "cuda" and cuda:
            return "cuda", f"cuda: {gpu_name} (chosen with KINETRACE_DEVICE)"
        if want == "mps" and mps:
            return "mps", "mps: Apple GPU (Metal) (chosen with KINETRACE_DEVICE)"
        if want in ("cuda", "mps"):
            return "cpu", f"cpu (KINETRACE_DEVICE={want} asked for a GPU that is not available)"
        if gpu is not None:
            return "cpu", f"cpu (chosen with KINETRACE_DEVICE; the {gpu_name} is available)"
        return "cpu", "cpu (chosen with KINETRACE_DEVICE)"
    if gpu is None:
        return "cpu", "cpu: no GPU found - tracking works but is several times slower"
    c = compare(torch)
    if c.get("gpu_ms") is None:
        return "cpu", f"cpu: the {gpu_name} could not run the models ({c.get('error', 'unknown error')})"
    speedup = float(c.get("speedup") or 0.0)
    if speedup < BENCH_MIN_SPEEDUP:
        return "cpu", (f"cpu: measured faster than the {gpu_name} here "
                       f"(GPU {c['gpu_ms']:.0f} ms vs CPU {c['cpu_ms']:.0f} ms per pass)")
    return gpu, f"{gpu}: {gpu_name} ({speedup:.0f}x faster than the CPU here)"


def sam_dtype(torch, device: str):
    """bfloat16 on CUDA (measured identical masks, half the memory); float32
    elsewhere - MPS and CPU kernels for bfloat16 are partial or slow."""
    return torch.bfloat16 if device == "cuda" else torch.float32


def supports_sam3d_body(device: str) -> bool:
    """Meta's SAM 3D Body code calls `.cuda()` on its tensors: NVIDIA only.
    KINETRACE_ALLOW_SAM3D_CPU=1 lifts the gate for someone who has patched
    that code (and for the test suite's CPU stand-in of its API)."""
    return device == "cuda" or os.environ.get("KINETRACE_ALLOW_SAM3D_CPU", "") == "1"


ALLTRACKER_DIM_GPU = 1024           # the app's verified working size on CUDA (tracker.ALLTRACKER_MAX_DIM)
ALLTRACKER_PEAK_GB_AT_1024 = 18.4   # measured peak RSS, CPU, one 16-frame window of a full 4K frame at 1024


def alltracker_max_dim(device: str, ram_bytes: int | None = None) -> int:
    """The longest side AllTracker works at on `device`.

    On CUDA the verified 1024 (memory lives on the card). On the CPU or an
    Apple GPU the window lives in RAM and grows with the FOURTH power of the
    side (every pixel pair of the window is correlated): measured 18.4 GB at
    1024 on a full 4K frame, which would exhaust a 16 GB laptop. So there the
    side is the largest of 1024 / 896 / 768 / 640 / 512 whose estimated peak
    plus 2 GB for the rest of the app fits in HALF the machine's RAM: 896 with
    32 GB, 768 with 16 GB, 512 with 8 GB. The native-resolution LK refinement
    stage still runs, so the loss is in the raw prediction only.
    `KINETRACE_ALLTRACKER_MAX_DIM` overrides (a number)."""
    env = os.environ.get("KINETRACE_ALLTRACKER_MAX_DIM", "").strip()
    if env.isdigit() and int(env) >= 256:
        return int(env)
    if device == "cuda":
        return ALLTRACKER_DIM_GPU
    ram = total_ram_bytes() if ram_bytes is None else int(ram_bytes)
    if ram <= 0:
        return 768
    budget_gb = ram / 1024 ** 3 / 2.0
    for d in (1024, 896, 768, 640, 512):
        if ALLTRACKER_PEAK_GB_AT_1024 * (d / 1024.0) ** 4 + 2.0 <= budget_gb:
            return d
    return 512


def total_ram_bytes() -> int:
    """Physical RAM, or 0 when it cannot be determined. No new dependency:
    ctypes on Windows, sysconf elsewhere (macOS and Linux both have
    SC_PHYS_PAGES)."""
    try:
        if os.name == "nt":
            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS(dwLength=ctypes.sizeof(_MS))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(ms.ullTotalPhys)
            return 0
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:  # noqa: BLE001 - any failure just means "unknown"
        return 0


def nvidia_driver() -> tuple[str, int] | None:
    """(driver version text, its major number) from nvidia-smi, or None when
    there is no NVIDIA driver on this machine. Used to explain a CUDA build
    of PyTorch that finds no usable GPU: the driver is older than the build."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        r = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15, creationflags=flags)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    text = (r.stdout or "").strip().splitlines()
    if not text:
        return None
    ver = text[0].strip()
    try:
        major = int(ver.split(".")[0])
    except ValueError:
        major = 0
    return ver, major


def probe(refresh: bool = False) -> dict:
    """Everything the user (or a support request) needs to know about what the
    models run on. Cached: the hardware does not change while the app runs.

    Keys: os, machine, python, torch (version or None), torch_error, cuda,
    mps, device, label, gpu, vram_gb, ram_gb, driver, forced, features
    (feature -> (state 'ok' | 'slow' | 'off', sentence)), notes (sentences
    about constraints, possibly empty)."""
    global _cache
    if _cache is not None and not refresh:
        return _cache
    p: dict = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": None, "torch_error": "", "torch_build": "",
        "cuda": False, "mps": False, "device": "cpu", "label": "",
        "gpu": "", "vram_gb": 0.0, "ram_gb": round(total_ram_bytes() / 1024 ** 3, 1),
        "driver": None, "forced": forced(), "features": {}, "notes": [],
        "bench": None,          # compare(): the GPU-vs-CPU measurement when a GPU exists
        "kind": "cpu",          # "gpu" | "cpu": what the status-bar badge shows
    }
    try:
        import torch
        p["torch"] = torch.__version__
        p["torch_build"] = ("CUDA " + torch.version.cuda) if getattr(torch.version, "cuda", None) \
            else ("Metal" if _mps_available(torch) else "CPU")
        p["cuda"] = bool(torch.cuda.is_available())
        p["mps"] = _mps_available(torch)
        p["device"], p["label"] = pick_device()
        if p["cuda"] or p["mps"]:
            p["bench"] = compare(torch)         # cached after the first call
        if p["cuda"]:
            p["gpu"] = torch.cuda.get_device_name(0)
            try:
                p["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
            except Exception:       # noqa: BLE001
                p["vram_gb"] = 0.0
    except Exception as e:      # noqa: BLE001 - the report says so instead of raising
        p["torch_error"] = f"{type(e).__name__}: {e}"
        p["label"] = "device unavailable: PyTorch could not be imported"
    p["driver"] = nvidia_driver()
    dev = p["device"]
    p["kind"] = "gpu" if dev in ("cuda", "mps") else "cpu"
    f = p["features"]
    slow = dev == "cpu"
    b = p["bench"] or {}
    if dev == "cpu" and (p["cuda"] or p["mps"]) and not p["forced"]:
        # a GPU exists but the CPU is doing the work: say why, first
        if b.get("gpu_ms") is None:
            p["notes"].append("A GPU is present but could not run the models (" + str(b.get("error", "")) +
                              "). The CPU is used for this run. Update the graphics driver; the GPU is tried "
                              "again every time Kinetrace starts.")
        else:
            p["notes"].append(f"The CPU measured faster than the GPU on this machine (GPU {b['gpu_ms']:.0f} ms vs "
                              f"CPU {b['cpu_ms']:.0f} ms per pass), so the CPU is used. Delete "
                              "models/device_benchmark.json to measure again, or force the GPU with KINETRACE_DEVICE.")
    at_dim = alltracker_max_dim(dev)
    at_note = (f"; AllTracker works on a {at_dim}-pixel picture here to fit the memory"
               if at_dim < ALLTRACKER_DIM_GPU else "")
    f["Point tracking (AllTracker / CoTracker3)"] = (
        ("slow", "runs on the CPU: several times slower than on a GPU (a 4K video takes hours)" + at_note)
        if slow else ("ok", "runs on the GPU" + at_note))
    f["Silhouettes (SAM 2.1 / SAM 3)"] = (
        ("slow", "runs on the CPU: SAM 2.1 base+ is the practical choice; SAM 3 is much slower here")
        if slow else ("ok", "runs on the GPU"))
    f["Body: ViTPose (2D joints)"] = (("slow", "runs on the CPU, slower") if slow else ("ok", "runs on the GPU"))
    f["Body: SAM 3D Body (3D joints, mesh)"] = (
        ("ok", "available (NVIDIA GPU present)") if supports_sam3d_body(dev)
        else ("off", "switched off: Meta's SAM 3D Body code runs only on an NVIDIA GPU"))
    f["Cameras, 3D, sync, calibration, exports"] = ("ok", "no model involved; any computer")
    notes = p["notes"]
    if p["torch_error"]:
        notes.append("PyTorch could not be imported, so no model can run. Delete the .venv folder and "
                     "start the launcher again to reinstall (" + p["torch_error"] + ").")
    elif dev == "cpu":
        if p["driver"] is not None and "CUDA" in p["torch_build"] and not p["cuda"]:
            ver, major = p["driver"]
            if major and major < DRIVER_MIN:
                notes.append(f"An NVIDIA graphics card is installed but its driver ({ver}) is older than "
                             f"this PyTorch build needs ({DRIVER_MIN} or newer). Update the NVIDIA driver, "
                             "or reinstall for the CPU (delete .venv, set KINETRACE_TORCH=cpu, start the launcher).")
            else:
                notes.append(f"An NVIDIA graphics card is installed (driver {ver}) but PyTorch cannot use it. "
                             "Restart the computer after a driver update; if it persists, delete .venv and "
                             "start the launcher again.")
        elif p["driver"] is not None and "CUDA" not in p["torch_build"]:
            notes.append("An NVIDIA graphics card is installed but the CPU build of PyTorch was installed. "
                         "Delete .venv and start the launcher again with an up-to-date NVIDIA driver to use it.")
        elif p["forced"] == "cpu":
            notes.append("The CPU was chosen with KINETRACE_DEVICE=cpu.")
        elif p["cuda"] or p["mps"]:
            pass                # a GPU exists: the note above says why the CPU is used (G95)
        else:
            notes.append("No GPU was found. Everything works; the models are several times slower and "
                         "SAM 3D Body is switched off.")
    elif dev == "mps":
        notes.append("Apple GPU (Metal). Point tracking and silhouettes run on it; an operation Metal does "
                     "not support falls back to the CPU on its own. SAM 3D Body is switched off (NVIDIA only).")
    if p["ram_gb"] and p["ram_gb"] < 16:
        notes.append(f"{p['ram_gb']:g} GB of memory: 4K videos will be slow to scrub and the largest models "
                     "may not fit. 16 GB or more is recommended.")
    _cache = p
    return p


def cached_device() -> str | None:
    """The device of an already-completed `probe()`, else None. For GUI-thread
    callers (dialogs deciding what to offer) that must not import torch
    themselves: the app probes in a background thread at start-up."""
    return None if _cache is None else str(_cache["device"])


def short_status(p: dict | None = None) -> str:
    """One sentence after the device label, for the status bar's tooltip."""
    p = p or probe()
    off = [k for k, (s, _) in p["features"].items() if s == "off"]
    slow = [k for k, (s, _) in p["features"].items() if s == "slow"]
    parts = []
    if slow:
        parts.append("slower on this computer: " + ", ".join(slow))
    if off:
        parts.append("not available on this computer: " + ", ".join(off))
    return "; ".join(parts) if parts else "every feature is available on this computer"


def describe(p: dict | None = None) -> str:
    """The System Check text. ASCII only: it is printed to Windows consoles."""
    p = p or probe()
    lines = ["Kinetrace system check", "",
             f"Computer:   {p['os']} ({p['machine']}), {p['ram_gb']:g} GB memory, Python {p['python']}"]
    if p["torch"]:
        lines.append(f"PyTorch:    {p['torch']} ({p['torch_build']} build)")
    else:
        lines.append("PyTorch:    NOT WORKING - " + p["torch_error"])
    if p["gpu"]:
        lines.append(f"GPU:        {p['gpu']} ({p['vram_gb']:g} GB)" + (
            f", driver {p['driver'][0]}" if p["driver"] else ""))
    elif p["driver"]:
        lines.append(f"GPU:        NVIDIA driver {p['driver'][0]} present, but not usable (see below)")
    elif p["mps"]:
        lines.append("GPU:        Apple GPU (Metal)")
    else:
        lines.append("GPU:        none found")
    lines.append(f"Models run on: {'GPU' if p['kind'] == 'gpu' else 'CPU'} - {p['label']}" + (
        f"   [forced by {FORCE_ENV}={p['forced']}]" if p["forced"] else ""))
    b = p.get("bench") or {}
    if b.get("cpu_ms") is not None:
        if b.get("gpu_ms") is not None:
            lines.append(f"Measured:   one pass of the models' workload takes {b['gpu_ms']:.1f} ms on the GPU, "
                         f"{b['cpu_ms']:.1f} ms on the CPU ({float(b['speedup'] or 0):.1f}x); the faster one is used")
        else:
            lines.append(f"Measured:   {b['cpu_ms']:.1f} ms on the CPU; the GPU could not run the workload")
    lines.append("")
    lines.append("Features on this computer:")
    tag = {"ok": "OK  ", "slow": "SLOW", "off": "OFF "}
    for name, (state, why) in p["features"].items():
        lines.append(f"  [{tag[state]}] {name}: {why}")
    if p["notes"]:
        lines.append("")
        for n in p["notes"]:
            lines.append("* " + n)
    return "\n".join(lines)


def cli() -> int:
    """`python -m kinetrace --check`: print the report (no window)."""
    p = probe()
    try:
        print(describe(p))
    except UnicodeEncodeError:      # a console that cannot show a GPU name's characters
        print(describe(p).encode("ascii", "replace").decode("ascii"))
    return 0 if p["torch"] else 1
