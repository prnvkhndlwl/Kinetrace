"""Install Kinetrace's Python packages into the running interpreter (the
folder's own .venv — run.bat / run.sh create it and call this script).

The only platform-specific part is PyTorch:

    Windows or Linux with an NVIDIA GPU  -> CUDA 13 build (download.pytorch.org/whl/cu130)
    Linux without an NVIDIA GPU          -> CPU build     (download.pytorch.org/whl/cpu)
    macOS on Apple Silicon               -> the PyPI build (runs on the GPU through Metal / MPS)
    macOS on Intel                       -> not supported (PyTorch no longer builds for it)

Override the choice with KINETRACE_TORCH=cuda | cpu | mps, e.g. to install the
CPU build on a machine whose NVIDIA driver is too old for CUDA 13 (R580+).
Everything else comes from requirements.txt.

Finally the source code of AllTracker (Harley et al., ICCV 2025, MIT licence),
the default point model, is fetched into models/alltracker at a pinned commit
(277 KB); its 63 MB checkpoint downloads the first time you press Track.
Without it the app falls back to CoTracker3. Standard library only.
"""
import io
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile

TORCH = "2.12.1"
TORCHVISION = "0.27.1"
HERE = os.path.dirname(os.path.abspath(__file__))
# the AllTracker commit Kinetrace's alltracker_backend.py was verified against
ALLTRACKER_SHA = "e7553135e7b361590dbccd10e2b274b024f41cd6"
ALLTRACKER_DIR = os.path.join(HERE, "models", "alltracker")


def has_nvidia() -> bool:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    try:
        return subprocess.run([exe, "-L"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def torch_choice() -> str:
    forced = os.environ.get("KINETRACE_TORCH", "").strip().lower()
    if forced in ("cuda", "cpu", "mps"):
        return forced
    if sys.platform == "darwin":
        return "mps"
    return "cuda" if has_nvidia() else "cpu"


def pip(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def fetch_alltracker() -> None:
    """AllTracker's code into models/alltracker (skipped when already there)."""
    if os.path.exists(os.path.join(ALLTRACKER_DIR, "nets", "alltracker.py")):
        return
    url = f"https://github.com/aharley/alltracker/archive/{ALLTRACKER_SHA}.zip"
    print(f"+ fetching AllTracker (MIT licence) from {url}", flush=True)
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
        z = zipfile.ZipFile(io.BytesIO(data))
        prefix = z.namelist()[0]
        for name in z.namelist():
            rel = name[len(prefix):]
            if not rel or name.endswith("/"):
                continue
            dest = os.path.join(ALLTRACKER_DIR, *rel.split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(z.read(name))
    except Exception as e:  # noqa: BLE001 - the app still works with CoTracker3
        print(f"  could not fetch AllTracker ({e}); the app uses CoTracker3 for now and tries again "
              "the next time it is started with an internet connection.", flush=True)
        shutil.rmtree(ALLTRACKER_DIR, ignore_errors=True)


def main() -> int:
    if sys.version_info < (3, 10) or sys.version_info >= (3, 15):
        print(f"Kinetrace needs Python 3.10 - 3.14 (3.12 recommended); this is {platform.python_version()}.")
        return 1
    if sys.platform == "darwin" and platform.machine() != "arm64":
        print("Kinetrace needs a Mac with Apple Silicon (M1 or later): PyTorch no longer builds for Intel Macs.")
        return 1
    choice = torch_choice()
    print(f"Kinetrace install: Python {platform.python_version()} on {platform.system()} {platform.machine()}, "
          f"PyTorch build: {choice}", flush=True)
    pip("--upgrade", "pip")
    wheels = [f"torch=={TORCH}", f"torchvision=={TORCHVISION}"]
    if choice == "cuda":
        pip(*wheels, "--index-url", "https://download.pytorch.org/whl/cu130")
    elif choice == "cpu" and sys.platform != "darwin":
        pip(*wheels, "--index-url", "https://download.pytorch.org/whl/cpu")
    else:
        pip(*wheels)
    pip("-r", os.path.join(HERE, "requirements.txt"))
    fetch_alltracker()
    print("Kinetrace install: done.", flush=True)
    return 0


if __name__ == "__main__":
    if "--alltracker-only" in sys.argv:        # the launchers retry a missed fetch on start
        fetch_alltracker()
        sys.exit(0)
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        print(f"\nInstallation failed ({e}). Check the internet connection and run the launcher again: "
              "it resumes where it stopped.")
        sys.exit(1)
