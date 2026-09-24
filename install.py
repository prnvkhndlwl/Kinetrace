"""Install Kinetrace's Python packages into the running interpreter (the
folder's own .venv — run.bat / run.sh create it and call this script).

The only platform-specific part is PyTorch:

    Windows or Linux with an NVIDIA GPU  -> CUDA 13 build (download.pytorch.org/whl/cu130)
    Linux without an NVIDIA GPU          -> CPU build     (download.pytorch.org/whl/cpu)
    macOS on Apple Silicon               -> the PyPI build (runs on the GPU through Metal / MPS)
    macOS on Intel                       -> not supported (PyTorch no longer builds for it)

Override the choice with KINETRACE_TORCH=cuda | cpu | mps, e.g. to install the
CPU build on a machine whose NVIDIA driver is too old for CUDA 13 (R580+).
Everything else comes from requirements.txt. Standard library only.
"""
import os
import platform
import shutil
import subprocess
import sys

TORCH = "2.12.1"
TORCHVISION = "0.27.1"
HERE = os.path.dirname(os.path.abspath(__file__))


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
    print("Kinetrace install: done.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        print(f"\nInstallation failed ({e}). Check the internet connection and run the launcher again: "
              "it resumes where it stopped.")
        sys.exit(1)
