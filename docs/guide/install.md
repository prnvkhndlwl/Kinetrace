<sub>Kinetrace guide — [Home](../../README.md) · **Install & run** · [Workflow: track an animal](workflow.md) · [Segments, silhouettes & skeletons](segments.md) · [Several cameras & 3D](cameras-3d.md) · [Human bodies](bodies.md) · [Export formats](exports.md) · [Projects & autosave](projects.md) · [Keyboard & mouse](shortcuts.md) · [Accuracy](accuracy.md) · [Performance (4K)](performance.md) · [Limitations](limitations.md) · [Testing](testing.md) · [Folder map & design](folder-map.md) · [User manual](../MANUAL.md) · [Licences](../../THIRD_PARTY_LICENSES.md)</sub>

# Install & run

**What you need**

| | Windows 10 / 11 | Ubuntu 22.04 or newer | macOS 14 or newer |
|---|---|---|---|
| Computer | 64-bit PC | 64-bit PC (x86-64 or ARM) | a Mac with **Apple Silicon** (M1 or later); Intel Macs are not supported by PyTorch any more |
| Python | 3.10 – 3.14 (3.12 recommended, from python.org) | the system `python3` (22.04 has 3.10, 24.04 has 3.12) plus `python3-venv` | 3.10 – 3.14 from python.org or Homebrew |
| Graphics | an NVIDIA GPU with a recent driver (R580 or newer) is used automatically; without one, the CPU | the same | the Mac's own GPU (Metal) |
| Disk | about 5 GB for the environment, plus the models you use | the same | about 3 GB, plus models |

**Get the code**, with git (`git clone <this repository's URL>`) or GitHub's
*Code → Download ZIP*, into a folder of your choice.

**Windows:** double-click `run.bat`.

**Ubuntu:** install the few system packages once, then start the launcher:

```bash
sudo apt install python3 python3-venv libxcb-cursor0 libegl1 libxkbcommon-x11-0 libgl1
```

```bash
./run.sh
```

**macOS:** open *Terminal* in the folder and run `./run.sh`. (If macOS refuses
to run it, `chmod +x run.sh` first.)

The **first run** creates a private Python environment in `.venv` inside the
folder and installs everything into it — up to ~4 GB, mostly PyTorch — which
takes a while; later launches start at once. `install.py` chooses the PyTorch
build for the machine: the CUDA 13 build when an NVIDIA GPU is present, the
CPU build on Linux without one, the standard build (Metal GPU) on a Mac. To
force a choice, set `KINETRACE_TORCH=cuda`, `cpu` or `mps` before the first
run (for example the CPU build on a PC whose NVIDIA driver is too old for CUDA
13). If the download is interrupted, run the launcher again: it resumes.

**Models** are downloaded the first time a feature needs them, into `models/`
inside the folder: the AllTracker point model (the default — the installer
fetches its code, and its 63 MB checkpoint downloads on the first Track),
CoTracker3 (~100 MB, only if you switch to it), SAM 2.1 for
silhouettes (~620 MB), ViTPose + RT-DETR for human poses (~500 MB). **SAM 3**
and **SAM 3D Body** are gated by Meta: request access on Hugging Face, then
put the weights in `models/sam3` / `models/sam-3d-body-*` (the Settings dialog
and the Body dialog say exactly what is missing). An internet connection is
needed only for these first downloads.

**Everything stays in the folder:** nothing is installed into the operating
system, and deleting the folder removes Kinetrace completely. Kinetrace sends
no usage data anywhere (Hugging Face telemetry is switched off); the only
network traffic is the model downloads above.

The status bar always shows which device (CUDA GPU, Apple GPU or CPU) the app
is using; on the CPU, tracking works but is several times slower.

---

[← Home](../../README.md) &nbsp;|&nbsp; [Contents](../../README.md#contents) &nbsp;|&nbsp; [Workflow: track an animal →](workflow.md)
