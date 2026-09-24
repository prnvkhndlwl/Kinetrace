# Licences of everything Kinetrace uses

Last checked: 2026-09-22, from the licence files of the packages installed in
`.venv` (the table below is generated from their metadata) and of the model
folders in `models/`, and from the model cards on Hugging Face.

**This page is information, not legal advice.** For a public release, your
institution's technology-transfer or legal office has the final word.

## 1. Kinetrace's own code

Everything in this repository — `kinetrace/`, `tests/`, `tools/`, the
installers and the docs — is the Kinetrace authors' own work. **No
licence has been chosen for it yet**, so until one is added (a `LICENSE` file
in this folder) all rights are reserved. Section 5 lists the options.

The repository contains **no third-party code, model weights or footage**:
`.venv/` and `models/` are created on each user's computer and never
committed (see `.gitignore`). Every model below is downloaded or supplied
there, under its own licence, the first time it is needed. There is no media
in the repository: the tests generate their synthetic videos with
`make_test_video.py`.

## 2. Models

| Model / code | What Kinetrace uses it for | How it gets onto a computer | Licence | Commercial use | What the licence asks of you |
|---|---|---|---|---|---|
| **CoTracker3** (Meta; code via `torch.hub` + weights) | point tracking (the second point model) | downloaded on first Track into `models/` | **CC BY-NC 4.0** | **No** | attribution; non-commercial use only; share the licence notice |
| **AllTracker** (Harley et al., ICCV 2025) | point tracking (the default model) | code fetched by `install.py` from `github.com/aharley/alltracker` at a pinned commit into `models/alltracker`, 63 MB checkpoint downloaded from `huggingface.co/aharley/alltracker` on the first Track | **MIT** (code; the README gives no separate licence for the weights) | yes | keep the copyright and licence notice |
| **SAM 2.1** hiera-base-plus (Meta) | silhouettes, ball markers (fallback) | downloaded on first use into `models/hf` | **Apache-2.0** | yes | keep the licence and NOTICE; state changes |
| **SAM 3 / SAM 3.1** (Meta, gated) | silhouettes, ball markers (preferred) | the user requests access on Hugging Face and places the weights in `models/sam3` | **SAM License** (Meta, 19 Nov 2025) | yes | pass on a copy of the SAM License with the weights or any derivative; **acknowledge SAM in publications**; comply with trade controls — no military, weapons, nuclear or espionage use; no reverse engineering |
| **SAM 3D Body** (Meta, gated; code + ViT-H or DINOv3 weights, incl. the MHR rig asset) | 3D human body pose + mesh | the user places Meta's repo in `models/sam-3d-body` and the weights in `models/sam-3d-body-*` | **SAM License** | yes | as for SAM 3 |
| **DINOv3** (Meta) | backbone of the SAM 3D Body DINOv3 variant | fetched with that variant into `models/` | **DINOv3 License** (19 Aug 2025) | yes | same terms as the SAM License: pass on a copy, acknowledge in publications, trade controls |
| **ViTPose** base-simple (`usyd-community`) | 2D human joints | downloaded on first use into `models/hf` | **Apache-2.0** | yes | keep the licence notice |
| **RT-DETR v2** r18vd (`PekingU`) | finding people for both body backends | downloaded on first use into `models/hf` | **Apache-2.0** | yes | keep the licence notice |

The full texts of the non-standard licences are in [`LICENSES/`](LICENSES/):
the SAM License, the DINOv3 License, CC BY-NC 4.0 (as shipped with CoTracker3)
and AllTracker's MIT licence.

**What this means in practice:**

- **Academic, non-commercial research can use every part of Kinetrace.**
- **Commercial use cannot use CoTracker3.** AllTracker (MIT), the default point
  model, and every other model allow commercial use, so the app still works
  without CoTracker3 — but whoever uses it commercially must make sure
  CoTracker3 is never selected (Track ▾ → point model).
- **Papers that use silhouettes, ball markers made with SAM 3, or SAM 3D Body
  must acknowledge SAM** (and DINOv3 for that SAM 3D Body variant). Cite
  CoTracker3 / AllTracker for point tracks as good practice.
- **Gated weights (SAM 3, SAM 3.1, SAM 3D Body) must not be put in this
  repository or handed out with the app** unless a copy of the SAM License goes
  with them; the simple rule is that each lab requests access and downloads
  them itself.

## 3. Python packages

Installed by `run.bat` / `run.sh` from `requirements.txt` into `.venv/` on the
user's computer; none of them is in this repository.

| Package | Version | Licence | Used for |
|---|---|---|---|
| aiohappyeyeballs | 2.7.1 | PSF-2.0 | indirect dependency |
| aiohttp | 3.14.3 | Apache-2.0 AND MIT | indirect dependency |
| aiosignal | 1.4.0 | Apache 2.0 | indirect dependency |
| annotated-doc | 0.0.5 | MIT | indirect dependency |
| antlr4-python3-runtime | 4.9.3 | BSD | indirect dependency |
| anyio | 4.15.1 | MIT | indirect dependency |
| attrs | 26.1.0 | MIT | indirect dependency |
| braceexpand | 0.1.7 | MIT | SAM 3D Body |
| certifi | 2026.7.22 | MPL-2.0 | indirect dependency |
| click | 8.5.0 | BSD-3-Clause | indirect dependency |
| colorama | 0.4.6 | BSD License | indirect dependency |
| einops | 0.8.2 | MIT | AllTracker model code |
| filelock | 3.29.5 | MIT | indirect dependency |
| frozenlist | 1.8.0 | Apache-2.0 | indirect dependency |
| fsspec | 2026.6.0 | BSD-3-Clause | indirect dependency |
| h11 | 0.16.0 | MIT | indirect dependency |
| hf-xet | 1.6.0 | Apache-2.0 | indirect dependency |
| httpcore | 1.0.9 | BSD-3-Clause | indirect dependency |
| httpx | 0.28.1 | BSD-3-Clause | indirect dependency |
| huggingface_hub | 1.31.0 | Apache-2.0 | model downloads into models/hf |
| idna | 3.19 | BSD-3-Clause | indirect dependency |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause | sound sync: a static ffmpeg binary (see below) |
| Jinja2 | 3.1.6 | BSD License | indirect dependency |
| lightning-utilities | 0.15.3 | Apache-2.0 | indirect dependency |
| markdown-it-py | 4.2.0 | MIT License | indirect dependency |
| MarkupSafe | 3.0.3 | BSD-3-Clause | indirect dependency |
| mdurl | 0.1.2 | MIT License | indirect dependency |
| mpmath | 1.3.0 | BSD | indirect dependency |
| multidict | 6.9.0 | Apache License 2.0 | indirect dependency |
| networkx | 3.6.1 | BSD-3-Clause | indirect dependency |
| numpy | 2.5.1 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | arrays everywhere |
| omegaconf | 2.3.1 | BSD License | SAM 3D Body config |
| opencv-python | 5.0.0.93 | Apache 2.0 | video decoding, image processing, calibration maths |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | indirect dependency |
| pillow | 12.3.0 | MIT-CMU | images for the SAM processor |
| propcache | 0.5.4 | Apache-2.0 | indirect dependency |
| Pygments | 2.21.0 | BSD-2-Clause | indirect dependency |
| PySide6 | 6.11.1 | see the package's LICENSE file | the user interface (Qt 6) |
| PySide6_Addons | 6.11.1 | see the package's LICENSE file | Qt 6 modules |
| PySide6_Essentials | 6.11.1 | see the package's LICENSE file | Qt 6 modules |
| pytorch-lightning | 2.6.6 | Apache-2.0 | SAM 3D Body checkpoint loading |
| PyYAML | 6.0.3 | MIT | indirect dependency |
| regex | 2026.9.10 | Apache-2.0 AND CNRI-Python | indirect dependency |
| rich | 15.0.0 | MIT | indirect dependency |
| roma | 1.6.1 | BSD-3-Clause | SAM 3D Body rotations |
| safetensors | 0.8.0 | Apache Software License | weight files |
| scipy | 1.18.0 | BSD License | optimisation, filtering, splines |
| shellingham | 1.5.4 | ISC License | indirect dependency |
| shiboken6 | 6.11.1 | see the package's LICENSE file | Qt binding runtime |
| sympy | 1.14.0 | BSD | indirect dependency |
| termcolor | 3.3.0 | MIT | SAM 3D Body logging |
| timm | 1.0.29 | Apache-2.0 | SAM 3D Body backbone |
| tokenizers | 0.23.2 | Apache Software License | transformers dependency |
| tomli | ≥ 2 | MIT | reads Anipose `calibration.toml`; installed only on Python 3.10 (3.11+ has `tomllib` built in) |
| torch | 2.12.1+cu130 | BSD-3-Clause | deep-learning runtime (CUDA build; its wheel bundles NVIDIA CUDA runtime libraries under NVIDIA's CUDA EULA) |
| torchmetrics | 1.9.0 | Apache-2.0 | indirect dependency |
| torchvision | 0.27.1+cu130 | BSD | image ops for the SAM image processor |
| tqdm | 4.70.0 | MPL-2.0 AND MIT | indirect dependency |
| transformers | 5.17.0 | Apache 2.0 License | SAM 2.1 / SAM 3, ViTPose, RT-DETR model code |
| typer | 0.27.2 | MIT | indirect dependency |
| typing_extensions | 4.16.0 | PSF-2.0 | indirect dependency |
| yacs | 0.1.8 | Apache Software License | SAM 3D Body config |
| yarl | 1.25.1 | Apache-2.0 | indirect dependency |

**A few licences need a sentence each:**

- **PySide6 / Qt 6 — LGPL-3.0** (also offered as GPL-2.0 / GPL-3.0 or a
  commercial licence). Kinetrace imports it as an ordinary, replaceable
  Python package, which the LGPL allows under any licence for Kinetrace's own
  code. If the app is ever shipped as a frozen bundle (PyInstaller or
  similar), the bundle must keep the Qt libraries replaceable and include the
  LGPL text and Qt's notices.
- **ffmpeg inside `imageio-ffmpeg`** — the Python wrapper is BSD-2-Clause, but
  the ffmpeg executable it carries is a **GPL** build (gyan.dev, ffmpeg 7.1).
  Kinetrace runs it as a separate program through a pipe (no linking), which
  does not bring the GPL onto Kinetrace's code. Whoever redistributes that
  executable itself must follow the GPL (licence text + source offer); a user
  who installs it with pip gets it from PyPI directly.
- The PyTorch CUDA wheel bundles NVIDIA CUDA runtime libraries, redistributed
  by PyTorch under NVIDIA's CUDA EULA; nothing extra is needed to use them.
- `certifi` and `tqdm` are MPL-2.0 (file-level copyleft); used unmodified, no
  obligation beyond keeping their notices.

## 4. Datasets and footage

No footage or annotation data is included. The tests build their own
synthetic videos with known ground truth.

## 5. Can Kinetrace be released as open source for academic use?

**Yes.** Nothing in the licences above stands in the way of publishing
Kinetrace's own source code for other labs:

1. The repository redistributes no third-party code or weights, so the model
   licences bind the **users** (each lab accepts them when it downloads a
   model), not the act of publishing Kinetrace.
2. Every package licence is permissive or weak-copyleft (LGPL, MPL), all of
   which allow Kinetrace's own code under any licence, open or closed.
3. The only non-commercial piece, CoTracker3, is optional: the default point
   model (AllTracker) is MIT.

**Choosing a licence for Kinetrace's own code** (the authors' decision):

| Option | Effect | Fits |
|---|---|---|
| **BSD-3-Clause** or **MIT** | anyone may use, change and redistribute, including commercially; keep the notice | widest academic uptake; the usual choice for research software |
| **Apache-2.0** | as MIT, plus an explicit patent grant and NOTICE file | the same, for code that may meet industry partners |
| **GPL-3.0** | open, and every redistributed derivative must stay open under the GPL | keeping forks open |
| **PolyForm Noncommercial 1.0** | source available, **no commercial use** (not "open source" by the OSI definition) | a "non-commercial use only" rule |

A permissive licence for the app plus a clear notice that CoTracker3 is
non-commercial is the common pattern (it is how most tools built on
CoTracker or SAM are published). If the "non-commercial only" rule must bind
the app itself, PolyForm Noncommercial says that in software terms; Creative
Commons licences are not recommended for code.

**Before the repository is made public**, add the `LICENSE` file, check the
institution's policy on releasing software written there and any grant's
software-sharing terms, and add a `CITATION.cff` so other labs can cite
Kinetrace alongside the model papers.
