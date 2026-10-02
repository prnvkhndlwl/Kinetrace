"""Everything Kinetrace downloads, in one place (I151-I154, G45; no Qt, stdlib only
at import): each file PINNED to one upstream commit and CHECKED before it is
used, fetched with a timeout, in chunks that report progress, and stoppable.

  * model weights (`FILES`): a pinned URL + the file's sha256; loaded with
    `load_weights` = torch.load(weights_only=True) -- tensors only, never code.
  * model code (`CODE`): a commit's zip from GitHub, unpacked only when every
    member stays inside the target folder, and used only when the digest of
    its Python files (`code_digest`) is the pinned one.
  * Hugging Face models (`HF_REVISIONS`): loaded at a pinned commit; with the
    commit's files in models/hf nothing is asked of the Hub, and a missing
    snapshot is fetched here with progress (`hf_snapshot`).

The pins were taken from the copies every test ran against (2026-09-30): the
checkpoints' sha256 equal the Hub's own (X-Linked-ETag), and the code folders
equal the commits' git blobs file for file. Bumping a model = new pin here.

A download never starts on its own: only when a model is first needed (Track,
S, a Body run) or by install.py. Nothing but the file itself is requested."""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TIMEOUT_S = 30              # no byte for this long = the download stalled
CHUNK = 1 << 20
RETRIES = 3


class DownloadError(Exception):
    """A download that did not complete, in a sentence for the user."""


class DownloadCancelled(DownloadError):
    pass


@dataclass(frozen=True)
class File:
    label: str
    url: str
    sha256: str
    size: int
    dest: Path


@dataclass(frozen=True)
class Code:
    label: str
    repo: str               # owner/name on GitHub
    commit: str
    digest: str             # code_digest() of the commit's Python files
    dest: Path
    marker: str             # a file that exists once the code is in place


FILES = {
    "alltracker": File("the AllTracker point model",
                       "https://huggingface.co/aharley/alltracker/resolve/"
                       "c8ba31225828cab9ec512d0627f0f826e641a877/alltracker.pth",
                       "ffd9ebcfb6d206d594b646999a150540f92c049cf9b2bf940facf7123f62aa1d", 66005722,
                       MODELS_DIR / "checkpoints" / "alltracker.pth"),
    "cotracker3": File("the CoTracker3 point model",
                       "https://huggingface.co/facebook/cotracker3/resolve/"
                       "bf55ea50d4390e1820a267f131cd6587240fb2c5/scaled_online.pth",
                       "205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218", 101695610,
                       MODELS_DIR / "checkpoints" / "scaled_online.pth"),
}

CODE = {
    "alltracker": Code("AllTracker's code (MIT)", "aharley/alltracker", "e7553135e7b361590dbccd10e2b274b024f41cd6",
                       "967075df6445e13acb0277408d92ab0f9bb9e8bc70de05248778f43770da3d11",
                       MODELS_DIR / "alltracker", "nets/alltracker.py"),
    # the folder name torch.hub gave it, so existing installs keep theirs
    "cotracker3": Code("CoTracker3's code (CC BY-NC 4.0)", "facebookresearch/co-tracker",
                       "82e02e8029753ad4ef13cf06be7f4fc5facdda4d",
                       "c5a1a84d1db69f4234566c5a5700ff1da38871de4421bcf46c2a81508c377943",
                       MODELS_DIR / "facebookresearch_co-tracker_main", "hubconf.py"),
}

# Hugging Face repo -> the commit every test ran against
HF_REVISIONS = {
    "facebook/sam2.1-hiera-base-plus": "b7320756a13354e7530a63935656d35b2f91a290",
    "facebook/sam2.1-hiera-large": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
    "usyd-community/vitpose-base-simple": "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45",
    "PekingU/rtdetr_v2_r18vd": "5650961749fa93567c0d46fc7f43ea4f9e914107",
}


def _mb(n: float) -> str:
    return f"{n / 1e6:.0f} MB"


def _no_net(label: str, e: Exception) -> DownloadError:
    return DownloadError(f"{label[0].upper()}{label[1:]} could not be downloaded ({_why(e)}). Check the internet "
                         "connection and try again: it is needed only once, then the file stays in models/.")


def _why(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        return f"the server answered {e.code}"
    if isinstance(e, urllib.error.URLError):          # before the file arrived: no connection at all
        return "no connection to the server"
    if isinstance(e, (TimeoutError, OSError)) and "timed out" in str(e).lower():
        return "the connection stalled"
    return type(e).__name__


def fetch(url: str, dest: Path, *, label: str, sha256: str | None = None, size: int | None = None,
          progress=None, cancel=lambda: False, timeout: float = TIMEOUT_S) -> Path:
    """`url` -> `dest`, through `dest.part` (a cut-off download resumes from it),
    in 1 MB chunks: `progress(done_bytes, total_bytes)` after each, `cancel()`
    checked between them, `timeout` seconds without a byte = stalled. With
    `sha256` the file is moved into place only when it matches."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(RETRIES):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "Kinetrace",
                                                   **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if have and r.status != 206:            # the server ignored the range: start again
                    have = 0
                total = size or (int(r.headers.get("Content-Length") or 0) + have) or 0
                with open(part, "ab" if have else "wb") as fh:
                    done = have
                    while True:
                        if cancel():
                            raise DownloadCancelled(f"The download of {label} was stopped.")
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        fh.write(buf)
                        done += len(buf)
                        if progress is not None:
                            progress(done, total)
            break
        except DownloadCancelled:
            raise
        except urllib.error.HTTPError as e:
            if e.code == 416:                            # the part file is already complete
                break
            last = e
            if e.code < 500:
                raise _no_net(label, e) from None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = e
        time.sleep(1.0 + attempt)
    else:
        raise _no_net(label, last) from None
    if sha256 is not None:
        got = file_sha256(part)
        if got != sha256:
            part.unlink(missing_ok=True)
            raise DownloadError(f"The downloaded file of {label} is not the expected one (its checksum differs), "
                                "so it was deleted and not used. Try again later; if it keeps happening, "
                                "update Kinetrace (Help > Check for Updates).")
    os.replace(part, dest)
    return dest


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(CHUNK), b""):
            h.update(buf)
    return h.hexdigest()


def ensure_file(key: str, progress=None, cancel=lambda: False) -> Path:
    """The model file `key` of FILES, downloaded and checked when missing.
    progress(label, done_bytes, total_bytes)."""
    f = FILES[key]
    if f.dest.is_file():
        return f.dest
    cb = None if progress is None else (lambda d, t: progress(f"Downloading {f.label}", d, t or f.size))
    return fetch(f.url, f.dest, label=f.label, sha256=f.sha256, size=f.size, progress=cb, cancel=cancel)


def load_weights(path: Path):
    """A checkpoint as tensors only: weights_only=True never runs code from
    the file (a full unpickle would)."""
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=True)


# ------------------------------------------------------------------ model code
def code_digest(folder: Path) -> str:
    """sha256 over the folder's Python files (path + content, line ends as
    LF: a git checkout on Windows has CRLF, the commit's zip LF)."""
    folder = Path(folder)
    h = hashlib.sha256()
    files = []
    for d, dirs, fs in os.walk(folder):
        dirs[:] = [x for x in dirs if x not in ("__pycache__", ".git")]
        files += [Path(d, f) for f in fs if f.endswith(".py")]
    for p in sorted(files, key=lambda p: p.relative_to(folder).as_posix()):
        data = p.read_bytes().replace(b"\r\n", b"\n")
        h.update(p.relative_to(folder).as_posix().encode() + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest()


def code_present(key: str) -> bool:
    c = CODE[key]
    return (c.dest / c.marker).is_file()


def ensure_code(key: str, progress=None, cancel=lambda: False) -> Path:
    """The model code `key` of CODE: the pinned commit's zip, unpacked beside
    the target and moved into place only when every member is safe and the
    Python files' digest is the pinned one."""
    c = CODE[key]
    if code_present(key):
        return c.dest
    url = f"https://github.com/{c.repo}/archive/{c.commit}.zip"
    tmp = c.dest.with_name(c.dest.name + ".download")
    zpath = tmp.with_suffix(".zip")
    cb = None if progress is None else (lambda d, t: progress(f"Downloading {c.label}", d, t))
    fetch(url, zpath, label=c.label, progress=cb, cancel=cancel)
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            top = names[0].split("/", 1)[0] + "/" if names else ""
            for name in names:
                rel = name[len(top):] if name.startswith(top) else None
                parts = PurePosixPath(rel or "").parts
                if rel is None or rel.startswith("/") or "\\" in rel or ":" in rel or ".." in parts:
                    raise DownloadError(f"{c.label}: the archive holds an unsafe path ({name[:80]!r}); not used.")
                if not rel or name.endswith("/"):
                    continue
                out = tmp.joinpath(*parts)
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(out, "wb") as fh:
                    shutil.copyfileobj(src, fh)
        if code_digest(tmp) != c.digest:
            raise DownloadError(f"The downloaded {c.label} is not the expected version (its checksum differs), "
                                "so it was not used. Try again later, or update Kinetrace.")
        if c.dest.exists():
            shutil.rmtree(c.dest)
        os.replace(tmp, c.dest)
    except zipfile.BadZipFile:
        raise DownloadError(f"The download of {c.label} is damaged; try again.") from None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        zpath.unlink(missing_ok=True)
    return c.dest


# ------------------------------------------------------------------ Hugging Face
def hf_revision(repo: str) -> str | None:
    return HF_REVISIONS.get(repo)


def hf_load_args(repo: str) -> dict:
    """from_pretrained's arguments for `repo`: its pinned commit, and -- once
    that commit's files are in models/hf -- local_files_only, so that a load
    asks the Hub nothing (without it a processor probes a dozen optional files
    it does not have: 77 s of retries with no network, measured)."""
    rev = hf_revision(repo)
    if rev is None:
        return {}
    return {"revision": rev, **({"local_files_only": True} if hf_cached(repo) else {})}


def hf_cached(repo: str) -> bool:
    """The pinned commit's snapshot (with its weights) is in models/hf."""
    rev = hf_revision(repo)
    hub = MODELS_DIR / "hf" / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
    if rev is None:
        return hub.is_dir() and any(any(p.glob("*.safetensors")) for p in hub.iterdir() if p.is_dir())
    return any((hub / rev).glob("*.safetensors")) if (hub / rev).is_dir() else False


def hf_snapshot(repo: str, label: str, progress=None, cancel=lambda: False) -> None:
    """Download `repo` at its pinned commit into models/hf, reporting bytes
    (the Hub's own progress bars, re-routed) and stoppable between chunks."""
    if hf_cached(repo):
        return
    from huggingface_hub import snapshot_download
    from tqdm import tqdm as _base

    seen = {}       # bar description -> (done, total): the Hub keeps a network bar and a written-bytes bar

    class _Bar(_base):
        def __init__(self, *a, **k):
            k["disable"] = False
            k.pop("name", None)
            super().__init__(*a, **k)

        def update(self, n=1):
            if cancel():
                raise DownloadCancelled(f"The download of {label} was stopped.")
            r = super().update(n)
            if progress is not None and self.unit == "B":
                seen[str(self.desc)[:12]] = (float(self.n), float(self.total or 0))
                done, total = max(seen.values())
                progress(f"Downloading {label}", done, total)
            return r

        def refresh(self, *a, **k):            # no console (pythonw): nothing is drawn
            return None

        def display(self, *a, **k):
            return None

    try:
        snapshot_download(repo, revision=hf_revision(repo), tqdm_class=_Bar,
                          cache_dir=str(MODELS_DIR / "hf" / "hub"))
    except DownloadCancelled:
        raise
    except Exception as e:  # noqa: BLE001 - the Hub's many errors, as one sentence
        low = f"{type(e).__name__} {e}".lower()
        if any(k in low for k in ("gated", "401", "403", "unauthorized", "restricted")):
            raise DownloadError(f"{label[0].upper()}{label[1:]} is gated on Hugging Face: request access to "
                                f"https://huggingface.co/{repo}, then paste a read token in Settings.") from None
        raise _no_net(label, e) from None
