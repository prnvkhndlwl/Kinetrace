"""Checking for, and installing, a newer published Kinetrace (G37; no Qt,
standard library only, so the update scripts work even when the app will not
start).

A new version is a GitHub Release `vX.Y.Z`, published by
`.github/workflows/release.yml` when `APP_VERSION` changes on main. The check
is ONE GET to GitHub's releases API, made only when the user asks (Help ->
Check for Updates..., or `python -m kinetrace.update`); nothing identifying is
sent and nothing is checked in the background.

Installing it depends on how the folder was obtained:

* a `git clone` -> `git fetch` the release tag and fast-forward to it, refused
  (with the reason) when a tracked file has local changes;
* a ZIP download -> download the release's ZIP into `update/`, check it really
  is a Kinetrace release, and write ONLY the files that changed. What each user
  owns is never touched (`PROTECTED`: the environment, models, recovery copies,
  logs, saved skeletons, footage). Files the previous release had and this one
  dropped are removed, using the list kept in `MANIFEST`.

When `requirements.txt` or `install.py` changed, the install marker
(`.venv/kinetrace-install.json`) is deleted, so the launcher runs `install.py`
again on the next start and installs whatever the new version needs (I142).

    python -m kinetrace.update            check, and install a newer version
    python -m kinetrace.update --check    only say whether there is one
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from kinetrace import APP_VERSION

REPO = "prnvkhndlwl/Kinetrace"
PAGE = f"https://github.com/{REPO}"
ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ".kinetrace-release.json"      # the files the installed release put here
STAGING = "update"                         # download + unpack folder, inside the install
# top-level folders (and files) an update never writes or removes: each user's own
PROTECTED = (".venv", "models", "recovery", "logs", "skeletons", "test_videos", "tests/out",
             ".git", STAGING, MANIFEST, "settings.ini")
# a changed one of these means the environment may need something new (I142)
INSTALL_INPUTS = ("requirements.txt", "install.py")
MARKER = Path(".venv") / "kinetrace-install.json"
TIMEOUT_S = 15


class UpdateError(Exception):
    """Something the user can act on, said in one or two plain sentences."""


@dataclass
class Release:
    version: str                   # "1.3.0"
    tag: str                       # "v1.3.0"
    name: str
    notes: str                     # markdown, as written on the release
    zip_url: str
    page_url: str
    published: str = ""            # ISO date


@dataclass
class ApplyResult:
    version: str
    how: str                       # "git" | "zip"
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    reinstall: bool = False        # the launcher will run install.py on the next start

    def sentence(self) -> str:
        if self.how == "git":
            what = "the folder was moved to the release"
        else:
            what = f"{len(self.changed)} file(s) updated" + (f", {len(self.removed)} removed" if self.removed else "")
        extra = (" The next start installs what the new version needs first (a few minutes, needs the internet)."
                 if self.reinstall else "")
        return f"Kinetrace {self.version} is installed ({what}). Restart Kinetrace to use it.{extra}"


# ------------------------------------------------------------------ versions

def parse_version(text: str) -> tuple[int, ...]:
    """"v1.10.2" -> (1, 10, 2); anything that is not a number counts as 0."""
    parts = re.split(r"[.\-+]", str(text).strip().lstrip("vV"))
    out = []
    for p in parts[:4]:
        m = re.match(r"\d+", p)
        out.append(int(m.group()) if m else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)


def is_newer(remote: str, local: str = APP_VERSION) -> bool:
    return parse_version(remote) > parse_version(local)


# (G179) Nothing is checked unless the user asks, so a user who never asks never hears of a fix. The
# app counts the days instead (no network): this long without a check, a note at start says so ...
REMIND_AFTER_DAYS = 30
REMIND_EVERY_DAYS = 7        # ... and not again sooner, until the user checks


def reminder_due(now: float, last_check: float, since: float, last_reminded: float) -> bool:
    """Whether to remind the user to check for updates (times in seconds since the epoch, 0 = never):
    `REMIND_AFTER_DAYS` since the last check (or, never checked, since `since`: the first start that
    counted), and `REMIND_EVERY_DAYS` since the last reminder. Offline: it only compares dates."""
    day = 86400.0
    base = max(float(last_check or 0), float(since or 0))
    return (base > 0 and now - base >= REMIND_AFTER_DAYS * day
            and now - float(last_reminded or 0) >= REMIND_EVERY_DAYS * day)


# ------------------------------------------------------------------- network

def api_base() -> str:
    """GitHub's API for this repository; `KINETRACE_UPDATE_API` points it at a
    test server."""
    return os.environ.get("KINETRACE_UPDATE_API", f"https://api.github.com/repos/{REPO}").rstrip("/")


def _open(url: str, timeout: float = TIMEOUT_S):
    req = urllib.request.Request(url, headers={
        "User-Agent": f"Kinetrace/{APP_VERSION}",          # GitHub refuses requests without one
        "Accept": "application/vnd.github+json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise _NotFound(url) from None
        if e.code in (403, 429):
            raise UpdateError(
                "GitHub is limiting how often it can be asked from this internet connection "
                "(60 times an hour). Try again in an hour.") from None
        raise UpdateError(f"GitHub answered with an error ({e.code} {e.reason}). Try again later.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        why = getattr(e, "reason", e)
        raise UpdateError(
            "Could not reach GitHub to look for a newer version. Check the internet connection "
            f"(a proxy or firewall can block it too). Details: {why}") from None


class _NotFound(UpdateError):
    """GitHub answered 404 for this URL (said in words by `latest_release`)."""


def latest_release(timeout: float = TIMEOUT_S) -> Release:
    try:
        r = _open(api_base() + "/releases/latest", timeout)
    except _NotFound:
        # GitHub answers 404 both for a repository without a release and for one it will not
        # show this computer (private): ask for the repository itself to tell them apart
        # (Mac install audit P0-3; "no release yet" was said for a private repository)
        try:
            _open(api_base(), timeout).close()
        except _NotFound:
            raise UpdateError(
                "GitHub does not show the Kinetrace repository to this computer: it is private (or "
                "was moved), so updates cannot be looked up or downloaded without an account that "
                f"was given access. Ask the person who shared Kinetrace with you. Project page: {PAGE}") from None
        raise UpdateError(
            "GitHub has no published version of Kinetrace to offer yet (the repository has no "
            f"release). The project page is {PAGE}.") from None
    with r:
        try:
            data = json.loads(r.read().decode("utf-8"))
        except ValueError:
            raise UpdateError("GitHub's answer could not be read. Try again later.") from None
        except (OSError, http.client.HTTPException):     # the connection dropped while reading (G94)
            raise UpdateError("The connection to GitHub was cut while reading its answer. Check the internet "
                              "connection and try again.") from None
    tag = str(data.get("tag_name") or "")
    if not tag or not data.get("zipball_url"):
        raise UpdateError("The newest release on GitHub has no version number or download.")
    return Release(version=tag.lstrip("vV"), tag=tag, name=str(data.get("name") or tag),
                   notes=str(data.get("body") or ""), zip_url=str(data["zipball_url"]),
                   page_url=str(data.get("html_url") or PAGE), published=str(data.get("published_at") or "")[:10])


def check(local: str = APP_VERSION, timeout: float = TIMEOUT_S) -> tuple[Release, bool]:
    """(the newest release, whether it is newer than `local`)."""
    rel = latest_release(timeout)
    return rel, is_newer(rel.version, local)


# ------------------------------------------------------------- how installed

def install_kind(root: Path = ROOT) -> str:
    return "git" if (Path(root) / ".git").exists() else "zip"


_GIT_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _git_candidates() -> list[list[str]]:
    """The gits to try, in order. A Mac app started from Finder sees only the system folders, so
    `which git` finds Apple's /usr/bin/git: a stand-in that needs the Command Line Tools and fails
    when they are missing or broken (owner's Mac 2026-10-09, after a macOS upgrade: "unable to load
    libxcrun ... fat file"). Homebrew's git, when there is one, is tried first."""
    out: list[list[str]] = []
    if sys.platform == "darwin":
        out += [[p] for p in ("/opt/homebrew/bin/git", "/usr/local/bin/git") if os.path.exists(p)]
    found = shutil.which("git")
    if found and [found] not in out:
        out.append([found])
    return out


_GIT_CMD: list[str] | None = None
_GIT_PROBLEM = ""             # what the last git tried said when none works (for `git_blocker`)


def git_cmd() -> list[str] | None:
    """The first git that RUNS (`git --version`), or None; remembered once one works."""
    global _GIT_CMD, _GIT_PROBLEM
    if _GIT_CMD is not None:
        return _GIT_CMD
    for cmd in _git_candidates():
        try:
            r = subprocess.run([*cmd, "--version"], capture_output=True, text=True, timeout=20,
                               creationflags=_GIT_FLAGS)
        except (OSError, subprocess.SubprocessError) as e:
            _GIT_PROBLEM = str(e)
            continue
        if r.returncode == 0:
            _GIT_CMD = cmd
            return cmd
        _GIT_PROBLEM = (r.stderr or r.stdout).strip()
    return None


def _git(root: Path, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    cmd = git_cmd() or ["git"]
    return subprocess.run([*cmd, "-C", str(root), *args], capture_output=True, text=True,
                          timeout=timeout, creationflags=_GIT_FLAGS)


def _no_git_sentence() -> str:
    """Why there is no working git here, and what to do -- in words (G94: never a raw error)."""
    manual = ("Or update by hand with a git that works (`git pull` in the Kinetrace folder), or download the "
              "new version from the release page.")
    if sys.platform == "darwin":
        broken = "xcrun" in _GIT_PROBLEM or "CommandLineTools" in _GIT_PROBLEM or "developer tools" in _GIT_PROBLEM
        return ("This folder is a git clone, so it is updated with git, but git does not work on this Mac: "
                + ("Apple's Command Line Tools, which provide it, need repairing (this often happens after a "
                   "macOS upgrade). " if broken or not _git_candidates() else "")
                + "To repair them, open Terminal and run `xcode-select --install` (if it says they are already "
                  "installed, run `sudo rm -rf /Library/Developer/CommandLineTools` first), then try again. "
                + manual)
    return "This folder is a git clone, so it is updated with git, but git does not work here. " + manual


def git_blocker(root: Path = ROOT) -> str | None:
    """Why a git checkout cannot be updated automatically, or None."""
    if git_cmd() is None:
        return _no_git_sentence()
    r = _git(root, "status", "--porcelain", "--untracked-files=no")
    if r.returncode != 0:
        said = (r.stderr or r.stdout).strip()
        return (f"git could not read this folder ({said[:400]}). Update it by hand (`git pull` in the "
                "Kinetrace folder), or download the new version from the release page.")
    if r.stdout.strip():
        n = len(r.stdout.strip().splitlines())
        return (f"{n} file(s) of Kinetrace itself were changed in this folder, so updating would "
                "overwrite those changes. Commit or undo them (git status lists them), then try again.")
    return None


# -------------------------------------------------------------------- apply

def _protected(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return any(rel == p or rel.startswith(p + "/") for p in PROTECTED)


def _read_manifest(root: Path) -> dict:
    try:
        return json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _replace(src: Path, dst: Path, tries: int = 10) -> None:
    """os.replace, retried: Windows refuses while another program has the file open."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.2)


GITHUB_HOSTS = ("api.github.com", "codeload.github.com", "github.com")


def trusted_url(url: str) -> bool:
    """A release archive may come from GitHub over https only -- or from the
    test server `KINETRACE_UPDATE_API` names (I155): the link is read from an
    answer, and an answer is not a reason to download from anywhere else."""
    from urllib.parse import urlsplit
    u = urlsplit(url)
    if u.scheme == "https" and (u.hostname or "").lower() in GITHUB_HOSTS:
        return True
    api = urlsplit(api_base())
    return bool(os.environ.get("KINETRACE_UPDATE_API")) and (u.scheme, u.netloc) == (api.scheme, api.netloc)


def download(url: str, dest: Path, progress=None, timeout: float = 60) -> Path:
    """Fetch `url` into `dest`; progress(done_bytes, total_bytes or 0). Built on
    `downloads.fetch` (R21): retries, a resume from the kept `.part`, a stall
    timeout, and a connection cut short counts as a failure (I236) -- all said as
    an UpdateError sentence (G94)."""
    if not trusted_url(url):
        raise UpdateError(f"The release's download link does not point to GitHub ({url[:80]}), so it was not "
                          f"used. Download the new version from {PAGE} instead.")
    from kinetrace import downloads
    try:
        return downloads.fetch(url, dest, label="the new version of Kinetrace", progress=progress, timeout=timeout,
                               hint="Check the internet connection and try again.")
    except downloads.DownloadError as e:
        raise UpdateError(str(e)) from None


def _release_members(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """{path inside the repository: member}. GitHub's archives put everything in
    one top folder (owner-repo-sha/); anything unsafe refuses the whole file."""
    out = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/")
        parts = name.split("/", 1)
        if len(parts) < 2 or not parts[1]:
            continue
        rel = parts[1]
        if rel.startswith("/") or ".." in rel.split("/") or ":" in rel:
            raise UpdateError(f"The downloaded file contains an unsafe path ({name}); nothing was changed.")
        out[rel] = info
    if "kinetrace/__init__.py" not in out or "install.py" not in out:
        raise UpdateError("The downloaded file is not a Kinetrace release; nothing was changed.")
    return out


def apply_zip(root: Path, zip_path: Path, version: str, tag: str = "") -> ApplyResult:
    """Install the release in `zip_path` over the folder `root` (see the module
    docstring for what is and is not touched)."""
    root = Path(root)
    res = ApplyResult(version=version, how="zip")
    try:
        zf = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile):
        raise UpdateError("The download is damaged (not a readable ZIP); nothing was changed. Try again.") from None
    old_manifest = _read_manifest(root)
    with zf:
        members = _release_members(zf)
        # (I218) what the installer was last run for: the digests the last COMPLETED update
        # recorded, else what is on disk. A half-failed update replaces install.py /
        # requirements.txt but never reaches the manifest, so a retry still sees the change.
        recorded = old_manifest.get("inputs")
        before = recorded if isinstance(recorded, dict) else {n: _digest(root / n) for n in INSTALL_INPUTS}
        after = {n: (hashlib.sha256(zf.read(members[n])).hexdigest() if n in members else None)
                 for n in INSTALL_INPUTS}
        # unpack every file that differs into the staging folder FIRST, so a bad
        # archive fails before anything in the install has been touched
        stage = root / STAGING / "files"
        shutil.rmtree(stage, ignore_errors=True)
        todo = []
        for rel, info in members.items():
            if _protected(rel):
                continue
            data = zf.read(info)
            target = root / rel
            if target.is_file() and hashlib.sha256(data).hexdigest() == _digest(target):
                continue
            s = stage / rel
            s.parent.mkdir(parents=True, exist_ok=True)
            s.write_bytes(data)
            mode = (info.external_attr >> 16) & 0o777
            if mode and os.name != "nt":
                os.chmod(s, mode)
            todo.append(rel)
    inputs_changed = any(after[n] != before.get(n) for n in INSTALL_INPUTS)
    if recorded is None and (root / ".venv").is_dir() and not (root / MARKER).exists():
        inputs_changed = True       # an earlier, half-failed update (before digests were recorded) dropped it
    dropped = False
    if inputs_changed:
        # (I218) the marker goes BEFORE the first replacement: if a file is locked half-way, the
        # next start runs install.py again instead of trusting the old marker with new inputs
        dropped = _drop_marker(root)
    old = set(old_manifest.get("files", []))
    old_by_lower: dict[str, list[str]] = {}
    for o in old:
        old_by_lower.setdefault(o.lower(), []).append(o)
    for rel in todo:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # (I219) a release that renames a file by letter case only: on a case-insensitive file
        # system the old name IS the target, so remove it just before the new name is written
        # (the staged copy exists), and never remove it afterwards
        for o in old_by_lower.get(rel.lower(), []):
            if o != rel and not _protected(o):
                try:
                    (root / o).unlink()
                except OSError:
                    pass
        _replace(stage / rel, target)
        res.changed.append(rel)
    member_lower = {m.lower() for m in members}
    for rel in sorted(old - set(members)):
        if _protected(rel) or rel.lower() in member_lower:      # (I219) a case-only rename is not a removal
            continue
        p = root / rel
        try:
            if p.is_file():
                p.unlink()
                res.removed.append(rel)
        except OSError:
            pass
    (root / MANIFEST).write_text(json.dumps({
        "version": version, "tag": tag, "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": sorted(members), "inputs": after}, indent=1), encoding="utf-8")
    # reinstall = there is an install to redo: the marker was there (dropped now) or an earlier,
    # half-failed update already dropped it (the environment exists, the marker does not)
    res.reinstall = inputs_changed and (dropped or (root / ".venv").is_dir())
    shutil.rmtree(root / STAGING, ignore_errors=True)
    return res


def apply_git(root: Path, tag: str, version: str) -> ApplyResult:
    root = Path(root)
    why = git_blocker(root)
    if why:
        raise UpdateError(why)
    before = {n: _digest(root / n) for n in INSTALL_INPUTS}
    r = _git(root, "fetch", "--quiet", "origin", "tag", tag, "--no-tags")
    if r.returncode != 0:
        raise UpdateError(f"git could not download {tag}: {(r.stderr or r.stdout).strip()[:300]}")
    r = _git(root, "merge", "--ff-only", "--quiet", tag)
    if r.returncode != 0:
        raise UpdateError(
            f"This folder's history has moved away from the published one, so git cannot simply move it "
            f"forward to {tag}. Update it by hand (git pull), or download a fresh copy. git said: "
            f"{(r.stderr or r.stdout).strip()[:300]}")
    res = ApplyResult(version=version, how="git")
    if any(_digest(root / n) != before[n] for n in INSTALL_INPUTS):
        res.reinstall = _drop_marker(root)
    return res


def _drop_marker(root: Path) -> bool:
    """The launcher starts the app only when the marker exists; without it, it
    runs install.py first (I142). True when there was an install to redo."""
    m = Path(root) / MARKER
    try:
        m.unlink()
        return True
    except FileNotFoundError:
        return False


def apply(release: Release, root: Path = ROOT, progress=None) -> ApplyResult:
    """Install `release` into `root`, the way this folder was obtained."""
    root = Path(root)
    if install_kind(root) == "git":
        return apply_git(root, release.tag, release.version)
    dest = root / STAGING / f"kinetrace-{release.version}.zip"
    try:
        download(release.zip_url, dest, progress)        # an UpdateError about the connection / the disk (G94)
        try:
            return apply_zip(root, dest, release.version, release.tag)
        except OSError as e:                             # only the WRITING phase says "could not be written"
            raise UpdateError(f"The update could not be written into {root} ({e}). Close any program using "
                              "files in the Kinetrace folder and try again.") from None
    finally:
        shutil.rmtree(root / STAGING, ignore_errors=True)


# ------------------------------------------------------------------ restart

def relaunch(root: Path = ROOT) -> None:
    """Start Kinetrace again through its launcher, detached: the launcher runs
    install.py first when the update asked for it."""
    root = Path(root)
    if sys.platform == "win32":
        # run.bat in a console of its own, as a double click would. Not `start "title" x`: an
        # unquoted first argument of start is the program, so "start Kinetrace run.bat" ran
        # nothing (I156). `cmd /c ""path""` takes the quoted path literally (& ^ in a folder name).
        subprocess.Popen(f'cmd /c ""{root / "run.bat"}""', cwd=str(root),
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
    elif sys.platform == "darwin":
        # Kinetrace.app when run.sh made it (X27): no Terminal window, and it opens Terminal by itself
        # when the update asked for an install step; -n: a NEW instance, not the one that is quitting
        app = root / "Kinetrace.app"
        if app.is_dir():
            subprocess.Popen(["open", "-n", str(app)], cwd=str(root))
        else:
            subprocess.Popen(["open", str(root / "Kinetrace.command")], cwd=str(root))
    else:
        log = root / "logs"
        log.mkdir(exist_ok=True)
        with open(log / "launcher.log", "ab") as fh:
            subprocess.Popen(["bash", str(root / "run.sh")], cwd=str(root), stdout=fh, stderr=fh,
                             stdin=subprocess.DEVNULL, start_new_session=True)


# ---------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(f"Kinetrace {APP_VERSION} in {ROOT}")
    try:
        return _main(argv)
    except UpdateError as e:
        print("\n" + str(e))
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 1
    except Exception as e:      # noqa: BLE001 - a sentence for update.bat, never a traceback (G94)
        print(f"\nThe update stopped unexpectedly ({type(e).__name__}: {e}). Check the internet connection and "
              f"try again; the release page is {PAGE}.")
        return 1


def _main(argv: list[str]) -> int:
    rel, newer = check()
    if not newer:
        print(f"This is the newest version (the newest published is {rel.version}).")
        return 0
    print(f"A newer version is available: {rel.version} ({rel.published}), {rel.page_url}")
    if "--check" in argv:
        return 0
    print("Installing it ...")

    def _p(done, total):
        print(f"\r  downloaded {done / 1e6:.1f}" + (f" of {total / 1e6:.1f}" if total else "") + " MB",
              end="", flush=True)

    res = apply(rel, ROOT, _p)
    print("\n" + res.sentence())
    return 0


if __name__ == "__main__":
    sys.exit(main())
