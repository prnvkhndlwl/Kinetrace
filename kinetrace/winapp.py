"""The Kinetrace shortcut for Windows (owner 2026-10-10: "an app install on windows so that everytime the
run does not have to be clicked"): the Windows counterpart of macapp.py's Kinetrace.app.

run.bat writes `Kinetrace.lnk` into the Kinetrace folder after a finished install and refreshes it when it
is missing, when this file's SHORTCUT_VERSION changes or when the folder was moved (`--if-stale`). It is
made ON the user's computer, so it carries no download mark and opens without a SmartScreen prompt. No
installer, nothing written outside the folder unless the user asks (Help -> Add Kinetrace to the Start
Menu, `add_start_menu`).

    Kinetrace.lnk                 target: the folder's own pythonw (.venv\\base, else the venv's), running
                                  kinetrace\\winlaunch.py isolated (stdlib only); working folder = the
                                  Kinetrace folder; the app's icon; the app's taskbar id
                                  (System.AppUserModel.ID = APP_ID, the one app.main registers), so a
                                  pinned shortcut and the running window are ONE taskbar button
    .venv\\kinetrace.ico           appicon.render at the sizes Windows asks for (PNG entries)
    .venv\\kinetrace-shortcut      SHORTCUT_VERSION + the folder the shortcut was written for

winlaunch.py starts the app with no console window, or opens run.bat in a console when an install or repair
step is due (it talks and takes minutes), as the Mac app opens Terminal.

The shortcut is written through the shell's own COM objects (IShellLinkW, IPersistFile, IPropertyStore)
with ctypes: no pywin32, and the only way to give a shortcut its taskbar id.

    python -m kinetrace.winapp [--if-stale]      (run.bat does this)
"""
from __future__ import annotations

import ctypes
import os
import struct
import sys
from ctypes import POINTER, byref, c_void_p, c_wchar_p
from pathlib import Path

from kinetrace import paths as _paths

ROOT = _paths.ROOT                                   # (R48) the one install folder
APP_ID = "Kinetrace.Kinetrace"          # = app.main's SetCurrentProcessExplicitAppUserModelID
SHORTCUT_VERSION = "1"                  # bump when the target / arguments / icon change: run.bat rewrites it
LINK_NAME = "Kinetrace.lnk"
ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
DESCRIPTION = "Kinetrace - motion tracking (starts the app from this folder)"


def link_path(root: Path = ROOT) -> Path:
    return Path(root) / LINK_NAME


def icon_path(root: Path = ROOT) -> Path:
    return Path(root) / ".venv" / "kinetrace.ico"


def stamp_path(root: Path = ROOT) -> Path:
    return Path(root) / ".venv" / "kinetrace-shortcut"


def start_menu_link() -> Path:
    """The per-user Start menu entry (only ever written when the user asks). KINETRACE_START_MENU_DIR
    replaces the Start menu folder (the tests: a menu-clicking sweep answers Yes to every question)."""
    override = os.environ.get("KINETRACE_START_MENU_DIR")
    if override:
        return Path(override) / LINK_NAME
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / LINK_NAME


def launcher_python(root: Path = ROOT) -> Path:
    """The pythonw the shortcut runs: the folder's own base interpreter when there is one (it starts even
    when a moved folder left .venv\\pyvenv.cfg pointing at the old place), else the venv's."""
    base = Path(root) / ".venv" / "base" / "pythonw.exe"
    return base if base.exists() else Path(root) / ".venv" / "Scripts" / "pythonw.exe"


def link_spec(root: Path = ROOT) -> dict:
    """What the shortcut holds -- one place for the writer, the stale check and the tests."""
    root = Path(root)
    return {"target": str(launcher_python(root)),
            "arguments": f'-I "{root / "kinetrace" / "winlaunch.py"}"',
            "workdir": str(root),
            "icon": str(icon_path(root)),
            "description": DESCRIPTION,
            "app_id": APP_ID}


# ------------------------------------------------------------------ COM, through ctypes

class _GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16), ("d3", ctypes.c_uint16),
                ("d4", ctypes.c_ubyte * 8)]

    def __init__(self, text: str):
        super().__init__()
        h = text.strip("{}").replace("-", "")
        self.d1, self.d2, self.d3 = int(h[:8], 16), int(h[8:12], 16), int(h[12:16], 16)
        for i in range(8):
            self.d4[i] = int(h[16 + 2 * i:18 + 2 * i], 16)


class _PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", ctypes.c_uint32)]


class _PROPVARIANT(ctypes.Structure):
    # vt + 3 reserved words + a 16-byte union (pointer-sized members first): 24 bytes on 64-bit
    _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort), ("r2", ctypes.c_ushort),
                ("r3", ctypes.c_ushort), ("pwszVal", c_void_p), ("pad", c_void_p)]


CLSID_ShellLink = "{00021401-0000-0000-C000-000000000046}"
IID_IShellLinkW = "{000214F9-0000-0000-C000-000000000046}"
IID_IPersistFile = "{0000010B-0000-0000-C000-000000000046}"
IID_IPropertyStore = "{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}"
PKEY_AppUserModel_ID = ("{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}", 5)
VT_LPWSTR = 31
CLSCTX_INPROC_SERVER = 1
# vtable slots (IUnknown takes 0-2)
_SL_GETPATH, _SL_GETDESCRIPTION, _SL_SETDESCRIPTION = 3, 6, 7
_SL_GETWORKDIR, _SL_SETWORKDIR, _SL_GETARGS, _SL_SETARGS = 8, 9, 10, 11
_SL_GETICON, _SL_SETICON, _SL_SETPATH = 16, 17, 20
_PF_LOAD, _PF_SAVE = 5, 6
_PS_GETVALUE, _PS_SETVALUE, _PS_COMMIT = 5, 6, 7


def _method(obj: c_void_p, slot: int, *argtypes):
    vtbl = ctypes.cast(obj, POINTER(c_void_p))[0]
    fn = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)(ctypes.cast(vtbl, POINTER(c_void_p))[slot])
    return lambda *a: fn(obj, *a)


def _release(obj: c_void_p) -> None:
    if obj:
        vtbl = ctypes.cast(obj, POINTER(c_void_p))[0]
        ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(ctypes.cast(vtbl, POINTER(c_void_p))[2])(obj)


def _query(obj: c_void_p, iid: str) -> c_void_p:
    out = c_void_p()
    _method(obj, 0, POINTER(_GUID), POINTER(c_void_p))(byref(_GUID(iid)), byref(out))
    return out


class _Com:
    """CoInitialize for this thread when it is not already (Qt's GUI thread is: RPC_E_CHANGED_MODE
    then, and nothing is undone)."""

    def __enter__(self):
        ole = ctypes.windll.ole32
        ole.CoInitializeEx.restype = ctypes.c_long
        hr = ole.CoInitializeEx(None, 2)          # COINIT_APARTMENTTHREADED
        self._undo = hr in (0, 1)                 # S_OK / S_FALSE
        return self

    def __exit__(self, *exc):
        if self._undo:
            ctypes.windll.ole32.CoUninitialize()


def _shell_link() -> c_void_p:
    obj = c_void_p()
    ole = ctypes.windll.ole32
    ole.CoCreateInstance.restype = ctypes.HRESULT
    ole.CoCreateInstance(byref(_GUID(CLSID_ShellLink)), None, CLSCTX_INPROC_SERVER,
                         byref(_GUID(IID_IShellLinkW)), byref(obj))
    return obj


def _key() -> _PROPERTYKEY:
    k = _PROPERTYKEY()
    k.fmtid, k.pid = _GUID(PKEY_AppUserModel_ID[0]), PKEY_AppUserModel_ID[1]
    return k


def write_link(path: Path, spec: dict) -> None:
    """A .lnk at `path` from `link_spec`, its taskbar id included; written to a temporary name and moved
    into place (a pinned copy keeps working while it is rewritten)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".new.lnk")
    with _Com():
        link = _shell_link()
        store = pf = c_void_p()
        try:
            _method(link, _SL_SETPATH, c_wchar_p)(spec["target"])
            _method(link, _SL_SETARGS, c_wchar_p)(spec["arguments"])
            _method(link, _SL_SETWORKDIR, c_wchar_p)(spec["workdir"])
            _method(link, _SL_SETDESCRIPTION, c_wchar_p)(spec["description"])
            _method(link, _SL_SETICON, c_wchar_p, ctypes.c_int)(spec["icon"], 0)
            store = _query(link, IID_IPropertyStore)
            buf = ctypes.create_unicode_buffer(spec["app_id"])
            pv = _PROPVARIANT()
            pv.vt, pv.pwszVal = VT_LPWSTR, ctypes.cast(buf, c_void_p)
            _method(store, _PS_SETVALUE, POINTER(_PROPERTYKEY), POINTER(_PROPVARIANT))(byref(_key()), byref(pv))
            _method(store, _PS_COMMIT)()
            pf = _query(link, IID_IPersistFile)
            _method(pf, _PF_SAVE, c_wchar_p, ctypes.c_int)(str(tmp), 1)
        finally:
            for o in (pf, store, link):
                _release(o)
    os.replace(tmp, path)


def read_link(path: Path) -> dict:
    """What a .lnk holds (the same keys as `link_spec`)."""
    out = {}
    with _Com():
        link = _shell_link()
        store = pf = c_void_p()
        try:
            pf = _query(link, IID_IPersistFile)
            _method(pf, _PF_LOAD, c_wchar_p, ctypes.c_uint32)(str(path), 0)
            buf = ctypes.create_unicode_buffer(1024)
            _method(link, _SL_GETPATH, c_wchar_p, ctypes.c_int, c_void_p, ctypes.c_uint32)(buf, 1024, None, 4)
            out["target"] = buf.value                     # SLGP_RAWPATH: as written, no %VAR% expansion
            _method(link, _SL_GETARGS, c_wchar_p, ctypes.c_int)(buf, 1024)
            out["arguments"] = buf.value
            _method(link, _SL_GETWORKDIR, c_wchar_p, ctypes.c_int)(buf, 1024)
            out["workdir"] = buf.value
            _method(link, _SL_GETDESCRIPTION, c_wchar_p, ctypes.c_int)(buf, 1024)
            out["description"] = buf.value
            idx = ctypes.c_int()
            _method(link, _SL_GETICON, c_wchar_p, ctypes.c_int, POINTER(ctypes.c_int))(buf, 1024, byref(idx))
            out["icon"] = buf.value
            store = _query(link, IID_IPropertyStore)
            pv = _PROPVARIANT()
            _method(store, _PS_GETVALUE, POINTER(_PROPERTYKEY), POINTER(_PROPVARIANT))(byref(_key()), byref(pv))
            out["app_id"] = ctypes.wstring_at(pv.pwszVal) if pv.vt == VT_LPWSTR and pv.pwszVal else ""
            ctypes.windll.ole32.PropVariantClear(byref(pv))
        finally:
            for o in (pf, store, link):
                _release(o)
    return out


# ------------------------------------------------------------------ the icon

def ico_bytes(images: list[tuple[int, bytes]]) -> bytes:
    """An .ico of PNG entries [(size, png bytes)] (Windows Vista and later read PNG entries at any size)."""
    head = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    dirs, data = b"", b""
    for size, png in images:
        side = 0 if size >= 256 else size             # 0 = 256 in an icon directory
        dirs += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
    return head + dirs + data


def write_icon(dest: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtGui import QGuiApplication
    app = QGuiApplication.instance() or QGuiApplication([])     # noqa: F841 - QImage needs one
    from kinetrace import appicon
    images = []
    for s in ICON_SIZES:
        buf = QBuffer()
        buf.open(QIODevice.WriteOnly)
        appicon.render(s).save(buf, "PNG")
        images.append((s, bytes(buf.data())))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".new")
    tmp.write_bytes(ico_bytes(images))
    os.replace(tmp, dest)


# ------------------------------------------------------------------ build / stale

def _stamp_text(root: Path) -> str:
    return f"{SHORTCUT_VERSION}\n{Path(root)}"


def stale_reason(root: Path = ROOT) -> str:
    """'' (up to date), 'missing' (no shortcut or icon), 'version' or 'moved' (rewrite)."""
    root = Path(root)
    if not link_path(root).exists() or not icon_path(root).exists():
        return "missing"
    try:
        version, _, where = stamp_path(root).read_text(encoding="utf-8").partition("\n")
    except OSError:
        return "version"
    if version.strip() != SHORTCUT_VERSION:
        return "version"
    return "" if where.strip() == str(root) else "moved"


def _stamped_root(root: Path) -> Path | None:
    try:
        return Path(stamp_path(root).read_text(encoding="utf-8").partition("\n")[2].strip() or ".")
    except OSError:
        return None


def build(root: Path = ROOT) -> Path:
    """(Re)write root/Kinetrace.lnk and its icon; a Start menu entry the user added for this folder (also
    when it still names the folder's old place) is rewritten too. Returns the shortcut's path."""
    root = Path(root)
    old = _stamped_root(root)
    write_icon(icon_path(root))
    spec = link_spec(root)
    write_link(link_path(root), spec)
    menu = start_menu_link()
    if menu.exists() and old is not None and (start_menu_points_to(old) or start_menu_points_to(root)):
        write_link(menu, spec)
    stamp_path(root).write_text(_stamp_text(root), encoding="utf-8")
    return link_path(root)


# ------------------------------------------------------------------ the Start menu (opt-in)

def start_menu_points_to(root: Path = ROOT) -> bool:
    """The Start menu holds a Kinetrace entry that starts THIS folder's Kinetrace."""
    menu = start_menu_link()
    if not menu.exists():
        return False
    try:
        args = read_link(menu).get("arguments", "")
    except OSError:
        return False
    return str(Path(root) / "kinetrace" / "winlaunch.py").lower() in args.lower()


def start_menu_other() -> str:
    """The Kinetrace folder another Start menu entry starts ('' when there is none, or it is this one's)."""
    menu = start_menu_link()
    if not menu.exists() or start_menu_points_to(ROOT):
        return ""
    try:
        return read_link(menu).get("workdir", "") or "another folder"
    except OSError:
        return "another folder"


def add_start_menu(root: Path = ROOT) -> Path:
    """Help -> Add Kinetrace to the Start Menu: the same shortcut in the user's Start menu."""
    root = Path(root)
    if stale_reason(root):
        build(root)
    write_link(start_menu_link(), link_spec(root))
    return start_menu_link()


def remove_start_menu() -> bool:
    menu = start_menu_link()
    try:
        menu.unlink()
    except FileNotFoundError:
        return False
    return True


def main(argv: list[str]) -> int:
    if sys.platform != "win32" and "--any-os" not in argv:
        print("The Kinetrace shortcut is for Windows.")
        return 0
    reason = stale_reason(ROOT)
    if "--if-stale" in argv and reason == "":
        return 0
    try:
        lnk = build()
    except (OSError, ImportError) as e:
        # never stops run.bat: the app still starts the old way
        print(f"The Kinetrace shortcut could not be written ({e}); start Kinetrace with run.bat.")
        return 1
    if "--quiet" not in argv:
        print(f"Made {lnk.name} in this folder: double-click it to start Kinetrace (no extra window); "
              "right-click it -> Pin to Start or Pin to taskbar to keep it there.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
