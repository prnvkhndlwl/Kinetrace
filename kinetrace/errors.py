"""Errors in words for the user (G54; no Qt): what failed, why, and what to do
about it -- instead of a library's exception text."""
from __future__ import annotations

import re
import traceback


def plain_error(e: BaseException, what: str, short: bool = False, reading: bool | None = None) -> str:
    """`what` failed because of `e`, in words, with what to do about it (G54):
    a file open in another program, a full drive, a missing folder, too little
    memory; anything else keeps the library's own words and points at the
    error log. `short`: one line, for a status line or a list. `reading`: the
    failed step was READING (a permission / lock error then is about reading,
    not writing); None = guessed from `what` ("... could not be read / opened /
    loaded / imported"). A ProjectFileError is already a sentence naming the
    file: it passes through unchanged, with no type name and nothing logged
    (G142)."""
    import errno
    from kinetrace.projectfile import ProjectFileError
    if isinstance(e, ProjectFileError):
        return str(e)
    if reading is None:
        reading = re.search(r"\b(read|reading|opened?|loaded?|imported?)\b", what, re.I) is not None
    name = getattr(e, "filename", None)
    where = f" ({name})" if name else ""
    if isinstance(e, PermissionError) or getattr(e, "errno", None) in (errno.EACCES, errno.EPERM):
        if reading:
            return (f"{what}: Kinetrace may not read there{where}. The file is probably open in another program "
                    "that locks it, or you have no permission to read it: close it, or check its permissions, "
                    "and try again.")
        return (f"{what}: Kinetrace may not write there{where}. The file is probably open in another program "
                "(Excel, MATLAB…) or the folder is read-only: close it, or choose another folder, and try again.")
    if getattr(e, "errno", None) == errno.ENOSPC:
        return f"{what}: the drive is full{where}. Free some space, or choose another drive, and try again."
    if isinstance(e, FileNotFoundError):
        return (f"{what}: a file or folder was not found{where} -- a disconnected drive, or a file moved "
                "or renamed since. Check it is there and try again.")
    if isinstance(e, MemoryError):
        return f"{what}: the computer ran out of memory. Close other programs, or work on a shorter stretch."
    import logging            # the error log (crashlog.py) keeps the traceback for the Error Report
    logging.getLogger("kinetrace.errors").warning(
        "handled: %s\n%s", what, "".join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip())
    if short:
        return f"{what}: {type(e).__name__}: {e}"
    return (f"{what}: {type(e).__name__}: {e}\n\nHelp > Error Report shows the details (nothing is sent "
            "anywhere); include it if you report the problem.")
