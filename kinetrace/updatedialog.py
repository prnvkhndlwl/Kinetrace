"""Help -> About Kinetrace (G36) and Help -> Check for Updates... (G37).

The network and the file writing run on plain daemon threads (never the GUI
thread); their results come back through a QObject signal, queued onto the GUI
thread. Daemon, not QThread: a check stuck on a slow connection must never keep
the app from closing (a QThread destroyed while running aborts the process).
Nothing is looked up until the dialog is actually shown."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QTextBrowser, QVBoxLayout)

from kinetrace import APP_NAME, APP_TAGLINE, APP_VERSION, update

LAB = "biomechLab@CMC"
LICENSE_NAME = "PolyForm Noncommercial License 1.0.0"
LICENSE_URL = f"{update.PAGE}/blob/main/LICENSE.md"
CREDITS_URL = f"{update.PAGE}/blob/main/THIRD_PARTY_LICENSES.md"
_log = logging.getLogger(__name__)


def about_html(version: str = APP_VERSION) -> str:
    return (
        f"<h2 style='margin-bottom:2px'>{APP_NAME}</h2>"
        f"<p style='margin-top:0'>Version {version} &middot; {APP_TAGLINE}</p>"
        f"<p>Developed at <b>{LAB}</b>.</p>"
        f"<p>Free for any non-commercial use (research, teaching, study) under the "
        f"<a href='{LICENSE_URL}'>{LICENSE_NAME}</a>; not for commercial use.</p>"
        f"<p>Built on AllTracker, CoTracker3, SAM 2.1 / SAM 3, SAM 3D Body, ViTPose, PyTorch, OpenCV "
        f"and Qt &mdash; each under its own licence (<a href='{CREDITS_URL}'>the list</a>). Please cite "
        f"the models you use in publications.</p>"
        f"<p><a href='{update.PAGE}'>{update.PAGE.replace('https://', '')}</a></p>")


def show_about(parent, on_check=None) -> QDialog:
    """The About box; `on_check` = what its Check for Updates... button does."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(f"About {APP_NAME}")
    lay = QVBoxLayout(dlg)
    top = QHBoxLayout()
    from kinetrace import appicon
    mark = QLabel()
    mark.setPixmap(appicon.icon().pixmap(96, 96))           # the app icon (G41)
    mark.setAlignment(Qt.AlignTop)
    top.addWidget(mark)
    text = QLabel(about_html())
    text.setWordWrap(True)
    text.setOpenExternalLinks(True)
    text.setMinimumWidth(440)
    top.addWidget(text, 1)
    lay.addLayout(top)
    dlg.about_mark = mark                     # for tests
    btns = QDialogButtonBox(QDialogButtonBox.Close)
    if on_check is not None:
        chk = btns.addButton("Check for Updates…", QDialogButtonBox.ActionRole)
        chk.clicked.connect(lambda: (dlg.accept(), QTimer.singleShot(0, on_check)))
    btns.rejected.connect(dlg.reject)
    lay.addWidget(btns)
    dlg.about_text = text                     # for tests
    dlg.exec()
    dlg.deleteLater()
    return dlg


class _Relay(QObject):
    """Carries a worker thread's result onto the GUI thread (queued)."""
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)


def _run(relay: _Relay, fn) -> None:
    """Run fn(progress) on a daemon thread; report through `relay`."""
    def body():
        def prog(done, total):
            try:
                relay.progress.emit(int(done), int(total))
            except RuntimeError:              # the dialog is gone
                pass
        try:
            out = fn(prog)
        except update.UpdateError as e:
            out, err = None, str(e)
        except Exception as e:                # noqa: BLE001 - said, and logged with the traceback
            _log.exception("update step failed")
            out, err = None, f"Something unexpected went wrong ({type(e).__name__}: {e}). Try again later."
        else:
            err = None
        try:
            relay.failed.emit(err) if err is not None else relay.done.emit(out)
        except RuntimeError:
            pass
    threading.Thread(target=body, name="kinetrace-update", daemon=True).start()


class UpdateDialog(QDialog):
    """checking -> uptodate | available -> applying -> done, or error.

    `busy()` returns a sentence while updating must wait (a run, a video
    opening), else None; `restart()` closes the app and starts it again."""

    def __init__(self, parent=None, root: Path = update.ROOT, local: str = APP_VERSION,
                 busy=None, restart=None):
        super().__init__(parent)
        self.setWindowTitle("Check for updates")
        self.setMinimumSize(560, 360)
        self._root, self._local = Path(root), local
        self._busy, self._restart = busy or (lambda: None), restart
        self.state = "new"
        self.release: update.Release | None = None
        self.result: update.ApplyResult | None = None
        lay = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight: 600; font-size: 11pt;")
        lay.addWidget(self.title)
        self.body = QTextBrowser()
        self.body.setOpenExternalLinks(True)
        lay.addWidget(self.body, 1)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.btns = QDialogButtonBox()
        lay.addWidget(self.btns)
        self._relay = _Relay(self)
        self._relay.progress.connect(self._on_progress)
        self._handlers = ()                  # the (done, failed) slots now connected
        self._started = False

    # ---------------------------------------------------------------- states
    def _buttons(self, *specs) -> dict:
        """specs = (label, role, slot); returns {label: button}."""
        self.btns.clear()
        out = {}
        for label, role, slot in specs:
            b = QPushButton(label)
            self.btns.addButton(b, role)
            b.clicked.connect(slot)
            out[label] = b
        self.buttons = out
        return out

    def showEvent(self, ev):                 # noqa: N802 - Qt override
        super().showEvent(ev)
        if not self._started:                # nothing goes to the network before the dialog is on screen
            self._started = True
            QTimer.singleShot(0, self.check)

    def check(self) -> None:
        self.state = "checking"
        self.title.setText("Looking for a newer version of Kinetrace…")
        self.body.setPlainText(f"You have Kinetrace {self._local}. Asking GitHub for the newest published "
                               "version (nothing about you or your computer is sent).")
        self.bar.setRange(0, 0)
        self.bar.setVisible(True)
        self._buttons(("Close", QDialogButtonBox.RejectRole, self.reject))
        self._connect(self._on_checked, self._on_error)
        _run(self._relay, lambda _p: update.check(self._local))

    def _connect(self, done, failed) -> None:
        """Route the next result to (done, failed), dropping the previous step's pair."""
        for sig, slot in zip((self._relay.done, self._relay.failed), self._handlers):
            sig.disconnect(slot)
        self._handlers = (done, failed)
        self._relay.done.connect(done)
        self._relay.failed.connect(failed)

    def _on_error(self, msg: str) -> None:
        was = self.state
        self.state = "error"
        self.bar.setVisible(False)
        self.title.setText("Could not install the update" if was == "applying" else "Could not check for updates")
        self.body.setPlainText(msg)
        self._buttons(("Try again", QDialogButtonBox.ActionRole, self.check),
                      ("Close", QDialogButtonBox.RejectRole, self.reject))

    def _on_checked(self, out) -> None:
        rel, newer = out
        self.release = rel
        self.bar.setVisible(False)
        if not newer:
            self.state = "uptodate"
            self.title.setText(f"Kinetrace {self._local} is the newest version")
            self.body.setHtml(f"<p>The newest published version is {rel.version}"
                              + (f" ({rel.published})" if rel.published else "")
                              + f". <a href='{rel.page_url}'>Its release page</a>.</p>")
            self._buttons(("Close", QDialogButtonBox.RejectRole, self.reject))
            return
        self.state = "available"
        self.title.setText(f"Kinetrace {rel.version} is available — you have {self._local}")
        how, blocker = update.install_kind(self._root), None
        if how == "git":
            blocker = update.git_blocker(self._root)
            note = (blocker or f"This folder is a git checkout: it will be moved forward to {rel.tag}.")
        else:
            note = ("Only Kinetrace's own files are replaced; your projects, models, recovery copies, "
                    "saved skeletons and settings are kept.")
        notes = rel.notes.strip() or "(no release notes)"
        self.body.setMarkdown(f"**What is new** ({rel.published or rel.tag})\n\n{notes}\n\n---\n\n{note}")
        b = self._buttons(("Update now", QDialogButtonBox.AcceptRole, self.install),
                          ("Open the release page", QDialogButtonBox.ActionRole,
                           lambda: QDesktopServices.openUrl(QUrl(rel.page_url))),
                          ("Later", QDialogButtonBox.RejectRole, self.reject))
        b["Update now"].setEnabled(blocker is None)
        b["Update now"].setDefault(True)

    def install(self) -> None:
        if self.release is None:
            return
        why = self._busy()
        if why:
            self.body.append(f"\n{why}")
            return
        self.state = "applying"
        rel = self.release
        self.title.setText(f"Installing Kinetrace {rel.version}…")
        self.bar.setRange(0, 0)
        self.bar.setVisible(True)
        self._buttons()                      # nothing to press while files are written
        self._connect(self._on_installed, self._on_error)
        root = self._root
        _run(self._relay, lambda p: update.apply(rel, root, p))

    def _on_progress(self, done: int, total: int) -> None:
        if self.state == "applying" and total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(min(done, total))

    def _on_installed(self, res) -> None:
        self.result = res
        self.state = "done"
        self.bar.setVisible(False)
        self.title.setText(f"Kinetrace {res.version} is installed")
        self.body.setPlainText(res.sentence())
        self._buttons(("Restart now", QDialogButtonBox.AcceptRole, self._restart_now),
                      ("Later", QDialogButtonBox.RejectRole, self.reject))

    def _restart_now(self) -> None:
        self.accept()
        if self._restart is not None:
            QTimer.singleShot(0, self._restart)

    def reject(self) -> None:
        if self.state == "applying":         # half-written files are worse than a wait of seconds
            return
        super().reject()
