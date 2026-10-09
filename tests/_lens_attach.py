"""Test helper (G176): `LensAttachDialog.exec` replaced by the user's own clicks inside the REAL dialog
(it is shown, the step in STATE["do"] clicks its tick boxes / buttons, then OK is clicked when it is
enabled, else -- or with STATE["cancel"] -- Cancel is clicked). What each dialog showed is kept in
STATE["seen"]."""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from kinetrace import lensattach

STATE = {"do": None, "seen": [], "cancel": False}


def tick(dlg, view: int) -> None:
    """Click the tick box of camera `view`'s row."""
    r = [row.view for row in dlg.rows].index(view)
    item = dlg.table.item(r, 0)
    dlg.table.scrollToItem(item)
    rect = dlg.table.visualItemRect(item)
    QTest.mouseClick(dlg.table.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(rect.left() + 11, rect.center().y()))
    QApplication.processEvents()


def click(button) -> None:
    QTest.mouseClick(button, Qt.LeftButton)
    QApplication.processEvents()


def _exec(self):
    self.show()
    QApplication.processEvents()
    if STATE["do"] is not None:
        STATE["do"](self)
    STATE["seen"].append({"rows": {r.view: (r.effect, r.can_attach) for r in self.rows}, "chosen": self.chosen(),
                          "summary": self.summary.text(),
                          "ok": self.box.button(QDialogButtonBox.Ok).text()})
    ok = self.box.button(QDialogButtonBox.Ok)
    if STATE["cancel"]:
        click(self.box.button(QDialogButtonBox.Cancel))
    elif ok.isEnabled():
        click(ok)
    else:
        click(self.box.button(QDialogButtonBox.Cancel))
    QApplication.processEvents()
    return self.result()


def install() -> None:
    lensattach.LensAttachDialog.exec = _exec
