"""Demo mode for Mesa-S1: show each interface element being acted on.

The UI session applies actions to Qt widgets directly, so nothing on screen
says which button was "pressed". For demos and launch videos, `DemoUiSession`
draws an overlay on the FreeCAD window before every action: a highlight
around the toolbar button, dialog field, dropdown, check box or OK/Cancel
button the model chose, a cursor dot on it, and a caption strip naming the
action. Numbers are typed into fields character by character. Interactions
in the 3D view and the sketch editor (which the widget tree cannot see) are
outlined over the 3D view instead. Captions show the element id the model
chose, verbatim.

`Recorder` grabs the main window (overlay included) at a fixed rate into
numbered JPEG frames for scripts/mesa_demo.py to turn into a video.

Runs inside the FreeCAD GUI (visible window; see scripts/mesa_demo.py).
"""

from __future__ import annotations

import time
from pathlib import Path

import FreeCADGui as Gui
from PySide import QtCore, QtGui, QtWidgets

try:  # Qt 6 (FreeCAD 1.x)
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except ImportError:  # Qt 5
    QOpenGLWidget = QtWidgets.QOpenGLWidget

from . import spec as S
from .session import UiSession

ACCENT = QtGui.QColor(42, 120, 214)  # theme s1 (viz/src/theme.ts)
INK = QtGui.QColor(11, 11, 11)
PAPER = QtGui.QColor(252, 252, 251)

HIDDEN_PANELS = ("Python console", "Report view")  # hidden in demo recordings


def _wait(seconds: float) -> None:
    """Keep the GUI (and the recorder's timer) running while pausing."""
    end = time.time() + seconds
    while time.time() < end:
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 10)
        time.sleep(0.005)


class Overlay(QtWidgets.QWidget):
    """Transparent layer over the main window: highlight box, cursor, caption."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.rect_: QtCore.QRect | None = None
        self.dashed = False
        self.caption = ""
        self.note = ""
        self.pressed = False
        self.setGeometry(parent.rect())
        parent.installEventFilter(self)
        self.raise_()
        self.show()

    def eventFilter(self, obj, ev):  # noqa: N802 (Qt name) — follow the main window's size
        if ev.type() == QtCore.QEvent.Resize:
            self.setGeometry(self.parent().rect())
        return False

    def target(self, rect: QtCore.QRect | None, caption: str, dashed: bool = False) -> None:
        self.rect_, self.caption, self.dashed, self.pressed = rect, caption, dashed, False
        self.raise_()
        self.update()

    def press(self) -> None:
        self.pressed = True
        self.update()

    def clear(self) -> None:
        self.rect_ = None
        self.update()

    def paintEvent(self, ev) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        if self.rect_ is not None:
            r = QtCore.QRectF(self.rect_).adjusted(-5, -5, 5, 5)
            fill = QtGui.QColor(ACCENT)
            fill.setAlpha(14 if self.dashed else (70 if self.pressed else 34))
            pen = QtGui.QPen(ACCENT, 3)
            if self.dashed:
                pen.setStyle(QtCore.Qt.DashLine)
            if not self.dashed:  # soft glow so the target catches the eye even when sped up
                for width, alpha in ((16, 40), (9, 70)):
                    glow = QtGui.QColor(ACCENT)
                    glow.setAlpha(alpha)
                    p.setPen(QtGui.QPen(glow, width))
                    p.setBrush(QtCore.Qt.NoBrush)
                    p.drawRoundedRect(r, 8, 8)
            pen.setWidth(4)
            p.setPen(pen)
            p.setBrush(fill)
            p.drawRoundedRect(r, 7, 7)
            if not self.dashed:  # cursor dot, larger ring while "pressing"
                c = r.center()
                p.setPen(QtGui.QPen(PAPER, 2))
                p.setBrush(INK)
                p.drawEllipse(c, 7, 7)
                if self.pressed:
                    p.setBrush(QtCore.Qt.NoBrush)
                    p.setPen(QtGui.QPen(ACCENT, 4))
                    p.drawEllipse(c, 22, 22)
        if self.caption:
            font = QtGui.QFont(self.font())
            font.setPixelSize(22)
            font.setWeight(QtGui.QFont.DemiBold)
            p.setFont(font)
            tag = "MESA-S1"
            fm = QtGui.QFontMetrics(font)
            text = self.caption + (f"   {self.note}" if self.note else "")
            w = fm.horizontalAdvance(tag) + fm.horizontalAdvance(text) + 70
            h = fm.height() + 24
            box = QtCore.QRectF((self.width() - w) / 2, self.height() - h - 36, w, h)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(INK)
            p.drawRoundedRect(box, h / 2, h / 2)
            p.setPen(ACCENT.lighter(150))
            p.drawText(box.adjusted(26, 0, 0, 0), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, tag)
            p.setPen(PAPER)
            p.drawText(box.adjusted(26 + fm.horizontalAdvance(tag) + 22, 0, 0, 0),
                       QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, text)
        p.end()


class Recorder:
    """Grabs the main window (overlay included) into frame_000000.jpg, ... at `fps`.

    A widget grab leaves the OpenGL 3D view blank, so each frame is
    composed: the window grab, the 3D view rendered by FreeCAD's offscreen
    renderer (refreshed every `view_every` seconds), the docks that float
    over the 3D view (FreeCAD 1.1's task panel) and the overlay on top."""

    def __init__(self, window: QtWidgets.QWidget, overlay: QtWidgets.QWidget | None = None,
                 view_every: float = 0.2, get_view=None) -> None:
        self.window = window
        self.get_view = get_view  # the session document's 3D view (targets are built in another document)
        self.overlay = overlay
        self.view_every = view_every
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.grab)
        self.dir: Path | None = None
        self.n = 0
        self.events: list[dict] = []  # frame ranges of each action (start, press, end) for editing
        self._view_img: QtGui.QImage | None = None
        self._view_t = 0.0

    def start(self, directory: str, fps: int = 15) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.events = []
        self._view_img = None
        self.timer.start(int(1000 / fps))

    def _gl_widget(self):
        best = None
        for gl in self.window.findChildren(QOpenGLWidget):
            if gl.isVisible() and gl.width() > 50 and (best is None or gl.width() * gl.height() > best.width() * best.height()):
                best = gl
        return best

    def _view_image(self, gl) -> QtGui.QImage | None:
        if self._view_img is not None and time.time() - self._view_t < self.view_every:
            return self._view_img
        view = self.get_view() if self.get_view is not None else None
        if view is None or not hasattr(view, "saveImage"):
            return self._view_img
        ratio = gl.devicePixelRatioF()
        path = str(self.dir / ".view.png")
        try:
            view.saveImage(path, int(gl.width() * ratio), int(gl.height() * ratio), "Current")
            self._view_img = QtGui.QImage(path)
            self._view_t = time.time()
        except Exception:
            pass
        return self._view_img

    def grab(self) -> None:
        if self.dir is None:
            return
        pix = self.window.grab()
        p = QtGui.QPainter(pix)
        gl = self._gl_widget()
        if gl is not None:
            rect = QtCore.QRect(gl.mapTo(self.window, QtCore.QPoint(0, 0)), gl.size())
            img = self._view_image(gl)
            if img is not None and not img.isNull():
                p.drawImage(rect, img)
                for dock in self.window.findChildren(QtWidgets.QDockWidget):  # panels floating over the view
                    if dock.isVisible() and not dock.isFloating():
                        drect = QtCore.QRect(dock.mapTo(self.window, QtCore.QPoint(0, 0)), dock.size())
                        if rect.contains(drect.center()):
                            dock.render(p, drect.topLeft(), QtGui.QRegion(), QtWidgets.QWidget.DrawChildren)
        if self.overlay is not None:
            self.overlay.render(p, QtCore.QPoint(0, 0), QtGui.QRegion(), QtWidgets.QWidget.DrawChildren)
        p.end()
        pix.save(str(self.dir / f"frame_{self.n:06d}.jpg"), "JPG", 92)
        self.n += 1

    def stop(self) -> int:
        self.timer.stop()
        self.grab()
        self.dir = None
        return self.n


class DemoUiSession(UiSession):
    """UiSession that shows every action on screen before performing it."""

    def __init__(self, dwell: float = 0.55, after: float = 0.3, type_delay: float = 0.07) -> None:
        super().__init__()
        self.dwell, self.after, self.type_delay = dwell, after, type_delay
        self.window = Gui.getMainWindow()
        self.overlay = Overlay(self.window)
        self.recorder = Recorder(self.window, self.overlay, get_view=self._session_view)

    # -- extra ops for the demo harness (see runtime/gui_server.py) ---------------

    def handle_extra(self, req: dict) -> dict | None:
        op = req.get("op")
        if op == "demo_window":
            self.window.showNormal()
            self.window.resize(int(req.get("width", 1600)), int(req.get("height", 1000)))
            for dock in self.window.findChildren(QtWidgets.QDockWidget):
                if dock.windowTitle() in HIDDEN_PANELS or dock.objectName() in HIDDEN_PANELS:
                    dock.hide()
            _wait(0.5)
            return {}
        if op == "record_start":
            self.recorder.start(req["dir"], int(req.get("fps", 15)))
            return {}
        if op == "record_stop":
            return {"frames": self.recorder.stop(), "events": self.recorder.events}
        if op == "demo_note":  # e.g. the model's decision time, shown in the caption
            self.overlay.note = req.get("text", "")
            return {}
        if op == "demo_wait":
            self.overlay.target(None, req.get("caption", ""))
            _wait(float(req.get("seconds", 1.0)))
            return {}
        return None

    # -- where on screen an element lives ------------------------------------------

    def _rect_of(self, w) -> QtCore.QRect | None:
        if w is None or not w.isVisible():
            return None
        top_left = w.mapTo(self.window, QtCore.QPoint(0, 0))
        return QtCore.QRect(top_left, w.size())

    def _toolbar_button(self, command: str):
        try:
            action = Gui.Command.get(command).getAction()[0]
        except Exception:
            return None
        for tb in self.window.findChildren(QtWidgets.QToolBar):
            if not tb.isVisible():
                continue
            for a in tb.actions():
                if a is action or (a.objectName() and a.objectName() == action.objectName()):
                    btn = tb.widgetForAction(a)
                    if btn is not None and btn.isVisible():
                        return btn
        return None

    def _workbench_selector(self):
        for w in self.window.findChildren(QtWidgets.QWidget):
            name = w.metaObject().className()
            if "Workbench" in name and ("Selector" in name or "ComboBox" in name or "Tab" in name) and w.isVisible():
                return w
        return None

    def _view_3d(self):
        best = None
        for w in self.window.findChildren(QtWidgets.QWidget):
            if "View3DInventor" in w.metaObject().className() and w.isVisible():
                best = w
        return best

    def _locate(self, element: str):
        """(widget, caption, dashed) for an element. The caption is the
        element id the model chose, as it is (plus the value typed into a
        field), so the video shows exactly what the model acted on."""
        role = S.role(element)
        if role == "cmd":
            return self._toolbar_button(S.underlying(element)), element, False
        if role == "wb":
            return self._workbench_selector(), element, False
        if role == "set":
            name = S.field_name(element)
            value = self.pending.targets.get(name) if self.pending else None
            return self._widget(name), f"{element}  \u2190  {self._fmt(value, name)}", False
        if role in ("opt", "toggle"):
            return self._widget(S.field_name(element)), element, False
        if role == "click":
            return self._button("OK" if element == S.OK else "Cancel"), element, False
        if role == "canvas":
            return self._view_3d(), element, True
        return None, element, False

    def _fmt(self, value, name: str) -> str:
        if value is None:
            return "a value"
        fl = next((f for fs in S.DIALOG_FIELDS.values() for f in fs if f.name == name), None)
        unit = {"mm": " mm", "deg": "°"}.get(fl.unit if fl else "", "")
        return f"{value:g}{unit}" if isinstance(value, (int, float)) else str(value)

    # -- show, then act -------------------------------------------------------------

    def step(self, element: str) -> dict:
        widget, caption, dashed = self._locate(element)
        event = {"element": element, "start": self.recorder.n}
        self.overlay.target(self._rect_of(widget), caption, dashed)
        _wait(self.dwell)
        if S.role(element) == "set" and widget is not None and self.pending is not None:
            self._type_visibly(widget, element)
        self.overlay.press()
        event["press"] = self.recorder.n
        _wait(0.12)
        info = super().step(element)
        self.overlay.target(None, self.overlay.caption)  # the widget may be gone (dialog closed): keep only the caption
        if element in (S.OK, "cmd:Sketcher_LeaveSketch", "Done") and not info["error"]:
            self._frame_part()  # cosmetic: keep the part in view for the recording
        _wait(self.after)
        self.overlay.clear()
        event["end"] = self.recorder.n
        self.recorder.events.append(event)
        return info

    def _session_view(self):
        if self.doc is None:
            return None
        gdoc = Gui.getDocument(self.doc.Name)
        return gdoc.ActiveView if gdoc is not None else None

    def _frame_part(self) -> None:
        view = self._session_view()
        if view is not None and Gui.getDocument(self.doc.Name).getInEdit() is None:
            view.viewIsometric()
            view.fitAll()

    def _type_visibly(self, widget, element: str) -> None:
        """Show the value appearing keystroke by keystroke (cosmetic: the
        real entry is made by UiSession._set_number right after)."""
        value = self.pending.targets.get(S.field_name(element))
        if value is None or not hasattr(widget, "lineEdit"):
            return
        text = f"{int(round(value))}" if float(value).is_integer() else f"{value:g}"
        line = widget.lineEdit()
        blocker = QtCore.QSignalBlocker(widget)  # don't let half-typed values reach the feature
        line.clear()
        for i in range(1, len(text) + 1):
            line.setText(text[:i])
            _wait(self.type_delay)
        del blocker
