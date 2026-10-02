"""Serve the worker protocol from inside a running FreeCAD GUI.

FreeCAD's GUI must be driven from its main thread, so requests arrive on a
localhost TCP socket and are polled by a Qt timer; each request runs in the
GUI thread against a `GuiSession` (real workbench, selection, edit mode and
command activity). Targets are built in a hidden headless session.

Start it from FreeCAD (Macro > Execute, or `FreeCAD scripts/freecad_gui_server.FCMacro`),
then drive it with `python scripts/gui_demo.py`.
"""

from __future__ import annotations

import json
import socket
import traceback

from .fcenv import PROTOCOL_PREFIX as PREFIX
from .gui_session import GuiSession
from .session import HeadlessSession
from .worker import Worker

_SERVER = None  # keep a reference so the Qt timer is not garbage-collected


class GuiServer:
    def __init__(self, port: int = 8765, ui: bool = False) -> None:
        """`ui`: serve a UI-level session (freecad_s1/ui) instead of GuiSession."""
        from PySide import QtCore

        if ui:
            from ..ui.session import UiSession
            from ..ui.spec import ui_step_budget

            self.worker = Worker(UiSession(), HeadlessSession(), budget_fn=ui_step_budget)
        else:
            self.worker = Worker(GuiSession(), HeadlessSession())
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(1)
        self.sock.setblocking(False)
        self.conn = None
        self.buf = b""
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(20)
        print(f"Taiga-S1 GUI server listening on 127.0.0.1:{port}")

    def poll(self) -> None:
        if self.conn is None:
            try:
                self.conn, _ = self.sock.accept()
                self.conn.setblocking(False)
            except BlockingIOError:
                return
        try:
            chunk = self.conn.recv(1 << 20)
        except BlockingIOError:
            return
        if not chunk:  # client disconnected
            self.conn.close()
            self.conn, self.buf = None, b""
            return
        self.buf += chunk
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            if line.strip():
                self.respond(json.loads(line))

    def respond(self, req: dict) -> None:
        if req.get("op") == "quit":
            self.conn.close()
            self.conn = None
            return
        if req.get("op") == "exit":  # quit FreeCAD itself (servers launched by freecad_s1.ui.env)
            import os

            self.conn.close()
            os._exit(0)
        try:
            resp = {"ok": True, **self.worker.handle(req)}
        except Exception as exc:
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
        self.conn.setblocking(True)
        self.conn.sendall((PREFIX + json.dumps(resp, separators=(",", ":")) + "\n").encode())
        self.conn.setblocking(False)


def start(port: int = 8765, ui: bool = False) -> GuiServer:
    global _SERVER
    if _SERVER is None:
        _SERVER = GuiServer(port, ui=ui)
    return _SERVER
