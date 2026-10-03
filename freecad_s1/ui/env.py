"""Torch-side handles on hidden FreeCAD GUIs serving UI-level sessions.

`UiEnv` has FreeCADEnv's send/recv/call/close interface, so `UiVecEnv` is a
drop-in for runtime.client.VecEnv in rollouts, evaluation and DAgger. Each
env is a hidden FreeCAD running scripts/ui_server.FCMacro on its own port,
under the watchdog in ui/launch.py.
"""

from __future__ import annotations

import json
import socket
import tempfile
import time
from pathlib import Path

from ..runtime.client import Episode, VecEnv, WorkerError
from ..runtime.fcenv import PROTOCOL_PREFIX as PREFIX
from ..runtime.fcenv import REPO_ROOT
from ..schema import Goal, State
from .launch import DEFAULT_MEM_LIMIT_GB, GuiProcess, launch_gui_script

SERVER_SCRIPT = REPO_ROOT / "scripts" / "ui_server.FCMacro"


class UiEnv:
    def __init__(self, port: int, mem_limit_gb: float = DEFAULT_MEM_LIMIT_GB, timeout: float = 6 * 3600,
                 connect_timeout: float = 120.0) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="s1-ui-env-")
        script = Path(self.tmp.name) / f"ui_server_{port}.FCMacro"  # unique path: the watchdog's handle on it
        script.write_text(SERVER_SCRIPT.read_text())
        self.proc: GuiProcess = launch_gui_script(script, {"S1_PORT": str(port)}, log=Path(self.tmp.name) / "log",
                                                  mem_limit_gb=mem_limit_gb, timeout=timeout)
        deadline = time.time() + connect_timeout
        while True:
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=600)
                break
            except OSError:
                if time.time() > deadline or self.proc.killed or self.proc.proc.poll() is not None:
                    self.proc.kill("could not connect")
                    raise WorkerError(f"UI server on port {port} did not come up ({self.proc.killed})")
                time.sleep(1)
        self.file = self.sock.makefile("r")

    def send(self, req: dict) -> None:
        self.sock.sendall((json.dumps(req) + "\n").encode())

    def recv(self) -> dict:
        line = self.file.readline()
        if not line:
            raise WorkerError(f"UI server closed the connection ({self.proc.killed or 'exited'})")
        resp = json.loads(line[len(PREFIX):])
        if not resp.get("ok"):
            raise WorkerError(resp.get("error", "unknown error"))
        return resp

    def call(self, req: dict) -> dict:
        self.send(req)
        return self.recv()

    def close(self) -> None:
        try:
            self.send({"op": "exit"})
        except OSError:
            pass
        try:
            self.proc.proc.wait(timeout=20)
        except Exception:
            pass
        self.proc.kill("closed")
        self.tmp.cleanup()


class UiVecEnv(VecEnv):
    """N hidden FreeCAD GUIs stepped in lockstep (ports base_port..+n-1).

    FreeCAD occasionally segfaults inside a task dialog. A slot whose FreeCAD
    dies is restarted; its episode ends with `info.crashed` (rollout.py
    reports it as outcome "crashed") and scores 0, so one crash never takes
    down a whole evaluation or DAgger round."""

    def __init__(self, n: int, base_port: int = 8800, mem_limit_gb: float = DEFAULT_MEM_LIMIT_GB) -> None:
        self.base_port = base_port
        self.mem_limit_gb = mem_limit_gb
        self.envs = []
        self.last: list[dict | None] = [None] * n
        self.crashed: set[int] = set()
        self.crashes = 0
        try:
            for i in range(n):
                self.envs.append(UiEnv(base_port + i, mem_limit_gb=mem_limit_gb))
        except BaseException:
            self.close()
            raise

    def _restart(self, i: int) -> None:
        self.crashes += 1
        self.crashed.add(i)
        try:
            self.envs[i].close()
        except Exception:
            pass
        self.envs[i] = UiEnv(self.base_port + i, mem_limit_gb=self.mem_limit_gb)

    def reset(self, specs: list[dict]):
        eps = []
        for i, spec in enumerate(specs):
            for attempt in range(3):
                try:
                    if attempt:
                        self._restart(i)
                    r = self.envs[i].call({"op": "reset", **spec})
                    break
                except WorkerError:
                    if attempt == 2:
                        raise
            self.crashed.discard(i)
            self.last[i] = r
            goal = Goal.from_json(r["goal"])
            eps.append(Episode(goal, r["budget"], spec.get("level", goal.level), State.from_json(r["state"]),
                               r["actions"], r["expert"], progress=r.get("progress", -1)))
        return eps

    def step(self, idx: list[int], actions: list[str], reward: bool = False) -> list[dict]:
        for i, a in zip(idx, actions):
            try:
                self.envs[i].send({"op": "step", "action": a, "reward": reward})
            except OSError:
                pass  # picked up as a failed recv below
        out = []
        for i in idx:
            try:
                r = self.envs[i].recv()
                self.last[i] = r
            except (WorkerError, OSError, ValueError):
                self._restart(i)
                prev = self.last[i] or {}
                r = {"ok": True, "info": {"changed": False, "error": "FreeCAD crashed", "done": True, "crashed": True},
                     "state": prev.get("state"), "actions": prev.get("actions", []), "expert": [],
                     "progress": prev.get("progress", -1)}
            out.append(r)
        return out

    def score(self, idx: list[int]) -> list[dict]:
        out = []
        for i in idx:
            if i in self.crashed:
                out.append({"ok": True, "iou": 0.0, "match": False, "crashed": True})
                continue
            try:
                out.append(self.envs[i].call({"op": "score"}))
            except (WorkerError, OSError, ValueError):
                self._restart(i)
                out.append({"ok": True, "iou": 0.0, "match": False, "crashed": True})
        return out
