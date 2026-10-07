"""Start a FreeCAD GUI process that runs a script, without stealing focus,
under a watchdog.

macOS: `open -g -j -n -W` starts a new, hidden FreeCAD instance in the
background (Qt's offscreen platform crashes once the 3D view needs OpenGL).
Elsewhere: the `freecad` binary under `xvfb-run` when available.

Guard rails: a hidden FreeCAD that leaks is invisible until the machine runs
out of memory (a runaway run once reached a 125 GB footprint), so every
launch is watched. A monitor thread samples FreeCAD's memory footprint (the
figure Activity Monitor shows, compressed and swapped pages included) and
kills FreeCAD with SIGKILL when it exceeds `mem_limit_gb` or runs longer
than `timeout`. FreeCAD is also killed when the launching Python process
exits. Defaults: S1_MEM_LIMIT_GB (4) and S1_TIMEOUT (seconds, 1800).

Inside FreeCAD, `self_guard()` adds a second layer (own footprint, deadline,
launcher still alive), for when the launcher itself dies hard.

Configuration reaches the script through environment variables (`S1_*`);
the script must put `S1_REPO` on sys.path itself and call `self_guard()`
first (see scripts/smoke_ui.FCMacro).
Override the application with FREECAD_APP (macOS bundle path or binary).

Pure stdlib.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..runtime.fcenv import REPO_ROOT, freecad_python

DEFAULT_MEM_LIMIT_GB = float(os.environ.get("S1_MEM_LIMIT_GB", "4"))
DEFAULT_TIMEOUT = float(os.environ.get("S1_TIMEOUT", "1800"))
_UNITS = {"B": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:
    from . import _win32


def freecad_app() -> str:
    app = os.environ.get("FREECAD_APP")
    if app:
        if _IS_WINDOWS:
            path = Path(app).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"FREECAD_APP does not name FreeCAD.exe: {path}")
            return str(path)
        return app
    if sys.platform == "darwin":
        return "/Applications/FreeCAD.app"
    if _IS_WINDOWS:
        _, lib = freecad_python()
        path = Path(lib) / "FreeCAD.exe"
        if path.is_file():
            return str(path.resolve())
        raise FileNotFoundError(f"FreeCAD.exe not found beside FreeCAD.pyd: {path}")
    return shutil.which("freecad") or shutil.which("FreeCAD") or "freecad"


def find_pids(script: str, root_pid: int | None = None) -> list[int]:
    """FreeCAD processes running `script` (it appears on their command line)."""
    if _IS_WINDOWS:
        return _win32.process_tree(root_pid) if root_pid is not None else []
    out = subprocess.run(["pgrep", "-f", script], capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if p.strip().isdigit() and int(p) != os.getpid()]


def memory_bytes(pid: int) -> int | None:
    """Memory footprint of `pid` (None if it is gone). macOS: `top`'s MEM
    (phys_footprint, includes compressed memory); Linux: VmRSS + VmSwap."""
    if _IS_WINDOWS:
        return _win32.memory_bytes(pid)
    if sys.platform == "darwin":
        out = subprocess.run(["top", "-l", "1", "-pid", str(pid), "-stats", "pid,mem"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == str(pid):
                m = re.match(r"([\d.]+)([BKMGT]?)", parts[1])
                if m:
                    return int(float(m.group(1)) * _UNITS.get(m.group(2) or "B", 1))
        return None
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    kb = sum(int(m.group(1)) for m in re.finditer(r"^(?:VmRSS|VmSwap):\s+(\d+) kB", status, re.M))
    return kb * 1024


def resident_bytes(pid: int) -> int | None:
    """Resident memory via `ps` (fast, but excludes compressed/swapped pages):
    the early signal, sampled often; `memory_bytes` is the full footprint."""
    if _IS_WINDOWS:
        return _win32.memory_bytes(pid)
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 if out.isdigit() else None


def _kill(pid: int, created_at: int | None = None) -> bool:
    if _IS_WINDOWS:
        return _win32.terminate(pid, created_at)
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False


class GuiProcess:
    """A watched FreeCAD GUI run. `wait()` returns once FreeCAD has exited
    (or was killed); `killed` says why it was killed, if it was.

    On Windows, process ownership is pinned to creation times so a recycled
    PID is never measured or killed. Other platforms retain the existing
    command-line membership check."""

    def __init__(self, proc: subprocess.Popen, script: str, mem_limit_gb: float, timeout: float,
                 poll: float = 0.2, footprint_every: int = 10) -> None:
        self.proc = proc
        self.script = script
        self.mem_limit = int(mem_limit_gb * (1 << 30))
        self.deadline = time.time() + timeout
        self.poll = poll
        self.footprint_every = footprint_every  # full footprint (slow `top`) every N polls; `ps` RSS every poll
        self.peak = 0
        self.killed: str | None = None
        self._root_created_at = _win32.process_created_at(proc.pid) if _IS_WINDOWS else None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        atexit.register(self.kill)

    def _owned_processes(self) -> list[tuple[int, int | None]]:
        if _IS_WINDOWS:
            if self._root_created_at is None:
                return []
            return _win32.process_tree_identities(self.proc.pid, self._root_created_at)
        return [(pid, None) for pid in find_pids(self.script, self.proc.pid)]

    def _watch(self) -> None:
        tick = 0
        while not self._stop.is_set():
            processes = self._owned_processes()
            if self.proc.poll() is not None and not processes:
                return
            for pid, _ in processes:
                mem = resident_bytes(pid)
                if tick % self.footprint_every == 0:
                    mem = max(mem or 0, memory_bytes(pid) or 0) or None
                if mem is None:
                    continue
                self.peak = max(self.peak, mem)
                if mem > self.mem_limit:
                    self.kill(f"memory {mem / (1 << 30):.1f} GB > limit {self.mem_limit / (1 << 30):.1f} GB")
            if time.time() > self.deadline:
                self.kill("timeout")
            tick += 1
            self._stop.wait(self.poll)

    def kill(self, reason: str = "parent exiting") -> None:
        hit = [pid for pid, created_at in reversed(self._owned_processes()) if _kill(pid, created_at)]
        if hit and self.killed is None:
            self.killed = reason
        if self.proc.poll() is None:
            self.proc.kill()

    def wait(self) -> int | None:
        try:
            self.proc.wait()
            self._thread.join(timeout=30)
            self.kill("still running after its launcher exited")
        except BaseException:  # Ctrl-C etc.: never leave a hidden FreeCAD behind
            self.kill("interrupted")
            raise
        return self.proc.returncode

    @property
    def peak_gb(self) -> float:
        return self.peak / (1 << 30)


def self_guard(poll: float = 0.25, mem_limit_gb: float | None = None, timeout: float | None = None) -> None:
    """Call first thing in a script launched by `launch_gui_script` (inside
    FreeCAD): a daemon thread SIGKILLs this FreeCAD when its own footprint
    exceeds S1_MEM_LIMIT_GB, when S1_TIMEOUT passes, or when the launching
    Python process (S1_PARENT_PID) is gone. Second layer behind the
    launcher's watchdog, for when the launcher itself dies hard."""
    limit = float(mem_limit_gb or os.environ.get("S1_MEM_LIMIT_GB", DEFAULT_MEM_LIMIT_GB)) * (1 << 30)
    deadline = time.time() + float(timeout or os.environ.get("S1_TIMEOUT", DEFAULT_TIMEOUT))
    parent = int(os.environ.get("S1_PARENT_PID", "0"))
    me = os.getpid()

    def parent_alive() -> bool:
        if not parent:
            return True
        if _IS_WINDOWS:
            return _win32.pid_alive(parent)
        try:
            os.kill(parent, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def watch() -> None:
        tick = 0
        while True:
            mem = resident_bytes(me) or 0
            if tick % 8 == 0:
                mem = max(mem, memory_bytes(me) or 0)
            tick += 1
            if mem > limit or time.time() > deadline or not parent_alive():
                sys.stderr.write(f"S1 self-guard: killing FreeCAD (mem {mem / (1 << 30):.1f} GB)\n")
                if _IS_WINDOWS:
                    _win32.terminate(me)
                else:
                    os.kill(me, signal.SIGKILL)
            time.sleep(poll)

    threading.Thread(target=watch, daemon=True, name="s1-self-guard").start()


def launch_gui_script(script: str | Path, env: dict[str, str], log: str | Path | None = None,
                      mem_limit_gb: float = DEFAULT_MEM_LIMIT_GB, timeout: float = DEFAULT_TIMEOUT,
                      hidden: bool = True) -> GuiProcess:
    """Run `script` in a new hidden FreeCAD GUI under the watchdog. The script
    should end with `os._exit` so FreeCAD quits when it is done."""
    script = str(Path(script).resolve())
    if find_pids(script):
        raise RuntimeError(f"a FreeCAD is already running {script}; kill it first (pkill -f {script})")
    env = {"S1_REPO": str(REPO_ROOT), "S1_MEM_LIMIT_GB": str(mem_limit_gb), "S1_TIMEOUT": str(timeout),
           "S1_PARENT_PID": str(os.getpid()),
           **{k: os.environ[k] for k in ("S1_DEFAULT_SNAP",) if k in os.environ},  # training-goal options
           **env}
    log = str(Path(log).resolve()) if log else os.devnull
    if sys.platform == "darwin":
        # -g: don't take focus; -j: launch hidden. Demos (hidden=False) need a real, rendered window.
        cmd = ["open", "-g", *(["-j"] if hidden else []), "-n", "-W", "--stdout", log, "--stderr", log]
        for k, v in env.items():
            cmd += ["--env", f"{k}={v}"]
        cmd += ["-a", freecad_app(), "--args", script]
        proc = subprocess.Popen(cmd)
    else:
        full_env = {**os.environ, **env}
        cmd = [freecad_app(), script]
        if shutil.which("xvfb-run") and not full_env.get("DISPLAY"):
            cmd = ["xvfb-run", "-a", *cmd]
        with open(log, "a") as fh:
            proc = subprocess.Popen(cmd, env=full_env, stdout=fh, stderr=subprocess.STDOUT)
    return GuiProcess(proc, script, mem_limit_gb, timeout)
