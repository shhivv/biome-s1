import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Win32 process contracts")


def _stop_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for(path: Path, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def _spawn_tree(tmp_path: Path) -> tuple[subprocess.Popen, int]:
    child_pid = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    root = subprocess.Popen([sys.executable, "-c", code, str(child_pid)])
    _wait_for(child_pid)
    return root, int(child_pid.read_text())


def test_process_tree_contains_only_root_and_descendants(tmp_path):
    from freecad_s1.ui import _win32

    root, child_pid = _spawn_tree(tmp_path)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        tree = _win32.process_tree(root.pid)

        assert root.pid in tree
        assert child_pid in tree
        assert unrelated.pid not in tree
    finally:
        _stop_tree(root.pid)
        _stop_tree(unrelated.pid)


def test_pid_alive_tracks_exit():
    from freecad_s1.ui import _win32

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert _win32.pid_alive(proc.pid)
    finally:
        proc.kill()
        proc.wait(timeout=10)
    assert not _win32.pid_alive(proc.pid)


def test_memory_bytes_reads_live_process():
    from freecad_s1.ui import _win32

    measured = _win32.memory_bytes(os.getpid())

    assert measured is not None
    assert 1 << 20 < measured < 64 << 30


def test_terminate_stops_process():
    from freecad_s1.ui import _win32

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert _win32.terminate(proc.pid)
        proc.wait(timeout=10)
        assert not _win32.pid_alive(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_gui_process_timeout_kills_only_owned_tree(tmp_path):
    from freecad_s1.ui import _win32
    from freecad_s1.ui.launch import GuiProcess

    root, child_pid = _spawn_tree(tmp_path)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        watched = GuiProcess(root, "standin.py", mem_limit_gb=4, timeout=0.2, poll=0.05)
        watched.wait()

        assert watched.killed == "timeout"
        assert not _win32.pid_alive(root.pid)
        assert not _win32.pid_alive(child_pid)
        assert _win32.pid_alive(unrelated.pid)
    finally:
        _stop_tree(root.pid)
        _stop_tree(unrelated.pid)
