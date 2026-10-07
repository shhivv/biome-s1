import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell contracts")

REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_SCRIPTS = REPO_ROOT / "scripts" / "windows"
POWERSHELL = shutil.which("powershell.exe") or "powershell.exe"


def _ps_file(name: str, *args: str, env=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WINDOWS_SCRIPTS / name),
            *args,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def _quote_ps(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _freecad_fixture(root: Path, *, executable_python=False) -> Path:
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "FreeCAD.exe").touch()
    if executable_python:
        shutil.copy2(Path(os.environ["SystemRoot"]) / "System32" / "where.exe", bin_dir / "python.exe")
    else:
        (bin_dir / "python.exe").touch()
    (bin_dir / "FreeCAD.pyd").touch()
    return root


def _wait_dead(pid: int, timeout: float = 10.0) -> bool:
    from freecad_s1.ui._win32 import pid_alive

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_common_discovers_installed_freecad(tmp_path):
    installed = _freecad_fixture(tmp_path / "Program Files" / "FreeCAD 1.1")
    env = os.environ.copy()
    env["ProgramW6432"] = str(tmp_path / "Program Files")
    env["ProgramFiles"] = str(tmp_path / "Other Program Files")
    env["LOCALAPPDATA"] = str(tmp_path / "Local AppData")
    for name in ("FREECAD_APP", "FREECAD_PYTHON", "FREECAD_LIB"):
        env.pop(name, None)
    command = (
        f". {_quote_ps(WINDOWS_SCRIPTS / 'common.ps1')}; "
        "(Find-S1FreeCADRoot) | Write-Output"
    )

    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == installed.resolve()


def test_preflight_preserves_paths_with_spaces_and_is_repeatable(tmp_path):
    root = _freecad_fixture(tmp_path / "Installed Tools" / "FreeCAD 1.1")

    first = _ps_file("preflight.ps1", "-FreeCADRoot", str(root), "-SkipImportCheck", "-Json")
    second = _ps_file("preflight.ps1", "-FreeCADRoot", str(root), "-SkipImportCheck", "-Json")

    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    payload = json.loads(first.stdout.strip().splitlines()[-1])
    assert Path(payload["Root"]) == root.resolve()
    assert Path(payload["App"]) == (root / "bin" / "FreeCAD.exe").resolve()
    assert Path(payload["Python"]) == (root / "bin" / "python.exe").resolve()
    assert Path(payload["Lib"]) == (root / "bin").resolve()
    assert Path(payload["ProjectPython"]) == (REPO_ROOT / ".venv" / "Scripts" / "python.exe").resolve()


@pytest.mark.parametrize("missing", ["FreeCAD.exe", "python.exe", "FreeCAD.pyd"])
def test_preflight_reports_each_missing_freecad_artifact(tmp_path, missing):
    root = _freecad_fixture(tmp_path / "FreeCAD")
    (root / "bin" / missing).unlink()

    result = _ps_file("preflight.ps1", "-FreeCADRoot", str(root), "-SkipImportCheck")

    assert result.returncode != 0
    assert missing in result.stderr


def test_preflight_reports_missing_project_environment(tmp_path):
    root = _freecad_fixture(tmp_path / "FreeCAD")
    missing_python = tmp_path / "missing venv" / "python.exe"

    result = _ps_file(
        "preflight.ps1",
        "-FreeCADRoot",
        str(root),
        "-ProjectPython",
        str(missing_python),
        "-SkipImportCheck",
    )

    assert result.returncode != 0
    assert str(missing_python) in result.stderr


def test_preflight_reports_failed_freecad_imports(tmp_path):
    root = _freecad_fixture(tmp_path / "FreeCAD", executable_python=True)

    result = _ps_file("preflight.ps1", "-FreeCADRoot", str(root))

    assert result.returncode != 0
    assert "FreeCAD and Part" in result.stderr


def test_preflight_rejects_busy_port_without_stopping_listener(tmp_path):
    root = _freecad_fixture(tmp_path / "FreeCAD")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    port = listener.getsockname()[1]
    try:
        result = _ps_file(
            "preflight.ps1",
            "-FreeCADRoot",
            str(root),
            "-Port",
            str(port),
            "-SkipImportCheck",
        )

        assert result.returncode != 0
        assert str(port) in result.stderr
        probe = socket.create_connection(("127.0.0.1", port), timeout=2)
        probe.close()
    finally:
        listener.close()


def _fake_launch_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    capture = tmp_path / "capture with spaces"
    capture.mkdir()
    scripts = tmp_path / "scripts with spaces"
    scripts.mkdir()
    server = scripts / "fake server.py"
    server.write_text(
        """
import json, os, pathlib, socket, subprocess, sys, time
capture = pathlib.Path(os.environ["S1_TEST_CAPTURE"])
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
(capture / "child.pid").write_text(str(child.pid))
(capture / "server.json").write_text(json.dumps({
    "pid": os.getpid(), "argv": sys.argv,
    "repo": os.environ.get("FREECAD_S1_REPO"),
    "ui": os.environ.get("FREECAD_S1_UI", "")
}))
sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("127.0.0.1", 8765))
sock.listen(1)
while True:
    time.sleep(1)
"""
    )
    runner = scripts / "fake runner.py"
    runner.write_text(
        """
import json, os, pathlib, sys
capture = pathlib.Path(os.environ["S1_TEST_CAPTURE"])
(capture / "runner.json").write_text(json.dumps({
    "argv": sys.argv, "ui": os.environ.get("FREECAD_S1_UI", "")
}))
raise SystemExit(int(os.environ.get("S1_TEST_RUNNER_EXIT", "0")))
"""
    )
    return capture, server, runner


@pytest.mark.parametrize(
    ("launcher", "model", "ui"),
    [("start-taiga.ps1", "release/hf", ""), ("start-mesa.ps1", "release/mesa-s1", "1")],
)
def test_launcher_preserves_arguments_mode_and_cleans_owned_tree(tmp_path, launcher, model, ui):
    root = _freecad_fixture(tmp_path / "FreeCAD With Spaces")
    capture, server, runner = _fake_launch_files(tmp_path)
    output = tmp_path / "Output With Spaces"

    result = _ps_file(
        launcher,
        "-FreeCADRoot",
        str(root),
        "-Out",
        str(output),
        "-SkipImportCheck",
        "-FreeCADApp",
        sys.executable,
        "-MacroPath",
        str(server),
        "-RunnerPath",
        str(runner),
        "-CapturePath",
        str(capture),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    server_data = json.loads((capture / "server.json").read_text())
    run_data = json.loads((capture / "runner.json").read_text())
    assert Path(server_data["argv"][0]) == server
    assert Path(server_data["repo"]) == REPO_ROOT
    assert server_data["ui"] == run_data["ui"] == ui
    assert run_data["argv"][run_data["argv"].index("--model") + 1].replace("\\", "/").endswith(model)
    assert Path(run_data["argv"][run_data["argv"].index("--out") + 1]) == output
    assert _wait_dead(server_data["pid"])
    assert _wait_dead(int((capture / "child.pid").read_text()))


def test_launcher_propagates_runner_failure_and_still_cleans_up(tmp_path):
    root = _freecad_fixture(tmp_path / "FreeCAD")
    capture, server, runner = _fake_launch_files(tmp_path)
    env = os.environ.copy()
    env["S1_TEST_RUNNER_EXIT"] = "7"

    result = _ps_file(
        "start-taiga.ps1",
        "-FreeCADRoot",
        str(root),
        "-SkipImportCheck",
        "-FreeCADApp",
        sys.executable,
        "-MacroPath",
        str(server),
        "-RunnerPath",
        str(runner),
        "-CapturePath",
        str(capture),
        env=env,
    )

    assert result.returncode == 7, result.stdout + result.stderr
    server_data = json.loads((capture / "server.json").read_text())
    assert _wait_dead(server_data["pid"])
    assert _wait_dead(int((capture / "child.pid").read_text()))
