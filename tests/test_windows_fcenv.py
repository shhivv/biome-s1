from pathlib import Path

import pytest

from freecad_s1.runtime import fcenv


def _freecad_tree(root: Path) -> tuple[Path, Path, Path]:
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    app = bin_dir / "FreeCAD.exe"
    python = bin_dir / "python.exe"
    module = bin_dir / "FreeCAD.pyd"
    for path in (app, python, module):
        path.touch()
    return app, python, bin_dir


def _windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fcenv, "_IS_WINDOWS", True, raising=False)
    monkeypatch.setenv("ProgramW6432", str(tmp_path / "Program Files 64"))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "Program Files"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local AppData"))
    for name in ("FREECAD_APP", "FREECAD_PYTHON", "FREECAD_LIB"):
        monkeypatch.delenv(name, raising=False)


def test_windows_discovers_installed_layout(tmp_path, monkeypatch):
    _, python, lib = _freecad_tree(tmp_path / "Program Files 64" / "FreeCAD 1.1")
    _windows(monkeypatch, tmp_path)

    assert fcenv.freecad_python() == (str(python.resolve()), str(lib.resolve()))


def test_windows_freecad_app_handles_spaces(tmp_path, monkeypatch):
    app, python, lib = _freecad_tree(tmp_path / "Custom Tools" / "FreeCAD 1.1")
    _windows(monkeypatch, tmp_path)
    monkeypatch.setenv("FREECAD_APP", str(app))

    assert fcenv.freecad_python() == (str(python.resolve()), str(lib.resolve()))


@pytest.mark.parametrize("override", ["python", "lib"])
def test_windows_partial_override_derives_sibling(tmp_path, monkeypatch, override):
    _, python, lib = _freecad_tree(tmp_path / "FreeCAD")
    _windows(monkeypatch, tmp_path)
    if override == "python":
        monkeypatch.setenv("FREECAD_PYTHON", str(python))
    else:
        monkeypatch.setenv("FREECAD_LIB", str(lib))

    assert fcenv.freecad_python() == (str(python.resolve()), str(lib.resolve()))


@pytest.mark.parametrize(
    ("name", "value"),
    [("FREECAD_PYTHON", "missing-python.exe"), ("FREECAD_LIB", "missing-lib")],
)
def test_windows_invalid_override_is_actionable(tmp_path, monkeypatch, name, value):
    _windows(monkeypatch, tmp_path)
    monkeypatch.setenv(name, str(tmp_path / value))

    with pytest.raises(FileNotFoundError, match=name):
        fcenv.freecad_python()


def test_windows_missing_install_is_actionable(tmp_path, monkeypatch):
    _windows(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError, match=r"FreeCAD not found on Windows.*FREECAD_APP"):
        fcenv.freecad_python()


def test_non_windows_python_override_keeps_discovered_library(tmp_path, monkeypatch):
    override_python = tmp_path / "custom" / "python3"
    override_python.parent.mkdir()
    override_python.touch()
    discovered_python = tmp_path / "freecad" / "bin" / "python3"
    discovered_lib = tmp_path / "freecad" / "lib"
    discovered_python.parent.mkdir(parents=True)
    discovered_python.touch()
    discovered_lib.mkdir()
    monkeypatch.setattr(fcenv, "_IS_WINDOWS", False, raising=False)
    monkeypatch.setattr(fcenv, "_CANDIDATES", [(str(discovered_python), str(discovered_lib))])
    monkeypatch.setenv("FREECAD_PYTHON", str(override_python))
    monkeypatch.delenv("FREECAD_LIB", raising=False)

    assert fcenv.freecad_python() == (str(override_python), str(discovered_lib))


def test_non_windows_lib_override_keeps_discovered_python(tmp_path, monkeypatch):
    discovered_python = tmp_path / "freecad" / "bin" / "python3"
    discovered_lib = tmp_path / "freecad" / "lib"
    override_lib = tmp_path / "custom" / "lib"
    discovered_python.parent.mkdir(parents=True)
    discovered_python.touch()
    discovered_lib.mkdir()
    override_lib.mkdir(parents=True)
    monkeypatch.setattr(fcenv, "_IS_WINDOWS", False, raising=False)
    monkeypatch.setattr(fcenv, "_CANDIDATES", [(str(discovered_python), str(discovered_lib))])
    monkeypatch.delenv("FREECAD_PYTHON", raising=False)
    monkeypatch.setenv("FREECAD_LIB", str(override_lib))

    assert fcenv.freecad_python() == (str(discovered_python), str(override_lib))


def test_freecad_env_preserves_existing_pythonpath(tmp_path, monkeypatch):
    _, python, lib = _freecad_tree(tmp_path / "FreeCAD With Spaces")
    _windows(monkeypatch, tmp_path)
    monkeypatch.setenv("FREECAD_PYTHON", str(python))
    monkeypatch.setenv("FREECAD_LIB", str(lib))
    monkeypatch.setenv("PYTHONPATH", "existing-path")

    env = fcenv.freecad_env()

    assert env["PYTHONPATH"].split(fcenv.os.pathsep) == [
        str(lib.resolve()),
        str(fcenv.REPO_ROOT),
        "existing-path",
    ]
