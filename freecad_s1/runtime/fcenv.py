"""Locate a FreeCAD Python interpreter that can `import FreeCAD`.

Override with FREECAD_PYTHON (interpreter) and FREECAD_LIB (directory that
contains FreeCAD.so / FreeCAD.pyd). Pure stdlib; used by the torch process to
spawn FreeCAD workers.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PREFIX = "@@S1 "  # marks worker responses on stdout

_CANDIDATES = [
    ("/Applications/FreeCAD.app/Contents/Resources/bin/python", "/Applications/FreeCAD.app/Contents/Resources/lib"),
    ("/usr/lib/freecad/bin/python3", "/usr/lib/freecad/lib"),
    ("/usr/lib/freecad-python3/bin/python3", "/usr/lib/freecad-python3/lib"),
]

_IS_WINDOWS = os.name == "nt"


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_python(path: str | Path, source: str) -> Path:
    candidate = _resolved(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"{source} does not name an existing Python executable: {candidate}")
    return candidate


def _require_lib(path: str | Path, source: str) -> Path:
    candidate = _resolved(path)
    if not candidate.is_dir():
        raise FileNotFoundError(f"{source} does not name an existing directory: {candidate}")
    if not (candidate / "FreeCAD.pyd").is_file():
        raise FileNotFoundError(f"{source} does not contain FreeCAD.pyd: {candidate}")
    return candidate


def _lib_near(python: Path) -> Path | None:
    for candidate in (python.parent, python.parent / "lib", python.parent.parent / "lib"):
        if candidate.is_dir() and (candidate / "FreeCAD.pyd").is_file():
            return candidate.resolve()
    return None


def _python_near(lib: Path) -> Path | None:
    for candidate in (lib / "python.exe", lib.parent / "bin" / "python.exe"):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _windows_layout(root_or_app: str | Path) -> tuple[str, str] | None:
    path = _resolved(root_or_app)
    roots = [path]
    if path.is_file():
        if path.name.casefold() != "freecad.exe":
            return None
        roots = [path.parent]
    elif path.is_dir():
        roots.extend(child for child in path.iterdir() if child.is_dir())
    for root in roots:
        for bin_dir in (root / "bin", root):
            if all((bin_dir / name).is_file() for name in ("FreeCAD.exe", "python.exe", "FreeCAD.pyd")):
                return str((bin_dir / "python.exe").resolve()), str(bin_dir.resolve())
    return None


def _windows_roots() -> tuple[Path, ...]:
    bases: list[Path] = []
    for value in (os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles")):
        if value:
            bases.append(Path(value))
    if os.environ.get("LOCALAPPDATA"):
        bases.append(Path(os.environ["LOCALAPPDATA"]) / "Programs")

    roots: list[Path] = []
    for base in bases:
        if not base.is_dir():
            continue
        roots.extend(sorted(base.glob("FreeCAD*"), reverse=True))
    return tuple(dict.fromkeys(root.resolve() for root in roots if root.is_dir()))


def freecad_python() -> tuple[str, str]:
    """Return (python executable, FreeCAD lib dir)."""
    py, lib = os.environ.get("FREECAD_PYTHON"), os.environ.get("FREECAD_LIB")
    if py and lib:
        if _IS_WINDOWS:
            return str(_require_python(py, "FREECAD_PYTHON")), str(_require_lib(lib, "FREECAD_LIB"))
        return py, lib
    if _IS_WINDOWS:
        if py:
            python = _require_python(py, "FREECAD_PYTHON")
            nearby = _lib_near(python)
            if nearby is not None:
                return str(python), str(nearby)
            raise FileNotFoundError(
                f"FREECAD_PYTHON has no sibling directory containing FreeCAD.pyd: {python}"
            )
        if lib:
            library = _require_lib(lib, "FREECAD_LIB")
            nearby = _python_near(library)
            if nearby is not None:
                return str(nearby), str(library)
            raise FileNotFoundError(f"FREECAD_LIB has no sibling python.exe: {library}")
        app = os.environ.get("FREECAD_APP")
        if app:
            layout = _windows_layout(app)
            if layout is not None:
                return layout
            raise FileNotFoundError(f"FREECAD_APP is not a supported FreeCAD Windows layout: {_resolved(app)}")
        for root in _windows_roots():
            layout = _windows_layout(root)
            if layout is not None:
                return layout
        raise FileNotFoundError(
            "FreeCAD not found on Windows; set FREECAD_APP or both FREECAD_PYTHON and FREECAD_LIB"
        )
    for cand_py, cand_lib in _CANDIDATES:
        if Path(cand_py).exists() and Path(cand_lib).exists():
            return py or cand_py, lib or cand_lib
    if lib:
        return py or shutil.which("python3") or "python3", lib
    raise FileNotFoundError("FreeCAD not found; set FREECAD_PYTHON and FREECAD_LIB")


def freecad_env() -> dict[str, str]:
    """Environment for a subprocess running FreeCAD's Python with this repo importable."""
    _, lib = freecad_python()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([lib, str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def freecad_available() -> bool:
    try:
        freecad_python()
        return True
    except FileNotFoundError:
        return False
