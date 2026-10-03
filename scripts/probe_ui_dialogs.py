"""Dump every feature task dialog's widgets from a hidden FreeCAD GUI (under
the watchdog), to check or update the field specs in freecad_s1/ui/spec.py.

    python scripts/probe_ui_dialogs.py > dialogs.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import shutil
import tempfile

from freecad_s1.ui.launch import launch_gui_script


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="s1-probe-") as tmp:
        script = Path(tmp) / "probe_ui_dialogs.FCMacro"  # unique path: the watchdog's handle on it
        shutil.copy(Path(__file__).with_suffix(".FCMacro"), script)
        out = Path(tmp) / "dialogs.json"
        proc = launch_gui_script(script, {"S1_OUT": str(out)}, log=Path(tmp) / "log", mem_limit_gb=2.0, timeout=300)
        proc.wait()
        if not out.exists():
            sys.exit(f"no output (killed: {proc.killed})")
        print(out.read_text())


if __name__ == "__main__":
    main()
