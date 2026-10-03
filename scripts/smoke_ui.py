"""Run scripts/smoke_ui.FCMacro in a hidden FreeCAD GUI and print its report.

    python scripts/smoke_ui.py --n 3 --seed 0
    python scripts/smoke_ui.py --goals 1 --n 0     # one fixed goal, for debugging

FreeCAD runs under the watchdog in freecad_s1/ui/launch.py: it is killed past
--mem-gb or --timeout, and when this script exits.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import argparse
import json
import shutil
import tempfile

from freecad_s1.ui.launch import launch_gui_script


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="sampled goals per level (on top of the fixed set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="also write the report here")
    ap.add_argument("--trace", help="write every step here before running it (to locate a crash)")
    ap.add_argument("--goals", type=int, default=0, help="only the first N goals (0 = all)")
    ap.add_argument("--mem-gb", type=float, default=2.0, help="kill FreeCAD above this memory footprint")
    ap.add_argument("--timeout", type=float, default=300.0, help="kill FreeCAD after this many seconds")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        script = Path(tmp) / "smoke_ui.FCMacro"  # unique path: the watchdog finds (and kills) FreeCAD by it
        shutil.copy(Path(__file__).with_name("smoke_ui.FCMacro"), script)
        proc = launch_gui_script(script,
                                 {"S1_OUT": str(out), "S1_N": str(args.n), "S1_SEED": str(args.seed),
                                  "S1_MAX_GOALS": str(args.goals),
                                  **({"S1_TRACE": str(Path(args.trace).resolve())} if args.trace else {})},
                                 log=Path(tmp) / "freecad.log", mem_limit_gb=args.mem_gb, timeout=args.timeout)
        proc.wait()
        print(f"FreeCAD peak memory {proc.peak_gb:.2f} GB" + (f"; KILLED: {proc.killed}" if proc.killed else ""),
              file=sys.stderr)
        if not out.exists():
            log = (Path(tmp) / "freecad.log").read_text()
            sys.exit("FreeCAD produced no report (crashed?):\n" + log[log.find("Running:"):][-3000:])
        report = json.loads(out.read_text())
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    if report.get("fatal") or report["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
