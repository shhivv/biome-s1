"""UI layer: element mapping, dialog teacher, and the FreeCAD watchdog.
No FreeCAD needed: the watchdog is exercised on stand-in Python processes."""
import subprocess
import sys
import textwrap

from freecad_s1.schema import GoalFeature
from freecad_s1.ui import spec as S
from freecad_s1.ui.launch import GuiProcess, memory_bytes
from freecad_s1.ui.teacher import command_elements, dialog_elements, dialog_expert, fields_on_plan


def test_command_elements_map_to_interface_roles():
    acts = ["Std_Workbench:PartDesignWorkbench", "PartDesign_Pad", "Select:Face+Z", "Sketcher_CreateCircle", "Done"]
    els = command_elements(acts)
    assert els == ["wb:PartDesignWorkbench", "cmd:PartDesign_Pad", "canvas:Select:Face+Z",
                   "canvas:Sketcher_CreateCircle", "Done"]
    assert [S.underlying(e) for e in els] == acts
    assert S.role("set:lengthEdit") == "set" and S.underlying("set:lengthEdit") is None


def _pocket_ui(mode="Dimension", length=5.0, reversed_=False):
    return {"dialog": "PartDesign_Pocket", "fields": {
        "sidesMode": {"kind": "choice", "value": "One sided", "options": ["One sided", "Two sided", "Symmetric"],
                      "enabled": True},
        "changeMode": {"kind": "choice", "value": mode, "enabled": True,
                       "options": ["Dimension", "Through all", "To first", "Up to face", "Up to shape"]},
        "lengthEdit": {"kind": "number", "unit": "mm", "value": length, "enabled": mode == "Dimension"},
        "checkBoxReversed": {"kind": "check", "value": reversed_, "enabled": True}}}


def test_dialog_elements_skip_reference_picks_and_disabled_fields():
    els = dialog_elements(_pocket_ui(mode="Through all"))
    assert "opt:changeMode=Through all" in els and "opt:changeMode=Up to face" not in els
    assert "opt:sidesMode=Two sided" not in els
    assert "set:lengthEdit" not in els  # disabled in through-all mode
    assert els[-2:] == [S.OK, S.CANCEL]


def test_dialog_teacher_fills_fields_then_ok_and_cancels_off_plan():
    hole = GoalFeature("hole", {"r": 3, "x": 0, "y": 0})
    pocket = GoalFeature("pocket_rect", {"w": 6, "d": 5, "x": 0, "y": 0, "depth": 3})
    t_hole = S.targets("PartDesign_Pocket", hole, 40)
    t_rect = S.targets("PartDesign_Pocket", pocket, 40)
    assert dialog_expert(_pocket_ui(), t_hole, True) == ["opt:changeMode=Through all"]
    assert dialog_expert(_pocket_ui(mode="Through all"), t_hole, True) == [S.OK]
    assert dialog_expert(_pocket_ui(length=5.0, reversed_=True), t_rect, True) == ["set:lengthEdit",
                                                                                    "toggle:checkBoxReversed"]
    assert dialog_expert(_pocket_ui(length=3.0), t_rect, True) == [S.OK]
    assert fields_on_plan(_pocket_ui(length=3.0), t_rect) and not fields_on_plan(_pocket_ui(), t_rect)
    assert dialog_expert(_pocket_ui(length=3.0), t_rect, False) == [S.CANCEL]


def test_every_dialog_command_has_targets_for_its_fields():
    f = GoalFeature("base_box", {"w": 20, "d": 10, "h": 5})
    for cmd, fields in S.DIALOG_FIELDS.items():
        assert set(S.targets(cmd, f, 20)) == {fl.name for fl in fields}, cmd


def _standin(tmp_path, body: str):
    script = tmp_path / "standin_freecad.py"
    script.write_text(textwrap.dedent(body))
    return script, subprocess.Popen([sys.executable, str(script)])


def test_watchdog_kills_on_memory_limit(tmp_path):
    script, proc = _standin(tmp_path, """
        import time
        hog = []
        for _ in range(60):  # ~1.5 GB at most, if the watchdog failed
            hog.append(bytearray(25 << 20))
            time.sleep(0.05)
        time.sleep(30)
    """)
    g = GuiProcess(proc, str(script), mem_limit_gb=0.3, timeout=60, poll=0.2)
    g.wait()
    assert g.killed and g.killed.startswith("memory"), g.killed
    assert proc.returncode is not None


def test_watchdog_kills_on_timeout(tmp_path):
    script, proc = _standin(tmp_path, "import time; time.sleep(60)")
    g = GuiProcess(proc, str(script), mem_limit_gb=4, timeout=1.0, poll=0.2)
    g.wait()
    assert g.killed == "timeout"


def test_watchdog_leaves_finished_runs_alone(tmp_path):
    script, proc = _standin(tmp_path, "print('ok')")
    g = GuiProcess(proc, str(script), mem_limit_gb=4, timeout=30, poll=0.2)
    assert g.wait() == 0 and g.killed is None


def test_memory_probe_reads_own_process():
    import os
    mem = memory_bytes(os.getpid())
    assert mem is not None and 1 << 20 < mem < 64 << 30


def _self_guarded(tmp_path, env: dict, body: str) -> int:
    import os
    from freecad_s1.runtime.fcenv import REPO_ROOT

    script = tmp_path / "guarded.py"
    script.write_text(f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\n"
                      "from freecad_s1.ui.launch import self_guard\nself_guard(poll=0.2)\n" + textwrap.dedent(body))
    proc = subprocess.run([sys.executable, str(script)], env={**os.environ, **env}, timeout=60)
    return proc.returncode


def test_self_guard_kills_its_own_process_over_memory(tmp_path):
    code = _self_guarded(tmp_path, {"S1_MEM_LIMIT_GB": "0.3"}, """
        import time
        hog = []
        for _ in range(60):
            hog.append(bytearray(25 << 20))
            time.sleep(0.05)
        time.sleep(30)
    """)
    assert code == -9


def test_self_guard_exits_when_launcher_is_gone(tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    code = _self_guarded(tmp_path, {"S1_PARENT_PID": str(dead.pid)}, "import time; time.sleep(30)")
    assert code == -9


def test_ui_model_scores_dialog_elements_with_live_values():
    import random

    import torch

    from freecad_s1.goals import sample_goal
    from freecad_s1.model.featurize import collate, make_example
    from freecad_s1.model.net import S1Config, S1Model
    from freecad_s1.schema import Node, ShapeInfo, State

    goal = sample_goal(2, random.Random(0))
    ui = _pocket_ui(length=5.0)
    st = State(doc_open=True, workbench="PartDesignWorkbench", has_body=True,
               tree=[Node("Body", "PartDesign::Body"), Node("Pocket", "PartDesign::Pocket", parent=0, depth=1)],
               recent=["cmd:PartDesign_Body", "canvas:Select:Plane:XY", "cmd:PartDesign_Pocket"],
               shape=ShapeInfo(True, 1000.0, 600.0, (10, 10, 10), 6, 12, 1, ["+Z"]), ui=ui)
    elements = dialog_elements(ui)
    model = S1Model(S1Config(ui=True)).eval()
    opts = model.cfg.feature_opts()
    ex = make_example(st, goal, elements, dialog_expert(ui, {"lengthEdit": 3.0}, True), **opts)
    assert ex.actions["role"].tolist().count(4) == 5  # One sided, Symmetric + Dimension, Through all, To first
    assert ex.actions["ui"][elements.index("set:lengthEdit")][0] > 0  # current length / scale
    assert ex.actions["ui"][elements.index("opt:changeMode=Dimension")][4] == 1.0  # current entry
    assert ex.target.tolist() == [e == "set:lengthEdit" for e in elements]
    logits, _ = model(collate([ex]))
    assert logits.shape == (1, len(elements)) and torch.isfinite(logits).all()
    # the same state with the field already at its target changes the option's features
    ui2 = _pocket_ui(length=3.0)
    st.ui = ui2
    ex2 = make_example(st, goal, dialog_elements(ui2), **opts)
    assert (ex2.actions["ui"] != ex.actions["ui"]).any()


def test_truncated_shard_keeps_complete_records(tmp_path):
    import gzip
    import json as _json

    from freecad_s1.data import read_records

    path = tmp_path / "shard000.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        for i in range(200):
            fh.write(_json.dumps({"ep": "e", "i": i, "pad": "x" * 200}) + "\n")
            if i == 99:
                fh.flush()  # what datagen does after each episode
    data = path.read_bytes()
    cut = tmp_path / "cut.jsonl.gz"
    cut.write_bytes(data[: len(data) * 2 // 3])  # as if SIGKILLed mid-write
    recs = list(read_records(str(cut)))
    assert len(recs) >= 100 and [r["i"] for r in recs] == list(range(len(recs)))
