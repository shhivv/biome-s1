import io
import sys

from scripts import gui_demo


def _run_demo(tmp_path, monkeypatch, *, done=True, match=True, action="Done"):
    state = {
        "doc_open": True,
        "workbench": "PartDesignWorkbench",
        "shape": {"bbox": [0, 0, 0]},
    }
    goal = {"features": [{"kind": "base_box", "params": {"w": 1, "d": 1, "h": 1}}]}

    class FakeEnv:
        def __init__(self, port):
            pass

        def call(self, request):
            if request["op"] == "reset":
                return {
                    "goal": goal,
                    "start": {},
                    "budget": 1,
                    "state": state,
                    "actions": [action],
                    "expert": [action],
                }
            if request["op"] == "step":
                return {"info": {"error": None, "done": done}, "expert": []}
            if request["op"] == "score":
                return {"match": match, "iou": 1.0 if match else 0.0}
            if request["op"] == "save":
                return {"fcstd": request["fcstd"], "png": request["png"]}
            raise AssertionError(request)

    class FakePolicy:
        def __init__(self, model, device):
            pass

        def act(self, states, goals, actions):
            return [action]

    monkeypatch.setattr(gui_demo, "SocketEnv", FakeEnv)
    monkeypatch.setattr(gui_demo, "from_pretrained", lambda path: object())
    monkeypatch.setattr(gui_demo, "Policy", FakePolicy)
    monkeypatch.setattr(gui_demo.time, "sleep", lambda delay: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gui_demo.py", "--model", "release/hf", "--out", str(tmp_path), "--delay", "0"],
    )
    return gui_demo.main()


def test_main_returns_zero_for_matching_completed_result(tmp_path, monkeypatch):
    assert _run_demo(tmp_path, monkeypatch, done=True, match=True) == 0


def test_main_returns_nonzero_for_mismatch(tmp_path, monkeypatch):
    assert _run_demo(tmp_path, monkeypatch, done=True, match=False) == 1


def test_status_output_survives_legacy_console_encoding(tmp_path, monkeypatch):
    data = io.BytesIO()
    stdout = io.TextIOWrapper(data, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)

    assert _run_demo(tmp_path, monkeypatch, action="Akcja ś") == 0
    stdout.flush()
    assert "OK" in data.getvalue().decode("cp1252")
