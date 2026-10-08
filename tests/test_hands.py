"""Screen-reading logic of the hands (no FreeCAD, no screen needed)."""
import sys

import pytest

if sys.platform != "darwin":
    pytest.skip("macOS only (pyobjc)", allow_module_level=True)
pytest.importorskip("Quartz")

from freecad_s1.ui.hands import SOLID_PREFIXES, Hands, _menu_title, parse_status  # noqa: E402
from freecad_s1.ui.screen import Text  # noqa: E402


def test_parse_status_face():
    h = parse_status("Preselected: S1Doc2.Body.Pad.Face3 (-1.00 mm, -0.37 mm, 8.00 mm)  ")
    assert (h.obj, h.sub, h.point) == ("Pad", "Face3", (-1.0, -0.37, 8.0))


def test_parse_status_units_and_objects():
    h = parse_status("Preselected: Doc.Body.Pocket001.Edge12 (1.5 cm, 0 mm, -20 µm)")
    assert (h.obj, h.sub) == ("Pocket001", "Edge12")
    assert h.point == pytest.approx((15.0, 0.0, -0.02))
    h = parse_status("Preselected: Doc.Body.XY_Plane (0.00 mm, 1.00 mm, 0.00 mm)")
    assert (h.obj, h.sub) == ("XY_Plane", "")


def test_parse_status_other_messages():
    assert parse_status("") is None
    assert parse_status("Selected: Doc.Body.Pad.Face3") is None


def test_menu_titles():
    assert _menu_title("2 Top") == "Top"
    assert _menu_title("3 Wireframe (V, 3)") == "Wireframe"
    assert _menu_title("Standard Views") == "Standard Views"
    assert _menu_title("Fit All (V, F)") == "Fit All"


def rows():
    return [Text("S1Doc2", 32, 244, 40, 12), Text("- • Body", 13, 260, 50, 12), Text("• & F Origin", 26, 276, 60, 12),
            Text("  XY_Plane (Base)", 40, 300, 80, 12), Text("• Pad001", 30, 320, 40, 12),
            Text("& Pocket", 30, 340, 50, 12)]


def test_tree_rows():
    assert Hands._find_row(rows(), r"XY[_ ]?Plane\d*")[1] == "XY_Plane"
    assert Hands._find_row(rows(), r"YZ[_ ]?Plane\d*") is None
    tip, word = Hands._find_row(rows(), "(?:" + "|".join(SOLID_PREFIXES) + r")\d*")
    assert word == "Pocket"  # the lowest feature row
    x, y = Hands._word_center(tip, word)
    assert tip.x + tip.w * 2 / 8 < x <= tip.x + tip.w  # on the name, right of the icon characters
    assert Hands._find_row([Text("PadXYZ", 0, 0, 10, 10)], "Pad\\d*") is None  # whole words only


def test_perceived_tree_puts_back_collapsed_sketches():
    from freecad_s1.ui.perceive import tree_objects, tree_state

    rows = [Text("S1Doc2", 32, 244, 40, 12), Text("- • Body", 13, 260, 50, 12), Text("• & F Origin", 26, 276, 60, 12),
            Text("• & Pad", 26, 292, 40, 12), Text("• & Pad001", 26, 308, 50, 12), Text("& Pocket", 26, 324, 50, 12),
            Text("& Sketch003", 26, 340, 60, 12)]
    tree = tree_state(tree_objects(rows))
    assert [n.type.split("::")[1] for n in tree] == ["Body", "SketchObject", "Pad", "SketchObject", "Pad",
                                                     "SketchObject", "Pocket", "SketchObject"]
    assert [n.num["consumed"] for n in tree if "Sketch" in n.type] == [1.0, 1.0, 1.0, 0.0]
    assert [n.num["tip"] for n in tree].index(1.0) == 6  # the Pocket
    assert tree[-1].num["visible"] == 1.0 and tree[2].num["visible"] == 0.0


def test_perceived_tree_nested_sketch_row():
    from freecad_s1.ui.perceive import tree_objects, tree_state

    rows = [Text("Body", 13, 260, 50, 12), Text("▾ Pad", 26, 292, 40, 12), Text("Sketch", 46, 308, 40, 12)]
    tree = tree_state(tree_objects(rows))
    assert [n.type.split("::")[1] for n in tree] == ["Body", "SketchObject", "Pad"]
