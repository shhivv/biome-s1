"""Canvas actions with the real mouse: what a person does in FreeCAD's 3D view
and model tree, done from outside the app using only what is on screen.

The session says what to pick ("Face+Z", "Plane:XY", "Tip", "Edges@Face+Z",
"Edges|Z", "Clear") and afterwards checks what got selected; it never tells
the hands where anything is. The hands find it the way a person would:

- 3D view: turn to a standard view, hover, and read FreeCAD's status bar,
  which names the element under the cursor and the 3D point hovered
  ("Preselected: Doc.Body.Pad.Face3 (1.00 mm, 2.00 mm, 8.00 mm)"). The status
  bar is read through accessibility (about 1 ms), not OCR.
- Model tree: read the rows with on-device text recognition (the tree can't
  be read through accessibility without crashing FreeCAD) and click them.

FreeCAD must be in front: the mouse is the real one.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from . import screen
from .ax import SKIP_ROLES, AccessibilityReader

_PRESELECTED = re.compile(r"Preselected:\s*(\S+)\s*\(([^)]*)\)")
_UNITS = {"nm": 1e-6, "µm": 1e-3, "um": 1e-3, "mm": 1.0, "cm": 10.0, "dm": 100.0, "m": 1000.0, "km": 1e6}
_AXIS = {"X": 0, "Y": 1, "Z": 2}
# Standard view that looks at a face whose outward normal is the key.
_VIEW_FOR = {"+Z": "Top", "-Z": "Bottom", "-Y": "Front", "+Y": "Rear", "+X": "Right", "-X": "Left"}
SOLID_PREFIXES = ("Pad", "Pocket", "Hole", "Revolution", "Groove", "Fillet", "Chamfer", "Draft", "Thickness",
                  "Mirrored", "LinearPattern", "PolarPattern", "MultiTransform")


@dataclass
class Hover:
    obj: str  # the feature (e.g. "Pad001")
    sub: str  # "Face3", "Edge7", "Vertex2" or "" for a whole object
    point: tuple[float, float, float]  # the 3D point under the cursor (mm)
    x: float = 0.0  # where the cursor was (screen points)
    y: float = 0.0


def _menu_title(t: str) -> str:
    """'2 Top' -> 'Top', '3 Wireframe (V, 3)' -> 'Wireframe' (menu numbering and shortcuts)."""
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t.replace("&", "")).strip()
    return re.sub(r"^\d+\s+", "", t)


def parse_status(text: str) -> Hover | None:
    m = _PRESELECTED.search(text or "")
    if not m:
        return None
    path = m.group(1).split(".")
    sub = path[-1] if re.match(r"(Face|Edge|Vertex)\d+$", path[-1]) else ""
    obj = path[-2] if sub else path[-1]
    nums = []
    for part in m.group(2).split(","):
        mm = re.match(r"\s*(-?[\d.]+(?:e-?\d+)?)\s*([a-zµ]*)", part.strip())
        if not mm:
            return None
        nums.append(float(mm.group(1)) * _UNITS.get(mm.group(2) or "mm", 1.0))
    if len(nums) != 3:
        return None
    return Hover(obj, sub, tuple(nums))


class Hands:
    def __init__(self, reader: AccessibilityReader, pid: int) -> None:
        self.r = reader
        self.AS = reader.AS
        self.pid = pid
        self.win = screen.Window(pid)
        self.mouse = screen.Mouse()
        self._status = None
        self.hovers = 0  # cursor positions read (for speed reports)

    # -- reading the app -------------------------------------------------------

    def _walk(self, pred, skip_trees: bool = True):
        stack = list(self.r._attr(self.r.app, "AXWindows") or [])
        while stack:
            el = stack.pop()
            if pred(el):
                return el
            role = str(self.r._attr(el, "AXRole") or "")
            ident = str(self.r._attr(el, "AXIdentifier") or "")
            if skip_trees and (role in SKIP_ROLES or "Tree" in ident):
                continue
            stack.extend(self.r._attr(el, "AXChildren") or [])
        return None

    def _frame(self, el) -> tuple[float, float, float, float] | None:
        p, z = self.r._attr(el, "AXPosition"), self.r._attr(el, "AXSize")
        if p is None or z is None:
            return None
        _, p = self.AS.AXValueGetValue(p, self.AS.kAXValueCGPointType, None)
        _, z = self.AS.AXValueGetValue(z, self.AS.kAXValueCGSizeType, None)
        return p.x, p.y, z.width, z.height

    def status_text(self) -> str:
        if self._status is None:
            self._status = self._walk(lambda e: str(self.r._attr(e, "AXIdentifier") or "")
                                      .endswith("statusBar.Gui::StatusBarLabel"))
            if self._status is None:
                raise RuntimeError("FreeCAD's status bar label not found")
        return str(self.r._attr(self._status, "AXValue") or "")

    def view_rect(self) -> tuple[float, float, float, float]:
        el = self._walk(lambda e: "Gui::View3DInventor" in str(self.r._attr(e, "AXIdentifier") or "")
                        and str(self.r._attr(e, "AXIdentifier") or "").endswith("QStackedWidget"))
        if el is None:
            raise RuntimeError("3D view not found")
        return self._frame(el)

    def tree_rect(self) -> tuple[float, float, float, float]:
        # The tree itself is an AXOutline; its frame is safe to read (its rows are not).
        el = self._walk(lambda e: str(self.r._attr(e, "AXRole") or "") == "AXOutline"
                        and "Tree" in str(self.r._attr(e, "AXIdentifier") or ""), skip_trees=False)
        if el is None:
            raise RuntimeError("model tree not found")
        return self._frame(el)

    # -- acting ----------------------------------------------------------------

    def front(self) -> None:
        if screen.frontmost_pid() != self.pid:
            self.win.bring_to_front()
            time.sleep(0.25)

    def menu(self, *path: str) -> None:
        """Pick a menu bar item by its titles, through accessibility (no mouse)."""
        bar = self.r._attr(self.r.app, "AXMenuBar")
        el = bar
        for i, title in enumerate(path):
            kids = self.r._attr(el, "AXChildren") or []
            if i > 0:  # menu bar item -> its menu -> items
                kids = [k for c in kids for k in ((self.r._attr(c, "AXChildren") or []) if
                                                   str(self.r._attr(c, "AXRole")) == "AXMenu" else [c])]
            nxt = next((k for k in kids if _menu_title(str(self.r._attr(k, "AXTitle") or "")) == title), None)
            if nxt is None:
                raise RuntimeError(f"menu item {' > '.join(path[:i + 1])} not found")
            el = nxt
        err = self.AS.AXUIElementPerformAction(el, "AXPress")
        if err != 0:
            raise RuntimeError(f"menu {' > '.join(path)} failed ({err})")
        time.sleep(0.15)

    def press_tool(self, command: str) -> None:
        el = self._walk(lambda e: str(self.r._attr(e, "AXHelp") or "") == command
                        and str(self.r._attr(e, "AXRole")) in ("AXButton", "AXMenuButton"))
        if el is None:
            raise RuntimeError(f"toolbar button {command} not found")
        self.AS.AXUIElementPerformAction(el, "AXPress")
        time.sleep(0.15)

    def look(self, view: str) -> None:
        """Turn the 3D view to a standard view and fit the part."""
        if view == "Isometric":
            self.menu("View", "Standard Views", "Axonometric", "Isometric")
        else:
            self.menu("View", "Standard Views", view)
        self.press_tool("Std_ViewFitAll")
        time.sleep(0.25)

    def hover(self, x: float, y: float, wait: float = 0.012) -> Hover | None:
        before = self.status_text()
        self.mouse.move(x, y, settle=wait)
        deadline = time.time() + 0.035
        text = self.status_text()
        while text == before and time.time() < deadline:
            time.sleep(0.004)
            text = self.status_text()
        self.hovers += 1
        h = parse_status(text)
        if h is not None:
            h.x, h.y = x, y
        return h

    def click(self, x: float, y: float, add: bool = False) -> None:
        self.mouse.click(x, y, command=add)  # Command-click adds to FreeCAD's selection on macOS
        time.sleep(0.15)

    # -- the canvas actions ------------------------------------------------------

    def grid(self, n: int = 14, m: int = 10, margin: float = 0.06) -> list[Hover]:
        x0, y0, w, h = self.view_rect()
        hits = []
        for j in range(m):
            for i in range(n):
                x = x0 + w * (margin + (1 - 2 * margin) * (i + 0.5) / n)
                y = y0 + h * (margin + (1 - 2 * margin) * (j + 0.5) / m)
                hit = self.hover(x, y)
                if hit is not None:
                    hits.append(hit)
        return hits

    def pick_face(self, direction: str) -> None:
        """The largest flat face whose outward normal is `direction`: look at the part from
        that side, hover a grid, keep faces whose hovered points all lie in one plane
        perpendicular to the view, and click the one covering most of the view."""
        self.look(_VIEW_FOR[direction])
        axis = _AXIS[direction[1]]
        sign = 1 if direction[0] == "+" else -1
        faces: dict[tuple, list[Hover]] = {}
        for n, m in ((14, 10), (28, 20)):
            for hit in self.grid(n, m):
                if hit.sub.startswith("Face"):
                    faces.setdefault((hit.obj, hit.sub), []).append(hit)
            flat = {k: v for k, v in faces.items() if max(p.point[axis] for p in v) - min(p.point[axis] for p in v) < 1e-3}
            if flat:
                break
        if not flat:
            raise RuntimeError(f"no flat face seen from the {_VIEW_FOR[direction].lower()}")
        key = max(flat, key=lambda k: (len(flat[k]), sign * flat[k][0].point[axis]))
        pts = flat[key]
        cx, cy = sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts)
        target = min(pts, key=lambda p: (p.x - cx) ** 2 + (p.y - cy) ** 2)  # a hovered point near its middle
        if self.hover(target.x, target.y) is None:
            raise RuntimeError("face moved under the cursor")
        self.click(target.x, target.y)

    def tree_rows(self, scroll: str | None = None) -> list[screen.Text]:
        """Rows of the model tree read from a screenshot; scroll="top"/"bottom" first."""
        x, y, w, h = self.tree_rect()
        if scroll:
            self.mouse.scroll(x + w / 2, y + h / 2, 40 if scroll == "top" else -40)
            time.sleep(0.2)
        shot = self.win.capture().crop(x, y, w, h)
        return [t for t in screen.ocr(shot, words=["XY_Plane", "XZ_Plane", "YZ_Plane", "Origin", "Body"])
                if t.confidence > 0.3]

    @staticmethod
    def _find_row(rows: list[screen.Text], pattern: str) -> tuple[screen.Text, str] | None:
        """The lowest row containing `pattern` as a word (icons are read as stray characters)."""
        rx = re.compile(r"(?:^|[^A-Za-z0-9_])(" + pattern + r")(?![A-Za-z0-9_])")
        hits = [(t, m.group(1)) for t in rows for m in [rx.search(t.text)] if m]
        return max(hits, key=lambda h: h[0].y) if hits else None

    @staticmethod
    def _word_center(t: screen.Text, word: str) -> tuple[float, float]:
        """Click the name, not the row's icons (the eye icon toggles visibility)."""
        i = t.text.rfind(word)
        cw = t.w / max(len(t.text), 1)
        return t.x + cw * (i + len(word) / 2), t.y + t.h / 2

    def pick_plane(self, plane: str) -> None:
        """Expand the Body's Origin in the model tree and click the plane."""
        pat = plane + r"[_ ]?Plane\d*"
        rows = self.tree_rows(scroll="top")
        hit = self._find_row(rows, pat)
        if hit is None:
            origin = self._find_row(rows, r"Origin\d*")
            if origin is None:
                raise RuntimeError("Origin not found in the model tree")
            self.click(*self._word_center(*origin))
            screen.key(124)  # Right arrow: expand the selected tree item
            time.sleep(0.4)
            hit = self._find_row(self.tree_rows(), pat)
            if hit is None:
                raise RuntimeError(f"{plane} plane not found in the model tree")
        self.click(*self._word_center(*hit))

    def pick_tip(self) -> None:
        """The Body's last solid feature: the lowest feature row in the model tree."""
        hit = self._find_row(self.tree_rows(scroll="bottom"), "(?:" + "|".join(SOLID_PREFIXES) + r")\d*")
        if hit is None:
            raise RuntimeError("no feature in the model tree")
        self.click(*self._word_center(*hit))

    def clear(self) -> None:
        """Click empty space in the 3D view."""
        x0, y0, w, h = self.view_rect()
        for fx, fy in ((0.04, 0.5), (0.04, 0.9), (0.5, 0.95), (0.3, 0.06), (0.04, 0.1)):
            x, y = x0 + w * fx, y0 + h * fy
            if self.hover(x, y) is None:
                self.click(x, y)
                return
        raise RuntimeError("no empty spot in the 3D view")

    def _flat_faces(self, hits: list[Hover], axis: int) -> dict[tuple, list[Hover]]:
        faces: dict[tuple, list[Hover]] = {}
        for hit in hits:
            if hit.sub.startswith("Face"):
                faces.setdefault((hit.obj, hit.sub), []).append(hit)
        return {k: v for k, v in faces.items() if max(p.point[axis] for p in v) - min(p.point[axis] for p in v) < 1e-3}

    @staticmethod
    def _bounds(hits: list[Hover], pad: float = 12.0) -> tuple[float, float, float, float]:
        return (min(h.x for h in hits) - pad, min(h.y for h in hits) - pad,
                max(h.x for h in hits) + pad, max(h.y for h in hits) + pad)

    def _edge_inward(self, cx, cy, dx, dy, tmax, level, axis=2):
        """Walk from outside the part (distance tmax from the face's middle) towards the
        middle; at the first point at the face's height, search finely just outside it
        for an edge at that height."""
        t = tmax
        while t > 0:
            hit = self.hover(cx + dx * t, cy + dy * t)
            if hit is not None and abs(hit.point[axis] - level) < 1e-3:
                if hit.sub.startswith("Edge"):
                    return hit
                for k in range(1, 13):  # 0.5 pt steps back outward
                    e = self.hover(cx + dx * (t + 0.5 * k), cy + dy * (t + 0.5 * k))
                    if e is not None and e.sub.startswith("Edge") and abs(e.point[axis] - level) < 1e-3:
                        return e
                return None
            t -= 3.0
        return None

    def pick_top_edges(self) -> None:
        """The outer edges of the largest upward face: look from the top and, along rays
        from outside the part towards the face's middle, take the first edge at the
        face's height."""
        import math

        self.look("Top")
        hits = self.grid(28, 20)
        flat = self._flat_faces(hits, 2)
        if not flat:
            raise RuntimeError("no flat face seen from the top")
        key = max(flat, key=lambda k: (len(flat[k]), flat[k][0].point[2]))
        pts, level = flat[key], flat[key][0].point[2]
        cx, cy = sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts)
        x0, y0, x1, y1 = self._bounds(hits)
        edges: dict[str, Hover] = {}
        for k in range(24):
            a = 2 * math.pi * (k + 0.5) / 24
            dx, dy = math.cos(a), math.sin(a)
            tx = ((x1 - cx) / dx if dx > 0 else (x0 - cx) / dx) if abs(dx) > 1e-9 else 1e9
            ty = ((y1 - cy) / dy if dy > 0 else (y0 - cy) / dy) if abs(dy) > 1e-9 else 1e9
            e = self._edge_inward(cx, cy, dx, dy, min(tx, ty), level)
            if e is not None:
                edges.setdefault(e.sub, e)
        if not edges:
            raise RuntimeError("no outer edges found")
        for i, e in enumerate(edges.values()):
            self.hover(e.x, e.y)
            self.click(e.x, e.y, add=i > 0)

    def pick_vertical_edges(self) -> None:
        """Straight vertical edges: in an isometric wireframe view, sweep across the part at
        mid-height and keep edges whose hovered points straight above and below share x, y."""
        self.look("Isometric")
        bx0, by0, bx1, by1 = self._bounds(self.grid(20, 14), pad=6)  # shaded: faces show the outline
        self.menu("View", "Draw Style", "Wireframe")
        try:
            found: dict[str, Hover] = {}
            for fy in (0.5, 0.65, 0.35):
                y = by0 + (by1 - by0) * fy
                x = bx0
                while x < bx1:
                    hit = self.hover(x, y)
                    x += 1.0
                    if hit is None or not hit.sub.startswith("Edge") or hit.sub in found:
                        continue
                    a, b = self.hover(hit.x, hit.y - 4), self.hover(hit.x, hit.y + 4)
                    if all(p is not None and p.sub == hit.sub and abs(p.point[0] - hit.point[0]) < 1e-3
                           and abs(p.point[1] - hit.point[1]) < 1e-3 and abs(p.point[2] - hit.point[2]) > 1e-3
                           for p in (a, b)):
                        found[hit.sub] = hit
            if not found:
                raise RuntimeError("no vertical edges found")
            for i, e in enumerate(found.values()):
                self.hover(e.x, e.y)
                self.click(e.x, e.y, add=i > 0)
        finally:
            self.menu("View", "Draw Style", "As is")

    def perform(self, target: str) -> None:
        self.front()
        if target == "Clear":
            self.clear()
        elif target.startswith("Plane:"):
            self.pick_plane(target[6:])
        elif target == "Tip":
            self.pick_tip()
        elif target.startswith("Face"):
            self.pick_face(target[4:])
        elif target == "Edges@Face+Z":
            self.pick_top_edges()
        elif target == "Edges|Z":
            self.pick_vertical_edges()
        else:
            raise RuntimeError(f"can't pick {target}")
