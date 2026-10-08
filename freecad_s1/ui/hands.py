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


class ViewImage:
    """A screenshot of the 3D view split into visible face patches: pixels that are neither the
    background nor FreeCAD's dark edge lines, in connected regions (edges separate faces)."""

    def __init__(self, shot: screen.Shot, rect: tuple[float, float, float, float], step: int = 4) -> None:
        import numpy as np

        w, h, _, raw = shot.pixels()
        self.full = np.frombuffer(raw, np.uint8).reshape(h, w, 4)[..., :3]
        img = self.full.astype(np.int16)
        self.shot, self.step, self.s = shot, step, shot.scale
        self.px1 = int((rect[0] + rect[2] - shot.x) * shot.scale)
        self.px0 = max(int((rect[0] - shot.x) * self.s), 0)
        self.py0 = max(int((rect[1] - shot.y) * self.s), 0)
        px1, py1 = int((rect[0] + rect[2] - shot.x) * self.s), int((rect[1] + rect[3] - shot.y) * self.s)
        full = img[self.py0:py1, self.px0:px1]
        Hf, Wf = (full.shape[0] // step) * step, (full.shape[1] // step) * step
        full = full[:Hf, :Wf]
        a = full[::step, ::step]
        bg = np.median(a[:, :3], axis=1, keepdims=True)  # per row, from the view's left edge
        part = np.abs(a - bg).max(axis=2) > 18
        # Edge lines are 1-2 px wide: find them at full resolution; a block with any is an edge block.
        dark_full = full.mean(axis=2) < 110
        dark = dark_full.reshape(Hf // step, step, Wf // step, step).any(axis=(1, 3))
        mask = part & ~dark
        H, W = mask.shape
        mask[: int(H * 0.32), int(W * 0.82):] = False  # navigation cube
        mask[int(H * 0.86):, int(W * 0.9):] = False  # axis cross
        self.mask, self.dark, self.H, self.W = mask, dark & part, H, W
        depth = np.zeros(mask.shape, np.int16)
        cur = mask.copy()
        for _ in range(40):
            nxt = cur.copy()
            nxt[1:] &= cur[:-1]; nxt[:-1] &= cur[1:]; nxt[:, 1:] &= cur[:, :-1]; nxt[:, :-1] &= cur[:, 1:]
            nxt[0] = nxt[-1] = False
            nxt[:, 0] = nxt[:, -1] = False
            if not nxt.any():
                break
            depth += nxt
            cur = nxt
        self.labels = np.zeros(mask.shape, np.int32)
        self.regions: list[dict] = []
        n = 0
        for sy, sx in zip(*np.nonzero(mask)):
            if self.labels[sy, sx]:
                continue
            n += 1
            self.labels[sy, sx] = n
            stack, cells = [(sy, sx)], []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not self.labels[ny, nx]:
                        self.labels[ny, nx] = n
                        stack.append((ny, nx))
            if len(cells) < 25:
                continue
            arr = np.array(cells)
            d = depth[arr[:, 0], arr[:, 1]]
            best = arr[int(d.argmax())]
            deep = arr[d >= d.max() // 2]
            far = deep[int(((deep - best) ** 2).sum(axis=1).argmax())]
            far2 = deep[int(((deep - far) ** 2).sum(axis=1).argmax())]
            self.regions.append({"label": n, "area": len(cells) * (step / self.s) ** 2,
                                 "points": [self.to_point(*best), self.to_point(*far), self.to_point(*far2)]})
        self.regions.sort(key=lambda r: -r["area"])

    def bounds(self) -> tuple[float, float, float, float] | None:
        """The part's extent on screen (points): x0, y0, x1, y1."""
        import numpy as np

        ys, xs = np.nonzero(self.mask | self.dark)
        if len(xs) == 0:
            return None
        (ax, ay), (bx, by) = self.to_point(ys.min(), xs.min()), self.to_point(ys.max(), xs.max())
        return ax, ay, bx, by

    def dark_crossings(self, y: float) -> list[float]:
        """x positions (points) where dark lines (edges) cross the row at height y."""
        import numpy as np

        py = int(round((y - self.shot.y) * self.s))
        if not 0 <= py < self.full.shape[0]:
            return []
        row = self.full[py, self.px0:self.px1].astype(np.int16)
        bg = np.median(row[:6], axis=0)
        dark = (row.mean(axis=1) < 90) & (np.abs(row - bg).max(axis=1) > 18)  # full resolution: lines are thin
        xs, run = [], []
        for px, d in enumerate(dark):
            if d:
                run.append(px)
            elif run:
                xs.append(self.shot.x + (self.px0 + sum(run) / len(run)) / self.s)
                run = []
        return xs

    def to_point(self, cy, cx) -> tuple[float, float]:
        return (self.shot.x + (self.px0 + cx * self.step) / self.s, self.shot.y + (self.py0 + cy * self.step) / self.s)

    def to_cell(self, x, y) -> tuple[int, int]:
        return (int(round(((y - self.shot.y) * self.s - self.py0) / self.step)),
                int(round(((x - self.shot.x) * self.s - self.px0) / self.step)))

    def last_exit(self, label: int, x: float, y: float, dx: float, dy: float) -> float | None:
        """Distance (points) along a ray from (x, y) to the last pixel of region `label`."""
        last, t = None, 0.0
        cell = self.step / self.s
        while True:
            cy, cx = self.to_cell(x + dx * t, y + dy * t)
            if not (0 <= cy < self.H and 0 <= cx < self.W):
                return last
            if self.labels[cy, cx] == label:
                last = t
            t += cell


class Hands:
    def __init__(self, reader: AccessibilityReader, pid: int) -> None:
        self.r = reader
        self.AS = reader.AS
        self.pid = pid
        self.win = screen.Window(pid)
        self.mouse = screen.Mouse()
        self._status = None
        self.hovers = 0  # cursor positions read (for speed reports)
        self._last = None  # (x, y, Hover | None) of the latest hover

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
        """The 3D view, minus a task panel floating over its right side (FreeCAD's overlay)."""
        el = self._walk(lambda e: "Gui::View3DInventor" in str(self.r._attr(e, "AXIdentifier") or "")
                        and str(self.r._attr(e, "AXIdentifier") or "").endswith("QStackedWidget"))
        if el is None:
            raise RuntimeError("3D view not found")
        x, y, w, h = self._frame(el)
        panel = self._walk(lambda e: str(self.r._attr(e, "AXIdentifier") or "").endswith((".OverlayRight", ".Tasks"))
                           and self._frame(e) is not None and self._frame(e)[2] > 0)
        f = self._frame(panel) if panel is not None else None
        if f is not None and f[2] > 0 and x < f[0] < x + w:
            w = f[0] - x
        return x, y, w, h

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
        if view in ("Isometric", "Dimetric", "Trimetric"):
            self.menu("View", "Standard Views", "Axonometric", view)
        else:
            self.menu("View", "Standard Views", view)
        self.press_tool("Std_ViewFitAll")
        self._last = None
        time.sleep(0.25)

    def hover(self, x: float, y: float, wait: float = 0.012) -> Hover | None:
        """What is under the cursor at (x, y). Moving to a new spot over geometry always changes the
        status bar (at least the coordinates); if it doesn't change, the cursor is over nothing (or
        over a panel) and the old message is stale."""
        if self._last is not None and self._last[:2] == (x, y):
            return self._last[2]
        before = self.status_text()
        self.mouse.move(x, y, settle=wait)
        deadline = time.time() + 0.035
        text = self.status_text()
        while text == before and time.time() < deadline:
            time.sleep(0.004)
            text = self.status_text()
        self.hovers += 1
        h = parse_status(text) if text != before else None
        if h is not None:
            h.x, h.y = x, y
        self._last = (x, y, h)
        return h

    def click(self, x: float, y: float, add: bool = False) -> None:
        self.mouse.click(x, y, command=add)  # Command-click adds to FreeCAD's selection on macOS
        self._last = None
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

    def _region_faces(self, img: ViewImage, axis: int, limit: int = 10) -> dict[tuple, dict]:
        """Hover a few interior points of each visible patch; keep patches that are one flat face
        perpendicular to the view. Returns {(obj, sub): {"area", "level", "point", "regions"}}."""
        faces: dict[tuple, dict] = {}
        for reg in img.regions[:limit]:
            hits = [self.hover(x, y) for x, y in reg["points"]]
            if not all(h is not None and h.sub.startswith("Face") for h in hits):
                continue
            if len({(h.obj, h.sub) for h in hits}) != 1 or max(h.point[axis] for h in hits) - min(h.point[axis] for h in hits) > 1e-3:
                continue
            f = faces.setdefault((hits[0].obj, hits[0].sub), {"area": 0.0, "level": hits[0].point[axis],
                                                              "point": hits[0], "regions": []})
            f["area"] += reg["area"]
            f["regions"].append(reg)
        return faces

    def pick_face(self, direction: str) -> None:
        """The largest flat face whose outward normal is `direction`: look at the part from that
        side, find the visible patches in a screenshot, hover a few points of each, and click the
        flat face covering the most of the view."""
        self.look(_VIEW_FOR[direction])
        axis = _AXIS[direction[1]]
        sign = 1 if direction[0] == "+" else -1
        img = ViewImage(self.win.capture(), self.view_rect())
        faces = self._region_faces(img, axis)
        if faces:
            key = max(faces, key=lambda k: (round(faces[k]["area"]), sign * faces[k]["level"]))
            target = faces[key]["point"]
        else:  # nothing recognisable in the picture: hover a grid
            flat: dict[tuple, list[Hover]] = {}
            for n, m in ((14, 10), (28, 20)):
                for hit in self.grid(n, m):
                    if hit.sub.startswith("Face"):
                        flat.setdefault((hit.obj, hit.sub), []).append(hit)
                flat = {k: v for k, v in flat.items()
                        if max(p.point[axis] for p in v) - min(p.point[axis] for p in v) < 1e-3}
                if flat:
                    break
            if not flat:
                raise RuntimeError(f"no flat face seen from the {_VIEW_FOR[direction].lower()}")
            key = max(flat, key=lambda k: (len(flat[k]), sign * flat[k][0].point[axis]))
            pts = flat[key]
            cx, cy = sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts)
            target = min(pts, key=lambda p: (p.x - cx) ** 2 + (p.y - cy) ** 2)
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

    def _expander(self, row: screen.Text) -> tuple[float, float] | None:
        """The expand/collapse triangle of a tree row: the leftmost dark mark on the row's line."""
        import numpy as np

        x, y, w, h = self.tree_rect()
        shot = self.win.capture()
        cy = row.y + row.h / 2
        band = shot.crop(x, cy - 3, min(row.x + row.w, x + w) - x, 6)
        bw, bh, _, raw = band.pixels()
        a = np.frombuffer(raw, np.uint8).reshape(bh, bw, 4)[..., :3].mean(axis=2)
        cols = np.nonzero((a < 120).any(axis=0))[0]
        if len(cols) == 0:
            return None
        run_end = cols[0]
        while run_end + 1 in cols:
            run_end += 1
        return x + (cols[0] + run_end) / 2 / band.scale, cy

    def pick_plane(self, plane: str) -> None:
        """Expand the Body's Origin in the model tree and click the plane."""
        pat = plane + r"[-_ ]?[Pp]lane\d*"  # FreeCAD 1.1 shows "XY-plane"
        rows = self.tree_rows(scroll="top")
        hit = self._find_row(rows, pat)
        if hit is None:
            origin = self._find_row(rows, r"Origin\d*")
            if origin is None:
                raise RuntimeError("Origin not found in the model tree")
            for wait in (0.4, 0.6, 0.8):  # a click right after the app came to the front can be lost
                arrow = self._expander(origin[0])
                if arrow is None:
                    raise RuntimeError("Origin's expand arrow not found")
                self.click(*arrow)  # the triangle left of the row's icons (clicking it doesn't select)
                time.sleep(wait)
                rows = self.tree_rows()
                hit = self._find_row(rows, pat)
                if hit is not None:
                    break
                origin = self._find_row(rows, r"Origin\d*") or origin
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

    def pick_top_edges(self) -> None:
        """The outer edges of the largest upward face: in a top view, march along rays from the
        face's middle to where its patch ends for the last time (the screenshot says where), and
        hover just around there for an edge at the face's height."""
        import math

        self.look("Top")
        img = ViewImage(self.win.capture(), self.view_rect())
        faces = self._region_faces(img, 2)
        if not faces:
            raise RuntimeError("no flat face seen from the top")
        key = max(faces, key=lambda k: (round(faces[k]["area"]), faces[k]["level"]))
        level = faces[key]["level"]
        edges: dict[str, Hover] = {}
        for reg in faces[key]["regions"]:
            cx, cy = reg["points"][0]
            for k in range(36):
                a = 2 * math.pi * (k + 0.5) / 36
                dx, dy = math.cos(a), math.sin(a)
                t = img.last_exit(reg["label"], cx, cy, dx, dy)
                if t is None:
                    continue
                for dt in (1.0, 2.0, 0.0, 3.0, 1.5, 2.5, 4.0, -1.0):
                    e = self.hover(cx + dx * (t + dt), cy + dy * (t + dt))
                    if e is not None and e.sub.startswith("Edge") and abs(e.point[2] - level) < 1e-3:
                        edges.setdefault(e.sub, e)
                        break
        if not edges:
            raise RuntimeError("no outer edges found")
        for i, e in enumerate(edges.values()):
            self.hover(e.x, e.y)
            self.click(e.x, e.y, add=i > 0)

    def pick_vertical_edges(self) -> None:
        """Straight vertical edges: in an isometric wireframe view (hidden edges drawn too), take
        rows across the part, hover where dark lines cross them in a screenshot, and keep edges
        whose points just above and below differ only in height."""
        self.look("Trimetric")  # in isometric, a box's back vertical edge hides behind the front one
        self.menu("View", "Draw Style", "Wireframe")
        try:
            time.sleep(0.3)
            img = ViewImage(self.win.capture(), self.view_rect())
            box = img.bounds()
            if box is None:
                raise RuntimeError("part not visible")
            x0, y0, x1, y1 = box
            found: dict[str, Hover] = {}
            for fy in (0.5, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74, 0.18, 0.82, 0.1, 0.9):  # short edges too
                y = y0 + (y1 - y0) * fy
                for x in img.dark_crossings(y):
                    for dx in (0.0, -1.0, 1.0):
                        hit = self.hover(x + dx, y)
                        if hit is not None and hit.sub.startswith("Edge"):
                            break
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
