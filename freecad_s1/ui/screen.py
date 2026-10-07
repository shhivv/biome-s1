"""Seeing and touching an application like a person does (macOS): window
screenshots, on-device text recognition, and the real mouse.

    win = Window(pid)            # FreeCAD's main window
    img = win.capture()          # works while the app is in the background
    for t in ocr(img): ...       # Text(text, x, y, w, h) in screen points
    mouse = Mouse(); mouse.click(x, y)

Captures and OCR never disturb the user. Mouse moves the real cursor: the
application must be in front, and the person shouldn't be using the computer.
Needs Screen Recording and Accessibility permission for the terminal.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import AppKit
import Quartz as Q


@dataclass
class Text:
    text: str
    x: float  # screen points, top-left origin (like mouse coordinates)
    y: float
    w: float
    h: float
    confidence: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass
class Shot:
    image: object  # CGImage
    x: float  # where the window's top-left corner is on screen (points)
    y: float
    w: float
    h: float

    @property
    def scale(self) -> float:  # pixels per point (2 on Retina)
        return Q.CGImageGetWidth(self.image) / self.w

    def crop(self, x: float, y: float, w: float, h: float) -> "Shot":
        """A region given in screen points."""
        s = self.scale
        rect = Q.CGRectMake((x - self.x) * s, (y - self.y) * s, w * s, h * s)
        return Shot(Q.CGImageCreateWithImageInRect(self.image, rect), x, y, w, h)

    def save(self, path: str) -> None:
        url = AppKit.NSURL.fileURLWithPath_(str(path))
        dest = Q.CGImageDestinationCreateWithURL(url, "public.png", 1, None)
        Q.CGImageDestinationAddImage(dest, self.image, None)
        Q.CGImageDestinationFinalize(dest)

    def pixels(self):
        """(width, height, bytes_per_row, raw RGBA bytes) of the image."""
        w, h = Q.CGImageGetWidth(self.image), Q.CGImageGetHeight(self.image)
        cs = Q.CGColorSpaceCreateDeviceRGB()
        ctx = Q.CGBitmapContextCreate(None, w, h, 8, w * 4, cs, Q.kCGImageAlphaPremultipliedLast)
        Q.CGContextDrawImage(ctx, Q.CGRectMake(0, 0, w, h), self.image)
        data = Q.CGBitmapContextGetData(ctx)
        return w, h, w * 4, bytes(data.as_buffer(w * h * 4))


class Window:
    """The largest normal window of a process."""

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def info(self) -> dict:
        wins = [w for w in Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionAll, Q.kCGNullWindowID)
                if w.get("kCGWindowOwnerPID") == self.pid and w.get("kCGWindowLayer") == 0]
        if not wins:
            raise RuntimeError(f"no window for pid {self.pid}")
        return max(wins, key=lambda w: w["kCGWindowBounds"]["Width"] * w["kCGWindowBounds"]["Height"])

    def capture(self) -> Shot:
        info = self.info()
        b = info["kCGWindowBounds"]
        img = Q.CGWindowListCreateImage(Q.CGRectNull, Q.kCGWindowListOptionIncludingWindow, info["kCGWindowNumber"],
                                        Q.kCGWindowImageBoundsIgnoreFraming | Q.kCGWindowImageBestResolution)
        if img is None:
            raise RuntimeError("window capture failed (Screen Recording permission?)")
        return Shot(img, b["X"], b["Y"], b["Width"], b["Height"])

    def bring_to_front(self) -> None:
        # macOS ignores activation requests from background processes; through accessibility
        # (AXFrontmost on the application, AXRaise on its window) it works.
        import ApplicationServices as AS

        ax = AS.AXUIElementCreateApplication(self.pid)
        AS.AXUIElementSetAttributeValue(ax, "AXFrontmost", True)
        err, wins = AS.AXUIElementCopyAttributeValue(ax, "AXWindows", None)
        if err == 0 and wins:
            AS.AXUIElementPerformAction(wins[0], "AXRaise")
        AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(self.pid).activateWithOptions_(
            AppKit.NSApplicationActivateIgnoringOtherApps)
        deadline = time.time() + 3
        while time.time() < deadline:
            if frontmost_pid() == self.pid:
                return
            time.sleep(0.05)
        raise RuntimeError("could not bring the application to the front")


def frontmost_pid() -> int | None:
    """Owner of the topmost normal window (NSWorkspace's answer goes stale without a run loop)."""
    for w in Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements,
                                          Q.kCGNullWindowID):
        if w.get("kCGWindowLayer") == 0:
            return int(w["kCGWindowOwnerPID"])
    return None


def bring_pid_to_front(pid: int) -> None:
    import ApplicationServices as AS

    AS.AXUIElementSetAttributeValue(AS.AXUIElementCreateApplication(pid), "AXFrontmost", True)


def ocr(shot: Shot, fast: bool = False, words: list[str] | None = None) -> list[Text]:
    """Text in a screenshot, with boxes in screen points (on-device Vision framework)."""
    import Vision

    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast if fast
                             else Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    if words:
        req.setCustomWords_(words)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(shot.image, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(f"text recognition failed: {err}")
    out = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)[0]
        bb = obs.boundingBox()  # normalised, bottom-left origin
        out.append(Text(str(cand.string()),
                        shot.x + bb.origin.x * shot.w,
                        shot.y + (1 - bb.origin.y - bb.size.height) * shot.h,
                        bb.size.width * shot.w, bb.size.height * shot.h, float(cand.confidence())))
    return out


class Mouse:
    """The real cursor (global events). Leaves it where it found it on restore()."""

    def __init__(self) -> None:
        self.home = self.position()

    @staticmethod
    def position() -> tuple[float, float]:
        p = Q.CGEventGetLocation(Q.CGEventCreate(None))
        return p.x, p.y

    @staticmethod
    def _post(kind, x, y, clicks: int = 1, flags: int = 0) -> None:
        ev = Q.CGEventCreateMouseEvent(None, kind, (x, y), Q.kCGMouseButtonLeft)
        if clicks > 1:
            Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventClickState, clicks)
        if flags:
            Q.CGEventSetFlags(ev, flags)
        Q.CGEventPost(Q.kCGHIDEventTap, ev)

    def move(self, x: float, y: float, settle: float = 0.05) -> None:
        self._post(Q.kCGEventMouseMoved, x, y)
        time.sleep(settle)

    def click(self, x: float, y: float, double: bool = False, command: bool = False) -> None:
        flags = Q.kCGEventFlagMaskCommand if command else 0
        self.move(x, y)
        for n in ((1, 2) if double else (1,)):
            self._post(Q.kCGEventLeftMouseDown, x, y, n, flags)
            time.sleep(0.03)
            self._post(Q.kCGEventLeftMouseUp, x, y, n, flags)
            time.sleep(0.05)

    def scroll(self, x: float, y: float, lines: int) -> None:
        """Scroll wheel over (x, y): positive scrolls up (towards the top)."""
        self.move(x, y)
        ev = Q.CGEventCreateScrollWheelEvent(None, Q.kCGScrollEventUnitLine, 1, lines)
        Q.CGEventPost(Q.kCGHIDEventTap, ev)
        time.sleep(0.05)

    def restore(self) -> None:
        self.move(*self.home)


def key(code: int, command: bool = False, shift: bool = False) -> None:
    """A real key press (global; goes to the frontmost application)."""
    flags = (Q.kCGEventFlagMaskCommand if command else 0) | (Q.kCGEventFlagMaskShift if shift else 0)
    for down in (True, False):
        ev = Q.CGEventCreateKeyboardEvent(None, code, down)
        if flags:
            Q.CGEventSetFlags(ev, flags)
        Q.CGEventPost(Q.kCGHIDEventTap, ev)
        time.sleep(0.015)
