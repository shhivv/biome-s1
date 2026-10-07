# Toward a fast executor for applications without an API

Mesa-S1 decides which interface element to use next. An **executor** turns that
decision into what a person would do: press the button, type the number, click
the face. This note records what was measured on FreeCAD 1.1 (macOS) while
building one from the outside, and what it implies for AutoCAD.

## The three channels

| Channel | How | Speed | Works with the app in the background? |
|---|---|---|---|
| Read the interface | macOS accessibility tree (`freecad_s1/ui/ax.py`) | ~20–25 ms for the whole window | yes |
| Press buttons, type values | accessibility "press" + key events posted to the process | one action ≈ the app's own reaction time | yes |
| Read what's under the cursor | the app's status bar through accessibility (`hands.py`) | ~1 ms per read | — |
| Screenshot | window capture (`screen.py`) | 35–75 ms | yes |
| Read text in a screenshot | on-device OCR (Vision) | 250–340 ms (accurate mode) | yes |
| Click in the canvas | the real mouse (`screen.Mouse`) | one hover ≈ 35 ms incl. readback | **no**: the app must be in front |

## What was measured

- **Reading through accessibility** gives Mesa-S1 the same view as reading Qt from
  inside: it decided identically on 535 of 536 steps, and built 15/15 parts.
- **Acting through accessibility** (toolbar buttons, number fields, check boxes,
  OK/Cancel): 174 actions, none failed; 14/15 parts (one FreeCAD crash). FreeCAD
  stays in the background and the user's mouse and keyboard are untouched.
- Things that needed care:
  - Typed lengths only take effect when editing finishes. In the background there
    is no focus change, and Return would press OK, so an up/down step commits them.
    The session checks the value the field will actually apply, not its text.
  - Dropdown popups ignore posted keys while the app is in the background;
    dropdowns stay with the session for now.
  - FreeCAD's Document Recovery dialog after a killed run blocks everything.
  - After the app has been in front once, its populated menu bar makes a naive
    accessibility walk loop back to the application.
- **Clicks sent to a background app never reach the 3D view.** In front with the
  real mouse they do. Hovering makes FreeCAD print the element and the 3D point
  under the cursor (`Preselected: Doc.Body.Pad.Face3 (-1.00 mm, -0.37 mm, 8.00 mm)`).
  That readback is the key to finding things without an API: picking "the largest
  upward face" = look from the top, hover a grid, keep faces whose points share
  one height, click the biggest. It got 2/2 face picks right, at ~13 s each
  (141 hovers): correct, but slow.
- **The model tree** can't be read through accessibility (reading its rows crashes
  Qt later) but reads fine with OCR from a background screenshot (~0.3 s).
- **A locked screen stops everything:** the app's windows disappear from the
  accessibility tree and nothing can come to the front. Unattended runs need an
  unlocked session (e.g. a dedicated user or VM).

## Still inside FreeCAD: sketches

Drawing sketch geometry and its constraints is still done by the session. The
planned path is FreeCAD's on-view parameters (1.0+): with a sketch tool active,
typed values (`x`, Tab, `y`, Enter, then the radius) place the geometry exactly,
and FreeCAD adds the matching constraints itself. Typing works with the app in
the background. Two open questions:
- Does typing reach the on-view boxes without the cursor over the view?
- Those constraints (e.g. radius plus position) differ from the ones the session
  books (lock plus diameter). The session would check the geometry instead of
  constraint kinds.

## What this means for AutoCAD

1. **Most of AutoCAD is keystrokes.** Its command line takes commands and
   coordinates (`CIRCLE`, `10,0`, `3`), so creating geometry is typing, the
   channel that already works in the background and is fast. FreeCAD's sketcher
   needs canvas clicks for the same thing.
2. **The command line is the feedback channel.** The status bar played this role
   for FreeCAD. The prompt and its options (`Specify center point or [3P/2P/Ttr]:`)
   are exactly an element list for an S1 model: the options are the candidates,
   and the numbers come from the parameter stage, as in Mesa-S1.
3. **The canvas is still opaque.** It's a GPU view, like FreeCAD's 3D view.
   Selections that can't be typed (window/crossing selection takes typed corners)
   need the front-app, real-mouse path plus a readback. That makes them the slow
   path, so the design should keep picks rare.
4. **Speed budget per action:** model ~1 ms, interface read ~20 ms, act =
   the app's reaction. Hover-based picks are the outlier (seconds). Faster ways:
   a coarse-to-fine search seeded by the screenshot's silhouette, and the app's own
   snaps or typed selection.

## Next experiments (AutoCAD for Mac)

1. Accessibility audit: ribbon/toolbar buttons, command line text, dialogs.
   How much of AutoCAD's interface is named?
2. Posted keystrokes into the command line with AutoCAD in the background, and
   reading the prompt back after each one.
3. The prompt-to-elements parser (command, options in brackets, value slots) and a
   teacher that drives AutoCAD (script files / AutoLISP if available) to generate
   training sessions. DXF export plus `ezdxf` to score the result.
4. A pick readback for the canvas (selection preview / properties palette) for the
   operations that need clicks.
