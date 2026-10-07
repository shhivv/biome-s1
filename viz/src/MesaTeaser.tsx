import React from "react";
import {AbsoluteFill, Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from "remotion";
import {fontFamily} from "./Frame";
import SPEC from "./mesaFrames.json";

// 10 s teaser: title -> Mesa-S1 operating FreeCAD's interface live (every click highlighted) -> end card.
// Build frames are the real FreeCAD window recorded by scripts/mesa_demo.py, picked per action by
// scripts/mesa_teaser_frames.py (interface clicks get the time, 3D-view steps whizz past).

const TITLE_END = 36;
const BUILD_START = 30;
const BUILD_END = BUILD_START + SPEC.frames.length; // 255
const RESULT_START = BUILD_END;
export const MESA_TEASER = {fps: 30, durationInFrames: RESULT_START + 45, width: 1920, height: 1080};

const ACCENT = "#2a78d6";
const INK = "#000000";
const INK2 = "#555555";
const BG = "#ffffff";

const fade = (f: number, a: number, b: number) =>
  interpolate(f, [a, b], [0, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});

const Title: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = (d: number) => spring({frame: f - d, fps, config: {damping: 200}});
  const out = 1 - fade(f, TITLE_END - 10, TITLE_END);
  return (
    <AbsoluteFill style={{justifyContent: "center", alignItems: "center", opacity: out}}>
      <div style={{textAlign: "center"}}>
        <div style={{opacity: s(0), transform: `translateY(${(1 - s(0)) * 20}px)`, color: INK, fontSize: 26,
          fontWeight: 600, letterSpacing: 4, display: "flex", gap: 14, alignItems: "center", justifyContent: "center"}}>
          <span style={{width: 40, height: 5, borderRadius: 3, background: INK}} />BIOME-S1
        </div>
        <div style={{opacity: s(2), transform: `translateY(${(1 - s(2)) * 30}px)`, fontSize: 168, fontWeight: 800,
          color: INK, letterSpacing: -5, marginTop: 10, lineHeight: 1.1}}>Mesa-S1</div>
        <div style={{opacity: s(5), transform: `translateY(${(1 - s(5)) * 20}px)`, fontSize: 40, color: INK2,
          fontWeight: 500, marginTop: 28}}>A 1.2M-parameter model that operates FreeCAD's interface</div>
      </div>
    </AbsoluteFill>
  );
};

// macOS-style pointer, drawn as a vector so it stays crisp at any zoom; ripple on each click
const Cursor: React.FC<{i: number; tx: number; ty: number; z: number}> = ({i, tx, ty, z}) => {
  const [cx, cy] = SPEC.cursor[i];
  const x = tx + cx * VIEW_W * z;
  const y = ty + cy * VIEW_H * z;
  const click = [...SPEC.clicks].reverse().find((c) => c <= i);
  const since = click === undefined ? 99 : i - click;
  const p = Math.min(1, since / 12);
  const pressScale = since < 4 ? 0.86 + 0.035 * since : 1;
  return (
    <>
      {since < 12 && (
        <div style={{position: "absolute", left: x - (14 + 40 * p), top: y - (14 + 40 * p), width: 2 * (14 + 40 * p),
          height: 2 * (14 + 40 * p), borderRadius: "50%", border: `${4 - 2 * p}px solid ${ACCENT}`,
          background: `rgba(42,120,214,${0.18 * (1 - p)})`, opacity: 1 - p}} />
      )}
      {/* named cursor, multiplayer-style: rounded accent arrow with a white rim, plus a "Mesa-S1" tag */}
      <svg width={44} height={56} viewBox="0 0 28 36"
        style={{position: "absolute", left: x - 7, top: y - 4, transform: `scale(${pressScale})`, transformOrigin: "7px 4px",
          filter: "drop-shadow(0 8px 14px rgba(10,40,90,0.30)) drop-shadow(0 2px 3px rgba(0,0,0,0.25))"}}>
        <path d="M5.5 4.2 C5.5 3.1 6.7 2.5 7.6 3.2 L24.3 17.3 C25.2 18.1 24.7 19.5 23.5 19.6 L16.4 20.2 L20.3 28.9 C20.7 29.8 20.3 30.8 19.4 31.2 L17.2 32.2 C16.3 32.6 15.3 32.2 14.9 31.3 L11.1 22.6 L7.4 27.4 C6.7 28.3 5.5 27.8 5.5 26.7 Z"
          fill={ACCENT} stroke="#fff" strokeWidth={2.2} strokeLinejoin="round" />
      </svg>
      <div style={{position: "absolute", top: y + 34, padding: "6px 14px", borderRadius: 999,
        ...(x > VIEW_W - 200 ? {right: VIEW_W - x + 8} : {left: x + 26}),  // flip the tag near the right edge
        background: ACCENT, color: "#fff", fontSize: 21, fontWeight: 700, letterSpacing: 0.3, whiteSpace: "nowrap",
        transform: `scale(${pressScale})`, transformOrigin: x > VIEW_W - 200 ? "100% 0" : "0 0",
        boxShadow: "0 8px 18px rgba(10,40,90,0.28), 0 0 0 2px #fff"}}>Mesa-S1</div>
    </>
  );
};

const VIEW_W = 1760;
const VIEW_H = 990; // the recorded window is 16:9

const Build: React.FC = () => {
  const f = useCurrentFrame();
  const i = Math.max(0, Math.min(SPEC.frames.length - 1, f - BUILD_START));
  const op = Math.min(fade(f, BUILD_START, BUILD_START + 8), 1 - fade(f, RESULT_START - 4, RESULT_START + 6));
  const element = SPEC.elements[i];
  const actionNo = SPEC.elements.slice(0, i + 1).filter((e, k, a) => k === 0 || a[k - 1] !== e).length;
  // camera: push in on the widget being clicked (precomputed and smoothed in mesa_teaser_frames.py)
  const z = SPEC.zoom[i];
  const [fx, fy] = SPEC.focus[i];
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
  const tx = clamp(VIEW_W / 2 - fx * VIEW_W * z, VIEW_W - VIEW_W * z, 0);
  const ty = clamp(VIEW_H / 2 - fy * VIEW_H * z, VIEW_H - VIEW_H * z, 0);
  return (
    <AbsoluteFill style={{opacity: op, alignItems: "center", justifyContent: "center"}}>
      <div style={{position: "absolute", top: 24, left: 80, right: 80, display: "flex", justifyContent: "space-between",
        alignItems: "center", fontSize: 22, fontWeight: 600, letterSpacing: 3, color: INK2}}>
        <span>LIVE IN FREECAD · EVERY CLICK IS THE MODEL'S</span>
        <span style={{display: "flex", alignItems: "center", gap: 18}}>
          <span style={{padding: "7px 16px", borderRadius: 999, background: INK, color: "#fff", fontSize: 20,
            letterSpacing: 1, fontWeight: 600}}>SLOWED DOWN {SPEC.slowdown}× · REAL RUN {SPEC.run_s} S</span>
          <span style={{fontVariantNumeric: "tabular-nums", letterSpacing: 1}}>
            <span style={{color: ACCENT}}>{element === "Done" ? "Done" : `action ${actionNo}`}</span> / {SPEC.actions}</span>
        </span>
      </div>
      <div style={{width: VIEW_W, height: VIEW_H, marginTop: 44, borderRadius: 14, overflow: "hidden", position: "relative",
        border: "1px solid rgba(0,0,0,0.12)", boxShadow: "0 30px 70px rgba(0,0,0,0.18), 0 8px 18px rgba(0,0,0,0.10)"}}>
        <Img src={staticFile(SPEC.frames[i])}
          style={{position: "absolute", left: 0, top: 0, width: VIEW_W, height: VIEW_H, transformOrigin: "0 0",
            transform: `translate(${tx}px, ${ty}px) scale(${z})`}} />
        <Cursor i={i} tx={tx} ty={ty} z={z} />
        {/* caption: the element the model chose, verbatim (drawn here so the zoom never crops it) */}
        <div style={{position: "absolute", bottom: 30, left: 0, right: 0, display: "flex", justifyContent: "center"}}>
          <div style={{display: "flex", gap: 20, alignItems: "baseline", padding: "16px 30px", borderRadius: 999,
            background: INK, color: "#fff", fontSize: 30, fontWeight: 600, boxShadow: "0 12px 30px rgba(0,0,0,0.25)"}}>
            <span style={{color: "#7fb2f0", letterSpacing: 1}}>MESA-S1</span>
            <span style={{fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontWeight: 500}}>
              {SPEC.captions[i]}</span>
            {SPEC.notes[i] && <span style={{color: "#b9b9b9", fontSize: 24, fontWeight: 500}}>{SPEC.notes[i]}</span>}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Result: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = (d: number) => spring({frame: f - RESULT_START - d, fps, config: {damping: 200}});
  return (
    <AbsoluteFill style={{flexDirection: "row", alignItems: "center", padding: "0 110px", gap: 70,
      opacity: fade(f, RESULT_START, RESULT_START + 6)}}>
      <Img src={staticFile(SPEC.final)}
        style={{width: 900, borderRadius: 14, opacity: s(0), transform: `scale(${0.94 + 0.06 * s(0)})`,
          border: "1px solid rgba(0,0,0,0.12)", boxShadow: "0 30px 60px rgba(0,0,0,0.18)"}} />
      <div>
        <div style={{opacity: s(2), transform: `translateY(${(1 - s(2)) * 24}px)`, fontSize: 130, fontWeight: 800,
          color: INK, letterSpacing: -4}}>Mesa-S1</div>
        <div style={{display: "flex", flexDirection: "column", gap: 12, marginTop: 14}}>
          {[`${SPEC.actions} actions, built exactly`, `Real time ${SPEC.run_s} s · model ${SPEC.model_s} s`, "1.2M parameters", "No LLM · no screenshots"].map((c, k) => (
            <div key={c} style={{opacity: s(5 + 3 * k), transform: `translateX(${(1 - s(5 + 3 * k)) * 20}px)`,
              fontSize: 34, color: INK2, fontWeight: 500, display: "flex", alignItems: "center", gap: 14}}>
              <span style={{width: 10, height: 10, borderRadius: 5, background: ACCENT}} />{c}</div>
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};

export const MesaTeaser: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{background: BG, fontFamily}}>
      {f < TITLE_END && <Title />}
      {f >= BUILD_START - 2 && f < RESULT_START + 8 && <Build />}
      {f >= RESULT_START && <Result />}
    </AbsoluteFill>
  );
};
