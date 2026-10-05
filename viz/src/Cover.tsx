import React from "react";
import {Img, staticFile} from "remotion";
import {fontFamily} from "./Frame";

const PARTS = ["enclosure", "flange", "slotted_wheel", "hex_nut"];
const ACCENT = "#2f7d5b";

export const Cover: React.FC = () => (
  <div style={{width: "100%", height: "100%", position: "relative", overflow: "hidden", fontFamily,
    background: "radial-gradient(1400px 800px at 80% 20%, #e3eee8 0%, #eef2ef 45%, #f7f7f4 100%)"}}>
    {/* faint engineering grid */}
    <div style={{position: "absolute", inset: 0, opacity: 0.35,
      backgroundImage: "linear-gradient(rgba(20,40,30,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(20,40,30,0.06) 1px, transparent 1px)",
      backgroundSize: "60px 60px", maskImage: "linear-gradient(180deg, transparent 0%, #000 35%, #000 70%, transparent 100%)"}} />
    <div style={{position: "absolute", left: 140, top: 130}}>
      <div style={{display: "flex", alignItems: "center", gap: 14, color: ACCENT, fontSize: 26, fontWeight: 600,
        letterSpacing: 3}}>
        <span style={{width: 44, height: 5, borderRadius: 3, background: ACCENT}} />
        SYSTEM-1 MODEL FOR CAD
      </div>
      <div style={{fontSize: 176, fontWeight: 800, color: "#111614", letterSpacing: -5, lineHeight: 1.05, marginTop: 18}}>
        Taiga-S1</div>
      <div style={{fontSize: 44, color: "#4d5a54", marginTop: 14, fontWeight: 500, letterSpacing: -0.5}}>
        A 1.2M-parameter model that builds CAD parts in FreeCAD</div>
      <div style={{display: "flex", gap: 16, marginTop: 38}}>
        {["1.2M parameters", "~1 ms per decision", "No LLM · no screenshots"].map((c) => (
          <div key={c} style={{padding: "12px 22px", borderRadius: 999, border: "1.5px solid rgba(17,22,20,0.12)",
            color: "#2a332f", fontSize: 26, fontWeight: 500, background: "rgba(255,255,255,0.7)"}}>{c}</div>
        ))}
      </div>
    </div>
    <div style={{position: "absolute", left: 70, right: 70, bottom: 80, display: "flex", justifyContent: "space-between",
      alignItems: "flex-end"}}>
      {PARTS.map((p, i) => (
        <Img key={p} src={staticFile(`parts/${p}.png`)}
          style={{width: 520, height: 480, objectFit: "contain",
            transform: `translateY(${i % 2 ? -18 : 0}px)`,
            filter: "drop-shadow(0 34px 38px rgba(20,40,30,0.22)) drop-shadow(0 6px 10px rgba(20,40,30,0.18))"}} />
      ))}
    </div>
  </div>
);

// Repo cover for the Biome-S1 family (Taiga-S1, Mesa-S1).
const MODELS: [string, string][] = [["Taiga-S1", "operates FreeCAD commands"], ["Mesa-S1", "operates FreeCAD's interface"]];

export const BiomeCover: React.FC = () => (
  <div style={{width: "100%", height: "100%", position: "relative", overflow: "hidden", fontFamily,
    background: "radial-gradient(1400px 800px at 80% 20%, #e3eee8 0%, #eef2ef 45%, #f7f7f4 100%)"}}>
    <div style={{position: "absolute", inset: 0, opacity: 0.35,
      backgroundImage: "linear-gradient(rgba(20,40,30,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(20,40,30,0.06) 1px, transparent 1px)",
      backgroundSize: "60px 60px", maskImage: "linear-gradient(180deg, transparent 0%, #000 35%, #000 70%, transparent 100%)"}} />
    <div style={{position: "absolute", left: 140, top: 130}}>
      <div style={{display: "flex", alignItems: "center", gap: 14, color: ACCENT, fontSize: 26, fontWeight: 600,
        letterSpacing: 3}}>
        <span style={{width: 44, height: 5, borderRadius: 3, background: ACCENT}} />
        SYSTEM-1 MODELS FOR COMPUTER USE
      </div>
      <div style={{fontSize: 176, fontWeight: 800, color: "#111614", letterSpacing: -5, lineHeight: 1.05, marginTop: 18}}>
        Biome-S1</div>
      <div style={{fontSize: 44, color: "#4d5a54", marginTop: 14, fontWeight: 500, letterSpacing: -0.5}}>
        Tiny, fast models that build CAD parts in FreeCAD</div>
      <div style={{display: "flex", gap: 16, marginTop: 38}}>
        {MODELS.map(([name, what]) => (
          <div key={name} style={{padding: "12px 22px", borderRadius: 999, border: `1.5px solid ${ACCENT}55`,
            color: "#2a332f", fontSize: 26, fontWeight: 500, background: "rgba(255,255,255,0.8)"}}>
            <span style={{fontWeight: 700, color: "#111614"}}>{name}</span> · {what}</div>
        ))}
        {["1.2M parameters", "~1 ms per decision"].map((c) => (
          <div key={c} style={{padding: "12px 22px", borderRadius: 999, border: "1.5px solid rgba(17,22,20,0.12)",
            color: "#2a332f", fontSize: 26, fontWeight: 500, background: "rgba(255,255,255,0.7)"}}>{c}</div>
        ))}
      </div>
    </div>
    <div style={{position: "absolute", left: 70, right: 70, bottom: 80, display: "flex", justifyContent: "space-between",
      alignItems: "flex-end"}}>
      {PARTS.map((p, i) => (
        <Img key={p} src={staticFile(`parts/${p}.png`)}
          style={{width: 520, height: 480, objectFit: "contain",
            transform: `translateY(${i % 2 ? -18 : 0}px)`,
            filter: "drop-shadow(0 34px 38px rgba(20,40,30,0.22)) drop-shadow(0 6px 10px rgba(20,40,30,0.18))"}} />
      ))}
    </div>
  </div>
);
