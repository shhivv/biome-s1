import React from "react";
import {Composition, Still} from "remotion";
import {AblationChart} from "./AblationChart";
import {Cover, BiomeCover} from "./Cover";
import {MESA_TEASER, MesaTeaser} from "./MesaTeaser";
import {LengthChart} from "./LengthChart";
import {TEASER, Teaser} from "./Teaser";

export const Root: React.FC = () => (
  <>
    <Still id="Cover" component={Cover} width={2400} height={1400} />
    <Still id="BiomeCover" component={BiomeCover} width={2400} height={1400} />
    <Still id="Length" component={LengthChart} width={1200} height={680} defaultProps={{mode: "light" as const}} />
    <Still id="Ablation" component={AblationChart} width={1200} height={680} defaultProps={{mode: "light" as const}} />
    <Composition id="Teaser" component={Teaser} {...TEASER} />
    <Composition id="MesaTeaser" component={MesaTeaser} {...MESA_TEASER} />
  </>
);
