import Plotly from "plotly.js-basic-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

// plotly.js-basic-dist-min only bundles scatter/bar/pie traces (no 3d/maps/
// finance) to stay inside the dashboard's 1.5 MB gzipped budget -- see
// portfolio_master_plan.md PF4 ("Plotly via npm, basic/partial bundle").
// This module (and react-plotly.js itself) is only ever reached via a
// React.lazy() page import, so it lives in its own chunk (vite.config.ts).
const Plot = createPlotlyComponent(Plotly);

export default Plot;
