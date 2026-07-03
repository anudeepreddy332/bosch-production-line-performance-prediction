import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Bundle budget (portfolio_master_plan.md PF4): <=1.5 MB gzipped total. Plotly
// is code-split into its own chunk so Story/Governance (no charts) never pay
// for it -- only Decision Explorer/Model Internals trigger the dynamic import.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes("plotly")) {
            return "plotly";
          }
        },
      },
    },
  },
});
