import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/apps": "http://localhost:8000",
      "/collections": "http://localhost:8000",
      "/query": "http://localhost:8000",
      "/generate": "http://localhost:8000",
      "/ingest": "http://localhost:8000",
    },
  },
});
