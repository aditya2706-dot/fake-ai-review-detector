import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API runs on :8000. Proxying rather than pointing the browser straight at
// it keeps the app same-origin in development, so CORS is one less thing that
// can be misconfigured -- the FastAPI side still sets it, for anyone serving
// the built bundle from elsewhere. Override the target with FRD_API_URL.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.FRD_API_URL ?? "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
