/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The only backend: FastAPI on loopback. Never derived from user input or env passed to the browser.
const API_TARGET = "http://127.0.0.1:8080";
// Agent turns take 5-20 s (first one ~30 s); allow generous proxy timeouts.
const proxy = {
  "/api": { target: API_TARGET, changeOrigin: false, timeout: 300_000, proxyTimeout: 300_000 },
  "/health": { target: API_TARGET, changeOrigin: false, timeout: 30_000, proxyTimeout: 30_000 },
};

export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", port: 5173, strictPort: true, proxy },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true, proxy },
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.tsx", "tests/**/*.test.ts"],
    restoreMocks: true,
  },
});
