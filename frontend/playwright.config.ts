import { defineConfig, devices } from "@playwright/test";

// REAL end-to-end: Chromium -> Vite (5173) -> FastAPI (8080) -> AgentRuntime -> the model named in .env + local
// Guardrails + PostgreSQL + the standalone MCP server (8765), which starts first. Turns take seconds, so timeouts are long.
export default defineConfig({
  testDir: "./e2e",
  timeout: 600_000,
  expect: { timeout: 120_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["json", { outputFile: "test-results/e2e-results.json" }]],
  use: { ...devices["Desktop Chrome"], baseURL: "http://127.0.0.1:5173", trace: "retain-on-failure" },
  webServer: [
    {
      command: "uv run python -m mcp_server",
      cwd: "..",
      url: "http://127.0.0.1:8765/mcp", // answers 401 without the token, which is enough to know it is up
      timeout: 60_000,
      reuseExistingServer: true,
    },
    {
      command: "uv run python -m app.api.serve",
      cwd: "..",
      url: "http://127.0.0.1:8080/health/live",
      timeout: 180_000,
      reuseExistingServer: false,
      stdout: "pipe",
    },
    { command: "npm run dev", url: "http://127.0.0.1:5173", timeout: 60_000, reuseExistingServer: false },
  ],
});
