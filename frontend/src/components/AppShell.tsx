import type { ReactNode } from "react";
import type { Ready } from "../api/types";

function ReadyBadge({ ready }: { ready: Ready | null }) {
  if (!ready) return <span className="badge badge-error" role="status">API unreachable</span>;
  const label = ready.state === "ready"
    ? `Ready · ${ready.model_server.model ?? "model"} (local)`
    : ready.state === "loading" ? "Loading local models…" : "Degraded";
  const tone = ready.state === "ready" ? "badge-ok" : ready.state === "loading" ? "badge-info" : "badge-error";
  const detail = [
    `PostgreSQL: ${ready.postgres.ok ? ready.postgres.version ?? "ok" : "unavailable"}`,
    `Model server: ${ready.model_server.ok ? ready.model_server.version ?? "ok" : "unavailable"}`,
    `Guardrails: ${ready.guardrails.loaded ? "loaded" : "not loaded"}`,
  ].filter(Boolean).join(" · ");
  return <span className={`badge ${tone}`} role="status" title={detail}>{label}</span>;
}

export function AppShell({ ready, children }: { ready: Ready | null; children: ReactNode }) {
  return (
    <div className="shell">
      <header className="shell-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 8l9-5 9 5v8l-9 5-9-5z" /><path d="M3 8l9 5 9-5" /><path d="M12 13v8" />
            </svg>
          </span>
          <div>
            <h1>Local Returns &amp; Exchanges Desk</h1>
            <p className="subtle">Two local AI agents · deterministic business rules · synthetic demo data, runs offline</p>
          </div>
        </div>
        <ReadyBadge ready={ready} />
      </header>
      <main className="shell-main">{children}</main>
    </div>
  );
}
