import type { TurnDiagnostics } from "../hooks/useDemo";

const show = (v: number | null) => (v === null ? "unavailable" : String(v));

export function DiagnosticsPanel({ turns }: { turns: TurnDiagnostics[] }) {
  return (
    <details className="card diagnostics">
      <summary>Diagnostics (measured per turn)</summary>
      {turns.length === 0 ? <p className="subtle">No turns measured yet.</p> : (
        <table>
          <thead>
            <tr><th scope="col">Turn</th><th scope="col">Time</th><th scope="col">Model calls</th>
              <th scope="col">Prompt tokens</th><th scope="col">Output tokens</th></tr>
          </thead>
          <tbody>
            {turns.map((t) => (
              <tr key={t.turn_number}>
                <td>{t.turn_number}</td><td>{(t.elapsed_ms / 1000).toFixed(1)} s</td><td>{t.llm_calls}</td>
                <td>{show(t.prompt_tokens)}</td><td>{show(t.completion_tokens)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="subtle small">Server-side wall time per turn; tokens as reported by the model engine.</p>
    </details>
  );
}
