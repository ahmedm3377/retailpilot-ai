import type { Notice } from "../hooks/useDemo";

export function ErrorNotice({ notice, canRetry, onRetry, onDismiss }: {
  notice: Notice;
  canRetry: boolean;
  onRetry: () => void;
  onDismiss: () => void;
}) {
  return (
    <div className={`notice notice-${notice.kind}`} role={notice.kind === "error" ? "alert" : "status"}
         data-testid="notice" data-code={notice.code}>
      <p>{notice.text}</p>
      <div className="row">
        {canRetry && <button type="button" onClick={onRetry}>Retry the same action</button>}
        <button type="button" className="secondary" onClick={onDismiss}>Dismiss</button>
      </div>
    </div>
  );
}
