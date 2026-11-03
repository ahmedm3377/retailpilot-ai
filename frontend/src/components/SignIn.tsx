import { useEffect, useState, type ChangeEvent, type FormEvent } from "react";
import { api } from "../api/client";
import type { CaseOption, CaseSelection, UploadedCase } from "../api/types";

const MAIN_STORE = "";
const UPLOAD = "upload";
const MAX_FILE_BYTES = 1_000_000;

/** A file holds one case, or the contract's { cases: [...] }. The server validates the case itself. */
function casesInFile(text: string): UploadedCase[] {
  const parsed: unknown = JSON.parse(text);
  const root = parsed as { cases?: unknown };
  const found = Array.isArray(root?.cases) ? (root.cases as UploadedCase[]) : [parsed as UploadedCase];
  if (!found.length || found.some((c) => !c || typeof c.case_id !== "string" || typeof c.customer_id !== "string")) {
    throw new Error("no case");
  }
  return found;
}

// Demo sign-in: the person enters their customer ID and the conversation starts on their data only.
// The server looks the ID up in the database and binds the session to it; nothing typed in the chat can change it.
// Optionally, the session works on an isolated copy of one case, listed or uploaded; the main store is then untouched.
export function SignIn({ busy, onSignIn }: { busy: boolean; onSignIn: (customerId: string, selection: CaseSelection) => void }) {
  const [customerId, setCustomerId] = useState("");
  const [cases, setCases] = useState<CaseOption[]>([]);
  const [choice, setChoice] = useState(MAIN_STORE);
  const [fileCases, setFileCases] = useState<UploadedCase[]>([]);
  const [fileCaseIndex, setFileCaseIndex] = useState(0);
  const [fileError, setFileError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api.cases().then((listed) => live && setCases(listed)).catch(() => undefined); // the upload and main store still work
    return () => { live = false; };
  }, []);

  const uploaded = choice === UPLOAD ? fileCases[fileCaseIndex] : undefined;
  const canStart = !busy && Boolean(customerId.trim()) && (choice !== UPLOAD || Boolean(uploaded));

  const chooseCase = (event: ChangeEvent<HTMLSelectElement>) => {
    const value = event.target.value;
    setChoice(value);
    const listed = cases.find((c) => c.case_id === value);
    if (listed) setCustomerId(listed.customer_id);
  };

  const pickFileCase = (found: UploadedCase[], index: number) => {
    setFileCases(found);
    setFileCaseIndex(index);
    if (found[index]) setCustomerId(found[index].customer_id);
  };

  const chooseFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setFileError(null);
    pickFileCase([], 0);
    if (!file) return;
    if (file.size > MAX_FILE_BYTES) {
      setFileError("That file is larger than 1 MB.");
      return;
    }
    try {
      pickFileCase(casesInFile(await file.text()), 0);
    } catch {
      setFileError("That file isn't a case: it needs JSON with case_id, customer_id and initial_state.");
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!canStart) return;
    const selection: CaseSelection = choice === MAIN_STORE ? {}
      : choice === UPLOAD ? { uploaded_case: uploaded! } : { scenario_id: choice };
    onSignIn(customerId.trim(), selection);
  };

  return (
    <section className="card sign-in" aria-labelledby="sign-in-heading">
      <h2 id="sign-in-heading">Returns &amp; exchanges</h2>
      <p>Enter your customer ID to start a conversation about your orders.</p>
      <form onSubmit={submit}>
        <div className="row">
          <label htmlFor="customer-id" className="visually-hidden">Customer ID</label>
          <input id="customer-id" value={customerId} maxLength={40} autoComplete="off" placeholder="Customer ID"
                 onChange={(e) => setCustomerId(e.target.value)} disabled={busy} />
          <button type="submit" disabled={!canStart}>Start</button>
        </div>
        <div className="case-choice">
          <label htmlFor="case-select">Case (optional)</label>
          <select id="case-select" value={choice} onChange={chooseCase} disabled={busy}>
            <option value={MAIN_STORE}>None: main store with all data</option>
            {cases.map((c, index) => <option key={c.case_id} value={c.case_id}>{`Case ${index + 1}: ${c.case_id}`}</option>)}
            <option value={UPLOAD}>New empty case (upload a JSON file)</option>
          </select>
          {choice === UPLOAD && (
            <>
              <label htmlFor="case-file">Case file (JSON, same structure as public_cases.json)</label>
              <input id="case-file" type="file" accept=".json,application/json" disabled={busy}
                     onChange={(e) => void chooseFile(e)} />
              {fileError && <p role="alert" className="small">{fileError}</p>}
              {fileCases.length > 1 && (
                <>
                  <label htmlFor="file-case-select">Case in this file</label>
                  <select id="file-case-select" value={fileCaseIndex} disabled={busy}
                          onChange={(e) => pickFileCase(fileCases, Number(e.target.value))}>
                    {fileCases.map((c, index) => <option key={`${c.case_id}-${index}`} value={index}>{c.case_id}</option>)}
                  </select>
                </>
              )}
            </>
          )}
        </div>
      </form>
      <p className="subtle small">Demo sign-in without a password; a real deployment would use the store&apos;s login.</p>
    </section>
  );
}
