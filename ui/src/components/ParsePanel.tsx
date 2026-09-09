import { useState } from "react";
import { ApiError, describeDetail, parseProblem, type ParseResponse, type SolveRequest } from "../api";

const MAX_PARSE_TEXT_CHARS = 2000;

interface ParsePanelProps {
  /** Called with the parsed, directly POSTable /solve request. */
  onRunSpec: (spec: SolveRequest) => void;
  /** Called on 503 provider_not_configured so the parent can hide the feature. */
  onUnavailable: () => void;
}

export default function ParsePanel({ onRunSpec, onUnavailable }: ParsePanelProps) {
  const [text, setText] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [parsed, setParsed] = useState<ParseResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const doParse = async () => {
    const trimmed = text.trim();
    if (trimmed.length === 0) {
      setStatus("error");
      setError("Enter a problem description first (1-2000 characters).");
      return;
    }
    setStatus("loading");
    setError(null);
    try {
      const res = await parseProblem(trimmed);
      setParsed(res);
      setStatus("done");
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) {
        onUnavailable();
        return;
      }
      setStatus("error");
      setParsed(null);
      if (err instanceof ApiError) {
        setError(`Parse failed (HTTP ${err.status}): ${describeDetail(err.detail) || err.message}`);
      } else {
        setError("Could not reach the API at /parse. Is the FastAPI backend running?");
      }
    }
  };

  return (
    <section className="panel parse-panel" aria-labelledby="parse-heading">
      <h2 id="parse-heading">Natural-language prompt</h2>
      <p className="truth-note">
        Parsing uses an LLM plus a mechanical verifier; the solver itself never uses one. A parse is
        never trusted blindly - check the verdict below.
      </p>
      <div className="field">
        <label htmlFor="nl-text">Describe the problem (1-{MAX_PARSE_TEXT_CHARS} characters)</label>
        <textarea
          id="nl-text"
          rows={3}
          maxLength={MAX_PARSE_TEXT_CHARS}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder='e.g. "solve a lasso regression with 60 rows, 10 variables, lambda 0.5, seed 7, using fista"'
        />
      </div>
      <button type="button" onClick={() => void doParse()} disabled={status === "loading"}>
        {status === "loading" ? "Parsing..." : "Parse description"}
      </button>

      {error !== null && (
        <p role="alert" className="parse-error">
          {error}
        </p>
      )}

      {parsed !== null && (
        <div className="parse-result">
          <p className="badge-row">
            {parsed.verified ? (
              <span className="badge badge-ok">verified: the text supports this spec</span>
            ) : (
              <span className="badge badge-err">
                NOT verified: the text does not fully support this spec
              </span>
            )}
            <span className="badge badge-neutral">parsed by: {parsed.parse_method === "llm" ? "LLM" : "stub (test provider)"}</span>
          </p>
          {!parsed.verified && parsed.mismatches.length > 0 && (
            <div className="mismatch-box" role="alert">
              <strong>Mismatches:</strong>
              <ul>
                {parsed.mismatches.map((m, i) => (
                  <li key={i}>{m}</li>
                ))}
              </ul>
              <p>Running it is allowed, but the spec may not say what you meant.</p>
            </div>
          )}
          <pre className="spec-preview">{JSON.stringify(parsed.problem_request, null, 2)}</pre>
          <button type="button" onClick={() => onRunSpec(parsed.problem_request)}>
            Run this spec
          </button>
        </div>
      )}
    </section>
  );
}
