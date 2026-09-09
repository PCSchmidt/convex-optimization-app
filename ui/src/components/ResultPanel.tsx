import type { SolveResponse } from "../api";

function fmt(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  if (Math.abs(v) < 1e-4 || Math.abs(v) >= 1e7) return v.toExponential(6);
  return v.toPrecision(10).replace(/0+$/, "").replace(/\.$/, "");
}

/**
 * Result readout. converged=false is a REAL, documented outcome (the fixed
 * 2000-iteration cap was hit) and is rendered as a distinct warning badge,
 * not an error.
 */
export default function ResultPanel({ result, clientMs }: { result: SolveResponse; clientMs: number | null }) {
  return (
    <section className="panel result-panel" aria-labelledby="result-heading">
      <h2 id="result-heading">
        Result: {result.problem} + {result.method}
      </h2>
      <p className="badge-row">
        {result.converged ? (
          <span className="badge badge-ok" data-testid="converged-badge">
            converged (tol {fmt(result.tol)} met)
          </span>
        ) : (
          <span className="badge badge-warn" data-testid="converged-badge">
            hit the 2000-iteration cap (not converged)
          </span>
        )}
      </p>
      <dl className="result-grid">
        <div>
          <dt>Iterations</dt>
          <dd>{result.iterations}</dd>
        </div>
        <div>
          <dt>Final objective</dt>
          <dd>{fmt(result.final_objective)}</dd>
        </div>
        <div>
          <dt>Ground-truth objective</dt>
          <dd>{fmt(result.ground_truth_objective)}</dd>
        </div>
        <div>
          <dt>Final objective gap</dt>
          <dd>{fmt(result.final_objective_gap)}</dd>
        </div>
        <div>
          <dt>Final residual</dt>
          <dd>{fmt(result.final_residual)}</dd>
        </div>
        <div>
          <dt>Client round-trip</dt>
          <dd>{clientMs === null ? "n/a" : `${clientMs} ms`}</dd>
        </div>
      </dl>
      <p className="truth-note">
        Ground truth source: {result.ground_truth_source}
      </p>
      <p className="truth-note">
        {result.parameters === null
          ? "Instance: frozen Stage 1 benchmark instance (no parameters in the response)."
          : `Instance parameters: ${Object.entries(result.parameters)
              .map(([k, v]) => `${k}=${v}`)
              .join(", ")}`}
      </p>
    </section>
  );
}
