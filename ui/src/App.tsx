import { useCallback, useEffect, useState } from "react";
import {
  ApiError,
  describeDetail,
  solveProblem,
  type SolveRequest,
  type SolveResponse,
} from "./api";
import ConvergenceChart from "./components/ConvergenceChart";
import MethodPanel from "./components/MethodPanel";
import ParsePanel from "./components/ParsePanel";
import ResultPanel from "./components/ResultPanel";
import StatusBanner, { type Status } from "./components/StatusBanner";
import { APPLICABLE } from "./methods";
import { EMPTY_FIELDS, randomSeed, validateParams, type ParamFields } from "./params";

type ProblemName = (typeof PROBLEMS)[number];
type MethodName = (typeof METHODS)[number];

const PROBLEMS = ["least_squares", "lasso", "logistic"] as const;
const METHODS = ["gd", "nesterov", "fista", "ista"] as const;
const TAIL_MAX = 20;

export default function App() {
  const [problem, setProblem] = useState<ProblemName>("least_squares");
  const [method, setMethod] = useState<MethodName>("gd");
  const [tail, setTail] = useState(TAIL_MAX);
  const [fields, setFields] = useState<ParamFields>(EMPTY_FIELDS);

  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [clientMs, setClientMs] = useState<number | null>(null);
  const [retrySeconds, setRetrySeconds] = useState(0);

  const [parseHidden, setParseHidden] = useState(false);

  const validated = validateParams(problem, fields);
  const pairInapplicable = !APPLICABLE[problem].includes(method as MethodName);

  // 429 Retry-After countdown.
  useEffect(() => {
    if (retrySeconds <= 0) return;
    const t = setInterval(() => setRetrySeconds((s) => (s <= 1 ? 0 : s - 1)), 1000);
    return () => clearInterval(t);
  }, [retrySeconds]);

  const runSolve = useCallback(async (req: SolveRequest) => {
    setStatus("loading");
    setMessage(null);
    const started = performance.now();
    try {
      const res = await solveProblem(req);
      setResult(res);
      setClientMs(Math.round(performance.now() - started));
      setStatus("success");
      setMessage(res.converged ? "Solve complete." : "Solve returned: the 2000-iteration cap was reached.");
    } catch (err) {
      setResult(null);
      setClientMs(null);
      setStatus("error");
      if (err instanceof ApiError) {
        const detailText = describeDetail(err.detail);
        if (err.status === 429) {
          const secs = err.retryAfter ?? 60;
          setRetrySeconds(secs);
          setMessage(
            `Rate limited (429). The backend allows a fixed number of requests per minute; retrying in ${secs}s.`,
          );
        } else if (err.status === 413) {
          setMessage("Request body too large (413). The backend caps request bodies at 64 KiB.");
        } else if (err.status === 422) {
          setMessage(`The backend rejected the request (422): ${detailText || err.message}`);
        } else {
          setMessage(`The backend returned ${err.status}: ${detailText || err.message}`);
        }
      } else {
        setMessage(err instanceof Error ? err.message : "Unknown failure.");
      }
    }
  }, []);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validated.errors.length > 0 || retrySeconds > 0) return;
    const req: SolveRequest = {
      problem,
      method,
      tail,
      params: validated.params ?? undefined,
    };
    void runSolve(req);
  };

  const runSpec = (spec: SolveRequest) => {
    // Mirror the parsed spec into the manual form so what ran stays visible and editable.
    setProblem(spec.problem);
    setMethod(spec.method);
    setTail(spec.tail);
    const p = spec.params ?? {};
    setFields({
      seed: p.seed != null ? String(p.seed) : "",
      nRows: p.n_rows != null ? String(p.n_rows) : "",
      nVars: p.n_vars != null ? String(p.n_vars) : "",
      lam: p.lam != null ? String(p.lam) : "",
      ridge: p.ridge != null ? String(p.ridge) : "",
      condition: p.condition != null ? String(p.condition) : "",
    });
    void runSolve(spec);
  };

  const setField = (key: keyof ParamFields) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setFields((f) => ({ ...f, [key]: e.target.value }));

  const solveDisabled = status === "loading" || validated.errors.length > 0 || retrySeconds > 0;

  return (
    <main className="workspace">
      <header className="workspace-header">
        <h1>convex_optimization workbench</h1>
        <p className="tagline">
          From-scratch first-order methods vs documented ground truths, over HTTP. Local/offline demo
          only - not a production service.
        </p>
      </header>

      <div className="layout">
        <div className="left-column">
          <form className="panel solve-form" onSubmit={onSubmit} aria-labelledby="config-heading">
            <fieldset>
              <legend id="config-heading">
                <h2>Problem + method</h2>
              </legend>

              <div className="field">
                <label htmlFor="problem">Problem</label>
                <select
                  id="problem"
                  value={problem}
                  onChange={(e) => setProblem(e.target.value as ProblemName)}
                >
                  {PROBLEMS.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label htmlFor="method">Method</label>
                <select
                  id="method"
                  value={method}
                  onChange={(e) => setMethod(e.target.value as MethodName)}
                >
                  {METHODS.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
                <span className="field-note">
                  Applicable for {problem}: {APPLICABLE[problem].join(", ")}.
                  {pairInapplicable && (
                    <strong className="hint-warn"> This pair is rejected by the API (HTTP 422).</strong>
                  )}
                </span>
              </div>

              <div className="field">
                <label htmlFor="tail">
                  History rows returned (tail, 1..{TAIL_MAX})
                </label>
                <input
                  id="tail"
                  type="number"
                  min={1}
                  max={TAIL_MAX}
                  value={tail}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (Number.isInteger(v) && v >= 1 && v <= TAIL_MAX) setTail(v);
                  }}
                />
              </div>
            </fieldset>

            <fieldset>
              <legend>
                <h2>Instance parameters (all optional)</h2>
              </legend>

              <div className="field">
                <label htmlFor="seed">seed (integer 0..2147483647)</label>
                <div className="field-row">
                  <input
                    id="seed"
                    type="text"
                    inputMode="numeric"
                    value={fields.seed}
                    onChange={setField("seed")}
                  />
                  <button
                    type="button"
                    onClick={() => setFields((f) => ({ ...f, seed: String(randomSeed()) }))}
                  >
                    Randomize seed
                  </button>
                </div>
              </div>

              <div className="field">
                <label htmlFor="n-rows">n_rows (integer 4..200)</label>
                <input
                  id="n-rows"
                  type="text"
                  inputMode="numeric"
                  value={fields.nRows}
                  onChange={setField("nRows")}
                />
              </div>

              <div className="field">
                <label htmlFor="n-vars">n_vars (integer 2..200)</label>
                <input
                  id="n-vars"
                  type="text"
                  inputMode="numeric"
                  value={fields.nVars}
                  onChange={setField("nVars")}
                />
              </div>

              <div className="field">
                <label htmlFor="lam">lam (lasso only, &gt;0..100)</label>
                <input
                  id="lam"
                  type="text"
                  inputMode="decimal"
                  value={fields.lam}
                  onChange={setField("lam")}
                  disabled={problem !== "lasso"}
                  placeholder={problem !== "lasso" ? "disabled: lasso only" : ""}
                />
              </div>

              <div className="field">
                <label htmlFor="ridge">ridge (logistic only, &gt;0..100)</label>
                <input
                  id="ridge"
                  type="text"
                  inputMode="decimal"
                  value={fields.ridge}
                  onChange={setField("ridge")}
                  disabled={problem !== "logistic"}
                  placeholder={problem !== "logistic" ? "disabled: logistic only" : ""}
                />
              </div>

              <div className="field">
                <label htmlFor="condition">condition (least_squares only, 1..1000000)</label>
                <input
                  id="condition"
                  type="text"
                  inputMode="decimal"
                  value={fields.condition}
                  onChange={setField("condition")}
                  disabled={problem !== "least_squares"}
                  placeholder={problem !== "least_squares" ? "disabled: least_squares only" : ""}
                />
              </div>

              <p className="frozen-note" data-testid="instance-note">
                {validated.params === null
                  ? "No parameters set: this will solve the FROZEN Stage 1 benchmark instance."
                  : "User-specified instance: only the filled fields are sent; omitted fields use the backend defaults."}
              </p>

              {validated.errors.length > 0 && (
                <p role="alert" className="param-errors">
                  {validated.errors.join("; ")}
                </p>
              )}

              <button type="submit" disabled={solveDisabled}>
                {status === "loading"
                  ? "Solving..."
                  : retrySeconds > 0
                    ? `Rate limited - wait ${retrySeconds}s`
                    : "Solve"}
              </button>

              <StatusBanner status={status} message={message} />
            </fieldset>
          </form>

          {parseHidden ? (
            <section className="panel parse-panel" aria-labelledby="parse-hidden-heading">
              <h2 id="parse-hidden-heading">Natural-language prompt</h2>
              <p className="parse-unavailable">
                Natural-language parsing is unavailable: the backend reports no LLM_API_KEY configured
                (503 provider_not_configured). Use the manual form above - the solver does not need it.
              </p>
            </section>
          ) : (
            <ParsePanel onRunSpec={runSpec} onUnavailable={() => setParseHidden(true)} />
          )}
        </div>

        <div className="results" aria-busy={status === "loading"}>
          {result === null ? (
            <section className="panel empty-panel" aria-labelledby="empty-heading">
              <h2 id="empty-heading">No result yet</h2>
              <p>
                Pick a problem and a method, optionally fill instance parameters, and press Solve.
                Without parameters the frozen Stage 1 benchmark instance is solved. The convergence
                chart, objective values and the SciPy-derived ground truth appear here after a solve.
              </p>
            </section>
          ) : (
            <>
              <ResultPanel result={result} clientMs={clientMs} />
              <section className="panel chart-panel" aria-labelledby="chart-heading">
                <h2 id="chart-heading">Convergence (history tail)</h2>
                <ConvergenceChart result={result} />
              </section>
            </>
          )}
          <MethodPanel />
        </div>
      </div>
    </main>
  );
}
