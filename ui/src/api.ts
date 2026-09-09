export type Problem = "least_squares" | "lasso" | "logistic";
export type Method = "gd" | "nesterov" | "fista" | "ista";

export interface HistoryRow {
  iteration: number;
  objective: number;
  residual: number;
}

export interface SolveRequest {
  problem: Problem;
  method: Method;
  tail: number;
  /** Resolved instance parameters; omitted entirely = frozen benchmark instance. */
  params?: Record<string, number> | null;
}

export interface SolveResponse {
  problem: string;
  method: string;
  ground_truth_source: string;
  iterations: number;
  converged: boolean;
  tol: number;
  final_objective: number;
  ground_truth_objective: number;
  final_objective_gap: number;
  final_residual: number;
  history_tail: HistoryRow[];
  parameters: Record<string, number> | null;
}

export interface ParsedParams {
  seed: number | null;
  n_rows: number | null;
  n_vars: number | null;
  lam: number | null;
  ridge: number | null;
  condition: number | null;
}

export interface ParseResponse {
  problem_request: SolveRequest;
  parse_method: "llm" | "stub";
  verified: boolean;
  mismatches: string[];
}

export class ApiError extends Error {
  status: number;
  detail: unknown;
  /** Seconds from the Retry-After header (429 responses only, else null). */
  retryAfter: number | null;
  constructor(status: number, message: string, detail: unknown = null, retryAfter: number | null = null) {
    super(message);
    this.status = status;
    this.detail = detail;
    this.retryAfter = retryAfter;
  }
}

/** Human-readable one-line rendering of the server's `detail` payload. */
export function describeDetail(detail: unknown): string {
  if (detail == null) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return JSON.stringify(detail);
  if (typeof detail === "object") {
    const d = detail as Record<string, unknown>;
    const parts: string[] = [];
    for (const key of ["error", "reason", "hint"]) {
      if (typeof d[key] === "string") parts.push(d[key] as string);
    }
    if (Array.isArray(d.applicable_methods)) {
      parts.push(`applicable methods: ${(d.applicable_methods as string[]).join(", ")}`);
    }
    if (parts.length > 0) return parts.join(" - ");
  }
  return JSON.stringify(detail);
}

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, `Could not reach the API at ${path}. Is the FastAPI backend running (e.g. \`make api\` or uvicorn)?`);
  }
  if (!res.ok) {
    let detail: unknown = null;
    try {
      const body = (await res.json()) as { detail?: unknown };
      detail = body?.detail ?? null;
    } catch {
      // keep the status-line detail
    }
    const retryAfterRaw = res.headers.get("retry-after");
    const retryAfter = retryAfterRaw !== null && /^\d+$/.test(retryAfterRaw) ? Number(retryAfterRaw) : null;
    throw new ApiError(res.status, `The backend returned ${res.status}.`, detail, retryAfter);
  }
  return (await res.json()) as T;
}

/** POST /solve. */
export function solveProblem(req: SolveRequest): Promise<SolveResponse> {
  return requestJson<SolveResponse>("/solve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

/** POST /parse (natural-language problem description -> /solve request). */
export function parseProblem(text: string): Promise<ParseResponse> {
  return requestJson<ParseResponse>("/parse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}
