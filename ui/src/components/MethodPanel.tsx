import type { Method } from "../api";
import { METHOD_EXPLANATIONS } from "../methods";

const ORDER: Method[] = ["gd", "nesterov", "fista", "ista"];

export default function MethodPanel() {
  return (
    <section className="panel method-panel" aria-labelledby="method-heading">
      <h2 id="method-heading">What each method does</h2>
      {ORDER.map((m) => (
        <p key={m}>
          <strong>{m}</strong> - {METHOD_EXPLANATIONS[m]}
        </p>
      ))}
      <p className="truth-note">
        Applicability (enforced by the API): least_squares and logistic take gd/nesterov; the nonsmooth
        lasso takes fista/ista. All methods run with the same fixed 1/L step, tol=1e-10, max_iter=2000
        and a zero start - nothing is tuned per request.
      </p>
    </section>
  );
}
