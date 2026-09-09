import type { Method } from "./api";

/** One compact, honest paragraph per method, written from the Stage 1 docstrings. */
export const METHOD_EXPLANATIONS: Record<Method, string> = {
  gd: "Gradient descent with the fixed 1/L step (L is the objective's Lipschitz constant): the largest step guaranteed to decrease an L-smooth convex objective monotonically, which keeps every run deterministic. For a mu-strongly-convex problem it converges at rate O((1 - 1/kappa)^k) with kappa = L/mu. An Armijo backtracking variant exists in the library, but the serving layer always uses the fixed step.",
  nesterov: "Accelerated gradient with constant momentum gamma = (sqrt(kappa) - 1) / (sqrt(kappa) + 1), computed from the problem's declared strong convexity (kappa = L/mu). It reaches O((1 - 1/sqrt(kappa))^k) instead of gradient descent's O((1 - 1/kappa)^k); this is why the benchmark problems declare their strong convexity - the momentum is a function of the known conditioning, not a tuned hyperparameter.",
  fista: "Accelerated proximal gradient (FISTA, Beck & Teboulle 2009) for min g(x) + h(x) with g L-smooth and h convex but possibly nonsmooth (here: the lasso L1 term). Each step applies prox_{h/L} to a momentum-extrapolated gradient step on g, with the FISTA t-sequence; guaranteed O(1/k^2) rate. The residual is the norm of the scaled prox-gradient map, zero exactly when a subgradient of the full objective vanishes.",
  ista: "The same proximal-gradient scheme as FISTA without the momentum sequence: identical 1/L step, identical prox, identical tolerance and stopping rule. The guaranteed rate is only O(1/k), but the objective decreases monotonically at every step. It exists so the benchmark can compare accelerated vs non-accelerated proximal gradient with everything else held fixed.",
};

/** Applicable (problem, method) pairs, mirroring the CLI/API APPLICABLE matrix. */
export const APPLICABLE: Record<string, Method[]> = {
  least_squares: ["gd", "nesterov"],
  lasso: ["fista", "ista"],
  logistic: ["gd", "nesterov"],
};
