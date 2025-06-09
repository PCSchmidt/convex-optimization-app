from __future__ import annotations

from typing import Dict, List, Tuple

import cvxpy as cp
import pulp
import numpy as np
import re

from parser import parse_matrix, parse_vector, parse_posynomial


def parse_expression(expr: str) -> List[Tuple[float, str]]:
    """Parse a simple algebraic expression into coefficient/variable terms."""
    expr = expr.replace(" ", "")
    terms = re.findall(r"([+-]?(?:\d*\.)?\d*)([a-zA-Z]\w*(?:\^2)?)", expr)
    return [
        (
            float(coef)
            if coef and coef not in ["+", "-"]
            else (1.0 if coef != "-" else -1.0),
            var,
        )
        for coef, var in terms
        if var
    ]


def solve_lp(objective: str, constraints: str) -> str:
    """Solve a linear program using PuLP.

    Args:
        objective: Objective function as a string, e.g. ``"3x + 2y"``.
        constraints: Newline separated constraint expressions.

    Returns:
        Human readable string describing solver status and variable values.
    """
    prob = pulp.LpProblem("Linear Program", pulp.LpMinimize)

    obj_terms = parse_expression(objective)
    variables = {var: pulp.LpVariable(var) for _, var in obj_terms}

    # Scan constraints for additional variables
    for constraint in constraints.splitlines():
        if "<=" in constraint:
            lhs, _ = constraint.split("<=")
        elif ">=" in constraint:
            lhs, _ = constraint.split(">=")
        else:
            continue
        for _, var in parse_expression(lhs):
            if var not in variables:
                variables[var] = pulp.LpVariable(var)

    prob += pulp.lpSum(coef * variables[var] for coef, var in obj_terms)

    for constraint in constraints.splitlines():
        if not constraint.strip():
            continue
        if "<=" in constraint:
            lhs, rhs = constraint.split("<=")
            lhs_terms = parse_expression(lhs)
            prob += (
                pulp.lpSum(coef * variables[var] for coef, var in lhs_terms)
                <= float(rhs.strip())
            )
        elif ">=" in constraint:
            lhs, rhs = constraint.split(">=")
            lhs_terms = parse_expression(lhs)
            prob += (
                pulp.lpSum(coef * variables[var] for coef, var in lhs_terms)
                >= float(rhs.strip())
            )

    prob.solve()

    status = pulp.LpStatus[prob.status]
    if status == "Optimal":
        result = f"Status: {status}\n"
        for var in prob.variables():
            result += f"{var.name} = {var.varValue}\n"
        result += f"Objective value: {pulp.value(prob.objective)}"
    else:
        result = f"Status: {status}"
    return result


def solve_qp(objective: str, constraints: str) -> str:
    """Solve a quadratic program using CVXPY.

    Args:
        objective: Quadratic objective as a string with ``^2`` for squared terms.
        constraints: Newline separated constraint expressions.

    Returns:
        Human readable string describing solver status and solution values.
    """
    obj_terms = parse_expression(objective)
    variables: Dict[str, cp.Variable] = {}
    quad_coeffs: Dict[str, float] = {}
    linear_coeffs: Dict[str, float] = {}

    for coef, term in obj_terms:
        if "^2" in term:
            var_name = term.split("^")[0]
            if var_name not in variables:
                variables[var_name] = cp.Variable()
            quad_coeffs[var_name] = coef
        else:
            if term not in variables:
                variables[term] = cp.Variable()
            linear_coeffs[term] = coef

    objective_expr = sum(
        coef * variables[var] ** 2 for var, coef in quad_coeffs.items()
    )
    objective_expr += sum(coef * variables[var] for var, coef in linear_coeffs.items())

    constraints_list: List[cp.Constraint] = []
    for constraint in constraints.splitlines():
        if not constraint.strip():
            continue
        if "<=" in constraint:
            lhs, rhs = constraint.split("<=")
            lhs_terms = parse_expression(lhs)
            lhs_expr = sum(coef * variables[var] for coef, var in lhs_terms)
            constraints_list.append(lhs_expr <= float(rhs.strip()))
        elif ">=" in constraint:
            lhs, rhs = constraint.split(">=")
            lhs_terms = parse_expression(lhs)
            lhs_expr = sum(coef * variables[var] for coef, var in lhs_terms)
            constraints_list.append(lhs_expr >= float(rhs.strip()))

    prob = cp.Problem(cp.Minimize(objective_expr), constraints_list)
    prob.solve()

    if prob.status == cp.OPTIMAL:
        result = f"Status: {prob.status}\n"
        for var_name, var in variables.items():
            result += f"{var_name} = {var.value}\n"
        result += f"Objective value: {prob.value}"
    else:
        result = f"Status: {prob.status}"
    return result


def solve_sdp(objective: str, constraints: str) -> str:
    """Solve a small semidefinite program.

    Parameters in ``objective`` and ``constraints`` should be provided as matrix
    strings using comma separated values with ``;`` separating rows.  Each
    constraint line is parsed as ``A <= b`` or ``A >= b`` meaning
    ``trace(A @ X) <= b`` etc.
    """

    C = parse_matrix(objective)
    n = C.shape[0]
    X = cp.Variable((n, n), PSD=True)

    constr: List[cp.Constraint] = []
    for line in constraints.splitlines():
        if not line.strip():
            continue
        if "<=" in line:
            A_str, b_str = line.split("<=")
            A = parse_matrix(A_str)
            constr.append(cp.trace(A @ X) <= float(b_str.strip()))
        elif ">=" in line:
            A_str, b_str = line.split(">=")
            A = parse_matrix(A_str)
            constr.append(cp.trace(A @ X) >= float(b_str.strip()))
        elif "==" in line:
            A_str, b_str = line.split("==")
            A = parse_matrix(A_str)
            constr.append(cp.trace(A @ X) == float(b_str.strip()))

    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constr)
    prob.solve(solver=cp.SCS)

    if prob.status == cp.OPTIMAL:
        result = f"Status: {prob.status}\n"
        result += f"X = {X.value}\n"
        result += f"Objective value: {prob.value}"
    else:
        result = f"Status: {prob.status}"
    return result


def solve_conic(objective: str, constraints: str) -> str:
    """Solve a basic conic program using second-order cone constraints."""

    c = parse_vector(objective)
    n = len(c)
    x = cp.Variable(n)

    constr: List[cp.Constraint] = []
    for line in constraints.splitlines():
        if not line.strip():
            continue
        if line.startswith("soc:"):
            # Format: soc:A|b|c -> norm(A@x + b) <= c
            _, rest = line.split(":", 1)
            A_str, b_str, c_str = rest.split("|")
            A = parse_matrix(A_str)
            b = parse_vector(b_str)
            t = float(c_str)
            constr.append(cp.norm(A @ x + b) <= t)
        elif "<=" in line:
            a_str, b_val = line.split("<=")
            a = parse_vector(a_str)
            constr.append(a @ x <= float(b_val.strip()))
        elif ">=" in line:
            a_str, b_val = line.split(">=")
            a = parse_vector(a_str)
            constr.append(a @ x >= float(b_val.strip()))

    prob = cp.Problem(cp.Minimize(c @ x), constr)
    prob.solve(solver=cp.SCS)

    if prob.status == cp.OPTIMAL:
        result = f"Status: {prob.status}\n"
        result += "\n".join(f"x{i} = {val}" for i, val in enumerate(x.value))
        result += f"\nObjective value: {prob.value}"
    else:
        result = f"Status: {prob.status}"
    return result


def solve_geometric(objective: str, constraints: str) -> str:
    """Solve a geometric program using CVXPY."""

    var_names = set(re.findall(r"[a-zA-Z]\w*", objective))
    for line in constraints.splitlines():
        var_names.update(re.findall(r"[a-zA-Z]\w*", line))

    variables = {name: cp.Variable(pos=True) for name in var_names}

    objective_expr = parse_posynomial(objective, variables)

    constr: List[cp.Constraint] = []
    for line in constraints.splitlines():
        if not line.strip():
            continue
        if "<=" in line:
            lhs, rhs = line.split("<=")
            lhs_expr = parse_posynomial(lhs, variables)
            constr.append(lhs_expr <= float(rhs.strip()))
        elif ">=" in line:
            lhs, rhs = line.split(">=")
            lhs_expr = parse_posynomial(lhs, variables)
            constr.append(lhs_expr >= float(rhs.strip()))
        elif "==" in line:
            lhs, rhs = line.split("==")
            lhs_expr = parse_posynomial(lhs, variables)
            constr.append(lhs_expr == float(rhs.strip()))

    prob = cp.Problem(cp.Minimize(objective_expr), constr)
    prob.solve(gp=True)

    if prob.status == cp.OPTIMAL:
        result = f"Status: {prob.status}\n"
        for name, var in variables.items():
            result += f"{name} = {var.value}\n"
        result += f"Objective value: {prob.value}"
    else:
        result = f"Status: {prob.status}"
    return result
