import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
import pulp
import cvxpy as cp
from app import parse_expression


def solve_lp(objective, constraints):
    prob = pulp.LpProblem('LP', pulp.LpMinimize)
    obj_terms = parse_expression(objective)
    variables = {var: pulp.LpVariable(var) for _, var in obj_terms}
    prob += pulp.lpSum(coef * variables[var] for coef, var in obj_terms)

    for line in constraints.strip().split('\n'):
        if not line.strip():
            continue
        if '<=' in line:
            lhs, rhs = line.split('<=')
            lhs_terms = parse_expression(lhs)
            for _, var in lhs_terms:
                if var not in variables:
                    variables[var] = pulp.LpVariable(var)
            prob += pulp.lpSum(coef * variables[var] for coef, var in lhs_terms) <= float(rhs.strip())
        elif '>=' in line:
            lhs, rhs = line.split('>=')
            lhs_terms = parse_expression(lhs)
            for _, var in lhs_terms:
                if var not in variables:
                    variables[var] = pulp.LpVariable(var)
            prob += pulp.lpSum(coef * variables[var] for coef, var in lhs_terms) >= float(rhs.strip())

    prob.solve()
    status = pulp.LpStatus[prob.status]
    solution = {v.name: v.varValue for v in prob.variables()}
    obj_value = pulp.value(prob.objective)
    return status, solution, obj_value


def solve_qp(objective, constraints):
    obj_terms = parse_expression(objective)
    variables = {}
    quad_coeffs = {}
    linear_coeffs = {}

    for coef, term in obj_terms:
        if '^2' in term:
            var_name = term.split('^')[0]
            if var_name not in variables:
                variables[var_name] = cp.Variable()
            quad_coeffs[var_name] = coef
        else:
            if term not in variables:
                variables[term] = cp.Variable()
            linear_coeffs[term] = coef

    objective_expr = sum(coef * variables[var]**2 for var, coef in quad_coeffs.items())
    objective_expr += sum(coef * variables[var] for var, coef in linear_coeffs.items())

    cons = []
    for line in constraints.strip().split('\n'):
        if not line.strip():
            continue
        if '<=' in line:
            lhs, rhs = line.split('<=')
            lhs_terms = parse_expression(lhs)
            expr = sum(coef * variables.setdefault(var, cp.Variable()) for coef, var in lhs_terms)
            cons.append(expr <= float(rhs.strip()))
        elif '>=' in line:
            lhs, rhs = line.split('>=')
            lhs_terms = parse_expression(lhs)
            expr = sum(coef * variables.setdefault(var, cp.Variable()) for coef, var in lhs_terms)
            cons.append(expr >= float(rhs.strip()))

    prob = cp.Problem(cp.Minimize(objective_expr), cons)
    prob.solve()
    return prob.status, {k: variables[k].value for k in variables}, prob.value


def test_solve_basic_lp():
    obj = '3x + 2y'
    cons = 'x + y >= 5\nx >= 0\ny >= 0'
    status, sol, val = solve_lp(obj, cons)
    assert status == 'Optimal'
    assert pytest.approx(sol['x'], abs=1e-4) == 0
    assert pytest.approx(sol['y'], abs=1e-4) == 5
    assert pytest.approx(val, abs=1e-4) == 10


def test_solve_basic_qp():
    obj = 'x^2 + y^2'
    cons = 'x >= 1\ny >= 1'
    status, sol, val = solve_qp(obj, cons)
    assert status == cp.OPTIMAL
    assert pytest.approx(sol['x'], abs=1e-4) == 1
    assert pytest.approx(sol['y'], abs=1e-4) == 1
    assert pytest.approx(val, abs=1e-4) == 2
