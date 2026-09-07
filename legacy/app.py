"""FastAPI application setup."""

from __future__ import annotations

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
import pulp
import cvxpy as cp
from parser import parse_polynomial, extract_linear_coeffs, extract_quadratic_terms
import sympy as sp
import numpy as np
import re
import plotly.io as pio
from visualize import (
    plot_linear_program,
    plot_3d_surface,
    gradient_descent_animation,
    feasible_region_animation,
    interior_point_animation,
    simplex_animation,
)

from routes import router
from starlette.middleware.sessions import SessionMiddleware
from solvers import parse_expression

templates = Jinja2Templates(directory="templates")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI()
    application.add_middleware(SessionMiddleware, secret_key="convex-secret")
    application.include_router(router)
    return application


app = create_app()
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/linear_program", response_class=HTMLResponse)
async def linear_program_post(request: Request, objective: str = Form(...), constraints: str = Form(...)):
    try:
        # Collect constraint lines
        constraint_lines = [c.strip() for c in constraints.split("\n") if c.strip()]

        obj_syms, obj_poly = parse_polynomial(objective)
        var_names = {str(s) for s in obj_syms}
        bounds = {}
        parsed_constraints = []

        for line in constraint_lines:
            c = line.replace(" ", "")
            m = re.match(r"^([+-]?\d+(?:\.\d+)?)<=([a-zA-Z]\w*)<=([+-]?\d+(?:\.\d+)?)$", c)
            if m:
                low, var, high = float(m.group(1)), m.group(2), float(m.group(3))
                bounds[var] = [low, high]
                var_names.add(var)
                continue
            m = re.match(r"^([a-zA-Z]\w*)(<=|>=)([+-]?\d+(?:\.\d+)?)$", c)
            if m:
                var, op, val = m.groups()
                val = float(val)
                b = bounds.setdefault(var, [None, None])
                if op == ">=":
                    b[0] = val
                else:
                    b[1] = val
                var_names.add(var)
                continue
            if "==" in c:
                lhs, rhs = c.split("==")
                op = "=="
            elif "<=" in c:
                lhs, rhs = c.split("<=")
                op = "<="
            elif ">=" in c:
                lhs, rhs = c.split(">=")
                op = ">="
            else:
                raise ValueError("Unsupported constraint format: %s" % line)
            parsed_constraints.append((lhs, op, float(sp.sympify(rhs))))
            var_names.update(str(s) for s in parse_polynomial(lhs)[0])

        # Create PuLP variables with bounds
        variables = {}
        for name in var_names:
            lb, ub = bounds.get(name, [None, None])
            variables[name] = pulp.LpVariable(name, lowBound=lb, upBound=ub)

        # Objective
        lin_coeffs = extract_linear_coeffs(obj_poly)
        prob = pulp.LpProblem("Linear Program", pulp.LpMinimize)
        prob += pulp.lpSum(coef * variables[var] for var, coef in lin_coeffs.items())

        # Constraints
        for lhs, op, rhs in parsed_constraints:
            lhs_syms, lhs_poly = parse_polynomial(lhs)
            # Ensure any variables found in this constraint exist
            for sym in lhs_syms:
                name = str(sym)
                if name not in variables:
                    lb, ub = bounds.get(name, [None, None])
                    variables[name] = pulp.LpVariable(name, lowBound=lb, upBound=ub)
            coeffs = extract_linear_coeffs(lhs_poly)
            lhs_expr = pulp.lpSum(coef * variables[var] for var, coef in coeffs.items())
            if op == "<=":
                prob += lhs_expr <= rhs
            elif op == ">=":
                prob += lhs_expr >= rhs
            else:
                prob += lhs_expr == rhs

        # Solve the problem
        prob.solve()

        # Prepare the result
        status = pulp.LpStatus[prob.status]
        if status == "Optimal":
            result = f"Status: {status}\n"
            for var in prob.variables():
                result += f"{var.name} = {var.varValue}\n"
            result += f"Objective value: {pulp.value(prob.objective)}"
        else:
            result = f"Status: {status}"

    except Exception as e:
        result = f"An error occurred: {str(e)}"

    return templates.TemplateResponse("linear_program.html", {"request": request, "result": result})

@app.get("/quadratic_program", response_class=HTMLResponse)
async def quadratic_program_get(request: Request):
    return templates.TemplateResponse("quadratic_program.html", {"request": request})

@app.post("/quadratic_program", response_class=HTMLResponse)
async def quadratic_program_post(request: Request, objective: str = Form(...), constraints: str = Form(...)):
    try:
        # Parse objective function
        obj_terms = parse_expression(objective)
        variables = {}
        quad_coeffs = {}
        linear_coeffs = {}
        cross_terms = []

        for coef, term in obj_terms:
            if '^2' in term:
                var_name = term.split('^')[0]
                if var_name not in variables:
                    variables[var_name] = cp.Variable()
                quad_coeffs[var_name] = quad_coeffs.get(var_name, 0) + coef
            elif len(term) == 2 and term.isalpha():
                var1, var2 = term[0], term[1]
                if var1 not in variables:
                    variables[var1] = cp.Variable()
                if var2 not in variables:
                    variables[var2] = cp.Variable()
                cross_terms.append((coef, var1, var2))
            else:
                if term not in variables:
                    variables[term] = cp.Variable()
                linear_coeffs[term] = linear_coeffs.get(term, 0) + coef

        # Construct objective function
        objective_expr = sum(coef * variables[var]**2 for var, coef in quad_coeffs.items())
        objective_expr += sum(coef * variables[var] for var, coef in linear_coeffs.items())
        objective_expr += sum(coef * variables[v1] * variables[v2] for coef, v1, v2 in cross_terms)
        lines = [c.strip() for c in constraints.split("\n") if c.strip()]

        obj_syms, obj_poly = parse_polynomial(objective)
        var_names = {str(s) for s in obj_syms}
        bounds = {}
        parsed_constraints = []

        for line in lines:
            c = line.replace(" ", "")
            m = re.match(r"^([+-]?\d+(?:\.\d+)?)<=([a-zA-Z]\w*)<=([+-]?\d+(?:\.\d+)?)$", c)
            if m:
                low, var, high = float(m.group(1)), m.group(2), float(m.group(3))
                bounds[var] = [low, high]
                var_names.add(var)
                continue
            m = re.match(r"^([a-zA-Z]\w*)(<=|>=)([+-]?\d+(?:\.\d+)?)$", c)
            if m:
                var, op, val = m.groups()
                val = float(val)
                b = bounds.setdefault(var, [None, None])
                if op == ">=":
                    b[0] = val
                else:
                    b[1] = val
                var_names.add(var)
                continue
            if "==" in c:
                lhs, rhs = c.split("==")
                op = "=="
            elif "<=" in c:
                lhs, rhs = c.split("<=")
                op = "<="
            elif ">=" in c:
                lhs, rhs = c.split(">=")
                op = ">="
            else:
                raise ValueError("Unsupported constraint format: %s" % line)
            parsed_constraints.append((lhs, op, float(sp.sympify(rhs))))
            var_names.update(str(s) for s in parse_polynomial(lhs)[0])

        variables = {name: cp.Variable(name=name) for name in var_names}
        constraints_list = []
        for name, (lb, ub) in bounds.items():
            if lb is not None:
                constraints_list.append(variables[name] >= lb)
            if ub is not None:
                constraints_list.append(variables[name] <= ub)

        quad_terms, lin_terms = extract_quadratic_terms(obj_poly)
        var_order = sorted(var_names)
        idx = {name: i for i, name in enumerate(var_order)}
        Q = np.zeros((len(var_order), len(var_order)))
        for (v1, v2), coef in quad_terms.items():
            i, j = idx[v1], idx[v2]
            if i == j:
                Q[i, j] += coef
            else:
                Q[i, j] += coef / 2
                Q[j, i] += coef / 2
        x_vec = cp.vstack([variables[name] for name in var_order])
        objective_expr = cp.quad_form(x_vec, Q)
        for v, coef in lin_terms.items():
            objective_expr += coef * variables[v]

        for lhs, op, rhs in parsed_constraints:
            lhs_syms, lhs_poly = parse_polynomial(lhs)
            coeffs = extract_linear_coeffs(lhs_poly)
            lhs_expr = sum(coef * variables[var] for var, coef in coeffs.items())
            if op == "<=":
                constraints_list.append(lhs_expr <= rhs)
            elif op == ">=":
                constraints_list.append(lhs_expr >= rhs)
            else:
                constraints_list.append(lhs_expr == rhs)

        prob = cp.Problem(cp.Minimize(objective_expr), constraints_list)
        prob.solve()

        # Prepare the result
        if prob.status == cp.OPTIMAL:
            result = f"Status: {prob.status}\n"
            for var_name, var in variables.items():
                result += f"{var_name} = {var.value}\n"
            result += f"Objective value: {prob.value}"
        else:
            result = f"Status: {prob.status}"

    except Exception as e:
        result = f"An error occurred: {str(e)}"

    return templates.TemplateResponse("quadratic_program.html", {"request": request, "result": result}) 
#  main

@app.get("/visualize", response_class=HTMLResponse)
async def visualize_get(request: Request):
    return templates.TemplateResponse("visualize.html", {"request": request})

@app.post("/visualize", response_class=HTMLResponse)
async def visualize_post(
    request: Request,
    objective: str = Form(...),
    constraints: str = Form(...),
    algorithm: str | None = Form(default=None),
):
    try:
        fig_region = plot_linear_program(objective, constraints)
        plot_html = pio.to_html(fig_region, include_plotlyjs="cdn")

        surface_fig = plot_3d_surface(objective)
        surface_html = pio.to_html(surface_fig, include_plotlyjs=False, full_html=False)

        animation_html = None
        if algorithm == "gradient_descent":
            anim_fig = gradient_descent_animation(objective)
        elif algorithm == "interior_point":
            anim_fig = interior_point_animation(objective, constraints)
        elif algorithm == "simplex":
            anim_fig = simplex_animation(objective, constraints)
        elif algorithm == "feasible_region":
            anim_fig = feasible_region_animation(objective, constraints)
        else:
            anim_fig = None

        if anim_fig is not None:
            animation_html = pio.to_html(anim_fig, include_plotlyjs=False, full_html=False)

        result = None
    except Exception as e:
        plot_html = None
        surface_html = None
        animation_html = None
        result = f"An error occurred: {e}"

    return templates.TemplateResponse(
        "visualize.html",
        {
            "request": request,
            "plot_html": plot_html,
            "surface_html": surface_html,
            "animation_html": animation_html,
            "result": result,
        },
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
