"""FastAPI application setup."""

from __future__ import annotations

# main
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
# main
import pulp
import cvxpy as cp
from parser import parse_polynomial, extract_linear_coeffs, extract_quadratic_terms
import sympy as sp
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from routes import router
from starlette.middleware.sessions import SessionMiddleware

templates = Jinja2Templates(directory="templates")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI()
    application.add_middleware(SessionMiddleware, secret_key="convex-secret")
    application.include_router(router)
    return application


app = create_app()

def parse_expression(expr):
    expr = expr.replace(' ', '')  # Remove all spaces
    terms = re.findall(r'([+-]?(?:\d*\.)?\d*)([a-zA-Z]\w*(?:\^2)?)', expr)
    return [(float(coef) if coef and coef not in ['+', '-'] else (1.0 if coef != '-' else -1.0), var) for coef, var in terms if var]

def evaluate_expression(terms, x, y):
    result = np.zeros_like(x, dtype=float)
    for coef, var in terms:
        if '^2' in var:
            base = var.split('^')[0]
            if base == 'x':
                result += coef * x**2
            elif base == 'y':
                result += coef * y**2
        else:
            if var == 'x':
                result += coef * x
            elif var == 'y':
                result += coef * y
    return result

def gradient(terms, point):
    grad = np.zeros(2)
    x, y = point
    for coef, var in terms:
        if '^2' in var:
            base = var.split('^')[0]
            if base == 'x':
                grad[0] += 2 * coef * x
            elif base == 'y':
                grad[1] += 2 * coef * y
        else:
            if var == 'x':
                grad[0] += coef
            elif var == 'y':
                grad[1] += coef
    return grad

# main
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
# main

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
async def visualize_post(request: Request, objective: str = Form(...), constraints: str = Form(...), animate: str = Form(None)):
    try:
        obj_terms = parse_expression(objective)

        x = np.linspace(-5, 5, 200)
        y = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(x, y)

        Z = evaluate_expression(obj_terms, X, Y)

        mask = np.ones_like(X, dtype=bool)
        for cons in constraints.split('\n'):
            if cons.strip():
                if '<=' in cons:
                    lhs, rhs = cons.split('<=')
                    lhs_terms = parse_expression(lhs)
                    lhs_val = evaluate_expression(lhs_terms, X, Y)
                    mask &= lhs_val <= float(rhs.strip())
                elif '>=' in cons:
                    lhs, rhs = cons.split('>=')
                    lhs_terms = parse_expression(lhs)
                    lhs_val = evaluate_expression(lhs_terms, X, Y)
                    mask &= lhs_val >= float(rhs.strip())

        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.contour(X, Y, Z, levels=20, cmap='viridis')
        ax.contourf(X, Y, mask, levels=[0.5, 1], colors=['#d0f0d0'], alpha=0.3)

        plot_buffer = BytesIO()
        fig.savefig(plot_buffer, format='png')
        plt.close(fig)
        plot_data = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')

        animation_data = None
        if animate:
            point = np.array([4.0, 4.0])
            path = [point.copy()]
            lr = 0.1
            for _ in range(30):
                g = gradient(obj_terms, point)
                point = point - lr * g
                path.append(point.copy())

            fig_anim, ax_anim = plt.subplots()
            ax_anim.contour(X, Y, Z, levels=20, cmap='viridis')
            ax_anim.contourf(X, Y, mask, levels=[0.5, 1], colors=['#d0f0d0'], alpha=0.3)
            path = np.array(path)
            ax_anim.plot(path[:,0], path[:,1], marker='o', color='red')
            anim_buffer = BytesIO()
            fig_anim.savefig(anim_buffer, format='png')
            plt.close(fig_anim)
            animation_data = base64.b64encode(anim_buffer.getvalue()).decode('utf-8')

        result = None
    except Exception as e:
        plot_data = None
        animation_data = None
        result = f"An error occurred: {str(e)}"

    return templates.TemplateResponse(
        "visualize.html",
        {
            "request": request,
            "plot_data": plot_data,
            "animation_data": animation_data,
            "result": result,
        },
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
