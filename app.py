from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import pulp
import cvxpy as cp
from parser import parse_polynomial, extract_linear_coeffs, extract_quadratic_terms
import sympy as sp
import numpy as np
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class LinearProgramInput(BaseModel):
    objective: str
    constraints: str

class QuadraticProgramInput(BaseModel):
    objective: str
    constraints: str

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/linear_program", response_class=HTMLResponse)
async def linear_program_get(request: Request):
    return templates.TemplateResponse("linear_program.html", {"request": request})

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

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
