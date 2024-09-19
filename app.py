from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import pulp
import cvxpy as cp
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class LinearProgramInput(BaseModel):
    objective: str
    constraints: str

class QuadraticProgramInput(BaseModel):
    objective: str
    constraints: str

def parse_expression(expr):
    expr = expr.replace(' ', '')  # Remove all spaces
    terms = re.findall(r'([+-]?(?:\d*\.)?\d*)([a-zA-Z]\w*(?:\^2)?)', expr)
    return [(float(coef) if coef and coef not in ['+', '-'] else (1.0 if coef != '-' else -1.0), var) for coef, var in terms if var]

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/linear_program", response_class=HTMLResponse)
async def linear_program_get(request: Request):
    return templates.TemplateResponse("linear_program.html", {"request": request})

@app.post("/linear_program", response_class=HTMLResponse)
async def linear_program_post(request: Request, objective: str = Form(...), constraints: str = Form(...)):
    try:
        # Create a PuLP problem
        prob = pulp.LpProblem("Linear Program", pulp.LpMinimize)

        # Parse objective function
        obj_terms = parse_expression(objective)
        variables = {var: pulp.LpVariable(var) for _, var in obj_terms}
        
        # Set objective
        prob += pulp.lpSum(coef * variables[var] for coef, var in obj_terms)

        # Parse and add constraints
        for constraint in constraints.split('\n'):
            if constraint.strip():
                if '<=' in constraint:
                    lhs, rhs = constraint.split('<=')
                    lhs_terms = parse_expression(lhs)
                    prob += pulp.lpSum(coef * variables[var] for coef, var in lhs_terms) <= float(rhs.strip())
                elif '>=' in constraint:
                    lhs, rhs = constraint.split('>=')
                    lhs_terms = parse_expression(lhs)
                    prob += pulp.lpSum(coef * variables[var] for coef, var in lhs_terms) >= float(rhs.strip())

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

        # Construct objective function
        objective_expr = sum(coef * variables[var]**2 for var, coef in quad_coeffs.items())
        objective_expr += sum(coef * variables[var] for var, coef in linear_coeffs.items())

        # Parse constraints
        constraints_list = []
        for constraint in constraints.split('\n'):
            if constraint.strip():
                if '<=' in constraint:
                    lhs, rhs = constraint.split('<=')
                    lhs_terms = parse_expression(lhs)
                    lhs_expr = sum(coef * variables[var] for coef, var in lhs_terms)
                    constraints_list.append(lhs_expr <= float(rhs.strip()))
                elif '>=' in constraint:
                    lhs, rhs = constraint.split('>=')
                    lhs_terms = parse_expression(lhs)
                    lhs_expr = sum(coef * variables[var] for coef, var in lhs_terms)
                    constraints_list.append(lhs_expr >= float(rhs.strip()))

        # Define and solve the CVXPY problem
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
