from __future__ import annotations

from fastapi import APIRouter, Form, Request, Query
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from solvers import solve_lp, solve_qp, solve_sdp, solve_conic, solve_geometric
import plotly.io as pio
import os
import json
import yaml
from visualize import (
    gradient_descent_animation,
    plot_linear_program,
    plot_quadratic_program,
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")


class ProgramRequest(BaseModel):
    objective: str
    constraints: str
    method: str | None = None
    algorithm: str | None = None
    max_iter: int | None = None
    tolerance: float | None = None


def validate_options(
    algorithm: str | None,
    valid: set[str],
    max_iter: int | None,
    tolerance: float | None,
) -> None:
    """Validate solver options from a request."""
    if algorithm and algorithm.upper() not in valid:
        raise ValueError(f"Unsupported method '{algorithm}'")
    if max_iter is not None and max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tolerance is not None and tolerance <= 0:
        raise ValueError("tolerance must be positive")

# Prefilled example problems used in the tutorial steps
TUTORIAL_EXERCISES: dict[int, dict[str, str]] = {
    1: {
        "type": "linear_program",
        "objective": "3x + 2y",
        "constraints": "x + y <= 4\nx >= 0\ny >= 0",
        "explanation": "This step introduces linear programming, where both the objective and constraints are linear.",
    },
    2: {
        "type": "quadratic_program",
        "objective": "x^2 + y^2 + x",
        "constraints": "x + y >= 1\nx >= 0\ny >= 0",
        "explanation": "Here we solve a quadratic program with a convex quadratic objective.",
    },
    3: {
        "type": "semidefinite_program",
        "objective": "1,0;0,1",
        "constraints": "1,0;0,1 >= 1",
        "explanation": "Semidefinite programming optimizes over positive semidefinite matrices.",
    },
    4: {
        "type": "conic_program",
        "objective": "1,1",
        "constraints": "soc:1,0;0,1|0,0|1",
        "explanation": "Conic programming allows cone constraints such as second-order cones.",
    },
}


def load_problems() -> list[dict]:
    """Load example problems from YAML/JSON files under ``problems``."""
    problems: list[dict] = []
    if not os.path.isdir("problems"):
        return problems

    for root, _dirs, files in os.walk("problems"):
        for fname in sorted(files):
            path = os.path.join(root, fname)
            if fname.endswith((".yaml", ".yml")):
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            elif fname.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                continue

            if isinstance(data, list):
                problems.extend(data)
            else:
                problems.append(data)

    return problems


def load_quiz_questions() -> list[dict]:
    """Load tutorial quiz questions from ``tutorial/quiz.yaml``."""
    path = os.path.join("tutorial", "quiz.yaml")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("questions", []) if isinstance(data, dict) else []


@router.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """Render the homepage with links to optimization problems."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/problems", response_class=HTMLResponse)
async def problem_library(
    request: Request, problem: str | None = Query(default=None)
) -> HTMLResponse:
    """Display available example problems and selected problem details."""
    problems = load_problems()
    selected = None
    if problem:
        for p in problems:
            if p.get("name") == problem:
                selected = p
                break
    return templates.TemplateResponse(
        "problem_library.html",
        {"request": request, "problems": problems, "selected": selected},
    )


@router.get("/tutorial/step/{step}", response_class=HTMLResponse)
async def tutorial_step_get(request: Request, step: int) -> HTMLResponse:
    """Render a tutorial step with a prefilled example problem."""
    exercise = TUTORIAL_EXERCISES.get(step)
    if not exercise:
        return HTMLResponse("Step not found", status_code=404)
    return templates.TemplateResponse(
        f"tutorial/step{step}.html",
        {
            "request": request,
            "objective": exercise["objective"],
            "constraints": exercise["constraints"],
            "progress": step,
            "total_steps": len(TUTORIAL_EXERCISES),
            "explanation": exercise.get("explanation"),
        },
    )


@router.post("/tutorial/step/{step}", response_class=HTMLResponse)
async def tutorial_step_post(
    request: Request,
    step: int,
    objective: str = Form(...),
    constraints: str = Form(...),
) -> HTMLResponse:
    """Solve the exercise for the given tutorial step."""
    exercise = TUTORIAL_EXERCISES.get(step)
    if not exercise:
        return HTMLResponse("Step not found", status_code=404)
    try:
        if exercise["type"] == "linear_program":
            result = solve_lp(objective, constraints)
        elif exercise["type"] == "quadratic_program":
            result = solve_qp(objective, constraints)
        elif exercise["type"] == "semidefinite_program":
            result = solve_sdp(objective, constraints)
        elif exercise["type"] == "conic_program":
            result = solve_conic(objective, constraints)
        else:
            result = "Unsupported exercise"
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
    current = request.session.get("tutorial_progress", 0)
    if step > current:
        request.session["tutorial_progress"] = step
    return templates.TemplateResponse(
        f"tutorial/step{step}.html",
        {
            "request": request,
            "objective": objective,
            "constraints": constraints,
            "result": result,
            "progress": step,
            "total_steps": len(TUTORIAL_EXERCISES),
            "explanation": exercise.get("explanation"),
        },
    )


@router.get("/tutorial/quiz", response_class=HTMLResponse)
async def tutorial_quiz_get(request: Request, q: int = Query(-1)) -> HTMLResponse:
    """Display a quiz question from the tutorial."""
    questions = load_quiz_questions()
    progress = int(request.session.get("quiz_progress", 0))
    if q == -1:
        q = progress
    if q < 0 or q >= len(questions):
        return HTMLResponse("Question not found", status_code=404)
    question = questions[q]
    return templates.TemplateResponse(
        "tutorial/quiz.html",
        {
            "request": request,
            "question": question,
            "index": q,
            "total": len(questions),
        },
    )


@router.post("/tutorial/quiz", response_class=HTMLResponse)
async def tutorial_quiz_post(
    request: Request, q: int = Form(...), answer: str = Form(...)
) -> HTMLResponse:
    """Check the submitted answer and show whether it is correct."""
    questions = load_quiz_questions()
    if q < 0 or q >= len(questions):
        return HTMLResponse("Question not found", status_code=404)
    question = questions[q]
    correct = answer.strip() == question.get("answer")
    progress = int(request.session.get("quiz_progress", 0))
    if correct and q == progress:
        request.session["quiz_progress"] = progress + 1
    next_index = q + 1 if q + 1 < len(questions) else None
    return templates.TemplateResponse(
        "tutorial/quiz.html",
        {
            "request": request,
            "question": question,
            "index": q,
            "answered": True,
            "correct": correct,
            "given": answer,
            "next_index": next_index,
            "total": len(questions),
        },
    )


@router.get("/linear_program", response_class=HTMLResponse)
async def linear_program_get(
    request: Request,
    objective: str | None = Query(default=None),
    constraints: str | None = Query(default=None),
    method: str | None = Query(default=None),
    algorithm: str | None = Query(default=None),
    max_iter: int | None = Query(default=None),
    tolerance: float | None = Query(default=None),
) -> HTMLResponse:
    """Display the linear programming input form."""
    return templates.TemplateResponse(
        "linear_program.html",
        {
            "request": request,
            "objective": objective,
            "constraints": constraints,
            "method": method,
            "algorithm": algorithm,
            "max_iter": max_iter,
            "tolerance": tolerance,
        },
    )


@router.post("/linear_program", response_class=HTMLResponse)
async def linear_program_post(
    request: Request,
    objective: str = Form(...),
    constraints: str = Form(...),
    method: str | None = Form(default=None),
    algorithm: str | None = Form(default=None),
    max_iter: int | None = Form(default=None),
    tolerance: float | None = Form(default=None),
) -> HTMLResponse:
    """Solve the provided linear program and return the result."""
    try:
        validate_options(algorithm, {"cbc", "glpk"}, max_iter, tolerance)
        result = solve_lp(
            objective,
            constraints,
            method=method,
            algorithm=algorithm,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        fig = plot_linear_program(objective, constraints)
        plot_html = pio.to_html(fig, include_plotlyjs="cdn")
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
        plot_html = None
    return templates.TemplateResponse(
        "linear_program.html",
        {
            "request": request,
            "result": result,
            "plot_html": plot_html,
            "method": method,
            "algorithm": algorithm,
            "max_iter": max_iter,
            "tolerance": tolerance,
        },
    )


@router.post("/api/linear_program")
async def api_linear_program(req: ProgramRequest) -> dict:
    """Solve a linear program and return JSON results."""
    try:
        validate_options(req.algorithm, {"cbc", "glpk"}, req.max_iter, req.tolerance)
        result = solve_lp(
            req.objective,
            req.constraints,
            method=req.method,
            algorithm=req.algorithm,
            max_iter=req.max_iter,
            tolerance=req.tolerance,
        )
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
        status = "error"
    return {"status": status, "result": result}


@router.get("/quadratic_program", response_class=HTMLResponse)
async def quadratic_program_get(
    request: Request,
    objective: str | None = Query(default=None),
    constraints: str | None = Query(default=None),
    method: str | None = Query(default=None),
    algorithm: str | None = Query(default=None),
    max_iter: int | None = Query(default=None),
    tolerance: float | None = Query(default=None),
) -> HTMLResponse:
    """Display the quadratic programming input form."""
    return templates.TemplateResponse(
        "quadratic_program.html",
        {
            "request": request,
            "objective": objective,
            "constraints": constraints,
            "method": method,
            "algorithm": algorithm,
            "max_iter": max_iter,
            "tolerance": tolerance,
        },
    )


@router.post("/quadratic_program", response_class=HTMLResponse)
async def quadratic_program_post(
    request: Request,
    objective: str = Form(...),
    constraints: str = Form(...),
    method: str | None = Form(default=None),
    algorithm: str | None = Form(default=None),
    max_iter: int | None = Form(default=None),
    tolerance: float | None = Form(default=None),
) -> HTMLResponse:
    """Solve the provided quadratic program and return the result."""
    try:
        validate_options(algorithm, {"ECOS", "OSQP", "SCS"}, max_iter, tolerance)
        result = solve_qp(
            objective,
            constraints,
            method=method,
            algorithm=algorithm,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        fig = plot_quadratic_program(objective)
        plot_html = pio.to_html(fig, include_plotlyjs="cdn")
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
        plot_html = None
    return templates.TemplateResponse(
        "quadratic_program.html",
        {
            "request": request,
            "result": result,
            "plot_html": plot_html,
            "method": method,
            "algorithm": algorithm,
            "max_iter": max_iter,
            "tolerance": tolerance,
        },
    )


@router.post("/api/quadratic_program")
async def api_quadratic_program(req: ProgramRequest) -> dict:
    """Solve a quadratic program and return JSON results."""
    try:
        validate_options(req.algorithm, {"ECOS", "OSQP", "SCS"}, req.max_iter, req.tolerance)
        result = solve_qp(
            req.objective,
            req.constraints,
            method=req.method,
            algorithm=req.algorithm,
            max_iter=req.max_iter,
            tolerance=req.tolerance,
        )
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
        status = "error"
    return {"status": status, "result": result}


@router.get("/semidefinite_program", response_class=HTMLResponse)
async def sdp_get(
    request: Request,
    objective: str | None = Query(default=None),
    constraints: str | None = Query(default=None),
    method: str | None = Query(default=None),
    algorithm: str | None = Query(default=None),
    max_iter: int | None = Query(default=None),
    tolerance: float | None = Query(default=None),
) -> HTMLResponse:
    """Display the semidefinite programming input form."""
    return templates.TemplateResponse(
        "semidefinite_program.html",
        {
            "request": request,
            "objective": objective,
            "constraints": constraints,
            "method": method,
            "algorithm": algorithm,
            "max_iter": max_iter,
            "tolerance": tolerance,
        },
    )


@router.post("/semidefinite_program", response_class=HTMLResponse)
async def sdp_post(
    request: Request,
    objective: str = Form(...),
    constraints: str = Form(...),
    method: str | None = Form(default=None),
    algorithm: str | None = Form(default=None),
    max_iter: int | None = Form(default=None),
    tolerance: float | None = Form(default=None),
) -> HTMLResponse:
    """Solve the provided semidefinite program and return the result."""
    try:
        validate_options(algorithm, {"SCS"}, max_iter, tolerance)
        result = solve_sdp(
            objective,
            constraints,
            method=method,
            algorithm=algorithm,
            max_iter=max_iter,
            tolerance=tolerance,
        )
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
    return templates.TemplateResponse(
        "semidefinite_program.html",
        {
            "request": request,
            "result": result,
            "method": method,
            "algorithm": algorithm,
            "max_iter": max_iter,
            "tolerance": tolerance,
        },
    )


@router.get("/conic_program", response_class=HTMLResponse)
async def conic_get(
    request: Request,
    objective: str | None = Query(default=None),
    constraints: str | None = Query(default=None),
    method: str | None = Query(default=None),
    algorithm: str | None = Query(default=None),
    max_iter: int | None = Query(default=None),
    tolerance: float | None = Query(default=None),
) -> HTMLResponse:
    """Display the conic programming input form."""
    return templates.TemplateResponse(
        "conic_program.html",
        {
            "request": request,
            "objective": objective,
            "constraints": constraints,
            "method": method,
            "algorithm": algorithm,
            "max_iter": max_iter,
            "tolerance": tolerance,
        },
    )


@router.post("/conic_program", response_class=HTMLResponse)
async def conic_post(
    request: Request,
    objective: str = Form(...),
    constraints: str = Form(...),
    method: str | None = Form(default=None),
    algorithm: str | None = Form(default=None),
    max_iter: int | None = Form(default=None),
    tolerance: float | None = Form(default=None),
) -> HTMLResponse:
    """Solve the provided conic program and return the result."""
    try:
        validate_options(algorithm, {"SCS"}, max_iter, tolerance)
        result = solve_conic(
            objective,
            constraints,
            method=method,
            algorithm=algorithm,
            max_iter=max_iter,
            tolerance=tolerance,
        )
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
    return templates.TemplateResponse(
        "conic_program.html",
        {
            "request": request,
            "result": result,
            "method": method,
            "algorithm": algorithm,
            "max_iter": max_iter,
            "tolerance": tolerance,
        }
    )


@router.get("/geometric_program", response_class=HTMLResponse)
async def gp_get(
    request: Request,
    objective: str | None = Query(default=None),
    constraints: str | None = Query(default=None),
    method: str | None = Query(default=None),
    algorithm: str | None = Query(default=None),
    max_iter: int | None = Query(default=None),
    tolerance: float | None = Query(default=None),
) -> HTMLResponse:
    """Display the geometric programming input form."""
    return templates.TemplateResponse(
        "geometric_program.html",
        {
            "request": request,
            "objective": objective,
            "constraints": constraints,
            "method": method,
            "algorithm": algorithm,
            "max_iter": max_iter,
            "tolerance": tolerance,
        },
    )


@router.post("/geometric_program", response_class=HTMLResponse)
async def gp_post(
    request: Request,
    objective: str = Form(...),
    constraints: str = Form(...),
    method: str | None = Form(default=None),
    algorithm: str | None = Form(default=None),
    max_iter: int | None = Form(default=None),
    tolerance: float | None = Form(default=None),
) -> HTMLResponse:
    """Solve the provided geometric program and return the result."""
    try:
        validate_options(algorithm, {"ECOS", "SCS"}, max_iter, tolerance)
        result = solve_geometric(
            objective,
            constraints,
            method=method,
            algorithm=algorithm,
            max_iter=max_iter,
            tolerance=tolerance,
        )
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
    return templates.TemplateResponse(
        "geometric_program.html",
        {
            "request": request,
            "result": result,
            "method": method,
            "algorithm": algorithm,
            "max_iter": max_iter,
            "tolerance": tolerance,
        },
    )


@router.get("/gradient_descent", response_class=HTMLResponse)
async def gradient_descent_get(
    request: Request,
    objective: str | None = Query(default=None),
) -> HTMLResponse:
    """Display the gradient descent animation form."""
    return templates.TemplateResponse(
        "gradient_descent.html",
        {"request": request, "objective": objective},
    )


@router.post("/gradient_descent", response_class=HTMLResponse)
async def gradient_descent_post(
    request: Request,
    objective: str = Form(...),
) -> HTMLResponse:
    """Generate an animation of gradient descent on the given objective."""
    try:
        fig = gradient_descent_animation(objective)
        plot_html = pio.to_html(fig, include_plotlyjs="cdn")
        result = None
    except Exception as exc:  # noqa: BLE001
        plot_html = None
        result = f"An error occurred: {exc}"
    return templates.TemplateResponse(
        "gradient_descent.html",
        {"request": request, "plot_html": plot_html, "result": result},
    )


@router.get("/benchmark", response_class=HTMLResponse)
async def benchmark_results(request: Request) -> HTMLResponse:
    """Run benchmarks and display the results."""
    from benchmark import run_benchmarks

    results = run_benchmarks(return_results=True)
    return templates.TemplateResponse(
        "benchmark.html", {"request": request, "results": results}
    )
