from __future__ import annotations

from fastapi import APIRouter, Form, Request, Query
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

# Prefilled example problems used in the tutorial steps
TUTORIAL_EXERCISES: dict[int, dict[str, str]] = {
    1: {
        "type": "linear_program",
        "objective": "3x + 2y",
        "constraints": "x + y <= 4\nx >= 0\ny >= 0",
    },
    2: {
        "type": "quadratic_program",
        "objective": "x^2 + y^2 + x",
        "constraints": "x + y >= 1\nx >= 0\ny >= 0",
    },
}


def load_problems() -> list[dict]:
    """Load example problems from the ``problems`` directory."""
    problems = []
    if not os.path.isdir("problems"):
        return problems
    for fname in sorted(os.listdir("problems")):
        path = os.path.join("problems", fname)
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
        else:
            result = "Unsupported exercise"
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
    return templates.TemplateResponse(
        f"tutorial/step{step}.html",
        {
            "request": request,
            "objective": objective,
            "constraints": constraints,
            "result": result,
        },
    )


@router.get("/tutorial/quiz", response_class=HTMLResponse)
async def tutorial_quiz_get(request: Request, q: int = Query(0)) -> HTMLResponse:
    """Display a quiz question from the tutorial."""
    questions = load_quiz_questions()
    if q < 0 or q >= len(questions):
        return HTMLResponse("Question not found", status_code=404)
    question = questions[q]
    return templates.TemplateResponse(
        "tutorial/quiz.html",
        {"request": request, "question": question, "index": q},
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
    return templates.TemplateResponse(
        "tutorial/quiz.html",
        {
            "request": request,
            "question": question,
            "index": q,
            "answered": True,
            "correct": correct,
            "given": answer,
        },
    )


@router.get("/linear_program", response_class=HTMLResponse)
async def linear_program_get(
    request: Request,
    objective: str | None = Query(default=None),
    constraints: str | None = Query(default=None),
    method: str | None = Query(default=None),
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
    max_iter: int | None = Form(default=None),
    tolerance: float | None = Form(default=None),
) -> HTMLResponse:
    """Solve the provided linear program and return the result."""
    try:
        result = solve_lp(
            objective,
            constraints,
            method=method,
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
            "max_iter": max_iter,
            "tolerance": tolerance,
        },
    )


@router.get("/quadratic_program", response_class=HTMLResponse)
async def quadratic_program_get(
    request: Request,
    objective: str | None = Query(default=None),
    constraints: str | None = Query(default=None),
    method: str | None = Query(default=None),
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
    max_iter: int | None = Form(default=None),
    tolerance: float | None = Form(default=None),
) -> HTMLResponse:
    """Solve the provided quadratic program and return the result."""
    try:
        result = solve_qp(
            objective,
            constraints,
            method=method,
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
            "max_iter": max_iter,
            "tolerance": tolerance,
        },
    )


@router.get("/semidefinite_program", response_class=HTMLResponse)
async def sdp_get(
    request: Request,
    objective: str | None = Query(default=None),
    constraints: str | None = Query(default=None),
    method: str | None = Query(default=None),
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
    max_iter: int | None = Form(default=None),
    tolerance: float | None = Form(default=None),
) -> HTMLResponse:
    """Solve the provided semidefinite program and return the result."""
    try:
        result = solve_sdp(
            objective,
            constraints,
            method=method,
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
    max_iter: int | None = Form(default=None),
    tolerance: float | None = Form(default=None),
) -> HTMLResponse:
    """Solve the provided conic program and return the result."""
    try:
        result = solve_conic(
            objective,
            constraints,
            method=method,
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
    max_iter: int | None = Form(default=None),
    tolerance: float | None = Form(default=None),
) -> HTMLResponse:
    """Solve the provided geometric program and return the result."""
    try:
        result = solve_geometric(
            objective,
            constraints,
            method=method,
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
