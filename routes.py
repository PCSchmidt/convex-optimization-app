from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from solvers import solve_lp, solve_qp, solve_sdp, solve_conic, solve_geometric
import plotly.io as pio
from visualize import (
    gradient_descent_animation,
    plot_linear_program,
    plot_quadratic_program,
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """Render the homepage with links to optimization problems."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/linear_program", response_class=HTMLResponse)
async def linear_program_get(request: Request) -> HTMLResponse:
    """Display the linear programming input form."""
    return templates.TemplateResponse("linear_program.html", {"request": request})


@router.post("/linear_program", response_class=HTMLResponse)
async def linear_program_post(
    request: Request,
    objective: str = Form(...),
    constraints: str = Form(...),
) -> HTMLResponse:
    """Solve the provided linear program and return the result."""
    try:
        result = solve_lp(objective, constraints)
        fig = plot_linear_program(objective, constraints)
        plot_html = pio.to_html(fig, include_plotlyjs="cdn")
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
        plot_html = None
    return templates.TemplateResponse(
        "linear_program.html",
        {"request": request, "result": result, "plot_html": plot_html},
    )


@router.get("/quadratic_program", response_class=HTMLResponse)
async def quadratic_program_get(request: Request) -> HTMLResponse:
    """Display the quadratic programming input form."""
    return templates.TemplateResponse(
        "quadratic_program.html", {"request": request}
    )


@router.post("/quadratic_program", response_class=HTMLResponse)
async def quadratic_program_post(
    request: Request,
    objective: str = Form(...),
    constraints: str = Form(...),
) -> HTMLResponse:
    """Solve the provided quadratic program and return the result."""
    try:
        result = solve_qp(objective, constraints)
        fig = plot_quadratic_program(objective)
        plot_html = pio.to_html(fig, include_plotlyjs="cdn")
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
        plot_html = None
    return templates.TemplateResponse(
        "quadratic_program.html",
        {"request": request, "result": result, "plot_html": plot_html},
    )


@router.get("/semidefinite_program", response_class=HTMLResponse)
async def sdp_get(request: Request) -> HTMLResponse:
    """Display the semidefinite programming input form."""
    return templates.TemplateResponse("semidefinite_program.html", {"request": request})


@router.post("/semidefinite_program", response_class=HTMLResponse)
async def sdp_post(
    request: Request,
    objective: str = Form(...),
    constraints: str = Form(...),
) -> HTMLResponse:
    """Solve the provided semidefinite program and return the result."""
    try:
        result = solve_sdp(objective, constraints)
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
    return templates.TemplateResponse(
        "semidefinite_program.html", {"request": request, "result": result}
    )


@router.get("/conic_program", response_class=HTMLResponse)
async def conic_get(request: Request) -> HTMLResponse:
    """Display the conic programming input form."""
    return templates.TemplateResponse("conic_program.html", {"request": request})


@router.post("/conic_program", response_class=HTMLResponse)
async def conic_post(
    request: Request,
    objective: str = Form(...),
    constraints: str = Form(...),
) -> HTMLResponse:
    """Solve the provided conic program and return the result."""
    try:
        result = solve_conic(objective, constraints)
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
    return templates.TemplateResponse(
        "conic_program.html", {"request": request, "result": result}
    )


@router.get("/geometric_program", response_class=HTMLResponse)
async def gp_get(request: Request) -> HTMLResponse:
    """Display the geometric programming input form."""
    return templates.TemplateResponse("geometric_program.html", {"request": request})


@router.post("/geometric_program", response_class=HTMLResponse)
async def gp_post(
    request: Request,
    objective: str = Form(...),
    constraints: str = Form(...),
) -> HTMLResponse:
    """Solve the provided geometric program and return the result."""
    try:
        result = solve_geometric(objective, constraints)
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
    return templates.TemplateResponse(
        "geometric_program.html", {"request": request, "result": result}
    )


@router.get("/gradient_descent", response_class=HTMLResponse)
async def gradient_descent_get(request: Request) -> HTMLResponse:
    """Display the gradient descent animation form."""
    return templates.TemplateResponse("gradient_descent.html", {"request": request})


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
