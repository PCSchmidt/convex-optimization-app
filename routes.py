from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from solvers import solve_lp, solve_qp

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
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
    return templates.TemplateResponse(
        "linear_program.html", {"request": request, "result": result}
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
    except Exception as exc:  # noqa: BLE001
        result = f"An error occurred: {exc}"
    return templates.TemplateResponse(
        "quadratic_program.html", {"request": request, "result": result}
    )
