import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple

from solvers import parse_expression


def _evaluate_expression(terms: List[Tuple[float, str]], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Evaluate a parsed expression on a mesh grid."""
    Z = np.zeros_like(X, dtype=float)
    for coef, var in terms:
        if var.endswith("^2"):
            base = var[:-2]
            if base == "x":
                Z += coef * X**2
            elif base == "y":
                Z += coef * Y**2
        else:
            if var == "x":
                Z += coef * X
            elif var == "y":
                Z += coef * Y
    return Z


def plot_linear_program(objective: str, constraints: str) -> go.Figure:
    """Generate a 2D plot of the feasible region and objective contours."""
    obj_terms = parse_expression(objective)

    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)

    Z = _evaluate_expression(obj_terms, X, Y)

    mask = np.ones_like(X, dtype=bool)
    for line in constraints.splitlines():
        if not line.strip():
            continue
        if "<=" in line:
            lhs, rhs = line.split("<=")
            lhs_terms = parse_expression(lhs)
            lhs_val = _evaluate_expression(lhs_terms, X, Y)
            mask &= lhs_val <= float(rhs.strip())
        elif ">=" in line:
            lhs, rhs = line.split(">=")
            lhs_terms = parse_expression(lhs)
            lhs_val = _evaluate_expression(lhs_terms, X, Y)
            mask &= lhs_val >= float(rhs.strip())

    fig = go.Figure()
    fig.add_trace(
        go.Contour(x=x, y=y, z=Z, contours=dict(showlabels=True), showscale=False)
    )
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=mask.astype(int),
            showscale=False,
            opacity=0.3,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,200,0,0.4)"]],
        )
    )
    fig.update_layout(
        title="Feasible Region and Objective Contours",
        xaxis_title="x",
        yaxis_title="y",
    )
    return fig


def plot_quadratic_program(objective: str) -> go.Figure:
    """Generate a 3D surface plot of a quadratic objective."""
    obj_terms = parse_expression(objective)

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = _evaluate_expression(obj_terms, X, Y)

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=Z, colorscale="Viridis")])
    fig.update_layout(
        title="Objective Surface",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x,y)"),
    )
    return fig


def gradient_descent_animation(objective: str, steps: int = 30, lr: float = 0.1) -> go.Figure:
    """Create an animated gradient descent path on the objective surface."""
    obj_terms = parse_expression(objective)

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = _evaluate_expression(obj_terms, X, Y)

    point = np.array([4.0, 4.0])
    path = [point.copy()]
    for _ in range(steps):
        grad = np.zeros(2)
        px, py = point
        for coef, var in obj_terms:
            if var.endswith("^2"):
                base = var[:-2]
                if base == "x":
                    grad[0] += 2 * coef * px
                elif base == "y":
                    grad[1] += 2 * coef * py
            else:
                if var == "x":
                    grad[0] += coef
                elif var == "y":
                    grad[1] += coef
        point = point - lr * grad
        path.append(point.copy())
    path = np.array(path)

    base_contour = go.Contour(x=x, y=y, z=Z, contours=dict(showlabels=True), showscale=False)

    frames = [
        go.Frame(data=[base_contour, go.Scatter(x=path[:1,0], y=path[:1,1], mode="markers+lines", line=dict(color="red"))])
    ]
    for i in range(1, len(path)):
        frames.append(
            go.Frame(
                data=[
                    base_contour,
                    go.Scatter(
                        x=path[: i + 1, 0],
                        y=path[: i + 1, 1],
                        mode="markers+lines",
                        line=dict(color="red"),
                    ),
                ]
            )
        )

    fig = go.Figure(
        data=[base_contour, go.Scatter(x=[path[0,0]], y=[path[0,1]], mode="markers+lines", line=dict(color="red"))],
        frames=frames,
    )
    fig.update_layout(
        title="Gradient Descent Animation",
        xaxis_title="x",
        yaxis_title="y",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play", method="animate", args=[None])],
            )
        ],
    )
    return fig
