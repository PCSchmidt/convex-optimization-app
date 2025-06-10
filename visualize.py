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


def _gradient_terms(terms: List[Tuple[float, str]], point: np.ndarray) -> np.ndarray:
    """Compute gradient of a parsed expression at a point."""
    grad = np.zeros(2)
    x, y = point
    for coef, var in terms:
        if var.endswith("^2"):
            base = var[:-2]
            if base == "x":
                grad[0] += 2 * coef * x
            elif base == "y":
                grad[1] += 2 * coef * y
        else:
            if var == "x":
                grad[0] += coef
            elif var == "y":
                grad[1] += coef
    return grad


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


def plot_3d_surface(expression: str) -> go.Figure:
    """Generate a generic 3D surface plot for a polynomial expression."""
    terms = parse_expression(expression)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = _evaluate_expression(terms, X, Y)
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=Z, colorscale="Viridis")])
    fig.update_layout(
        title="3D Surface",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x,y)"),
    )
    return fig


def feasible_region_animation(objective: str, constraints: str) -> go.Figure:
    """Animate the construction of the feasible region."""
    obj_terms = parse_expression(objective)
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = _evaluate_expression(obj_terms, X, Y)

    lines = [c for c in constraints.splitlines() if c.strip()]
    masks = []
    mask = np.ones_like(X, dtype=bool)
    for line in lines:
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
        masks.append(mask.copy())

    contour = go.Contour(x=x, y=y, z=Z, contours=dict(showlabels=True), showscale=False)
    frames = []
    for m in masks:
        frames.append(
            go.Frame(
                data=[
                    contour,
                    go.Heatmap(
                        x=x,
                        y=y,
                        z=m.astype(int),
                        showscale=False,
                        opacity=0.3,
                        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,200,0,0.4)"]],
                    ),
                ]
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="Feasible Region Animation",
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


def interior_point_animation(objective: str, constraints: str, steps: int = 20) -> go.Figure:
    """Simple interior-point style animation for demonstration."""
    obj_terms = parse_expression(objective)
    cons = []
    for c in constraints.splitlines():
        if not c.strip():
            continue
        if "<=" in c:
            lhs, rhs = c.split("<=")
            cons.append((parse_expression(lhs), "<=", float(rhs.strip())))
        elif ">=" in c:
            lhs, rhs = c.split(">=")
            cons.append((parse_expression(lhs), ">=", float(rhs.strip())))

    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = _evaluate_expression(obj_terms, X, Y)

    mask = np.ones_like(X, dtype=bool)
    for t, op, val in cons:
        vals = _evaluate_expression(t, X, Y)
        if op == "<=":
            mask &= vals <= val
        else:
            mask &= vals >= val

    point = np.array([0.0, 0.0])
    path = [point.copy()]
    lr = 0.05
    t = 1.0
    for _ in range(steps):
        grad = _gradient_terms(obj_terms, point)
        barrier = np.zeros(2)
        for terms, op, val in cons:
            g_val = _evaluate_expression(terms, np.array([[point[0]]]), np.array([[point[1]]]))[0, 0]
            diff = g_val - val if op == "<=" else val - g_val
            grad_g = _gradient_terms(terms, point)
            barrier += grad_g / (diff if diff != 0 else 1e-6)
        point = point - lr * (grad + t * barrier)
        path.append(point.copy())
        t *= 0.9

    path = np.array(path)
    contour = go.Contour(x=x, y=y, z=Z, contours=dict(showlabels=True), showscale=False)
    region = go.Heatmap(
        x=x,
        y=y,
        z=mask.astype(int),
        showscale=False,
        opacity=0.3,
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,200,0,0.4)"]],
    )
    frames = [go.Frame(data=[contour, region, go.Scatter(x=path[:1,0], y=path[:1,1], mode="markers+lines", line=dict(color="red"))])]
    for i in range(1, len(path)):
        frames.append(
            go.Frame(
                data=[
                    contour,
                    region,
                    go.Scatter(x=path[: i + 1, 0], y=path[: i + 1, 1], mode="markers+lines", line=dict(color="red")),
                ]
            )
        )
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="Interior Point Animation",
        xaxis_title="x",
        yaxis_title="y",
        updatemenus=[dict(type="buttons", showactive=False, buttons=[dict(label="Play", method="animate", args=[None])])],
    )
    return fig


def simplex_animation(objective: str, constraints: str) -> go.Figure:
    """Naive simplex-style animation using polygon vertices."""
    obj_terms = parse_expression(objective)
    lines = [c for c in constraints.splitlines() if c.strip()]

    def parse_coeffs(expr: str) -> tuple[float, float]:
        terms = parse_expression(expr)
        a = b = 0.0
        for coef, var in terms:
            if var == "x":
                a += coef
            elif var == "y":
                b += coef
        return a, b

    coeffs = []
    for c in lines:
        if "<=" in c:
            lhs, rhs = c.split("<=")
            a, b = parse_coeffs(lhs)
            coeffs.append((a, b, float(rhs.strip()), "<="))
        elif ">=" in c:
            lhs, rhs = c.split(">=")
            a, b = parse_coeffs(lhs)
            coeffs.append((a, b, float(rhs.strip()), ">="))

    vertices = []
    for i in range(len(coeffs)):
        for j in range(i + 1, len(coeffs)):
            a1, b1, c1, _ = coeffs[i]
            a2, b2, c2, _ = coeffs[j]
            det = a1 * b2 - a2 * b1
            if det == 0:
                continue
            px = (c1 * b2 - c2 * b1) / det
            py = (a1 * c2 - a2 * c1) / det
            feasible = True
            for a, b, c, op in coeffs:
                val = a * px + b * py
                if op == "<=" and val > c + 1e-6:
                    feasible = False
                    break
                if op == ">=" and val < c - 1e-6:
                    feasible = False
                    break
            if feasible:
                vertices.append((px, py))

    if not vertices:
        raise ValueError("No feasible region")

    vals = [
        _evaluate_expression(obj_terms, np.array([[vx]]), np.array([[vy]]))[0, 0]
        for vx, vy in vertices
    ]
    order = np.argsort(vals)
    path = np.array([vertices[i] for i in order])

    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = _evaluate_expression(obj_terms, X, Y)
    mask = np.ones_like(X, dtype=bool)
    for a, b, c, op in coeffs:
        vals = a * X + b * Y
        if op == "<=":
            mask &= vals <= c
        else:
            mask &= vals >= c

    contour = go.Contour(x=x, y=y, z=Z, contours=dict(showlabels=True), showscale=False)
    region = go.Heatmap(x=x, y=y, z=mask.astype(int), showscale=False, opacity=0.3, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,200,0,0.4)"]])
    frames = [go.Frame(data=[contour, region, go.Scatter(x=path[:1,0], y=path[:1,1], mode="markers+lines", line=dict(color="red"))])]
    for i in range(1, len(path)):
        frames.append(go.Frame(data=[contour, region, go.Scatter(x=path[: i + 1, 0], y=path[: i + 1, 1], mode="markers+lines", line=dict(color="red"))]))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="Simplex Animation",
        xaxis_title="x",
        yaxis_title="y",
        updatemenus=[dict(type="buttons", showactive=False, buttons=[dict(label="Play", method="animate", args=[None])])],
    )
    return fig
