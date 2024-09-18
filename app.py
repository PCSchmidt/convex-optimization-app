from fasthtml import FastHTML, TemplateResponse
import cvxpy as cp
import numpy as np
import plotly.graph_objects as go
import json

app = FastHTML()

@app.route('/')
async def home(request):
    return TemplateResponse('index.html')

@app.route('/linear_program')
async def linear_program(request):
    c1 = float(request.query_params.get('c1', 2))
    c2 = float(request.query_params.get('c2', 3))
    b = float(request.query_params.get('b', 10))
    x, y, result = solve_linear_program(c1, c2, b)
    plot = create_linear_program_plot(x, y, c1, c2, b)
    return TemplateResponse('linear_program.html', {'plot': plot, 'result': result, 'c1': c1, 'c2': c2, 'b': b})

@app.route('/quadratic_program')
async def quadratic_program(request):
    a = float(request.query_params.get('a', 1))
    b = float(request.query_params.get('b', 1))
    c = float(request.query_params.get('c', 1))
    x, y, result = solve_quadratic_program(a, b, c)
    plot = create_quadratic_program_plot(x, y, a, b, c)
    return TemplateResponse('quadratic_program.html', {'plot': plot, 'result': result, 'a': a, 'b': b, 'c': c})

def solve_linear_program(c1, c2, b):
    x = cp.Variable()
    y = cp.Variable()
    objective = cp.Minimize(c1*x + c2*y)
    constraints = [x + y <= b, x >= 0, y >= 0]
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    return x.value, y.value, result

def solve_quadratic_program(a, b, c):
    x = cp.Variable()
    y = cp.Variable()
    objective = cp.Minimize((a*x)**2 + (b*y)**2)
    constraints = [x + y == c, x - y >= 1]
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    return x.value, y.value, result

def create_linear_program_plot(x, y, c1, c2, b):
    x_range = np.linspace(0, b, 100)
    y1 = b - x_range  # x + y = b
    y2 = np.zeros_like(x_range)  # y = 0
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_range, y=y1, mode='lines', name=f'x + y = {b}'))
    fig.add_trace(go.Scatter(x=x_range, y=y2, mode='lines', name='y = 0'))
    fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', name='Optimal Point', marker=dict(size=10, color='red')))
    
    # Add objective function contour
    X, Y = np.meshgrid(x_range, x_range)
    Z = c1*X + c2*Y
    fig.add_trace(go.Contour(x=x_range, y=x_range, z=Z, colorscale='Viridis', opacity=0.5, name='Objective Function'))
    
    fig.update_layout(title='Linear Program Visualization', xaxis_title='x', yaxis_title='y')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_quadratic_program_plot(x, y, a, b, c):
    x_range = np.linspace(-2, 4, 100)
    y1 = c - x_range  # x + y = c
    y2 = x_range - 1  # x - y = 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_range, y=y1, mode='lines', name=f'x + y = {c}'))
    fig.add_trace(go.Scatter(x=x_range, y=y2, mode='lines', name='x - y = 1'))
    
    # Plot the objective function contours
    X, Y = np.meshgrid(x_range, x_range)
    Z = (a*X)**2 + (b*Y)**2
    fig.add_trace(go.Contour(x=x_range, y=x_range, z=Z, colorscale='Viridis', opacity=0.5, name='Objective Function'))
    
    fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', name='Optimal Point', marker=dict(size=10, color='red')))
    
    fig.update_layout(title='Quadratic Program Visualization', xaxis_title='x', yaxis_title='y')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

if __name__ == "__main__":
    app.run(debug=True)
