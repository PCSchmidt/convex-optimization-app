# Convex Optimization App

## Introduction

The Convex Optimization App is an educational and practical tool designed to help users understand and solve convex optimization problems, with a focus on linear and quadratic programming. This application provides both a theoretical foundation and hands-on experience with optimization techniques widely used in various fields such as operations research, machine learning, and economics.

## What is Convex Optimization?

Convex optimization is a subfield of mathematical optimization that deals with the minimization of convex functions over convex sets. It's particularly important because:

1. Many real-world problems can be modeled as convex optimization problems.
2. Convex problems have desirable properties, such as the guarantee that any local optimum is also a global optimum.
3. Efficient algorithms exist for solving convex optimization problems.

Key concepts in convex optimization include:

- Convex Sets: A set is convex if the line segment between any two points in the set is also contained in the set.
- Convex Functions: A function is convex if its epigraph (the set of points above the graph) is a convex set.
- Objective Function: The function we aim to minimize or maximize.
- Constraints: Conditions that the solution must satisfy, often defining the feasible set.

## Features

### 1. Linear Programming Solver

Linear Programming (LP) is a method to achieve the best outcome (maximum profit or lowest cost) in a mathematical model whose requirements are represented by linear relationships.

Our LP solver allows users to input:
- An objective function in the form of a linear equation
- Constraints as linear inequalities

The solver uses the PuLP library to find the optimal solution, providing:
- The values of decision variables
- The optimal value of the objective function
- The solution status

**Example Input**
```text
Objective: 3x + 2y
Constraints:
x + y <= 10
x >= 0
y >= 0
```

### 2. Quadratic Programming Solver

Quadratic Programming (QP) extends linear programming to allow for quadratic terms in the objective function while maintaining linear constraints.

Our QP solver enables users to input:
- A quadratic objective function
- Linear constraints

The solver utilizes the CVXPY library to solve the quadratic program, offering:
- Optimal values for decision variables
- The minimum value of the objective function
- The solution status
- Support for cross terms such as `xy`

**Example Input**
```text
Objective: 2x^2 + 3y^2 + 4x + 5y
Constraints:
x + y <= 10
x >= 0
y >= 0
```

### Solver Options

Each solver accepts optional parameters:

- **method**: choose the backend algorithm (e.g. `cbc` for LP, `ECOS` or `SCS` for convex programs).
- **max_iter**: limit the number of iterations the solver will run.
- **tolerance**: stopping tolerance passed to the solver.

If an unsupported method is supplied, the application will return an error.
### 3. Semidefinite Programming Solver

The SDP solver handles problems with matrix variables constrained to be positive semidefinite. Input the objective matrix and constraints using comma-separated values.

### 4. Conic Programming Solver

Use the conic solver for second-order cone problems. Constraints prefixed with `soc:` define cone constraints.

### 5. Geometric Programming Solver

The GP solver accepts posynomial objectives and constraints, allowing you to solve log-convex programs.

### 6. Visualization Tool

The visualization tool generates 2D plots of the feasible region and contour lines for small two-variable problems. You can optionally animate gradient descent to illustrate convergence.

Navigate to `/visualize` to try it out.

### 7. Educational Content

Cross terms in quadratic objectives (e.g., `xy`) are supported.

Key routes include `/visualize` for plots, `/benchmark` for performance testing, and the tutorial at `/tutorial/step/1`.

The app provides educational content on:
- The basics of convex optimization
- Detailed explanations of linear and quadratic programming
- Real-world applications of these optimization techniques
- The mathematical foundations behind the solvers

## How to Use the App

1. Navigate to the home page for an introduction to Convex Optimization.
2. Follow the interactive tutorial at `/tutorial/step/1` for a guided walkthrough.
3. Choose between the Linear Programming or Quadratic Programming solver.
4. Input your problem:
   - For LP: Enter the coefficients of your objective function and constraints.
   - For QP: Provide the quadratic and linear terms of your objective function and linear constraints.
5. Click "Solve" to get the optimal solution and interpretation.

## API Usage

The application exposes solver endpoints that accept JSON payloads. Each request
returns a response with `status` and `result` fields.

Example request to solve a linear program:

```bash
curl -X POST http://localhost:8000/api/linear_program \
     -H 'Content-Type: application/json' \
     -d '{"objective": "x", "constraints": "x >= 1"}'
```

Available endpoints:

- `POST /api/linear_program`
- `POST /api/quadratic_program`
- `POST /api/semidefinite_program`
- `POST /api/conic_program`
- `POST /api/geometric_program`

All endpoints accept the following JSON keys:

- `objective` – objective function or matrix/vector string
- `constraints` – newline separated constraints
- `method` *(optional)* – solver backend
- `max_iter` *(optional)* – iteration limit
- `tolerance` *(optional)* – solver tolerance

## Installation and Setup

This project requires **Python 3.11**.

1. Clone the repository:
   ```
   git clone https://github.com/PCSchmidt/convex-optimization-app.git
   ```

2. Navigate to the project directory:
   ```
   cd convex-optimization-app
   ```

3. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Run the application:
   ```
   uvicorn app:app --reload
   ```

6. Open a web browser and go to `http://localhost:8000`.

## Deployment

### Local Docker

Build and run the Docker image locally:

```bash
docker build -t convex-optimization-app .
docker run -p 8000:8000 convex-optimization-app
```

Then visit `http://localhost:8000` in your browser. You can expose a different
host port by adjusting the `-p` flag:

```bash
docker run -p 5000:8000 convex-optimization-app
```

### Hosting Options

The container can be deployed to any service that supports Docker images.
Below is an example workflow for [Heroku](https://www.heroku.com/) using the
Heroku CLI:

```bash
heroku create my-convex-app
heroku container:push web -a my-convex-app
heroku container:release web -a my-convex-app
```

Heroku and similar platforms provide the port for your application through the
`PORT` environment variable. To run the container with this variable locally you
can execute:

```bash
docker run -e PORT=8000 -p 8000:8000 convex-optimization-app \
    uvicorn app:app --host 0.0.0.0 --port $PORT
```

Example environment variables:

```text
PORT=8000
```

## Future Enhancements

We plan to expand the Convex Optimization App with the following features:

1. Interactive visualizations:
   - 2D and 3D plots of feasible regions and optimal solutions
   - Step-by-step algorithm visualizations

2. Problem library:
   - A collection of common optimization problems for practice
   - Real-world case studies

3. Advanced solver options:
   - Choice of different algorithms
   - Custom stopping criteria and solver parameters

4. Tutorial system:
   - Guided walkthroughs for beginners
   - Quizzes and exercises to test understanding

5. API access:
   - Allowing programmatic access to the solvers for integration with other applications

6. Performance benchmarking:
   - Comparison of different solvers and algorithms
   - Scalability tests for large-scale problems

## Current Limitations

The app currently provides command-line style input forms and basic textual output.
Interactive visualizations and advanced solver options are still under development.
Cross terms in quadratic objectives (e.g., `xy`) are supported by the parser.

## Running Tests

This project uses `pytest` for its test suite. After installing the dependencies, run:

```bash
pytest -q
```

## Benchmarking

The `benchmark.py` script runs all sample problems with multiple solvers
and records their runtime. Execute:

```bash
python benchmark.py --output results.csv
```

This writes the benchmark results to `results.csv`. Omitting `--output`
prints a CSV table to the terminal instead.


## Contributing

We welcome contributions to the Convex Optimization App! Whether you're fixing bugs, improving documentation, or proposing new features, your efforts are appreciated. Please see our CONTRIBUTING.md file for guidelines on how to submit your contributions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

We'd like to thank the developers of PuLP, CVXPY, and FastAPI, whose excellent libraries make this application possible. We also acknowledge the broader optimization community for their ongoing research and educational resources that inspire and inform this project.

## Contact

For questions, suggestions, or collaborations, please open an issue on our GitHub repository or contact the maintainers directly at [p.christopher.schmidt@gmail.com].

Thank you for using and contributing to the Convex Optimization App!
