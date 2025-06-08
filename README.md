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

**Example Input**
```text
Objective: 2x^2 + 3y^2 + 4x + 5y
Constraints:
x + y <= 10
x >= 0
y >= 0
```

<!-- codex/add-route-for-2d-plots-visualization -->
### 3. Visualization Tool

The visualization tool generates 2D plots of the feasible region and contour lines for small two-variable problems. You can optionally animate gradient descent to illustrate convergence.

Navigate to `/visualize` to try it out.

### 4. Educational Content

Cross terms in the objective function (e.g., `xy`) are currently unsupported.

### 3. Educational Content
<!-- main -->

The app provides educational content on:
- The basics of convex optimization
- Detailed explanations of linear and quadratic programming
- Real-world applications of these optimization techniques
- The mathematical foundations behind the solvers

## How to Use the App

1. Navigate to the home page for an introduction to Convex Optimization.
2. Choose between the Linear Programming or Quadratic Programming solver.
3. Input your problem:
   - For LP: Enter the coefficients of your objective function and constraints.
   - For QP: Provide the quadratic and linear terms of your objective function and linear constraints.
4. Click "Solve" to get the optimal solution and interpretation.

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

To build and run the Docker image:

```bash
docker build -t convex-optimization-app .
docker run -p 8000:8000 convex-optimization-app
```

Then visit `http://localhost:8000` in your browser.

## Future Enhancements

We plan to expand the Convex Optimization App with the following features:

1. Support for more optimization problem types:
   - Semidefinite Programming (SDP)
   - Conic Optimization
   - Geometric Programming

2. Interactive visualizations:
   - 2D and 3D plots of feasible regions and optimal solutions
   - Step-by-step algorithm visualizations

3. Problem library:
   - A collection of common optimization problems for practice
   - Real-world case studies

4. Advanced solver options:
   - Choice of different algorithms
   - Custom stopping criteria and solver parameters

5. Tutorial system:
   - Guided walkthroughs for beginners
   - Quizzes and exercises to test understanding

6. API access:
   - Allowing programmatic access to the solvers for integration with other applications

7. Performance benchmarking:
   - Comparison of different solvers and algorithms
   - Scalability tests for large-scale problems

## Current Limitations

The app currently provides command-line style input forms and basic textual output.
Interactive visualizations and advanced solver options are still under development.

## Running Tests

This project uses `pytest` for its test suite. After installing the dependencies, run:

```bash
pip install pytest
pytest
```


## Contributing

We welcome contributions to the Convex Optimization App! Whether you're fixing bugs, improving documentation, or proposing new features, your efforts are appreciated. Please see our CONTRIBUTING.md file for guidelines on how to submit your contributions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

We'd like to thank the developers of PuLP, CVXPY, and FastAPI, whose excellent libraries make this application possible. We also acknowledge the broader optimization community for their ongoing research and educational resources that inspire and inform this project.

## Contact

For questions, suggestions, or collaborations, please open an issue on our GitHub repository or contact the maintainers directly at [p.christopher.schmidt@gmail.com].

Thank you for using and contributing to the Convex Optimization App!