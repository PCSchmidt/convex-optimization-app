# Convex Optimization App

This application is designed to showcase Convex Optimization concepts in a creative and interactive way. It focuses on visually illustrating core concepts such as feasible sets, convex sets, objective functions, and optimal solutions through interactive visualizations of linear and quadratic programming problems.

## Project Goals

1. Demonstrate key concepts of convex optimization through interactive visualizations.
2. Provide users with hands-on experience in modifying optimization problems and observing results.
3. Offer clear explanations of convex optimization principles and their applications.
4. Create an educational tool for students and professionals learning about convex optimization.

## Tech Stack

- **Frontend**: FastHTML for building the UI/UX
- **Backend**: Python with FastHTML as the web framework
- **Optimization Solver**: CVXPY
- **Visualization**: Plotly
- **Mathematical Expressions**: MathJax

## Features

1. **Interactive Visuals**: 
   - Illustrate convex sets and optimization problems (linear programming, quadratic programming)
   - Use interactive charts powered by Plotly

2. **Optimization Solvers**: 
   - Integrate CVXPY to solve convex optimization problems

3. **User Inputs**: 
   - Allow users to modify parameters of optimization problems
   - Visualize how the solution changes based on user inputs

4. **Explanations**: 
   - Provide bullet-pointed summaries and explanations of key concepts
   - Offer insights into the mathematical foundations of convex optimization

5. **Math Rendering**: 
   - Use MathJax for clean, readable mathematical expressions

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/convex-optimization-app.git
   cd convex-optimization-app
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:8000` (or the URL shown in the terminal).

3. Explore the different optimization problems by clicking on the links on the home page.

4. Modify the parameters of each problem using the input forms and observe how the optimal solution changes.

## Current Implemented Features

- Home page with an introduction to convex optimization and key concepts
- Interactive linear programming problem solver and visualizer
- Interactive quadratic programming problem solver and visualizer
- User input forms to modify problem parameters
- Visualizations of feasible regions and optimal solutions
- Explanations of problem formulations and solutions

## Potential Future Enhancements

1. **Additional Problem Types**: 
   - Implement more types of convex optimization problems (e.g., semidefinite programming, geometric programming)

2. **Advanced Visualizations**: 
   - 3D visualizations for problems with three variables
   - Animated visualizations to show the optimization process

3. **Tutorial System**: 
   - Guided walkthroughs of solving optimization problems
   - Step-by-step explanations of solution methods

4. **User Accounts**: 
   - Allow users to save and share their custom problems

5. **API Integration**: 
   - Provide an API for solving custom convex optimization problems

6. **Mobile Responsiveness**: 
   - Improve the UI for better usability on mobile devices

7. **Performance Optimization**: 
   - Implement caching and other optimizations for faster problem solving and visualization

8. **Internationalization**: 
   - Add support for multiple languages

## Contributing

Contributions to improve the app or add new features are welcome. Please feel free to submit a pull request or open an issue to discuss potential changes or enhancements.

## License

This project is open-source and available under the MIT License.