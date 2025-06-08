# Contributing to Convex Optimization App

Thank you for considering contributing! The following guidelines will help you get started quickly.

## Setup
1. Clone the repository and create a virtual environment:
   ```bash
   git clone https://github.com/yourusername/convex-optimization-app.git
   cd convex-optimization-app
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the application locally:
   ```bash
   uvicorn app:app --reload
   ```

## Coding Style
- Use [PEP 8](https://peps.python.org/pep-0008/) formatting.
- Keep functions small and focused.
- Write clear commit messages describing the change.

## Sending Pull Requests
1. Fork the repository and create a branch for your change.
2. Make your edits and ensure the app runs without errors:
   ```bash
   python -m py_compile $(git ls-files '*.py')
   ```
3. Commit your changes and open a pull request targeting the `main` branch.
4. Provide a concise description of what your PR does and reference any relevant issues.

We appreciate your contributions to improving this project!
