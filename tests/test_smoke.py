"""Stage 0 smoke tests: interpreter floor and package importability."""

import sys

import convex_optimization


def test_python_version_floor():
    assert sys.version_info >= (3, 11)


def test_package_imports_and_exposes_version():
    assert hasattr(convex_optimization, "__version__")
    assert isinstance(convex_optimization.__version__, str)
    assert convex_optimization.__version__ == "0.1.0"
