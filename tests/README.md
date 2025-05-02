# Test Suite for QA System

This directory contains automated tests for the QA System project. The tests are designed to verify the correctness, reliability, and maintainability of the system's core components, including configuration, logging, and main application flows.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Running Tests](#running-tests)
- [Test Structure](#test-structure)
- [Example Commands](#example-commands)
- [Troubleshooting](#troubleshooting)

## Overview
The test suite uses [pytest](https://docs.pytest.org/) as the primary test runner. Tests are written in Python and are located in this `tests/` directory. Each test file targets a specific module or feature of the QA System.

## Requirements
- Python 3.8+
- All project dependencies installed (see `requirements.txt` or your environment setup)
- [pytest](https://docs.pytest.org/) installed in your environment

To install dependencies (if using a virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

You can run all tests from the project root or from within the `tests/` directory. The recommended way is to use `pytest`:

```bash
# From the project root
pytest -v

# Or from the tests directory
cd tests
pytest -v
```

If your tests import the main package (e.g., `from qa_system...`), you may need to set the `PYTHONPATH` to the project root:

```bash
export PYTHONPATH=$(pwd)
pytest -v
```

Or, in a single command:

```bash
PYTHONPATH=$(pwd) pytest -v
```

## Test Structure
- `test_config.py`: Tests for configuration loading and access
- `test_logging_setup.py`: Tests for logging setup and log file creation
- `test_main.py`: Tests for the main application flows (add, list, remove, query)
- `conftest.py`: Shared fixtures and setup for tests
- `test_config.yaml`: Example configuration file used in tests

## Example Commands

Run a specific test file:

```bash
pytest tests/test_config.py
```

Run a specific test function:

```bash
pytest tests/test_main.py::TestMain::test_successful_execution
```

Show detailed output:

```bash
pytest -v
```

Generate a test coverage report (if `pytest-cov` is installed):

```bash
pytest --cov=qa_system
```

## Troubleshooting
- **ModuleNotFoundError**: If you see errors like `ModuleNotFoundError: No module named 'qa_system'`, ensure your `PYTHONPATH` includes the project root.
- **Missing dependencies**: Install all required packages using `pip install -r requirements.txt`.
- **Virtual environment**: Activate your virtual environment before running tests.

## Additional Resources
- [pytest documentation](https://docs.pytest.org/)
- [Python unittest documentation](https://docs.python.org/3/library/unittest.html)

---

For any issues or questions, please refer to the main project README or contact the maintainers. 