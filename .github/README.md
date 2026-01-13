# GitHub Actions CI/CD Pipeline

This directory contains the GitHub Actions workflows for continuous integration and continuous deployment.

## CI Workflow (`workflows/ci.yml`)

The CI workflow runs automatically on:
- Push to `main` or `master` branches
- Pull requests to `main` or `master` branches
- Manual trigger via `workflow_dispatch`

### What it does:

1. **Multi-version Python Testing**: Tests the codebase on Python 3.9, 3.10, 3.11, and 3.12
2. **Dependency Installation**: 
   - Installs all base dependencies from `requirements.txt`
   - Installs optional `dev` dependencies (ruff, pre-commit, black, pytest, pytest-cov)
   - Installs optional `hyperopt` dependencies (gpy, ray, hebo) with special handling for Python 3.12
3. **Linting**: Runs ruff code linter on `src/` and `tests/` directories
4. **Testing**: Executes all 46 pytest tests with verbose output
5. **Coverage**: Generates test coverage reports (on Python 3.10 only)
6. **Coverage Upload**: Uploads coverage to Codecov (if token is configured)

### Adding a CI Badge

To add a CI status badge to your README.md, include:

```markdown
[![CI](https://github.com/abenechehab/AdaPTS/actions/workflows/ci.yml/badge.svg)](https://github.com/abenechehab/AdaPTS/actions/workflows/ci.yml)
```

### Notes on Python 3.12 Compatibility

Some hyperopt dependencies (specifically GPy) have known compatibility issues with Python 3.12. The workflow handles this gracefully with `continue-on-error: true` to ensure the CI doesn't fail due to these known issues. The core functionality and tests still run successfully on Python 3.12.

### Local Testing

You can test the workflow steps locally:

```bash
# Install dependencies
pip install -e .
pip install -e .[dev]
pip install -e .[hyperopt]  # May fail on Python 3.12

# Run linting
ruff check src/ tests/ --exit-zero

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=adapts --cov-report=html
```
