# AdaPTS Tests

This directory contains the pytest test suite for the AdaPTS project.

## Running Tests

To run all tests:
```bash
pytest tests/ -v
```

To run a specific test file:
```bash
pytest tests/test_adapters.py -v
```

To run tests with coverage:
```bash
pytest tests/ --cov=adapts --cov-report=html
```

## Test Structure

- `test_adapters.py` - Tests for adapter classes (IdentityTransformer, MultichannelProjector)
- `test_preprocessing.py` - Tests for preprocessing utilities (AxisScaler, AxisPCA, utility functions)
- `test_adapts.py` - Tests for the main ADAPTS class

## Test Coverage

The test suite covers:
- Core transformers and adapters
- Preprocessing utilities
- Shape validations
- Data transformations
- Integration tests

Total: 46 tests across all components
