# Contributing

## Development Setup

```bash
# Clone and enter the repo
git clone https://github.com/Achref-Yak/DynaFX.git
cd reasoning_engine

# Create virtual environment
uv venv
source .venv/bin/activate

# Install with all extras
uv pip install -e ".[all]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dynafx --cov-report=term-missing

# Run a specific test file
pytest tests/test_dsl.py
```

## Code Quality

```bash
# Lint
ruff check src/

# Type check
pyright src/dynafx
```

## Pull Request Checklist

- [ ] Tests pass (`pytest`)
- [ ] New tests added for new functionality
- [ ] Ruff lint passes (`ruff check src/`)
- [ ] Pyright type-check passes (`pyright src/dynafx`)
- [ ] For model changes: verify with `pytest tests/ -x`
- [ ] Update `CHANGELOG.md` if introducing user-facing changes
