# Development

## Setup

```bash
git clone https://github.com/Achref-Yak/DynaFX
cd reasoning_engine
uv sync --all-extras
```

## Running Tests

```bash
uv run pytest                     # all tests
uv run pytest --no-header -q      # quiet, fast
uv run pytest tests/test_kb_*.py  # knowledge base tests only
uv run pytest -k "inference"      # inference-related tests
```

## Code Quality

```bash
uv run ruff check src/            # lint
uv run ruff check src/ --fix      # lint + auto-fix
uv run pyright src/dynafx         # type check
uv run mypy src/dynafx            # strict type check (baseline)
```

## Pre-commit

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## CI

The CI workflow (`.github/workflows/ci.yml`) runs three checks on every push:

1. **Lint**: `ruff check src/`
2. **Type check**: `pyright src/dynafx`
3. **Test**: `pytest --no-header -q`

## PR Checklist

- [ ] `ruff check src/` passes
- [ ] `pyright src/dynafx` passes
- [ ] All tests pass
- [ ] New code has tests
- [ ] No `print()` statements (use `logger`)
