# Development

## Setup

```bash
git clone https://github.com/Achref-Yak/DynaFX
cd DynaFX
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
```

## CI

The CI workflow (`.github/workflows/ci.yml`) runs three checks on every push:

1. **Lint**: `ruff check src/`
2. **Type check**: `pyright src/dynafx`
3. **Test**: `pytest --no-header -q`

## Docs Deployment

Documentation is built with mkdocs (Material theme) and deployed to GitHub Pages automatically.

- Workflow: `.github/workflows/deploy.yml`
- Triggers: push to `main` touching `docs/**` or `mkdocs.yml`, or manual `workflow_dispatch`
- Build locally: `uv run mkdocs build --strict`
- Live site: <https://achref-yak.github.io/DynaFX/>

The GitHub Pages source is set to **GitHub Actions** (repo → Settings → Pages). The deploy workflow builds `site/`, uploads it as an artifact, and publishes it with `actions/deploy-pages`.

## PR Checklist

- [ ] `ruff check src/` passes
- [ ] `pyright src/dynafx` passes
- [ ] All tests pass
- [ ] New code has tests
- [ ] No `print()` statements (use `logger`)
