# Contributing

Thanks for your interest in DynaFX! Contributions of all kinds are welcome —
code, models, docs, tutorials, issues, and research collaborations. DynaFX is
a research platform first: if you're here to explore an idea, see
[Open Research Problems](docs/open-problems.md) for concrete questions we'd
love help with.

This project is governed by the [Contributor Covenant](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

## Quick Links

- **Full development reference:** [Development](docs/development.md)
- **Research directions:** [Open Research Problems](docs/open-problems.md)
- **Security issues:** see [SECURITY.md](SECURITY.md) — do **not** open a public issue

## Development Setup

```bash
# Clone and enter the repo
git clone https://github.com/Achref-Yak/DynaFX.git
cd DynaFX

# Create a virtual environment and install all extras
uv sync --all-extras
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## Running Checks

```bash
# All tests
uv run pytest

# Lint
uv run ruff check src/

# Type check
uv run pyright src/dynafx

# Docs build (strict — fails on broken links/warnings)
uv run mkdocs build --strict
```

These four checks run in CI on every push and pull request
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)), so passing them
locally is the fastest way to get a green CI run.

## Where to Start

- **Good first issues** — browse the [issue tracker](https://github.com/Achref-Yak/DynaFX/issues) for `good first issue` labels.
- **Models & recipes** — add a `.sysd` model or a new `dynafx/patterns/` factory; every model ships with a test.
- **Docs & tutorials** — the tutorial code blocks are *executed* against the package before publishing; keep that contract.
- **Research collaborations** — open an issue describing the problem and how DynaFX could support it.

## Pull Request Checklist

- [ ] `uv run pytest` passes
- [ ] New functionality has tests
- [ ] `uv run ruff check src/` passes
- [ ] `uv run pyright src/dynafx` passes
- [ ] `uv run mkdocs build --strict` passes (if docs changed)
- [ ] `CHANGELOG.md` updated for user-facing changes
- [ ] Commit message describes the *why*, not just the *what*
