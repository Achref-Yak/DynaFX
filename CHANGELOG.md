# Changelog

## 0.2.0 (unreleased)

### Added
- Pyright type checking configuration
- Ruff linting configuration
- GitHub Actions CI workflow (pytest + ruff + pyright on 3.12/3.13)
- GitHub Pages docs deployment workflow (`.github/workflows/deploy.yml`, mkdocs + Material)
- `CONTRIBUTING.md` with dev setup and PR checklist
- `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)
- `CITATION.cff` for machine-readable citation (GitHub "Cite this repository" button)
- `SECURITY.md`, GitHub issue templates (bug/feature) and PR template
- `docs/citation.md` — citation guide + planned Zenodo DOI path

### Changed
- README + docs/ (index, architecture, development, knowledge, digital-twin, examples) rewritten from scratch
- Updated `pyproject.toml` metadata: version bump, license, keywords, classifiers, URLs, `[project.urls]`, `authors`
- README gains a Citation section (BibTeX + plain text); docs nav gains Citation page
- `CONTRIBUTING.md` rewritten with accurate `uv run` commands and contributor links

### Removed
- **Subjective Logic / epistemics** — `dynafx/epistemics/`, `dynafx/sl/`, `knowledge/confidence.py`, `Opinion`/`FusionSituation`, and all epistemics tests deleted; docs rewritten project-wide
- **CLI** — `__main__.py` entrypoints, `[project.scripts]`, CLI tests, Makefile `run` target, and all doc references
- `reason/store.py` (CorpusStore — zero source imports) and associated test
- `tests/test_csv.py` + `tests/test_csv_import.py` merged into `tests/test_csv_io.py`
- `examples/output.json` (101KB generated artifact)
- Stale mypy overrides in `pyproject.toml`

### Fixed
- `dsl.py`: lazy self-imports now import from `_parser` instead of circular `dsl`
- DES metrics merged into `params` so post-hoc aux replay sees them
