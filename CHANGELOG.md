# Changelog

## 0.2.0 (unreleased)

### Added
- Pyright type checking configuration
- Ruff linting configuration
- GitHub Actions CI workflow (pytest + ruff + pyright on 3.12/3.13)
- `CONTRIBUTING.md` with dev setup and PR checklist
- `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)

### Changed
- Expanded README from 90→227 lines with badges, feature tables, CLI reference, architecture diagram
- Updated `pyproject.toml` metadata: version bump, license, keywords, classifiers, URLs, `[project.urls]`

### Fixed
- `__main__.py`: replaced dead import of deleted `api.main` with delegation to `system.__main__:main`
- `dsl.py`: lazy self-imports now import from `_parser` instead of circular `dsl`

### Removed
- `reason/store.py` (CorpusStore — zero source imports) and associated test
- `tests/test_csv.py` + `tests/test_csv_import.py` merged into `tests/test_csv_io.py`
- 33-line `__main__` block in `tests/test_relate_redesign.py` that duplicated pytest
- `examples/output.json` (101KB generated artifact)
- Stale mypy overrides in `pyproject.toml` (deleted `dynafx.domains.*`, `core.diff`, `core.pipeline`)
