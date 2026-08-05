PYTHON = python3.12
PYTHONPATH = src

.PHONY: test test-v run clean wheel wheel-clean install-wheel

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/ -q --tb=short

test-v:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/ -v

clean:
	rm -rf .pytest_cache __pycache__
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

wheel:
	uv run python -m build --wheel --outdir dist/

wheel-clean:
	rm -rf dist/ build/ src/dynafx.egg-info/

install-wheel:
	uv pip install dist/dynafx-*.whl
