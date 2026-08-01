PYTHON = python3.12
PYTHONPATH = src

.PHONY: test test-v clean

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/ -q --tb=short

test-v:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/ -v

clean:
	rm -rf .pytest_cache __pycache__
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
