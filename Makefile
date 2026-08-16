.PHONY: env install download-data lint format test clean

ENV_NAME = ecosystem-complexity
PYTHON ?= python

env:
	conda env create -f environment.yaml

env-update:
	conda env update -f environment.yaml --prune

install:
	pip install -e .
	$(MAKE) download-data

# Required observation inputs are staged on installation.  Fetch is idempotent:
# existing files are retained, and only a missing IntCal20 file is fetched.
download-data:
	PYTHONPATH=src $(PYTHON) apps/fetch.py israd
	PYTHONPATH=src $(PYTHON) apps/fetch.py atm14c

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
