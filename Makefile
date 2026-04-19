.PHONY: install test lint format demo backtest status ci

install:
	pip install -e '.[dev]'

test:
	python -m pytest -q

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

demo:
	python scripts/run_demo.py

backtest:
	python scripts/backtest.py

dashboard:
	python scripts/build_dashboard.py
	@echo "open docs/dashboard.html in your browser"

live-validation:
	python scripts/live_validation.py

status:
	analyst status

ci: lint test backtest
