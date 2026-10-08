.PHONY: install test lint format

install:
	pip install -r requirements.txt

test:
	pytest tests/

lint:
	ruff check src tests

format:
	black src tests
