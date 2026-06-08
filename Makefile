.PHONY: install install-dev run stop lint test

install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

install-dev: install
	.venv/bin/pip install -r requirements-dev.txt

run:
	./scripts/launch.sh

stop:
	./scripts/stop.sh

lint:
	.venv/bin/ruff check .

test:
	.venv/bin/pytest tests/ -v
