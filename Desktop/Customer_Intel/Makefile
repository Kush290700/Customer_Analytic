SHELL := /bin/bash

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: dev test bench e2e profile load lint type contract sec all report

venv:
	python -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt

dev: venv
	FLASK_ENV=development FLASK_APP=wsgi.py $(PY) -m flask run

test: venv
	mkdir -p perf/artifacts/coverage
	APP_ENABLE_SCHEDULER=false ENABLE_METRICS=false TESTING=1 $(VENV)/bin/pytest -q --junitxml=perf/artifacts/junit.xml -m "not e2e"

bench: venv
	APP_ENABLE_SCHEDULER=false ENABLE_METRICS=false TESTING=1 $(PY) perf/bench_endpoints.py

e2e: venv
	mkdir -p perf/artifacts/e2e
	$(VENV)/bin/python -m playwright install chromium || true
	APP_ENABLE_SCHEDULER=false ENABLE_METRICS=false TESTING=1 $(VENV)/bin/pytest -q -m e2e

profile: venv
	bash scripts/dev_profile.sh

load: venv
	mkdir -p perf/artifacts/load
	LOCUST_P95_BUDGET_MS=200 $(VENV)/bin/locust -f tests/load/locust_tabs.py --headless -u 30 -r 3 -t 2m --host http://localhost:5000 || true
	@echo "Load test artifacts (if any) under perf/artifacts/load/"

lint: venv
	$(VENV)/bin/ruff check . && $(VENV)/bin/ruff format --check .

type: venv
	$(VENV)/bin/mypy

contract: venv
	mkdir -p perf/artifacts/contract
	APP_ENABLE_SCHEDULER=false ENABLE_METRICS=false TESTING=1 $(VENV)/bin/pytest -q -m contract

sec: venv
	@echo "[bandit] scanning .." && $(VENV)/bin/bandit -q -r ci_app || true
	@echo "[safety] checking .." && $(VENV)/bin/safety check -r requirements.txt --full-report || echo "safety offline or failed; non-fatal in local runs"
	@command -v gitleaks >/dev/null 2>&1 && gitleaks detect --no-banner || echo "gitleaks not found; TODO install to run secret scan"

all: lint type test e2e bench contract

report:
	@echo "Artifacts: perf/artifacts"

.PHONY: ci-local
ci-local: venv
	bash scripts/local_ci.sh
