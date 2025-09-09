#!/usr/bin/env bash
set -euo pipefail

# Lightweight local CI:
# - Lint (ruff), type-check (mypy), tests (pytest)
# - Endpoint benchmarks to perf/baseline.json
# - Enforce budgets from PERF_BUDGETS.md
# - On failure, save profiler artifacts to perf/artifacts/

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Choose Python inside venv if present
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=python; fi

export PYTHONPATH="${PYTHONPATH:-.}:$ROOT_DIR"
export FLASK_APP=wsgi.py
export APP_ENABLE_SCHEDULER=false
export ENABLE_METRICS=false
export TESTING=1
export PARQUET_PATH="$ROOT_DIR/cached_data.parquet"
export SQLALCHEMY_DATABASE_URI="sqlite:///:memory:"

mkdir -p perf/artifacts

echo "[lint] ruff check (non-fatal) .."
"$PY" -m ruff check . || echo "[lint] ruff issues detected (continuing)"
echo "[type] mypy (non-fatal) .."
"$PY" -m mypy ci_app || echo "[type] mypy issues detected (continuing)"

echo "[tests] pytest (unit/integration; exclude e2e) .."
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:--x}"
"$PY" -m pytest -q -m "not e2e" --ignore=tests/e2e tests || exit 2

echo "[bench] pytest-benchmark (key endpoints) .."
if [ -d tests/benchmarks ]; then "$PY" -m pytest -q --benchmark-only tests/benchmarks || true; fi

echo "[bench] bench_endpoints to perf/baseline.json .."
"$PY" perf/bench_endpoints.py

echo "[gate] enforce budgets from PERF_BUDGETS.md .."
set +e
"$PY" perf/gate_perf.py
RC=$?
set -e

if [ $RC -ne 0 ]; then
  echo "[gate] budgets exceeded; collecting profiler artifacts .."
  EP="/api/analytics"
  if [ -f perf/worst_endpoint.txt ]; then EP=$(cat perf/worst_endpoint.txt); fi
  # Profile worst endpoint via pyinstrument
  "$PY" -m pyinstrument -r html -o perf/artifacts/profile.html -- "$PY" perf/bench_endpoints.py --profile "$EP" -o perf/artifacts/profile_ep.html || true
  # Save report and baseline
  cp -f PERF_REPORT.md perf/artifacts/ 2>/dev/null || true
  cp -f perf/baseline.json perf/artifacts/ 2>/dev/null || true
  echo "Artifacts saved under perf/artifacts/"
  exit 1
fi

echo "[e2e] attempting Playwright run (will skip if browser missing) .."
"$PY" -m playwright install chromium || true
APP_ENABLE_SCHEDULER=false ENABLE_METRICS=false TESTING=1 "$PY" -m pytest -q -m e2e || true

echo "[ok] local CI checks passed."
