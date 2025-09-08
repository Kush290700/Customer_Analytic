Overview

This repository includes a complete local-only testing system covering unit, integration/API, contract, E2E/UI, performance, load, and resilience.

Suites

- Unit: utils, analytics helpers; property-based via Hypothesis.
- Integration/API: Flask test client; schemas, caching (ETag/Cache-Control), pagination/sorting where applicable.
- Contract: OpenAPI validation; Schemathesis short fuzz for critical endpoints.
- E2E/UI: Playwright Python, headless; fails on console errors, unhandled rejections, or 4xx/5xx during navigation.
- Performance: pytest-benchmark on hot endpoints with budgets in PERF_BUDGETS.md.
- Load: Locust headless scenario switching tabs & filters; exports under perf/artifacts/load/.
- Resilience: cache-down fallback, internal-error JSON envelope, concurrency/race checks.

Commands

- `make dev` – run app locally (dev).
- `make test` – unit + integration (no e2e).
- `make e2e` – Playwright sweep (headless).
- `make bench` – pytest-benchmark suite.
- `make load` – Locust headless small run.
- `make contract` – OpenAPI validate + Schemathesis smoke.
- `make lint` – ruff + black --check.
- `make type` – mypy.
- `make sec` – bandit, safety (warn if offline), optional gitleaks.
- `make all` – lint, type, test, e2e, bench (short), contract (short).
- `make report` – open artifacts path hints.

Coverage & Artifacts

- Coverage thresholds: lines >= 85% (see .coveragerc) and branch coverage enabled.
- HTML & XML coverage under `perf/artifacts/coverage/`.
- Pytest JUnit XML at `perf/artifacts/junit.xml`.
- Playwright traces & screenshots at `perf/artifacts/e2e/`.
- Locust stats at `perf/artifacts/load/`.
- Schemathesis logs at `perf/artifacts/contract/`.

Debugging Failures

- Re-run a single test: `pytest -q tests/path::test_name -k keyword -vv`.
- Capture Playwright trace: traces saved per test in `playwright-artifacts/` or `perf/artifacts/e2e/` via make target.
- Contract violations: see assertion diff in contract tests; update `api/openapi.yaml` or fix handlers.
- Perf regressions: see `perf/artifacts/` (profile.html, baseline.json). Adjust budgets in PERF_BUDGETS.md if justified.

Notes

- No cloud dependencies are required. Redis is used only if `REDIS_URL` is present; otherwise `SimpleCache` is used.
- Safety may require network for vulnerability DB; we mark it non-fatal when offline.
- Gitleaks is optional; the `make sec` target will no-op with a TODO if the binary is not present.

