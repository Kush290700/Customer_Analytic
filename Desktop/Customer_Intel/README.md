# Customer Intel

Customer analytics, forecasting, and insights with Flask + pandas.

- Quickstart: run `flask run` (or `python run.py`) after creating a `.venv` and installing `requirements.txt`.
- See CONTRIBUTING.md for guidelines on date handling, numeric coercion, and caching keys.

## Production Deployment

Use this checklist to deploy safely and reliably.

- Environment:
  - `SECRET_KEY`: strong random string.
  - Database: either `DATABASE_URL` or MSSQL envs (`MSSQL_SERVER`, `MSSQL_DB`, `MSSQL_USER`+`MSSQL_PASSWORD` or `MSSQL_TRUSTED=1`).
  - Optional: `APP_TIMEZONE` (default `America/Los_Angeles`), `CSP_ALLOW_UNSAFE_*` off, `RATE_LIMIT` as needed.
  - Email (optional): SMTP settings for KPI digests.

- Run server (Gunicorn):
  - Linux example: `gunicorn -c gunicorn.conf.py wsgi:app` (behind a reverse proxy like NGINX).
  - Windows/IIS: use `waitress-serve` or run via container.

- Reverse proxy (NGINX):
  - Terminate TLS at the proxy.
  - Set headers: `X-Forwarded-Proto`, `X-Forwarded-For`, and enable gzip/brotli.

- Static + CSP:
  - The app sets a strict Content Security Policy with nonces (see `ci_app/security.py`).
  - If your environment blocks CDNs, self-host the assets referenced in `base.html` and adjust CSP accordingly.

- Data + migrations:
  - Migrations: `alembic upgrade head` (see `migrations/`).
  - Verify connectivity using a simple script or start the app and check `/api/readyz`.

- Health endpoints:
  - Liveness: `GET /api/healthz`.
  - Readiness: `GET /api/readyz`.

- Logging & Monitoring:
  - Configure log destination via your process manager (systemd, Docker). Sample log file `etl_ml_app.log` shows patterns.
  - Add metrics/alerts for 5xx, latency, and rate-limit denials.

- Background jobs (optional):
  - If using scheduled tasks or ML training, run `python ci_app/tasks.py` under a supervisor.

- Docker:
  - `docker build -t customer-intel .`
  - `docker run -p 8000:8000 --env-file .env customer-intel`
  - Or use `docker-compose.yml` for local dev.

## Performance, Testing & Caching

- Server-side caching via Flask-Caching uses Redis when `REDIS_URL` is set, and `SimpleCache` for local dev.
- Heavy analytics endpoints memoize intermediate frames using keys that include active filters and the parquet mtime to ensure correctness across refreshes.
- JSON APIs set `ETag` and short `Cache-Control` headers to enable client revalidation.
- Gzip/deflate is enabled when `flask-compress` is installed.
- A prewarm task warms key API bundles on start and periodically to reduce cold-start latency.

Useful env vars:
- `CACHE_TYPE` (auto-selects RedisCache if `REDIS_URL`, else SimpleCache)
- `REDIS_URL`, `RATELIMIT_STORAGE_URL`
- `ANALYTICS_CACHE_TTL`, `FORECAST_CACHE_TTL`
- `PREWARM_ON_START`, `PREWARM_HORIZONS`, `PREWARM_REQUIRED`
- `SLOW_REQUEST_MS` (slow request logging threshold)

Profiling & benchmarks:
- `python perf/bench_endpoints.py` to measure p50/p95 and update `PERF_REPORT.md`.
- `python perf/bench_endpoints.py --profile /api/analytics -o perf/profile_analytics.html` for pyinstrument traces.
- `pytest -q --benchmark-only` for pytest-benchmark runs.

Testing & QA quick commands:
- Unit+integration: `make test`
- E2E/UI (headless): `make e2e`
- Contract (OpenAPI + Schemathesis): `make contract`
- Load test (Locust headless): `make load`
- All gates (short): `make all` or `make ci-local`

Artifacts & reports:
- Coverage HTML/XML at `perf/artifacts/coverage/`
- Pytest JUnit XML at `perf/artifacts/junit.xml`
- Playwright traces at `perf/artifacts/e2e/`

## Local Performance

- Caching auto-detect: Uses Redis when `REDIS_URL` is set, otherwise `SimpleCache` (in-process) so it works without Docker.
- JSON: orjson-backed responses with `ETag` and short `Cache-Control` for stable client revalidation.
- Compression: `flask-compress` enabled when installed.
- Warm-up: `flask warmup` to precompute common bundles locally.
- Profiling: `make profile` runs `scripts/dev_profile.sh` to start the app under pyinstrument for ~60s while generating light traffic and saves `perf/profile_server.html`.

Environment (local friendly):
- `PERF_ENABLED=true` (no-op hint flag)
- `ENABLE_METRICS=true` enables `/metrics` if the exporter is installed; otherwise a no-op.
- `REDIS_URL` optional; without it, the app still runs fine with in-memory caching and rate limits.

## Performance & Metrics

- SLOs: p95 < 200ms for key analytics endpoints at 50 VUs; error rate < 0.5%.
- Prometheus: `/metrics` served via `prometheus_flask_exporter` when installed; otherwise built-in Prometheus client.
- Tracing: Optional OpenTelemetry spans for Flask + SQLAlchemy when `OTEL_EXPORTER_OTLP_ENDPOINT` set.
- Errors: Optional Sentry via `SENTRY_DSN`.

Env vars:
- `REDIS_URL`: enables Redis backend for Flask-Caching and rate limits.
- `SENTRY_DSN`, `SENTRY_TRACES_SAMPLE_RATE`.
- `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `OTEL_ENABLE`.
- `CACHE_DEFAULT_TIMEOUT`, `ANALYTICS_CACHE_TTL`, `FORECAST_CACHE_TTL`.

See `PERF_BUDGETS.md` for budgets and tuning tips.

## Using the App

- Filters: Use the left sidebar to narrow date range, regions, methods, and customers. Click Apply.
- Views: Save presets locally or to your account (requires login) and re-apply later.
- Sections: Dashboard overview, KPIs, Segmentation & RFM, Cohorts, CLV, Forecasting, Product/Region drilldowns.
- Exports: Export current page or all filtered rows as CSV/XLSX from each page where available.
- Shortcuts: Toggle sidebar with Ctrl+B (⌘B on macOS).

### Product Drilldown — Advanced ML

- Forecasts: In Velocity, see both classic and ML forecasts (LightGBM/XGBoost-backed) for next-month revenue/orders.
- Pricing: New Pricing tab provides price optimization using constant-elasticity economics (log–log regression with seasonal controls),
  unit-cost estimation, a guardrail band (±20% by default), and projected revenue/profit impact.
- Bias control: All ML training and anomaly/elasticity analyses exclude the ongoing month by default to avoid partial-period skew.
- API endpoints:
  - `GET /pages/api/product?product=...` now includes `forecast_ml` and `pricing` blocks.
  - `GET /pages/api/product/price?product=...` returns only pricing recommendation JSON.

## Dev Notes

- Templates: use `fmt_date`/`fmt_datetime` filters for safe rendering.
- Python: use `fmt_date_or_none`/`fmt_datetime_or_none` when building JSON or CSV labels.
- Prefer excluding the ongoing period for monthly/weekly training windows to avoid bias.
