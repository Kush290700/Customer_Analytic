#!/usr/bin/env python
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    # Ensure headless by default
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("APP_ENABLE_SCHEDULER", "false")
    os.environ.setdefault("ENABLE_METRICS", "false")
    os.environ.setdefault("PARQUET_PATH", os.path.join(os.getcwd(), "cached_data.parquet"))

    # Install chromium for Playwright if not present (non-fatal if offline)
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)
    except Exception:
        pass

    # Run only e2e tests
    cmd = [sys.executable, "-m", "pytest", "-q", "-m", "e2e"]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

