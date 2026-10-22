#!/usr/bin/env python3
"""Run the existing signals-only market scanner hourly as a long-lived service.

The scanner uses hourly candles, so polling every few minutes would often
repeat the same data and alerts. This runner scans immediately on startup, then
at five minutes past each UTC hour to allow the previous hourly candle time to
settle. It does not place trades.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import bot

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("advanced-quant-service")


def next_hourly_run(now: datetime | None = None) -> datetime:
    """Return the next UTC run time at HH:05."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    target = now.replace(minute=5, second=0, microsecond=0)
    if target <= now:
        target += timedelta(hours=1)
    return target


def run_forever() -> None:
    # Force a full scan on every runner cycle, independent of session-open windows.
    os.environ["FORCE_MARKET_SCAN"] = "true"
    while True:
        started = datetime.now(timezone.utc)
        log.info("Starting hourly market scan at %s", started.isoformat())
        try:
            exit_code = bot.main()
            if exit_code:
                log.error("Scanner returned exit code %s; service will continue.", exit_code)
            else:
                log.info("Scanner cycle completed successfully.")
        except KeyboardInterrupt:
            log.info("Shutdown requested.")
            raise
        except Exception:
            # Keep the service alive after transient provider/config errors;
            # systemd also restarts the runner if the process itself exits.
            log.exception("Unhandled scanner error; continuing at next scheduled cycle.")

        target = next_hourly_run(datetime.now(timezone.utc))
        delay = max(1.0, (target - datetime.now(timezone.utc)).total_seconds())
        log.info("Next scan scheduled for %s (sleep %.0f seconds).", target.isoformat(), delay)
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            log.info("Shutdown requested.")
            raise


if __name__ == "__main__":
    run_forever()
