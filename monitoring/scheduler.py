"""Run the drift job, then the pseudo-label job, on a fixed interval.
Phase 2 added the drift job, Phase 5 the pseudo-label job. Airflow replaces this in Phase 6.

Usage:
    python -m monitoring.scheduler --every 60      # minutes
"""
from __future__ import annotations

import argparse
import logging
import os
import time

from monitoring.drift_job import run

log = logging.getLogger("cardioguard.scheduler")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=60, help="minutes between runs")
    args = ap.parse_args()
    log.info("Drift checks every %s minutes. Ctrl+C to stop.", args.every)
    while True:
        try:
            r = run()
            log.info("Drift run %s finished: %s", r["run_id"], r["status"])
        except Exception:
            log.exception("Drift run failed")  # keep the scheduler alive
        # After the drift check, because the pseudo-label job skips when drift is critical
        if os.getenv("PSEUDO_LABEL_ENABLED", "1") == "1":
            try:
                from semisup.pseudo_label import run as pseudo_run
                s = pseudo_run()
                log.info("Pseudo-label run %s finished: %s, accepted %s", s["run_id"], s["status"],
                         s.get("n_accepted", 0))
            except Exception:
                log.exception("Pseudo-label run failed")
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()