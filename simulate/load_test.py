"""Load test: steady traffic so the Grafana dashboard has something to show (Phase 3).

Request mix (realistic for this service):
  80% POST /predict            one patient
  10% POST /predict/explain    one patient + SHAP
   5% POST /predict/batch      10 patients
   5% invalid input            rejected with 422 (counts as 4xx)

Usage (API must be running):
    python -m simulate.load_test --rps 20 --duration 120
    python -m simulate.load_test --rps 20 --duration 600 --scenario bp_shift   # drifted inputs
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time
from collections import Counter

import httpx
import numpy as np
import pandas as pd

from common.features import INPUT_FEATURES
from simulate.traffic import SCENARIOS, apply_scenario

INT_FIELDS = ("gender", "cholesterol", "gluc", "smoke", "alco", "active")


def to_payload(row: dict) -> dict:
    p = {k: float(row[k]) for k in INPUT_FEATURES}
    for k in INT_FIELDS:
        p[k] = int(p[k])
    return p


async def one_request(client: httpx.AsyncClient, patients: list[dict], stats: Counter, lat: list[float]):
    r = random.random()
    if r < 0.80:
        path, body = "/predict", random.choice(patients)
    elif r < 0.90:
        path, body = "/predict/explain", random.choice(patients)
    elif r < 0.95:
        path, body = "/predict/batch", {"patients": random.sample(patients, 10)}
    else:
        bad = dict(random.choice(patients))
        bad["ap_lo"] = bad["ap_hi"] + 10          # diastolic above systolic -> 422
        path, body = "/predict", bad
    t0 = time.perf_counter()
    try:
        resp = await client.post(path, json=body)
        stats[f"{resp.status_code // 100}xx"] += 1
    except Exception:
        stats["failed"] += 1
    lat.append(time.perf_counter() - t0)


async def run(args):
    rng = np.random.default_rng(args.seed)
    test = pd.read_parquet("data/processed/test.parquet")
    df = apply_scenario(test.sample(min(5000, len(test)), random_state=args.seed), args.scenario, rng)
    patients = [to_payload(r) for r in df[INPUT_FEATURES].to_dict("records")]

    stats, lat, tasks = Counter(), [], []
    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.api, timeout=30, limits=limits) as client:
        start = time.perf_counter()
        n = int(args.rps * args.duration)
        for i in range(n):
            # open-loop: requests are sent on schedule, whether or not earlier ones finished
            delay = start + i / args.rps - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one_request(client, patients, stats, lat)))
            if (i + 1) % int(args.rps * 10) == 0:
                print(f"  {time.perf_counter() - start:5.0f}s  sent {i + 1:6d}  {dict(stats)}")
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    lat_ms = np.array(lat) * 1000
    print(f"\n{len(lat)} requests in {elapsed:.0f}s ({len(lat) / elapsed:.1f} req/s), scenario '{args.scenario}'")
    print(f"Status: {dict(stats)}")
    print(f"Latency  p50 {np.percentile(lat_ms, 50):.0f} ms   p95 {np.percentile(lat_ms, 95):.0f} ms   "
          f"p99 {np.percentile(lat_ms, 99):.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", type=float, default=10, help="requests per second")
    ap.add_argument("--duration", type=float, default=60, help="seconds")
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--scenario", choices=SCENARIOS, default="normal")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--seed", type=int, default=0)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
