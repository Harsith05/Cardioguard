"""Run every dashboard query against live Prometheus + Postgres and report empty panels.

    python infra/grafana/check_dashboard.py
    python infra/grafana/check_dashboard.py --prom http://localhost:9090 \
        --pg postgresql+psycopg2://cardio:cardio@localhost:5432/cardio

Useful after changing the dashboard, or when a Grafana panel says "No data".
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import httpx

DASH = Path(__file__).parent / "dashboards" / "cardioguard.json"


def pg_sql(q: str) -> str:
    # Grafana macro -> plain SQL for the last 6 hours
    return re.sub(r"\$__timeFilter\((\w+)\)", r"\1 > now() - interval '6 hours'", q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prom", default="http://localhost:9090")
    ap.add_argument("--pg", default=os.getenv("DATABASE_URL",
                                              "postgresql+psycopg2://cardio:cardio@localhost:5432/cardio"))
    args = ap.parse_args()
    dash = json.loads(DASH.read_text())
    queries = [(p["title"], t) for p in dash["panels"] if p["type"] != "row" for t in p.get("targets", [])]
    queries += [(f"annotation: {a['name']}", a["target"]) for a in dash["annotations"]["list"] if "target" in a]

    engine = None
    ok = empty = failed = 0
    for title, t in queries:
        if "expr" in t:
            try:
                r = httpx.get(f"{args.prom}/api/v1/query", params={"query": t["expr"]}, timeout=10).json()
                if r["status"] != "success":
                    raise RuntimeError(r.get("error"))
                n = len(r["data"]["result"])
                src = "prom"
            except Exception as e:
                print(f"FAIL   {title} [{t['refId']}]: {e}")
                failed += 1
                continue
        else:
            from sqlalchemy import create_engine, text
            engine = engine or create_engine(args.pg)
            try:
                with engine.connect() as conn:
                    n = len(conn.execute(text(pg_sql(t["rawSql"]))).fetchall())
                src = "sql "
            except Exception as e:
                print(f"FAIL   {title}: {str(e).splitlines()[0]}")
                failed += 1
                continue
        status = "ok   " if n else "EMPTY"
        ok += bool(n)
        empty += not n
        print(f"{status}  {src}  {n:3d} series/rows  {title} [{t['refId']}]")
    print(f"\n{ok} with data, {empty} empty, {failed} failed (of {len(queries)} queries)")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
