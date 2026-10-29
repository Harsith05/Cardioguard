[← Observability](03-observability.md) · [All docs](README.md) · [Semi Supervised →](05-semi-supervised.md)

# Phase 4: Containerization

Everything now runs in Docker. No `uvicorn` terminal, no `DATABASE_URL` to set:
inside Docker the services find each other by name (`postgres:5432`, `pushgateway:9091`).

```
                 ┌──────────── Docker network ─────────────────────────────────────────┐
 your browser ──▶│ nginx :8000 ──▶ api (1..N) ──▶ postgres, mongodb                    │
 simulators      │                    ▲   └──metrics──▶ prometheus ──▶ grafana :3000   │
                 │ drift-scheduler ───┘ (same image) ──push──▶ pushgateway             │
                 │ bootstrap (one-off: reference data → Postgres)                      │
                 │ trainer   (on demand: retrain inside the image)                     │
                 └─────────────────────────────────────────────────────────────────────┘
```

## Start

```bash
docker compose up -d --build          # first time: builds the app image (~5 min)
docker compose ps                     # api / nginx / postgres / mongodb "healthy"
```

| URL | What |
|---|---|
| http://localhost:8000 | Prediction page (through Nginx) |
| http://localhost:8000/monitoring | Drift monitoring page |
| http://localhost:3000 | Grafana |
| http://localhost:9090 | Prometheus |

Simulators and load tests still run from your computer against `localhost:8000`:
`python -m simulate.traffic --scenario normal`, `python -m simulate.load_test --rps 15`.

## Services

| Service | Image | Role |
|---|---|---|
| `bootstrap` | cardioguard-app | runs once: loads reference data into Postgres, then exits |
| `api` | cardioguard-app | FastAPI (uvicorn); can be scaled |
| `drift-scheduler` | cardioguard-app | drift check every `DRIFT_EVERY_MINUTES` (default 60) |
| `nginx` | nginx:1.27-alpine | single entry point on port 8000; spreads load over api containers; hides `/metrics` |
| `trainer` | cardioguard-app | on demand (`--profile tools`): data prep + train + load reference |
| postgres, mongodb, pushgateway, prometheus, grafana | official images | as in Phases 1–3 |

One image (`docker/app.Dockerfile`) serves all application roles:
- **multi-stage build:** packages are installed in a builder stage, and only the virtualenv is copied into the final image;
- **runs as a non-root user** (`app`);
- **exact library versions** from `requirements-runtime.txt`;
- **models and reference data are mounted read-only**, not baked in, so retraining doesn't need a rebuild;
- **Evidently reports** live in a named volume shared by `api` and `drift-scheduler`.

Startup order is enforced with health checks: postgres healthy → bootstrap finished → api healthy → nginx and drift-scheduler.

## Common tasks

```bash
docker compose --profile tools run --rm trainer   # retrain inside the container (matches library versions)
docker compose restart api                        # load the new model
docker compose up -d --scale api=3                # three API containers behind Nginx
docker compose logs -f api drift-scheduler        # follow logs
docker compose exec api python -m monitoring.drift_job   # run a drift check now
docker compose down                               # stop (data is kept in volumes)
docker compose down -v                            # stop AND delete all data
```

Settings come from `.env` (copy `.env.example`): `POSTGRES_PORT`, `API_PORT`, `DRIFT_EVERY_MINUTES`,
`GRAFANA_PASSWORD`, `ALERT_WEBHOOK_URL`, `CHAOS_ENABLED`.

## Verified (full stack built and run with `docker compose`)

| Test | Result |
|---|---|
| `docker compose up` | correct order; bootstrap loaded 47,553 rows and exited 0; api, nginx, postgres, mongodb healthy |
| Through Nginx | `/`, `/monitoring`, `/docs` 200; `/metrics` 404 (scraped internally instead) |
| Traffic + drift checks (from the page button and from the scheduler container) | normal OK → bp_shift critical → retraining recommended; reports from both containers served by the API |
| `--scale api=3` | Nginx spread 30 requests 10 / 12 / 8; Prometheus discovered all 3 by DNS |
| One API container crashed during 10 req/s load | **0 failed requests**; Docker restarted it, healthy again |
| Retrain with the `trainer` service | same model quality (test AUC 0.8024); library versions recorded; API reloads it without warnings |
| `docker compose down` + `up` | 3,373 predictions, 9 drift runs, reference data and Evidently reports all kept |

Two problems found and fixed during these tests:
1. **`PredictionAPIDown` could never fire.** With DNS discovery, a stopped container vanishes from the targets instead of reporting `up == 0`. The alert now uses `absent(up == 1)`, and a separate `PredictionAPIInstanceDown` warning covers one replica of several. Both are covered by promtool tests.
2. **A stale drift alert after retraining.** Drift metrics were grouped by model version, so the old model's "critical" would have stayed in Pushgateway forever. Each push now replaces the previous one.

[← Observability](03-observability.md) · [All docs](README.md) · [Semi Supervised →](05-semi-supervised.md)
