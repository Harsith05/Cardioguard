[← Project README](../README.md)

# CardioGuard documentation

| # | Doc | Read it to… |
|---|---|---|
| ▶ | [Demo walkthrough](demo.md) | Run the whole system end to end in about 20 minutes |
| 1 | [Prediction service](01-prediction-service.md) | Understand the data, the model choice, the API and SHAP explanations |
| 2 | [Drift monitoring](02-drift-monitoring.md) | See how drift is detected and which thresholds are used |
| 3 | [Observability](03-observability.md) | Metrics, alerts and the Grafana dashboard |
| 4 | [Containerization](04-containerization.md) | Docker services, images and scaling |
| 5 | [Semi-supervised learning](05-semi-supervised.md) | How unconfirmed patients get trustworthy pseudo-labels |
| 6 | [Self-healing](06-self-healing.md) | Diagnosis, retraining strategies, gates, clinical safety, canary, rollback |
| ⚙ | [Configuration](configuration.md) | Every setting in `.env` and the YAML files |
| ? | [Troubleshooting](troubleshooting.md) | Problems met while building this, and their fixes |

The phase docs were written as each phase was built. Results, bugs found and design decisions
are recorded where they happened.
