# MLflow tracking server + model registry (Phase 6).
#   Metadata (runs, versions, aliases) -> Postgres database "mlflow"
#   Model files                        -> /mlartifacts (named volume), served by this server,
#                                         so the API and Airflow need no shared folder.
FROM python:3.13-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv && pip install mlflow==3.17.2 psycopg2-binary==2.9.13
RUN useradd --system --create-home --home-dir /home/mlflow mlflow \
 && mkdir -p /mlartifacts && chown mlflow:mlflow /mlartifacts
USER mlflow
EXPOSE 5000
# the command (backend store, allowed hosts...) is set in docker-compose.yml
