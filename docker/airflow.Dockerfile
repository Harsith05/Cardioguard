# Airflow 3 + the CardioGuard code (Phase 6).
#
# Two Python environments in one image:
#   Airflow's own Python      scheduler, API server, DAG parsing
#   /opt/cardio (a venv)      the ML steps (pandas, LightGBM, MLflow client...), pinned exactly
#                             like the app image, via @task.external_python in the DAG.
# So Airflow's dependencies never clash with the model's, and a model trained here loads
# in the API container with identical library versions.
FROM apache/airflow:3.2.2-python3.12

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements-runtime.txt /tmp/requirements-runtime.txt
# pendulum: Airflow checks for it in the venv (only to pass dates; avoids a noisy log line)
RUN python -m venv /opt/cardio \
 && /opt/cardio/bin/pip install -r /tmp/requirements-runtime.txt pendulum==3.1.0

# The project code, importable from the venv through a .pth file
WORKDIR /opt/cardioguard
COPY common/ common/
COPY app/ app/
COPY monitoring/ monitoring/
COPY training/ training/
COPY semisup/ semisup/
COPY mlops/ mlops/
RUN /opt/cardio/bin/python -c "import site; print(site.getsitepackages()[0])" > /tmp/sp \
 && echo /opt/cardioguard > "$(cat /tmp/sp)/cardioguard.pth" \
 && mkdir -p models data/processed data/reference reports \
 && chown -R airflow:root /opt/cardioguard

COPY --chown=airflow:root dags/ /opt/airflow/dags/

ENV CARDIO_PYTHON=/opt/cardio/bin/python \
    CARDIO_HOME=/opt/cardioguard \
    OMP_WAIT_POLICY=PASSIVE \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib
USER airflow
WORKDIR /opt/airflow
