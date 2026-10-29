# CardioGuard application image (Phase 4).
# One image, several roles (chosen by the compose "command"):
#   api              uvicorn app.main:app
#   drift-scheduler  python -m monitoring.scheduler
#   bootstrap        python -m training.load_reference
#   trainer          data_prep + train (on demand)
#   (drift-scheduler also runs the Phase 5 pseudo-label job: python -m semisup.pseudo_label)
#   bootstrap also registers the first champion in MLflow (Phase 6: python -m mlops.bootstrap)
#
# Build:  docker compose build      (or: docker build -f docker/app.Dockerfile -t cardioguard-app .)

# ---------- stage 1: install Python packages into a virtualenv ----------
FROM python:3.13-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install -r /tmp/requirements-runtime.txt

# ---------- stage 2: small runtime image (no compilers, no pip cache) ----------
FROM python:3.13-slim

# LightGBM / XGBoost need the OpenMP runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user, never as root
RUN groupadd --system app && useradd --system --gid app --home-dir /app --no-create-home app

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_WAIT_POLICY=PASSIVE
# OMP_WAIT_POLICY=PASSIVE: idle LightGBM/OpenMP threads sleep instead of spinning. Without it,
# the API and the pseudo-label job (separate containers, same CPUs) can slow each other 10x+.

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY common/ common/
COPY app/ app/
COPY monitoring/ monitoring/
COPY training/ training/
COPY simulate/ simulate/
COPY semisup/ semisup/
COPY mlops/ mlops/

# models/ and data/reference/ are mounted at run time (see docker-compose.yml);
# reports/ holds the Evidently HTML reports (a named volume shared by api + scheduler)
RUN mkdir -p models data/reference data/raw data/processed reports/drift \
 && chown -R app:app /app

USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
