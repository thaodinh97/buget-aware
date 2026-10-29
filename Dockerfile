FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ml.txt requirements-preprocessing.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 \
    && pip install --no-cache-dir -r requirements-ml.txt -r requirements-preprocessing.txt

COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY tests/ tests/

CMD ["python", "scripts/run_controlled_phase.py", "--output", "runs/docker/controlled_phase_v2_seed42"]
