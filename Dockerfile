FROM python:3.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MPLCONFIGDIR=/tmp/usv_faults_matplotlib_cache \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        gfortran \
        git \
        libgomp1 \
        libopenblas-dev \
        liblapack-dev \
        pkg-config \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -n "$TORCH_INDEX_URL" ]; then python -m pip install torch --index-url "$TORCH_INDEX_URL"; fi \
    && python -m pip install -e .

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY *.md ./

RUN mkdir -p /tmp/usv_faults_matplotlib_cache data artifacts runs

CMD ["python", "-m", "usv_faults.cli", "--help"]
