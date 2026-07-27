FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Local package sources (not on PyPI, installed straight from the repo tree)
COPY subs ./subs
COPY dep/carbs ./dep/carbs
COPY dep/scan_bench_suite ./dep/scan_bench_suite

RUN pip install --upgrade pip \
    # CPU-only torch/torchvision first, otherwise pip's default CUDA build
    # drags in several GB of nvidia-* wheels neither neps nor carbs need here.
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install ./subs ./dep/carbs \
    # autogluon's umbrella package pulls autogluon.multimodal/.timeseries, which
    # conflict with neps' networkx pin; scan_benchmark's code only ever imports
    # autogluon.tabular, so install its declared deps directly instead.
    && pip install --no-deps ./dep/scan_bench_suite \
    && pip install \
        xgboost==3.1.3 \
        tabpfn==6.4.1 \
        matplotlib \
        seaborn \
        tqdm \
        lightgbm==4.6.0 \
        "autogluon.tabular==1.5.0"

# Rest of the application
COPY src ./src
COPY scripts ./scripts

CMD ["python", "-m", "src.run", "--help"]
