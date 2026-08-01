#!/bin/bash
# Shared preamble for every script in scripts/: locate the repo, activate the
# conda env, and make sure `python -m src.*` resolves.
#
# Everything is overridable from the environment so the scripts run unmodified
# on someone else's machine/cluster:
#   CONDA_ENV        conda env to activate (default: scaling)
#   CONDA_SH         path to conda's profile.d/conda.sh (default: autodetected)
#   SCAN_RESULT_DIR  results root, read by src/benchmarks/configs.py
#                    (default: the cluster path in that file)
# SCAN_RESULT_DIR is not set here on purpose -- it is simply inherited by the
# python processes below, so `SCAN_RESULT_DIR=/my/path scripts/foo.sh` works.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-scaling}"
if [[ -z "${CONDA_SH:-}" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_SH="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
    else
        CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
fi

if [[ -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
else
    echo "warning: no conda.sh at $CONDA_SH -- using the current interpreter" >&2
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"
