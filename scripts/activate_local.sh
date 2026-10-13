#!/usr/bin/env bash
# Source this file to activate the locally deployed simulation environment.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Run: source scripts/activate_local.sh' >&2
    exit 1
fi
PRESSB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/js/miniforge3/etc/profile.d/conda.sh
conda activate "$PRESSB_ROOT/.conda/envs/pressb"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS="${PRESSB_CPU_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export CUDA_VISIBLE_DEVICES="${PRESSB_CUDA_DEVICES:-0}"
export TMPDIR="$PRESSB_ROOT/.cache/tmp"
mkdir -p "$TMPDIR"
