#!/usr/bin/env bash
# GPU 0 is the available rendering device on this workstation.
set -euo pipefail
PRESSB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${PRESSB_CPU_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# Keep Warp's CUDA enumeration on the physical rendering GPU.
export CUDA_VISIBLE_DEVICES="${PRESSB_CUDA_DEVICES:-0}"
exec bash "$PRESSB_ROOT/scripts/run.sh" --gpu 0 "$@"
