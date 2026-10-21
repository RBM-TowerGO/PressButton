#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mode=${1:-test}
output=${2:-outputs/astribot_closedloop_20261018/test40}
case "$mode" in
  smoke) options=(--smoke --max-seconds 2) ;;
  test) options=(--max-seconds 40) ;;
  *) echo 'Usage: run_astribot_closedloop.sh smoke|test NEW_OUTPUT_DIRECTORY' >&2; exit 2 ;;
esac
exec env -u PYTHONPATH -u PYTHONHOME \
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1 \
  OMNI_KIT_ACCEPT_EULA=YES OMNICLIENT_HUB_MODE=disabled \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TMPDIR="$PROJECT_ROOT/.cache/sim_tmp" \
  HF_HOME="$PROJECT_ROOT/.cache/hf" \
  PATH="$PROJECT_ROOT/.conda/envs/pressb/bin:$PROJECT_ROOT/.conda/envs/lerobot/bin:$PATH" \
  .conda/envs/pressb/bin/python scripts/eval_astribot_closedloop.py \
  --model-run models/astribot_head_right_joint8_ft_20261018 \
  --output "$output" --execute-horizon 7 "${options[@]}"
