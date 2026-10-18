#!/usr/bin/env bash
set -euo pipefail
cd /home/js/robot/VLA/PressButton/PressB
exec env -u PYTHONPATH -u PYTHONHOME \
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  .venv-astribot-inference/bin/python scripts/astribot_policy_server.py \
  --run models/astribot_head_right_joint8_ft_20261018 --device cuda:1 --port 8794
