#!/usr/bin/env bash
#SBATCH --job-name=s1b-real-joint10
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --qos=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc2hdd/home/jluo628/robot/VLA/PressButton/logs/s1b-real-%j.log
set -euo pipefail
PROJECT=/hpc2hdd/home/jluo628/robot/VLA/PressButton
ENV_ROOT=/hpc2hdd/home/jluo628/.conda/envs/vlajepa-piper
export PATH="$ENV_ROOT/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJECT"
# Keep the running job immutable if the working branch is updated later.
SNAPSHOT="$PROJECT/outputs/s1b_real_jobs/$SLURM_JOB_ID"
mkdir -p "$SNAPSHOT"
git rev-parse HEAD > "$SNAPSHOT/git_commit.txt"
cp -a training/s1b_real "$SNAPSHOT/code"
cp configs/vlajepa_s1b_real_joint10_hpc2.yaml "$SNAPSHOT/config.yaml"
export PYTHONPATH="$SNAPSHOT/code/runtime"
export S1B_PROJECT="$PROJECT"

python "$SNAPSHOT/code/prepare.py" --dataset "$PROJECT/data/astribot_s1b_real_joint10_v1" --output "$PROJECT/training/s1b_real/dataset_contract.json"
python -m pytest -q "$SNAPSHOT/code/tests"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
SMOKE=s1b_real_joint10_8000_v1_smoke
RUN=s1b_real_joint10_8000_v1
CONFIG="$SNAPSHOT/config.yaml"
# Freeze the hashed contract path into this job's config.
python - "$CONFIG" "$SNAPSHOT/code/dataset_contract.json" <<'PY'
from omegaconf import OmegaConf
import sys
c=OmegaConf.load(sys.argv[1]);c.datasets.vla_data.contract_path=sys.argv[2];OmegaConf.save(c,sys.argv[1])
PY
python - "$SNAPSHOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);hashes={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob('*')) if f.is_file() and '__pycache__' not in str(f)}
(p/'source_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
PY
printf 'PHASE: smoke fresh 2 steps\n'
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m starVLA.training.train_s1b_real --config "$CONFIG" --run-id "$SMOKE" --stop-after-step 2
# Preserve the 8->10 loading proof before the resume report replaces it.
cp "$PROJECT/runs/$SMOKE/pretrained_loading.json" "$PROJECT/runs/$SMOKE/transfer_loading.json"
printf 'PHASE: smoke resume to step 3\n'
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m starVLA.training.train_s1b_real --config "$CONFIG" --run-id "$SMOKE" \
    --resume "$PROJECT/runs/$SMOKE/checkpoints/last" --stop-after-step 3
python "$SNAPSHOT/code/check_smoke.py" "$PROJECT/runs/$SMOKE"
printf 'PHASE: formal training 8000 steps\n'
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m starVLA.training.train_s1b_real --config "$CONFIG" --run-id "$RUN"
printf 'PHASE: completed\n'
