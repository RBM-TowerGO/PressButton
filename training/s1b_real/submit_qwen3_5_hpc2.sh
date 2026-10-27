#!/usr/bin/env bash
#SBATCH --job-name=s1b-qwen35-joint10
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --qos=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc2hdd/home/jluo628/robot/VLA/PressButton/logs/s1b-qwen35-%j.log
set -euo pipefail
PROJECT=${S1B_PROJECT:-/hpc2hdd/home/jluo628/robot/VLA/PressButton}
ENV_ROOT=${S1B_QWEN35_ENV:-/hpc2hdd/home/jluo628/.conda/envs/vlajepa-qwen35}
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    echo "Missing Qwen3.5 environment: $ENV_ROOT" >&2
    exit 1
fi
export PATH="$ENV_ROOT/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJECT"
# Keep the running job immutable if the working branch is updated later.
SNAPSHOT="$PROJECT/outputs/s1b_qwen35_jobs/$SLURM_JOB_ID"
mkdir -p "$SNAPSHOT"
git rev-parse HEAD > "$SNAPSHOT/git_commit.txt"
cp -a training/s1b_real "$SNAPSHOT/code"
cp "${S1B_QWEN35_CONFIG:-configs/vlajepa_s1b_real_joint10_qwen3_5_hpc2.yaml}" "$SNAPSHOT/config.yaml"
export PYTHONPATH="$SNAPSHOT/code/runtime"
export S1B_PROJECT="$PROJECT"

DATASET=$(python -c 'from omegaconf import OmegaConf; import sys; print(OmegaConf.load(sys.argv[1]).datasets.vla_data.data_root_dir)' "$SNAPSHOT/config.yaml")
python "$SNAPSHOT/code/prepare.py" --dataset "$DATASET" --output "$SNAPSHOT/dataset_contract.json"
python -m pytest -q "$SNAPSHOT/code/tests"
python -m pip freeze > "$SNAPSHOT/environment.txt"
python -c "from transformers import Qwen3_5ForConditionalGeneration; import torch; print(torch.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
RUN=$(python -c 'from omegaconf import OmegaConf; import sys; print(OmegaConf.load(sys.argv[1]).run_id)' "$SNAPSHOT/config.yaml")
SMOKE=${RUN}_smoke_${SLURM_JOB_ID}
CONFIG="$SNAPSHOT/config.yaml"
# Freeze the hashed contract path into this job's config.
python - "$CONFIG" "$SNAPSHOT/dataset_contract.json" <<'PY'
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
# Preserve the backbone/conditioning migration report before resume.
cp "$PROJECT/runs/$SMOKE/pretrained_loading.json" "$PROJECT/runs/$SMOKE/transfer_loading.json"
printf 'PHASE: smoke resume to step 3\n'
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m starVLA.training.train_s1b_real --config "$CONFIG" --run-id "$SMOKE" \
    --resume "$PROJECT/runs/$SMOKE/checkpoints/last" --stop-after-step 3
python "$SNAPSHOT/code/check_smoke.py" "$PROJECT/runs/$SMOKE"
if [[ "${S1B_SMOKE_ONLY:-1}" == "1" ]]; then
    echo "Smoke passed; set S1B_SMOKE_ONLY=0 for a fresh formal run."
    exit 0
fi
printf 'PHASE: formal training 8000 steps\n'
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m starVLA.training.train_s1b_real --config "$CONFIG" --run-id "$RUN"
printf 'PHASE: completed\n'
