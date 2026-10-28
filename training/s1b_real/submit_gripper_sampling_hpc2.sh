#!/usr/bin/env bash
#SBATCH --job-name=s1b-gripper-4500-5500
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --qos=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc2hdd/home/jluo628/robot/VLA/PressButton/logs/s1b-gripper-%j.log
set -euo pipefail
PROJECT=${S1B_PROJECT:-/hpc2hdd/home/jluo628/robot/VLA/PressButton}
ENV_ROOT=${S1B_QWEN35_ENV:-$PROJECT/.venv-qwen3-5}
: "${S1B_GRIPPER_BUNDLE:?Set S1B_GRIPPER_BUNDLE to the immutable uploaded bundle}"
export PATH="$ENV_ROOT/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export S1B_PROJECT="$PROJECT"
cd "$PROJECT"
SNAPSHOT="$PROJECT/outputs/s1b_gripper_jobs/$SLURM_JOB_ID"
mkdir -p "$SNAPSHOT"
cp -a "$S1B_GRIPPER_BUNDLE/." "$SNAPSHOT/"
export PYTHONPATH="$SNAPSHOT/training/s1b_real/runtime"
CODE="$SNAPSHOT/training/s1b_real"
CONFIG="$SNAPSHOT/configs/vlajepa_s1b_gripper_sampling_4500_5500_hpc2.yaml"
PARENT="$PROJECT/runs/s1b_real_joint10_mixed_qwen3_5_4b_native_8gpu_6000_v2/checkpoints/step_004500"
python - "$SNAPSHOT" "$PARENT" <<'PY'
import json,sys
from pathlib import Path
from starVLA.training.train_s1b_real import sha256
snapshot,parent=map(Path,sys.argv[1:])
for name,digest in json.loads((snapshot/'bundle_manifest.json').read_text()).items():
    assert sha256(snapshot/name)==digest, name
manifest=json.loads((parent/'manifest.json').read_text())
assert manifest['complete'] and manifest['step']==4500
assert manifest['model_sha256']=='f29f20924bc6614c4a20b7f6dbe62aea329c8ec935953f33eedc710ad1248699'
for name in ('model','training'):
    assert sha256(parent/(name+'.pt'))==manifest[name+'_sha256'],name
print('PREFLIGHT: bundle and parent checkpoint verified',flush=True)
PY
cp "$CONFIG" "$SNAPSHOT/resolved_config.yaml"
CONFIG="$SNAPSHOT/resolved_config.yaml"
python "$CODE/prepare.py" --dataset "$PROJECT/data/astribot_s1b_real_joint10_mixed_v2" --output "$SNAPSHOT/dataset_contract.json"
python - "$CONFIG" "$SNAPSHOT/dataset_contract.json" "$PARENT" <<'PY'
import json,sys
from pathlib import Path
from omegaconf import OmegaConf
from starVLA.training.train_s1b_real import training_contract,validate_continuation_contract,sha256
config,contract,parent=sys.argv[1:];c=OmegaConf.load(config)
c.datasets.vla_data.contract_path=contract
old=json.loads((Path(parent).parents[1]/'training_contract.json').read_text())['contract']
validate_continuation_contract(old,training_contract(OmegaConf.to_container(c,resolve=True))[0])
expected=json.loads((Path(parent)/'manifest.json').read_text())['data_contract_sha256']
assert sha256(contract)==expected, 'Dataset contract changed'
OmegaConf.save(c,config)
PY
python -m pytest -q "$CODE/tests"
python -m pip freeze > "$SNAPSHOT/environment.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
RUN=$(python -c 'from omegaconf import OmegaConf; import sys; print(OmegaConf.load(sys.argv[1]).run_id)' "$CONFIG")
SMOKE="${RUN}_smoke_${SLURM_JOB_ID}"
printf 'PHASE: continuation smoke 4500 to 4502\n'
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m starVLA.training.train_s1b_real --config "$CONFIG" --run-id "$SMOKE" \
    --continue-from "$PARENT" --stop-after-step 4502 --set trainer.logging_frequency=1
cp "$PROJECT/runs/$SMOKE/initialization.json" "$PROJECT/runs/$SMOKE/initialization_4500.json"
printf 'PHASE: smoke resume 4502 to 4503\n'
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m starVLA.training.train_s1b_real --config "$PROJECT/runs/$SMOKE/config.yaml" \
    --resume "$PROJECT/runs/$SMOKE/checkpoints/last" --stop-after-step 4503
python "$CODE/check_gripper_smoke.py" "$PROJECT/runs/$SMOKE"
printf 'PHASE: formal continuation 4500 to 5500\n'
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m starVLA.training.train_s1b_real --config "$CONFIG" --run-id "$RUN" --continue-from "$PARENT"
printf 'PHASE: completed\n'
