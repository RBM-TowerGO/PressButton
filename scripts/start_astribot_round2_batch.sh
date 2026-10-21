#!/usr/bin/env bash
# Two independent seeds share a frozen policy server; each gets 50k + val-40.
set -uo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/activate_local.sh
export CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1
POLICY_PID="${1:-}"
RESUME_FLAG=()
if [[ "${2:-}" == --resume ]]; then RESUME_FLAG=(--resume-job); fi
if [[ "${2:-}" == --adopt ]]; then RESUME_FLAG=(--adopt-running-job); fi
RUN_ROOT="outputs/astribot_online_rl"
for seed in 20261021 20261022; do
    if [[ -e "$RUN_ROOT/residual_50k_shaped_20261020_seed$seed" && ${#RESUME_FLAG[@]} -eq 0 ]]; then
        echo "Output already exists; preserve evidence: seed $seed" >&2
        exit 1
    fi
done
python scripts/run_astribot_round2_job.py \
    --output "$RUN_ROOT/residual_50k_shaped_20261020_seed20261021" \
    --physics-backend gpu --seed 20261021 \
    --doc docs/astribot_round2_20261020.md "${RESUME_FLAG[@]}" >> logs/round2-seed20261021-supervisor.log 2>&1 &
PRIMARY_PID=$!
python scripts/run_astribot_round2_job.py \
    --output "$RUN_ROOT/residual_50k_shaped_20261020_seed20261022" \
    --physics-backend gpu --seed 20261022 \
    --doc docs/astribot_round2_20261020_seed20261022.md "${RESUME_FLAG[@]}" >> logs/round2-seed20261022-supervisor.log 2>&1 &
REPLICA_PID=$!
printf '%s\n' "$PRIMARY_PID" > logs/round2-primary-supervisor.pid
printf '%s\n' "$REPLICA_PID" > logs/round2-replica-supervisor.pid
trap 'kill -TERM "$PRIMARY_PID" "$REPLICA_PID" 2>/dev/null || true' TERM INT
wait "$PRIMARY_PID"
PRIMARY_RESULT=$?
wait "$REPLICA_PID"
REPLICA_RESULT=$?
python - "$PRIMARY_RESULT" "$REPLICA_RESULT" "$POLICY_PID" <<'PY'
from datetime import datetime, timezone, timedelta
import json, os
from pathlib import Path
import signal, sys
root = Path('outputs/astribot_online_rl')
rows = []
for seed, result in zip((20261021, 20261022), map(int, sys.argv[1:3])):
    output = root / f'residual_50k_shaped_20261020_seed{seed}'
    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}
    rows.append(dict(seed=seed, supervisor_exitcode=result, job=read(output/'job.json'),
                     evaluation=read(output/'run/eval/summary.json'),
                     comparison=read(output/'run/eval/paired_comparison.json')))
report = dict(finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), runs=rows)
(root/'round2_batch_summary_20261020.json').write_text(json.dumps(report, indent=2)+'\n')
doc = Path('docs/astribot_round2_20261020.md')
lines = ['\n\n## 双种子最终汇总\n', '| seed | 状态 | val 成功 | 基线 |', '|---|---|---:|---:|']
for row in rows:
    lines.append(f'| {row["seed"]} | {row["job"].get("phase", "supervisor failed")} | '
                 f'{row["evaluation"].get("successes", "—")}/40 | 23/40 |')
with doc.open('a') as stream:
    stream.write('\n'.join(lines)+'\n')
if sys.argv[3]:
    pid = int(sys.argv[3])
    try:
        if b'scripts/astribot_policy_server.py' in Path(f'/proc/{pid}/cmdline').read_bytes():
            os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
held = os.environ.get('ROUND2_HELD_BATCH_PID')
if held:
    pid = int(held)
    try:
        if b'start_astribot_round2_batch.sh' in Path(f'/proc/{pid}/cmdline').read_bytes():
            os.kill(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
PY
if (( PRIMARY_RESULT || REPLICA_RESULT )); then exit 1; fi
