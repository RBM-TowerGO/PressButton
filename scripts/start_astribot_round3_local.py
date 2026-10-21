"""Validate the local camera/learner chain before starting round three."""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
env = os.environ.copy()
for key in ('PYTHONPATH', 'PYTHONHOME'):
    env.pop(key, None)
env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMNI_KIT_ACCEPT_EULA='YES',
           CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           MKL_NUM_THREADS='1', TMPDIR=str(ROOT / '.cache/tmp'))
python = str(ROOT / '.conda/envs/pressb/bin/python')
for attempt in range(120):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8796/health', timeout=3) as response:
            health = json.load(response)
        assert health['ready'] and health['model_sha256'] == 'db98b7c5488c666c3e10fa721dc29b25e069c41237bf8cb42b638611863eaaa8'
        break
    except Exception:
        time.sleep(5)
else:
    raise RuntimeError('Local policy server did not become ready')
smoke = ROOT / 'outputs/astribot_online_rl/round3_local_smoke_20261020'
command = [python, 'scripts/run_astribot_online_rl.py', '--model-run',
           'models/astribot_head_right_joint8_ft_20261018', '--checkpoint-step', '8000',
           '--endpoint', 'http://127.0.0.1:8796', '--output', str(smoke), '--gpu', '0',
           '--physics-backend', 'gpu', '--render-cameras', 'head', 'wrist_right',
           '--scale', *(['0.01'] * 7), '0.025', '--policy-lr', '1e-5',
           '--init-alpha', '0.005', '--no-auto-alpha', '--action-penalty', '1.0',
           '--shaping', '1.0', '--seed', '20261023', '--smoke']
with (ROOT / 'logs/round3-local-smoke.log').open('w') as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
if result.returncode or (smoke / 'infrastructure_error.json').exists():
    raise RuntimeError('Local smoke failed; inspect logs/round3-local-smoke.log')
status = json.loads((smoke / 'status.json').read_text())
assert status['transitions'] > 0 and status['updates'] > 0, status
print('Smoke passed:', status['transitions'], 'transitions,', status['updates'], 'updates', flush=True)
pid = (ROOT / 'logs/astribot-policy-round3-local.pid').read_text().strip()
os.execve(python, [python, 'scripts/run_astribot_round3_job.py', '--output',
    'outputs/astribot_online_rl/residual_100k_conservative_local_20261020_seed20261023',
    '--physics-backend', 'gpu', '--seed', '20261023', '--managed-policy-pid', pid], env)
