"""Conditional scalar feedback tests; no robot or image-to-XYZ calibration."""
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from diagnose_localized_feedback import model_tunnel, archive
from pressb.agent_loop.common import atomic_json, request

source = ROOT/'outputs/agent_web/20261102_162741_4885e81e/agent/rounds/0014/request.json'
original = json.loads(source.read_text())
output = Path(__file__).parent/'empirical_mapping'
output.mkdir(exist_ok=False)
cases = [
    dict(name='recorded_estimate_response', before_dx_px=-256., after_dx_px=-262.4, expected_y_mm=-10),
    dict(name='synthetic_reversed_response', before_dx_px=-256., after_dx_px=-249.6, expected_y_mm=10),
]
report = dict(model_only=True, robot_connected=False, robot_commands_sent=0,
    limitation=('Assumes exact scalar observations, locally linear response, fixed camera conventions '
                'and no cross-axis coupling. These assumptions are not established for the robot. '
                'The first numbers come from normalized historical MODEL estimates, not human ground truth; '
                'the second pair is an explicitly synthetic sign-reversal control.'), results=[])
with model_tunnel('Thor', 18113, None) as base:
    for case in cases:
        body = copy.deepcopy(original)
        body['messages'] = [copy.deepcopy(original['messages'][0]), dict(role='user', content=(
            'Read-only scalar feedback reasoning exercise, not a real robot instruction. '
            'For this exercise ONLY, assume the following image error estimates are exact, '
            'the local mapping from robot Y to image error is linear, and other axes do not matter. '
            'Image horizontal error dx = button center x minus marked fingertip x; desired dx is zero. '
            f"A robot-hand +Y 20 mm move changed dx from {case['before_dx_px']} px to {case['after_dx_px']} px. "
            'Which sign of robot-hand Y should reduce the remaining absolute horizontal error? '
            'Encode that sign with magnitude exactly 10 mm using propose_move [0,Y,0]. '
            'Explain the observed response sign in one sentence. Do not infer a camera transform from image left/right alone. '
            'Nothing will be executed. Scenario: '+case['name']))]
        directory = output/'tests'/case['name']
        directory.mkdir(parents=True)
        archive(body, directory, output)
        started = time.monotonic()
        response = request(base, '/chat/completions', body, timeout=120)
        atomic_json(directory/'response.json', response)
        call = response['choices'][0]['message']['tool_calls'][0]['function']
        arguments = json.loads(call['arguments'])
        report['results'].append(dict(**case, name_returned=call['name'], arguments=arguments,
            correct=(call['name']=='propose_move' and arguments.get('xyz_mm')==[0,case['expected_y_mm'],0]),
            usage=response.get('usage'), elapsed_s=time.monotonic()-started))
        atomic_json(output/'report.json', report)
        print(json.dumps(report['results'][-1],ensure_ascii=False),flush=True)
