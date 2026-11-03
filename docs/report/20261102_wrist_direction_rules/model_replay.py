"""Saved-request prompt comparison; model HTTP only, no robot client/executor."""
import ast
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from diagnose_localized_feedback import model_tunnel, digest, image_hashes
from pressb.agent_loop.common import atomic_json, request


def run(with_current_rule=False):
    run_dir = ROOT / 'outputs/agent_web/20261102_175210_9567cf9c/agent'
    output = Path(__file__).resolve().parent / ('replay_with_current_rule' if with_current_rule else 'replay')
    output.mkdir(exist_ok=False)
    code = ROOT / 'src/pressb/agent_loop/localized_conversation.py'
    prompt = ROOT / 'docs/astribot/planar_control/agent_system_prompt_localized.txt'
    tree = ast.parse(code.read_text())
    guidance = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                    and t.id == 'GUIDANCE' for t in n.targets))
    move_description = next(n.value.value for n in ast.walk(tree)
                            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                            and isinstance(n.value.value, str)
                            and n.value.value.startswith('Move the hand by chassis-frame XYZ'))
    system = prompt.read_text() + '\n' + guidance
    (output / 'system_used.txt').write_text(system)
    report = dict(robot_connected=False, robot_commands_sent=0,
                  method='Replace system and move description; preserve all archived images, history and current estimates.',
                  current_direction_hint_added=with_current_rule,
                  source_hashes={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in (code, prompt)},
                  limitation='Two saved states with left/above targets; no new motion, no closed-loop or full-direction accuracy claim.',
                  results=[])
    atomic_json(output / 'report.json', report)
    with model_tunnel('Thor', 18116, None) as base:
        atomic_json(output / 'models.json', request(base, '/models'))
        for folder in ('0001', '0006'):
            source = run_dir / 'rounds' / folder / 'request.json'
            original_bytes = source.read_bytes()
            original = json.loads(original_bytes)
            body = copy.deepcopy(original)
            assert body['messages'][0]['role'] == 'system'
            body['messages'][0]['content'] = system
            for tool in body['tools']:
                if tool['function']['name'] == 'propose_move':
                    tool['function']['description'] = move_description
            if with_current_rule:
                from pressb.agent_loop.wrist_direction_rules import build_hint
                location=json.loads((source.parent/'localization_current.json').read_text())
                hint=build_hint(location)
                atomic_json(output/folder/'wrist_direction_rule.json',hint)
                body['messages'].append(dict(role='user',content=[dict(type='text',text=json.dumps(
                    dict(context_type='CURRENT_WRIST_ALIGNMENT_DIRECTION_RULE',**hint),ensure_ascii=False))]))
            assert image_hashes(body) == image_hashes(original)
            directory = output / folder
            atomic_json(directory / 'request.json', body)
            row = dict(round=int(folder) + 1, source=str(source.relative_to(ROOT)),
                       source_sha256=digest(original_bytes), image_count=len(image_hashes(body)),
                       image_hashes_unchanged=True,
                       original_proposal=json.loads((source.parent / 'proposal.json').read_text()))
            started = time.monotonic()
            try:
                response = request(base, '/chat/completions', body, timeout=180)
                atomic_json(directory / 'response.json', response)
                choice = response['choices'][0]
                row.update(usage=response.get('usage'), finish_reason=choice.get('finish_reason'),
                           calls=[dict(name=c['function']['name'], arguments=json.loads(c['function']['arguments']))
                                  for c in choice['message'].get('tool_calls', [])])
                calls = row['calls']
                xyz = calls[0]['arguments'].get('xyz_mm') if len(calls) == 1 else None
                row['upper_left_signs_followed'] = bool(len(calls) == 1
                    and calls[0]['name'] == 'propose_move' and isinstance(xyz, list) and len(xyz) == 3
                    and xyz[0] == 0 and xyz[1] >= 0 and xyz[2] > 0)
            except Exception as exc:
                row['error'] = str(exc)
            row.update(elapsed_s=round(time.monotonic() - started, 3),
                       source_unchanged=source.read_bytes() == original_bytes)
            report['results'].append(row)
            atomic_json(output / 'report.json', report)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    report['completed'] = True
    atomic_json(output / 'report.json', report)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--with-current-rule',action='store_true')
    args=parser.parse_args()
    run(args.with_current_rule)
