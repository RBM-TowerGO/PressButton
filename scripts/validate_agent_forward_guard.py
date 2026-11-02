#!/usr/bin/env python3
"""Check saved wrist images against the forward rule without creating a robot client."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pressb.agent_loop import forward_guard
from pressb.agent_loop.common import atomic_json
from pressb.agent_loop.conversation import image_count, live_message
from pressb.agent_loop.model_context import text
from s1b_agent import StopControl, tunnel


RECORDED_CASES = (
    ('round02_current', '8f60ad68c5e04f5084dda8aec27b08c3'),
    ('round10_current', '7d333ceb2b404b47805868ae37181846'),
    ('round14_after', 'e4656703f2174344b4d9e52bbb7d23de'),
)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def read(path):
        path = Path(path)
        value = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(value).hexdigest()
        return value

    report = dict(policy=forward_guard.POLICY, robot_connected=False, robot_commands_sent=0,
        action_model_called=False, thinking=False, dry_run=args.dry_run, results=[],
        limitations=[
            'All inputs are saved images. CURRENT identifies the test frame, not a live camera.',
            'Only wrist-image horizontal containment is checked; height alignment is not required.',
            'A saved frame with poor height alignment may legitimately pass this selected horizontal rule.',
            'Allowed is a program rule applied to model judgments, not proof of contact, activation or task success.',
        ])

    def save():
        atomic_json(output / 'report.json', report)

    for relative in ('scripts/validate_agent_forward_guard.py',
                     'src/pressb/agent_loop/forward_guard.py',
                     'src/pressb/agent_loop/grounding.py'):
        read(ROOT / relative)

    cases = []
    try:
        root = Path(args.recording) / 'agent' / 'observations'
        for name, oid in RECORDED_CASES:
            folder = root / oid
            manifest = json.loads(read(folder / 'manifest.json'))
            if manifest.get('observation_id') != oid:
                raise ValueError('saved_manifest_observation_mismatch:' + name)
            cases.append((name, manifest, read(folder / 'head.jpg'), read(folder / 'right.jpg'),
                          dict(kind='saved_run_observation', observation_id=oid, directory=str(folder.resolve()))))
        folder = Path(args.demo) / 'operations' / '000010' / 'before'
        oid = 'saved-demo-10-before'
        cases.append(('demo_step10_before', dict(observation_id=oid, state={}, depth={}),
                      read(folder / 'head.jpg'), read(folder / 'right.jpg'),
                      dict(kind='saved_demonstration_boundary', observation_id=oid,
                           step='000010', boundary='before', directory=str(folder.resolve()))))

        host = None if args.model_base or args.dry_run else args.thor_host
        with tunnel(host, args.port), StopControl() as control:
            base = args.model_base or f'http://127.0.0.1:{args.port}/v1'
            for name, observation, head, wrist, provenance in cases:
                folder = output / name
                folder.mkdir()
                oid = observation['observation_id']
                row = dict(case=name, observation_id=oid, source=provenance)
                started = time.monotonic()
                try:
                    current = live_message(observation, head, wrist)
                    body = forward_guard.build_request(args.goal, current)
                    body['messages'][1]['content'].insert(0, text(dict(
                        context_type='SAVED_IMAGE_REPLAY_NOT_LIVE_CAMERA',
                        note='CURRENT below is the saved test frame. No robot is connected or controlled.')))
                    pictures = [part for message in body['messages']
                                if isinstance(message.get('content'), list)
                                for part in message['content'] if part.get('type') == 'image_url']
                    if image_count(body['messages']) != 1:
                        raise ValueError('forward_replay_requires_exactly_one_wrist_image')
                    sent = base64.b64decode(pictures[0]['image_url']['url'].split(',', 1)[1])
                    if sent != wrist:
                        raise ValueError('forward_replay_image_is_not_original_wrist')
                    atomic_json(folder / 'source.json', provenance)
                    atomic_json(folder / 'request.json', body)
                    row.update(image_count=1, wrist_sha256=hashlib.sha256(wrist).hexdigest(),
                               original_wrist_bytes_verified=True)
                    if args.dry_run:
                        row['status'] = 'dry_run_no_model_request'
                    else:
                        assessment = control.call(forward_guard.inspect, base, body, oid, folder, control.cancel)
                        if control.cancel.is_set():
                            raise InterruptedError('operator_stopped')
                        decision = forward_guard.evaluate(assessment, oid)
                        atomic_json(folder / 'decision.json', decision)
                        row.update(status='read_only_check_completed', assessment=assessment,
                                   decision=decision, allowed=decision['allowed'])
                except InterruptedError:
                    row.update(status='interrupted', error='operator_stopped')
                    raise
                except Exception as exc:
                    row.update(status='error', error_type=type(exc).__name__, error=str(exc))
                finally:
                    row['elapsed_s'] = round(time.monotonic() - started, 3)
                    if (folder / 'metrics.json').exists():
                        row['metrics'] = json.loads((folder / 'metrics.json').read_text())
                    report['results'].append(row)
                    save()
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        report['source_integrity'] = dict(sha256=hashes, unchanged=all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in hashes.items()))
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--recording', default=str(ROOT / 'outputs/agent_web/20261102_000527_84b15208'))
    parser.add_argument('--demo', default=str(ROOT / 'outputs/keyboard_sessions/20261101_182424_352b5047'))
    parser.add_argument('--goal', default='按下电梯外呼向下按钮并撤回')
    parser.add_argument('--model-base', help='Existing model /v1 URL; disables automatic SSH tunneling')
    parser.add_argument('--thor-host', default='Thor')
    parser.add_argument('--port', type=int, default=18006)
    parser.add_argument('--dry-run', action='store_true', help='Build saved wrist requests without any network calls')
    run(parser.parse_args())
