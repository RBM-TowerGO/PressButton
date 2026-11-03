#!/usr/bin/env python3
"""Replay the wrist overlap check on saved images; never connect to a robot.

The default three frames are visibly non-overlapping. They test rejection only,
not recognition accuracy or safe contact. Exact model requests are archived by
the production read-only inspector; original run files remain untouched.
"""
import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pressb.agent_loop.common import atomic_json, request
from pressb.agent_loop import wrist_overlap_guard


DEFAULT_RUN = ROOT / 'outputs/agent_web/20261102_172331_124cf810/agent'
DEFAULT_ROUNDS = (2, 5, 11)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def image_part(raw):
    return dict(type='image_url', image_url=dict(
        url='data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii')))


@contextmanager
def model_endpoint(base, host, port):
    if base:
        yield base.rstrip('/')
        return
    tunnel = subprocess.Popen(['ssh', '-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
        '-L', f'127.0.0.1:{port}:127.0.0.1:8000', host])
    try:
        endpoint = f'http://127.0.0.1:{port}/v1'
        for _ in range(75):
            if tunnel.poll() is not None:
                raise RuntimeError('model_tunnel_failed')
            try:
                request(endpoint, '/models', timeout=1)
                break
            except OSError:
                time.sleep(.2)
        else:
            raise TimeoutError('model_tunnel_timeout')
        yield endpoint
    finally:
        tunnel.terminate()
        try:
            tunnel.wait(3)
        except subprocess.TimeoutExpired:
            tunnel.kill()
            tunnel.wait()


def saved_observation(run, round_number, output):
    """Resolve each decision's explicit observation binding, never infer from history."""
    source = run / 'rounds' / f'{round_number - 1:04d}' / 'decision_context.json'
    context_raw = source.read_bytes()
    context = json.loads(context_raw)
    oid = context['observation_id']
    directory = run / 'observations' / oid
    original = {source: context_raw}
    original.update({directory / name: (directory / name).read_bytes()
                     for name in ('manifest.json', 'head.jpg', 'right.jpg')})
    manifest = json.loads(original[directory / 'manifest.json'])
    if manifest.get('observation_id') != oid:
        raise ValueError('saved_observation_binding_mismatch')
    copied = output / 'observations' / oid
    copied.mkdir(parents=True)
    for path, raw in original.items():
        (copied / path.name).write_bytes(raw)
    # Production expects a dual-view observation container; the new inspector
    # independently extracts only the CURRENT wrist image. No case or old action
    # reason is included in this message.
    current = dict(role='user', content=[dict(type='text', text=json.dumps(dict(
        context_type='LIVE_OBSERVATION', observation_id=oid))),
        dict(type='text', text='CURRENT HEAD original saved image'),
        image_part(original[directory / 'head.jpg']),
        dict(type='text', text='CURRENT RIGHT WRIST original saved image'),
        image_part(original[directory / 'right.jpg'])])
    evidence = dict(round_number=round_number, observation_id=oid,
        decision_context_path=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        source_files=[dict(path=str(path), sha256=sha256(raw), bytes=len(raw))
                      for path, raw in original.items()])
    return current, evidence, original


def run(args):
    source_run = Path(args.run).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'script_used.py').write_bytes(Path(__file__).read_bytes())
    module_path = Path(wrist_overlap_guard.__file__)
    (output / 'wrist_overlap_guard_used.py').write_bytes(module_path.read_bytes())
    report = dict(format='pressb-wrist-overlap-saved-probe-v1', model_only=True,
        robot_client_created=False, robot_commands_sent=0, source_run=str(source_run),
        policy=wrist_overlap_guard.POLICY, forward_distance_cap_mm=None,
        coordinate_units='Explicitly declared by model; production normalization only, no scale guessing.',
        expected_all_blocked=args.expect_blocked, results=[], completed=False,
        limitations=['Saved frames do not show results of any new command.',
            'Default frames test non-overlap rejection only. No real overlapping arrow frame is available here.',
            'A rejected candidate is not evidence of exact fingertip localization or physical contact safety.',
            'Passing overlap has no positive-X distance cap. These probes send no movement.',
            'Timings describe sequential model calls; no speed or task success claim.'])
    originals = {}
    try:
        with model_endpoint(args.model_base, args.thor_host, args.port) as base:
            for number in args.rounds:
                current, source, raw_files = saved_observation(source_run, number, output)
                originals.update(raw_files)
                oid = source['observation_id']
                body = wrist_overlap_guard.build_request(args.goal, current)
                image_parts = [part for message in body['messages']
                    if isinstance(message.get('content'), list) for part in message['content']
                    if part.get('type') == 'image_url']
                if len(image_parts) != 1:
                    raise ValueError('wrist_overlap_probe_requires_one_wrist_image')
                wrist_raw = raw_files[source_run / 'observations' / oid / 'right.jpg']
                actual_image = base64.b64decode(image_parts[0]['image_url']['url'].split(',', 1)[1], validate=True)
                if actual_image != wrist_raw:
                    raise ValueError('model_image_does_not_equal_saved_wrist')
                directory = output / 'probes' / f'round_{number:02d}'
                assessment = wrist_overlap_guard.inspect(base, body, oid, directory)
                decision = wrist_overlap_guard.evaluate(assessment, oid)
                atomic_json(directory / 'decision.json', decision)
                metrics = json.loads((directory / 'metrics.json').read_text())
                entry = dict(**source, assessment=assessment, decision=decision, metrics=metrics,
                    image_count=len(image_parts), input_matches_original=True,
                    wrist_image_sha256=sha256(wrist_raw),
                    expected_allowed=False if args.expect_blocked else None,
                    false_positive=(decision.get('allowed') is True) if args.expect_blocked else None)
                report['results'].append(entry)
                atomic_json(output / 'report.json', report)
                print(json.dumps(dict(round=number, observation_id=oid, decision=decision,
                                      metrics=metrics), ensure_ascii=False), flush=True)
        report['completed'] = True
    finally:
        report['source_unchanged'] = all(path.read_bytes() == raw for path, raw in originals.items())
        report['false_positive_count'] = sum(item.get('false_positive') is True for item in report['results'])
        atomic_json(output / 'report.json', report)
    if not report['source_unchanged']:
        raise RuntimeError('saved_source_changed_during_probe')
    if report['false_positive_count']:
        raise RuntimeError('nonoverlap_frame_incorrectly_authorized: see complete archived report')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=str(DEFAULT_RUN))
    parser.add_argument('--rounds', nargs='+', type=int, default=list(DEFAULT_ROUNDS),
                        help='One-based rounds; exact observation comes from decision_context.json.')
    parser.add_argument('--output', required=True, help='New artifact directory; never overwrite an earlier replay.')
    parser.add_argument('--goal', default='按下电梯外呼向下按钮并撤回')
    parser.add_argument('--model-base')
    parser.add_argument('--thor-host', default='Thor')
    parser.add_argument('--port', type=int, default=18115)
    parser.add_argument('--expect-blocked', action='store_true',
                        help='Record independent human review that all selected saved frames lack overlap.')
    args = parser.parse_args()
    if any(number < 1 for number in args.rounds) or len(set(args.rounds)) != len(args.rounds):
        parser.error('--rounds must be distinct positive one-based round numbers')
    run(args)
