#!/usr/bin/env python3
"""Replay recorded alignment decisions using only the model; never create a robot client."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pressb.agent_loop.client import propose
from pressb.agent_loop.common import atomic_json
from pressb.agent_loop.conversation import image_count
from pressb.agent_loop.demonstration import Demonstration
from pressb.agent_loop.short_loop import ShortConversation
from s1b_agent import StopControl, tunnel


class Recording:
    def __init__(self, root):
        self.root = Path(root) / 'agent'
        self.hashes = {}

    def read(self, path):
        path = Path(path)
        raw = path.read_bytes()
        self.hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return raw

    def data(self, path):
        return json.loads(self.read(path))

    def round(self, number, name):
        # Public round numbers are one-based, on-disk directories are zero-based.
        return self.data(self.root / 'rounds' / f'{number - 1:04d}' / name)

    def observation(self, oid):
        path = self.root / 'observations' / oid
        return (self.data(path / 'manifest.json'), self.read(path / 'head.jpg'),
                self.read(path / 'right.jpg'))


def restored_conversation(recording, number, session, prompt, demo, folder, base, cancel):
    """Replay original judgments/results without validating or executing old proposals.

    Old decisions may fail today's validator. They nevertheless happened and must
    stay in the factual command history. Every appended result carries the actual
    recorded command; no new model proposal is ever appended as an executed action.
    """
    first = recording.round(1, 'decision_context.json')['observation_id']
    obs, head, wrist = recording.observation(first)
    conversation = ShortConversation('按下电梯外呼向下按钮并撤回', obs, head, wrist,
                                     session, prompt, demo=demo)
    # Restoring old state requires no model tokenization. Reset before the real
    # decision so its complete actual input is counted normally.
    conversation.estimated_tokens = 0
    for previous in range(1, number):
        if cancel.is_set():
            raise InterruptedError('operator_stopped')
        oid = recording.round(previous, 'decision_context.json')['observation_id']
        if conversation.current_id != oid:
            raise ValueError(f'recorded_observation_chain_mismatch:round_{previous}')
        assessment = recording.round(previous, 'perception/assessment.json')
        if assessment['observation_id'] != oid:
            raise ValueError(f'recorded_assessment_binding_mismatch:round_{previous}')
        # The selected recording has no lamp-on judgments. Reject a different
        # recording here rather than silently making extra verification calls.
        if assessment['lamp'] == 'on':
            raise ValueError('history_restoration_requires_no_activation_verification')
        conversation.assessments[oid] = copy.deepcopy(assessment)
        conversation.prepare_decision(base, folder / 'restored_history' / f'{previous:02d}', cancel)
        original = recording.round(previous, 'proposal.json')
        raw = recording.round(previous, 'assistant.json')
        if 'choices' in raw:
            raw = raw['choices'][0]['message']
        conversation.pending = copy.deepcopy(raw)
        result = recording.round(previous, 'result.json')
        if result.get('status') != 'completed':
            raise ValueError(f'history_contains_unfinished_command:round_{previous}')
        result = copy.deepcopy(result)
        result['replay'] = dict(
            note='Historical result only. No command was sent by this validation.',
            recorded_actual_command={k: v for k, v in result['request'].items()
                                     if k in ('type', 'xyz_mm', 'head_deg', 'key')})
        after = result['capture']['observation_id']
        obs, head, wrist = recording.observation(after)
        conversation.append_result(original, result, obs, head, wrist)
    expected = recording.round(number, 'decision_context.json')['observation_id']
    if conversation.current_id != expected:
        raise ValueError(f'recorded_current_observation_mismatch:round_{number}')
    conversation.estimated_tokens = None
    return conversation


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    recording = Recording(args.recording)
    session = recording.data(recording.root / 'session.json')
    prompt = recording.read(ROOT / 'docs/astribot/planar_control/agent_system_prompt.txt').decode()
    demo = Demonstration(args.demo).select_activation_tail(3)
    demo.save(output)
    recording.hashes.update({str((demo.root / name).resolve()): digest
                             for name, digest in demo.source_sha256.items()})
    numbers = args.rounds or ([4, 5, 6] if args.full_perception else [4, 5, 10, 13, 15])
    report = dict(robot_connected=False, robot_commands_sent=0, thinking=False, failure_case=None,
        mode='full_perception_and_action' if args.full_perception else 'action_with_recorded_perception',
        dry_run=args.dry_run, rounds=numbers, selected_demo_steps=demo.selection, results=[],
        limitations=[
            'All input frames are from a saved run, not a live camera.',
            'Historical judgments are model judgments, not human-confirmed geometry.',
            'Later recorded images reflect original commands, never these new suggestions.',
            'No suggestion is executed; this cannot establish task success or physical improvement.',
            'Action-only mode reuses the original current assessment; full mode recomputes it.',
        ])
    for relative in ('src/pressb/agent_loop/short_loop.py', 'src/pressb/agent_loop/client.py',
                     'src/pressb/agent_loop/grounding.py', 'scripts/validate_agent_alignment_fix.py',
                     'src/pressb/agent_loop/decision_validation.py',
                     'src/pressb/agent_loop/localization.py',
                     'src/pressb/agent_loop/control_policy.py'):
        if (ROOT / relative).exists():
            recording.read(ROOT / relative)

    def save():
        atomic_json(output / 'report.json', report)

    try:
        host = None if args.model_base or args.dry_run else args.thor_host
        with tunnel(host, args.port), StopControl() as control:
            base = args.model_base or f'http://127.0.0.1:{args.port}/v1'
            for number in numbers:
                folder = output / f'round_{number:02d}'
                folder.mkdir()
                start = time.monotonic()
                row = dict(round=number)
                try:
                    c = restored_conversation(recording, number, session, prompt, demo, folder,
                                              base, control.cancel)
                    original_assessment = recording.round(number, 'perception/assessment.json')
                    original = dict(proposal=recording.round(number, 'proposal.json'),
                                    assessment=original_assessment,
                                    action_metrics=recording.round(number, 'action_metrics.json'),
                                    perception_metrics=recording.round(number, 'perception/metrics.json'))
                    row.update(observation_id=c.current_id, original=original)
                    atomic_json(folder / 'original.json', original)
                    if not args.full_perception or args.dry_run:
                        c.assessments[c.current_id] = copy.deepcopy(original_assessment)
                        atomic_json(folder / 'recorded_current_assessment.json', original_assessment)
                    if args.dry_run:
                        c.estimated_tokens = 0
                    body = control.call(c.prepare_decision, base, folder, control.cancel)
                    row.update(current_assessment=copy.deepcopy(c.assessments[c.current_id]),
                               current_assessment_source='new_model' if args.full_perception and not args.dry_run
                                                         else 'saved_model_judgment',
                               action_images=image_count(body['messages']),
                               restored_commands=len(c.turns), control_memory=copy.deepcopy(c.memory))
                    if args.dry_run:
                        atomic_json(folder / 'request.json', body)
                        row['status'] = 'dry_run_no_model_request'
                    else:
                        obs, _, _ = recording.observation(c.current_id)
                        proposal = control.call(propose, base, body, obs, session, folder,
                                                cancel=control.cancel, conversation=c)
                        row.update(status='proposal_not_executed', proposal=proposal)
                except InterruptedError:
                    row.update(status='interrupted', error='operator_stopped')
                    raise
                except Exception as exc:
                    row.update(status='error', error_type=type(exc).__name__, error=str(exc))
                finally:
                    row['elapsed_s'] = round(time.monotonic() - start, 3)
                    for key, path in (('action_metrics', folder / 'action_metrics.json'),
                                      ('perception_metrics', folder / 'perception/metrics.json')):
                        if path.exists():
                            row[key] = json.loads(path.read_text())
                    report['results'].append(row)
                    save()
                    print(json.dumps({k: v for k, v in row.items() if k != 'control_memory'},
                                     ensure_ascii=False), flush=True)
    finally:
        report['source_integrity'] = dict(sha256=recording.hashes, unchanged=all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
            for path, digest in recording.hashes.items()))
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--recording', default=str(ROOT / 'outputs/agent_web/20261101_234200_c94a9727'))
    parser.add_argument('--demo', default=str(ROOT / 'outputs/keyboard_sessions/20261101_182424_352b5047'))
    parser.add_argument('--full-perception', action='store_true', help='Reinspect current frames; defaults to rounds 4/5/6')
    parser.add_argument('--rounds', type=int, nargs='+', help='One-based recorded decision rounds')
    parser.add_argument('--dry-run', action='store_true', help='Build and inspect restored contexts without any network call')
    parser.add_argument('--thor-host', default='Thor')
    parser.add_argument('--port', type=int, default=18006)
    parser.add_argument('--model-base', help='Existing /v1 model URL; disables the SSH tunnel')
    arguments = parser.parse_args()
    if arguments.rounds and any(number < 1 for number in arguments.rounds):
        parser.error('--rounds uses positive one-based indices')
    run(arguments)
