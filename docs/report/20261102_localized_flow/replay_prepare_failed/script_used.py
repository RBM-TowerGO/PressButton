#!/usr/bin/env python3
"""Replay localized decisions on saved A6/A7/A8 and B5 images; model HTTP only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from pressb.agent_loop.client import propose
from pressb.agent_loop.common import atomic_json
from pressb.agent_loop.conversation import image_count
from pressb.agent_loop.demonstration import Demonstration
from pressb.agent_loop.model_context import text
from s1b_agent import LOCALIZED_PROMPT, StopControl, tunnel
from validate_agent_view_recovery import overlay


GOAL = '按下电梯外呼向下按钮并撤回'
SOURCES = {
    'A': (ROOT/'outputs/agent_web/20261102_120741_0ca36af5/agent', [6, 7, 8]),
    'B': (ROOT/'outputs/agent_web/20261102_142505_13668097/agent', [5]),
}
REFERENCE_POINTS = {
    ('A', 8): {'button': [420, 117], 'tip': [470, 280]},
    ('B', 5): {'button': [164, 60], 'tip': [470, 280]},
}
COMMAND_FIELDS = ('type', 'xyz_mm', 'head_deg', 'key')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_feedback(recorded, proposal):
    """Completed status belongs to saved command, never to the new proposal."""
    result = copy.deepcopy(recorded)
    command = {key: value for key, value in result['request'].items() if key in COMMAND_FIELDS}
    result['replay'] = dict(
        proposal_executed=False,
        recorded_actual_command=command,
        recorded_command_status=result.get('status'),
        saved_response_images_not_result_of_new_proposal=True,
        proposed_action_was_terminal=proposal['name'] in ('report_done', 'report_unable'),
        note=('MODEL-ONLY REPLAY: completed status and observed next images belong to the '
              'recorded_actual_command from the historical run. The new proposal was NOT executed. '
              'The next saved frame is supplied for further diagnosis, even after a terminal proposal.'))
    return result


def prepare(output, demo_path):
    output.mkdir(parents=True, exist_ok=False)
    original_hashes = {}

    def archive_file(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        target.write_bytes(raw)
        original_hashes[str(source.resolve())] = hashlib.sha256(raw).hexdigest()

    def observation(source, oid, target_root):
        folder = source/'observations'/oid
        if load(folder/'manifest.json')['observation_id'] != oid:
            raise ValueError('saved_observation_id_mismatch')
        for path in folder.rglob('*'):
            if path.is_file():
                archive_file(path, target_root/'observations'/oid/path.relative_to(folder))

    sequences = []
    for label, (source, rounds) in SOURCES.items():
        target = output/label
        archive_file(source/'session.json', target/'session.json')
        sequence = dict(label=label, source_run=str(source), rounds=[])
        for number in rounds:
            rd = source/'rounds'/f'{number-1:04d}'
            context = load(rd/'decision_context.json')
            oid = context['observation_id']
            observation(source, oid, target)
            saved = target/'source_rounds'/f'{number:02d}'
            for name in ('decision_context.json', 'proposal.json', 'assistant.json', 'result.json', 'error.json'):
                if (rd/name).is_file():
                    archive_file(rd/name, saved/name)
            after = None
            if number != rounds[-1]:
                result = load(rd/'result.json')
                after = result['capture']['observation_id']
                following = load(source/'rounds'/f'{number:04d}'/'decision_context.json')['observation_id']
                if after != following:
                    raise ValueError('saved_rounds_are_not_contiguous')
                observation(source, after, target)
            sequence['rounds'].append(dict(number=number, observation_id=oid, after_observation_id=after))
        sequences.append(sequence)
    demo = Demonstration(demo_path).select_activation_tail(3)
    demo.save(output)
    archive_file(LOCALIZED_PROMPT, output/'prompt.txt')
    archive_file(Path(__file__).resolve(), output/'script_used.py')
    plan = dict(format='pressb-localized-replay-v1', status='prepared',
        model_only=True, robot_connected=False, robot_commands_sent=0, new_proposals_executed=0,
        goal=GOAL, decision_mode='localized', thinking=False, context_budget_tokens=65536,
        demo=str(Path(demo_path).resolve()), demo_selection=demo.selection, failure_case=None,
        sequences=sequences, approximate_wrist_reference_points={f'{k[0]}{k[1]}':v for k,v in REFERENCE_POINTS.items()},
        limitations=[
            'Saved subsequent images belong to the original recorded commands, NOT new proposals.',
            'A6/A7/A8 share one action conversation; B5 starts an independent conversation.',
            'All format corrections follow production behavior without a fixed correction count.',
            'Schema validity is not visual correctness or physical success.',
            'Approximate wrist reference points are assistant image annotations, not calibrated contact truth.',
            'HEAD accuracy must be reviewed separately; successful wrist results do not validate HEAD.'])
    atomic_json(output/'plan.json', plan)
    atomic_json(output/'source_hashes.json', original_hashes)
    print('Prepared model-only replay: '+str(output), flush=True)
    return plan


def observation(root, oid):
    folder = root/'observations'/oid
    return load(folder/'manifest.json'), (folder/'head.jpg').read_bytes(), (folder/'right.jpg').read_bytes()


def summarize_stages(folder):
    paths = sorted(folder.rglob('metrics.json'))
    if (folder/'action_metrics.json').exists():
        paths.append(folder/'action_metrics.json')
    return {str(path.relative_to(folder)): load(path) for path in paths}


def reference_error(assessment, label, number):
    reference = REFERENCE_POINTS.get((label, number))
    if reference is None:
        return None
    view = assessment['wrist']
    width, height = view['image_size']
    box, tip = view.get('button_box'), view.get('contact_tip')
    points = dict(button=None if box is None else [(box[0]+box[2])*width/2, (box[1]+box[3])*height/2],
                  tip=None if tip is None else [tip[0]*width, tip[1]*height])
    return dict(reference=reference, predicted=points,
        distance_px={key: None if value is None else sum((a-b)**2 for a,b in zip(value,reference[key]))**.5
                     for key,value in points.items()},
        semantics='Assistant approximate reference on WRIST only; not calibrated contact accuracy.')


def execute(args):
    from pressb.agent_loop.localized_conversation import LocalizedConversation
    output = args.output
    plan = load(output/'plan.json')
    if (output/'report.json').exists():
        raise ValueError('replay_output_already_started_choose_a_new_directory')
    originals = load(output/'source_hashes.json')
    if any(sha(path) != value for path, value in originals.items()):
        raise ValueError('source_changed_since_preparation')
    demo = Demonstration(plan['demo']).select_activation_tail(3)
    demo.save(output)
    prompt = (output/'prompt.txt').read_text()
    report = dict(model_only=True, robot_connected=False, robot_commands_sent=0,
                  new_proposals_executed=0, status='running', rows=[], limitations=plan['limitations'])
    code_paths = [Path(__file__).resolve(), ROOT/'scripts/s1b_agent.py',
                  ROOT/'src/pressb/agent_loop/localized_conversation.py',
                  ROOT/'src/pressb/agent_loop/spatial_localization.py',
                  ROOT/'src/pressb/agent_loop/spatial_progress.py',
                  ROOT/'src/pressb/agent_loop/localized_forward_guard.py',
                  ROOT/'src/pressb/agent_loop/client.py', ROOT/'src/pressb/agent_loop/conversation.py']
    report['implementation_sha256'] = {str(path.relative_to(ROOT)): sha(path) for path in code_paths}
    for path in code_paths:
        target = output/'implementation_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())

    def save():
        atomic_json(output/'report.json', report)

    save()
    try:
        with tunnel(None if args.model_base else args.thor_host, args.port), StopControl() as control:
            base = args.model_base or f'http://127.0.0.1:{args.port}/v1'
            for sequence in plan['sequences']:
                label, steps = sequence['label'], sequence['rounds']
                source = output/label
                session = load(source/'session.json')
                obs, head, wrist = observation(source, steps[0]['observation_id'])
                c = LocalizedConversation(GOAL, obs, head, wrist, session, prompt, demo,
                                          budget=65536, decision_mode='localized')
                c.fixed[1]['content'].append(text(dict(context_type='MODEL_ONLY_REPLAY_NOT_PHYSICAL_CONTROL',
                    instruction='New proposals are not executed. Later images and completed results refer only to recorded_actual_command.')))
                for step in steps:
                    if control.cancel.is_set():
                        raise InterruptedError('operator_stopped')
                    number, oid = step['number'], step['observation_id']
                    if c.current_id != oid:
                        raise ValueError('replay_conversation_observation_mismatch')
                    obs, _, _ = observation(source, oid)
                    folder = source/'rounds'/f'{number:02d}'
                    folder.mkdir(parents=True)
                    row = dict(label=label, round=number, observation_id=oid, proposal_executed=False)
                    started = time.monotonic()
                    print(f'Model-only {label}{number}: {oid}', flush=True)
                    try:
                        body = control.call(c.prepare_decision, base, folder, control.cancel)
                        row['action_images'] = image_count(body['messages'])
                        proposal = control.call(propose, base, body, obs, session, folder,
                                                cancel=control.cancel, conversation=c)
                        row['proposal'] = proposal
                        row['assessment'] = copy.deepcopy(c.localizations[oid])
                        row['progress'] = copy.deepcopy(c.localized_progress[oid])
                        row['approximate_wrist_error'] = reference_error(row['assessment'], label, number)
                        for view, filename in (('head', 'head.jpg'), ('wrist', 'right.jpg')):
                            overlay(source/'observations'/oid/filename, folder/f'{view}_overlay.png',
                                    row['assessment'][view], f'{label}{number} {view.upper()}')
                        if step['after_observation_id']:
                            recorded = load(source/'source_rounds'/f'{number:02d}'/'result.json')
                            feedback = replay_feedback(recorded, proposal)
                            after, ah, aw = observation(source, step['after_observation_id'])
                            atomic_json(folder/'replay_result.json', feedback)
                            c.append_result(proposal, feedback, after, ah, aw)
                            atomic_json(folder/'replay_ledger.json', c.turns[-1]['ledger'])
                    except BaseException as exc:
                        row.update(error_type=type(exc).__name__, error=str(exc))
                        raise
                    finally:
                        row['elapsed_s'] = time.monotonic()-started
                        row['stage_metrics'] = summarize_stages(folder)
                        report['rows'].append(row); save()
                        print(json.dumps({k:row.get(k) for k in ('label','round','proposal','error','elapsed_s')}, ensure_ascii=False), flush=True)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 'error',
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['source_hashes_unchanged'] = all(sha(path) == value for path,value in originals.items())
        totals = dict(requests=0, corrections=0, prompt_tokens=0, completion_tokens=0, elapsed_s=0.)
        for row in report['rows']:
            for metric in row.get('stage_metrics', {}).values():
                for key in ('prompt_tokens', 'completion_tokens', 'elapsed_s'):
                    totals[key] += metric.get(key, 0) or 0
                attempts = metric.get('attempts', 0) or 0
                totals['requests'] += attempts
                totals['corrections'] += max(0, attempts-1)
        report['model_stage_totals'] = totals
        report['elapsed_decision_s'] = sum(row['elapsed_s'] for row in report['rows'])
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--demo', type=Path, default=ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047')
    parser.add_argument('--thor-host', default='Thor')
    parser.add_argument('--port', type=int, default=18026)
    parser.add_argument('--model-base')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--execute-prepared', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute_prepared:
        prepare(args.output, args.demo)
    if not args.prepare_only:
        execute(args)


if __name__ == '__main__':
    main()
