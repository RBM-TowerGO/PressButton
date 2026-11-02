#!/usr/bin/env python3
"""Exercise the bounded control loop on saved frames, without a robot client."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pressb.agent_loop.client import propose
from pressb.agent_loop.common import atomic_json
from pressb.agent_loop.conversation import live_message, image_count
from pressb.agent_loop.demonstration import Demonstration
from pressb.agent_loop.grounding import build_request, inspect, verify_activation
from pressb.agent_loop.short_loop import ShortConversation
from s1b_agent import StopControl, tunnel


def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    source = Path(args.recording)/'agent'
    hashes = {}
    def read(path):
        raw = path.read_bytes(); hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest(); return raw
    def data(path): return json.loads(read(path))
    def observation(oid):
        path = source/'observations'/oid
        return data(path/'manifest.json'), read(path/'head.jpg'), read(path/'right.jpg')
    def round_data(index, name): return data(source/'rounds'/f'{index:04d}'/name)
    demo = Demonstration(args.demo).select_activation_tail(3); demo.save(out)
    session = data(source/'session.json')
    goal = '按下电梯外呼向下按钮并撤回'
    prompt = (ROOT/'docs/astribot/planar_control/agent_system_prompt.txt').read_text()
    report = dict(robot_connected=False, robot_commands_sent=0, results=[], selected_steps=demo.selection,
        limitations=['No proposal is executed; later saved images reflect the recorded actual command.',
                     'One sample per input; not evidence of physical success or a validated millimeter step.'])
    def save(): atomic_json(out/'report.json', report)
    def decision(base, control, conversation, folder, label):
        obs, _, _ = observation(conversation.current_id)
        row = dict(condition=label, observation_id=conversation.current_id)
        start = time.monotonic()
        try:
            body = control.call(conversation.prepare_decision, base, folder, control.cancel)
            proposal = control.call(propose, base, body, obs, session, folder,
                                    cancel=control.cancel, conversation=conversation)
            row.update(proposal=proposal, action_images=image_count(body['messages']),
                       assessment=data(folder/'perception/assessment.json'),
                       perception_metrics=data(folder/'perception/metrics.json'),
                       action_metrics=data(folder/'action_metrics.json'))
            return proposal
        except Exception as exc:
            row['error'] = str(exc)
            raise
        finally:
            row['elapsed_s'] = time.monotonic()-start
            report['results'].append(row); save()
            print(json.dumps(row, ensure_ascii=False), flush=True)
    try:
        with tunnel(None if args.model_base else args.thor_host, args.port), StopControl() as control:
            base = args.model_base or f'http://127.0.0.1:{args.port}/v1'
            first = round_data(args.start_round, 'decision_context.json')['observation_id']
            obs, h, w = observation(first)
            c = ShortConversation(goal, obs, h, w, session, prompt, demo)
            for offset in range(0 if args.activation_only else 3):
                index = args.start_round+offset; folder = out/'shifted'/f'{offset:04d}'
                print(f'Saved shifted decision {index+1}', flush=True)
                proposed = decision(base, control, c, folder, 'shifted_'+str(index+1))
                recorded = round_data(index, 'result.json')
                after, h, w = observation(recorded['capture']['observation_id'])
                result = dict(status='not_executed_replay', request=dict(command_id='replay-'+str(index),
                    type=proposed['name'].removeprefix('propose_'),
                    **{k:v for k,v in proposed['arguments'].items() if k not in ('reason','evidence')}),
                    motion=dict(action_completed=False), capture=dict(observation_id=after['observation_id']),
                    replay=dict(note='Saved next image is NOT the result of this new suggestion.',
                        recorded_actual_command={k:v for k,v in recorded['request'].items()
                                                  if k in ('type','xyz_mm','head_deg','key')}))
                atomic_json(folder/'replay_result.json', result)
                c.append_result(proposed, result, after, h, w)
            if not args.activation_only:
                first = round_data(0, 'decision_context.json')['observation_id']
                obs, h, w = observation(first)
                initial = ShortConversation(goal, obs, h, w, session, prompt, demo)
                decision(base, control, initial, out/'normal_start', 'normal_start')
            activation_images = {}
            for phase in ('before', 'after'):
                path = Path(args.demo)/'operations/000010'/phase
                oid = 'saved-demo-10-'+phase
                current = live_message(dict(observation_id=oid, state={}, depth={}), read(path/'head.jpg'), read(path/'right.jpg'))
                activation_images[phase] = current
                if args.activation_only:
                    continue
                folder = out/('activation_'+phase)
                assessment = control.call(inspect, base, build_request(goal, current), oid, folder, control.cancel)
                report['activation_'+phase] = dict(assessment=assessment, metrics=data(folder/'metrics.json'))
                save()
            step = data(Path(args.demo)/'operations/000010/operation.json')
            folder = out/'activation_pair'
            assessment = control.call(verify_activation, base, activation_images['before'], activation_images['after'],
                dict(type=step['kind'], **step['command']), folder, control.cancel)
            report['activation_pair'] = dict(assessment=assessment, metrics=data(folder/'metrics.json'))
            operation = data(Path(args.demo)/'operations/000011/operation.json')
            report['interrupted_retreat'] = dict(after_status=operation['after']['status'],
                withdrawal='unknown', reason='No complete after observation exists; no visual confirmation was fabricated.')
    finally:
        inputs = {k:v for k,v in hashes.items() if not Path(k).is_relative_to(out.resolve())}
        report['source_integrity'] = dict(sha256=inputs, unchanged=all(
            hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in inputs.items()))
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--recording', default=str(ROOT/'outputs/agent_web/20261101_211122_639d06d8'))
    parser.add_argument('--demo', default=str(ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047'))
    parser.add_argument('--start-round', type=int, default=11)
    parser.add_argument('--thor-host', default='Thor'); parser.add_argument('--port', type=int, default=18006)
    parser.add_argument('--model-base')
    parser.add_argument('--activation-only', action='store_true', help='Only verify the saved successful off-to-on pair')
    run(parser.parse_args())
