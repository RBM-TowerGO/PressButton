#!/usr/bin/env python3
"""Saved-frame model-only checks for progress review; never constructs a robot client."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from pressb.agent_loop import progress
from pressb.agent_loop.client import propose
from pressb.agent_loop.common import atomic_json, SESSION_SCHEMA
from pressb.agent_loop.conversation import Conversation, live_message
from pressb.agent_loop.demonstration import Demonstration
from pressb.agent_loop.model_context import execution
from s1b_agent import tunnel


def load(path):
    return json.loads(Path(path).read_text())


def observation(run, oid):
    folder = run/'observations'/oid
    obs = load(folder/'manifest.json')
    return obs, (folder/'head.jpg').read_bytes(), (folder/'right.jpg').read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-base')
    parser.add_argument('--thor-host', default='Thor')
    parser.add_argument('--thor-port', type=int, default=18020)
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/agent_web/20261102_113021_61b08bf5/agent')
    parser.add_argument('--arrow-snapshot', type=Path, default=ROOT/'outputs/arrow_marker_20261102_114636')
    parser.add_argument('--arrow-only', action='store_true', help='only test the saved white-arrow initial decision')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/report'/(
        'progress_monitor_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.model_base or f'http://127.0.0.1:{args.thor_port}/v1'
    result = dict(robot_connected=False, commands_sent=0, model_only=True, cases=[],
                  limitation='Archived later frames are actual old-run outcomes, never outcomes of new model suggestions.')
    atomic_json(args.output/'validation.json', result)
    try:
        with tunnel(None if args.model_base else args.thor_host, args.thor_port):
            # Three independent pairs spanning improvement, reaching the image row,
            # and crossing it. Old red markers remain accurately labelled as old.
            for current_round in (() if args.arrow_only else (10, 19, 20)):
                previous_dir = args.run/'rounds'/f'{current_round-2:04d}'
                current_dir = args.run/'rounds'/f'{current_round-1:04d}'
                before_id = load(previous_dir/'decision_context.json')['observation_id']
                current_id = load(current_dir/'decision_context.json')['observation_id']
                before = live_message(*observation(args.run, before_id))
                current = live_message(*observation(args.run, current_id))
                actual = execution(load(previous_dir/'result.json'))
                body = progress.build_review_request('按下电梯外呼向下按钮并撤回', before, current,
                                                     actual, contact_reference='red_marker')
                folder = args.output/f'review_before_round_{current_round:02d}'
                assessment = progress.inspect(base, body, current_id, before_id, folder)
                candidate = load(current_dir/'proposal.json')
                errors = progress.continuation_errors(assessment, candidate, actual['command'])
                record = dict(case=f'before_round_{current_round}', assessment=assessment,
                              archived_next_proposal=candidate, continuation_errors=errors,
                              metrics=load(folder/'metrics.json'))
                result['cases'].append(record)
                atomic_json(args.output/'validation.json', result)
                print(json.dumps(record, ensure_ascii=False), flush=True)

            # Feed actual old model calls and actual execution results for 3 steps;
            # no made-up progress fields are added to these historical calls.
            if not args.arrow_only:
                first = load(args.run/'rounds/0015/decision_context.json')['observation_id']
                obs, head, wrist = observation(args.run, first)
                session = load(args.run/'session.json')
                old_prompt = load(args.run/'task.json')['prompt']
                conv = Conversation('按下电梯外呼向下按钮并撤回', obs, head, wrist, session,
                                    old_prompt+'\n'+progress.GUIDANCE)
                for index in range(15, 18):
                    rd = args.run/'rounds'/f'{index:04d}'
                    raw = load(rd/'response-0.json')['response']
                    actual = load(rd/'result.json')
                    after_id = actual['capture']['observation_id']
                    obs, head, wrist = observation(args.run, after_id)
                    conv.accept(raw)
                    conv.append_result(load(rd/'proposal.json'), actual, obs, head, wrist)
                conv.monitor_progress = True
                folder = args.output/'archived_recovery_decision'
                try:
                    body = conv.prepare_decision(base, folder)
                    candidate = propose(base, body, obs, session, folder, conversation=conv)
                    conv.check_execution(candidate, obs['observation_id'])
                    result['archived_recovery_decision'] = dict(proposal=candidate, executed=False)
                except (ValueError, RuntimeError) as exc:
                    result['archived_recovery_decision'] = dict(status='stopped', error=str(exc), executed=False)
                result['archived_recovery_decision']['review'] = conv.progress_reviews.get(conv.current_id)
                if (folder/'action_metrics.json').exists():
                    result['archived_recovery_decision']['metrics'] = load(folder/'action_metrics.json')
                atomic_json(args.output/'validation.json', result)

            # New white arrow image: state/depth were NOT captured and remain null.
            snap = args.arrow_snapshot
            obs = dict(observation_id=snap.name, state=None, depth=None)
            demo = Demonstration(ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047')
            demo.select_activation_tail(3)
            session = dict(schema=SESSION_SCHEMA, speed_mm_s=20., head_speed_deg_s=20., control_hz=250.)
            conv = Conversation('按下电梯外呼向下按钮并撤回', obs, (snap/'head.jpg').read_bytes(),
                (snap/'right.jpg').read_bytes(), session,
                (ROOT/'docs/astribot/planar_control/agent_system_prompt_red_marker.txt').read_text(),
                demo=demo, monitor_progress=True)
            folder = args.output/'white_arrow_initial_decision'
            atomic_json(folder/'snapshot_source.json', dict(directory=str(snap.resolve()),
                        capture=load(snap/'snapshot.json'), measured_state='not_captured', depth='not_captured'))
            body = conv.prepare_decision(base, folder)
            candidate = propose(base, body, obs, session, folder, conversation=conv)
            conv.check_execution(candidate, obs['observation_id'])
            result['white_arrow_initial_decision'] = dict(proposal=candidate,
                progress=conv.progress_records[conv.current_id], metrics=load(folder/'action_metrics.json'), executed=False)
            result['status'] = 'completed'
    except Exception as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        atomic_json(args.output/'validation.json', result)
        print('Model-only report: '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
