#!/usr/bin/env python3
"""Model-only paired replay. Never imports RobotClient or sends robot commands."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pressb.agent_loop.common import atomic_json
from pressb.agent_loop.client import TOOLS, propose
from pressb.agent_loop.conversation import Conversation, tokenize, image_count
from pressb.agent_loop.demonstration import Demonstration, image_content
from pressb.agent_loop.failure_case import FailureCase
from s1b_agent import tunnel


def baseline(goal, obs, head, wrist, session, history, prompt, demo, failure):
    content = []
    for case in (demo, failure):
        content.append(dict(type='text', text=json.dumps(case.summary, ensure_ascii=False, allow_nan=False)))
        content.extend(case.content[1:])
    context = dict(goal=goal, context_type='LIVE_OBSERVATION', observation_id=obs['observation_id'],
        state=obs['state'], fixed_point=obs['selection'], depth=obs['depth'], session=session, recent_history=history[-5:])
    content += [dict(type='text',text=json.dumps(context,ensure_ascii=False)),
        dict(type='text',text='LIVE HEAD camera; current observation, not a demonstration.'),image_content(head),
        dict(type='text',text='LIVE RIGHT WRIST camera; original image. The depth sample pixel is metadata, not a marked contact point.'),image_content(wrist)]
    return dict(model='qwen38',temperature=0,max_tokens=1024,stream=False,
        chat_template_kwargs=dict(enable_thinking=False),tools=TOOLS,tool_choice='required',
        messages=[dict(role='system',content=prompt),dict(role='user',content=content)])


def load_observation(run, oid):
    folder = run/'observations'/oid
    return json.loads((folder/'manifest.json').read_text()), (folder/'head.jpg').read_bytes(), (folder/'right.jpg').read_bytes()


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    demo, failure = Demonstration(args.demo), FailureCase(args.failure_case)
    source = Path(args.recording)/'agent'
    session = json.loads((source/'session.json').read_text())
    prompt = (ROOT/'docs/astribot/planar_control/agent_system_prompt.txt').read_text()
    baseline_prompt = Path(args.baseline_prompt).read_text()
    old_chars = sum(len(json.dumps(c.summary,ensure_ascii=False)) for c in (demo,failure))
    new_chars = sum(len(c.content[0]['text']) for c in (demo,failure))
    demo.save(output);failure.save(output)
    records = [json.loads((source/'rounds'/f'{n:04d}'/'execution.json').read_text()) for n in range(4)]
    observations = [load_observation(source, r['decision_observation_id']) for r in records]
    goal = '按下电梯外呼向下按钮并撤回'
    obs,head,wrist = observations[0]
    c = Conversation(goal,obs,head,wrist,session,prompt,demo,failure)
    report = dict(robot_connected=False, robot_commands_sent=0, case_text_before_chars=old_chars,
                  case_text_after_chars=new_chars, case_text_reduction=1-new_chars/old_chars, rounds=[],
                  limitation='Sequential saved observations; model proposals were NOT executed. Off-policy protocol/latency check, not task success or a controlled cache-reset benchmark.')
    histories = {'baseline': [], 'conversation': []}
    previous_tokens = None
    with tunnel(args.thor_host, args.port):
        base = f'http://127.0.0.1:{args.port}/v1'
        for n in range(3):
            obs,head,wrist = observations[n]
            # Alternate order to avoid always timing one mode first.
            for mode in (['baseline','conversation'] if n%2==0 else ['conversation','baseline']):
                rd = output/mode/'rounds'/f'{n:04d}'
                body = (baseline(goal,obs,head,wrist,session,histories[mode],baseline_prompt,demo,failure)
                        if mode=='baseline' else c.prepare(base,rd))
                token_data = tokenize(base, body, return_tokens=True)
                row = dict(round=n+1, mode=mode, image_count=image_count(body['messages']), tokenize_count=token_data['count'])
                if mode=='conversation':
                    tokens = token_data['tokens']
                    if previous_tokens is not None:
                        shared = next((i for i,(a,b) in enumerate(zip(previous_tokens,tokens)) if a!=b), min(len(previous_tokens),len(tokens)))
                        row.update(previous_input_tokens=len(previous_tokens), shared_prefix_tokens=shared,
                                   changed_previous_tail_tokens=len(previous_tokens)-shared)
                        if len(previous_tokens)-shared > 128:
                            raise ValueError('unstable_rendered_history_prefix')
                    previous_tokens = tokens
                    if n == 0 and token_data['count'] > 32768:
                        raise ValueError('initial_context_exceeds_32k')
                if args.tokenize_only:
                    atomic_json(rd/'request.json',body)
                    report['rounds'].append(row)
                    continue
                started = time.monotonic()
                proposal = propose(base,body,obs,session,rd,timeout=180,
                                   conversation=c if mode=='conversation' else None)
                row['wall_s'] = time.monotonic()-started
                response_file = rd/('response-1.json' if (rd/'response-1.json').exists() else 'response-0.json')
                response = json.loads(response_file.read_text())
                row.update(usage=response['response'].get('usage'), model_request_s=response['elapsed_s'], proposal=proposal)
                after,new_head,new_wrist = observations[n+1]
                recorded = json.loads((source/'rounds'/f'{n:04d}'/'result.json').read_text())
                # Explicitly avoid pretending recorded motion was execution of this model's proposal.
                result = dict(status='not_executed_replay', request=dict(command_id=f'replay-{mode}-{n}',
                    type=proposal['name'], **{k:v for k,v in proposal['arguments'].items() if k not in ('reason','evidence')}),
                    motion=dict(action_completed=False,final_state=after['state']),
                    capture=dict(observation_id=after['observation_id']), error=None,
                    replay=dict(note='Proposal NOT executed. Next observation supplied from an earlier recording.',
                        recorded_actual_command={k:v for k,v in recorded['request'].items() if k in ('type','xyz_mm','head_deg','key')}))
                atomic_json(rd/'replay_result.json',result)
                if mode=='conversation':
                    c.append_result(proposal,result,after,new_head,new_wrist)
                histories[mode].append(dict(proposal=proposal,result=result))
                report['rounds'].append(row)
                atomic_json(output/'report.json',report)
                print(json.dumps(row,ensure_ascii=False),flush=True)
            if args.tokenize_only:
                break
    atomic_json(output/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--baseline-prompt',required=True)
    p.add_argument('--demo',default=str(ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047'))
    p.add_argument('--failure-case',default=str(ROOT/'outputs/agent_web/20261101_184824_dcb23038'))
    p.add_argument('--recording',default=str(ROOT/'outputs/agent_web/20261101_184824_dcb23038'))
    p.add_argument('--thor-host',default='Thor');p.add_argument('--port',type=int,default=18001)
    p.add_argument('--tokenize-only',action='store_true')
    run(p.parse_args())
