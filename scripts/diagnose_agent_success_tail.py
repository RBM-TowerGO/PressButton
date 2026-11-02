#!/usr/bin/env python3
"""Model-only success-tail probes, with and without preceding action history."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pressb.agent_loop.common import atomic_json, request
from pressb.agent_loop.client import parse_proposal
from pressb.agent_loop.conversation import image_count
from pressb.agent_loop.demonstration import Demonstration, image_content
from pressb.agent_loop.model_context import text
from s1b_agent import StopControl, tunnel


def success_tail(demo):
    # Last three COMPLETE operations ending at user-confirmed activation, not the interrupted retreat.
    review = demo.summary.get('case_review', {})
    if not review.get('button_activation_verified'):
        raise ValueError('reviewed_activation_required')
    ends = {p.split('/')[1] for p in review.get('evidence', [])
            if isinstance(p, str) and p.startswith('operations/') and '/after/' in p}
    if len(ends) != 1:
        raise ValueError('one_reviewed_activation_boundary_required')
    last = next(iter(ends))
    rows = demo.model_summary['operations']
    end = next(i for i, row in enumerate(rows) if row['step'] == last)
    selected = rows[end-2:end+1]
    if len(selected) != 3 or any(row['status'] != 'completed' for row in selected):
        raise ValueError('three_complete_activation_steps_required')
    steps = [row['step'] for row in selected]
    summary = copy.deepcopy(demo.model_summary)
    summary['operations'] = copy.deepcopy(selected)
    summary['case_review'] = {k: review[k] for k in (
        'case_id', 'analysis_version', 'outcome', 'outcome_source', 'outcome_scope',
        'button_activation_verified', 'withdrawal_verified', 'image_review', 'limitations', 'measurement_point') if k in review}
    summary['selected_steps'] = steps
    summary['selection_note'] = ('Only the last three completed actions through reviewed activation are examples. '
        'Later interrupted retreat is not included and does not prove withdrawal. These are historical command targets, not measured poses.')
    content = [text(summary)]
    pictures = []
    for step in steps:
        index = next(i for i, row in enumerate(rows) if row['step'] == step)
        content.extend(copy.deepcopy(demo.content[1+2*index:3+2*index]))
        pictures.append(demo.images[index])
    if image_count([dict(content=content)]) != 3:
        raise ValueError('three_case_images_required')
    return summary, content, pictures


def run(args):
    output = Path(args.output);output.mkdir(parents=True,exist_ok=False)
    demo = Demonstration(args.demo)
    summary, examples, pictures = success_tail(demo)
    atomic_json(output/'demonstration/model_reference.json', summary)
    for name, blob in pictures:(output/'demonstration'/name).write_bytes(blob)
    session = json.loads(Path(args.session).read_text())
    report = dict(robot_connected=False, robot_commands_sent=0, selected_steps=summary['selected_steps'],
        failure_case_included=False, enable_thinking=False, results=[], sources=[],
        limitations=['Saved requests, no physical action; a direction proposal is not task success.',
                     'with_history keeps authentic preceding observations/reasons; current_only omits them.',
                     'One sample per condition per observation, shared model caches not reset.'])
    def save():atomic_json(output/'report.json',report)
    try:
        with tunnel(None if args.model_base else args.thor_host,args.port),StopControl() as control:
            base=args.model_base or f'http://127.0.0.1:{args.port}/v1'
            for index in range(3):
                source=Path(args.source)/f'{index:04d}'/'request.json'
                raw=source.read_bytes();original=json.loads(raw)
                current=original['messages'][-1]
                observation=json.loads(current['content'][0]['text'])
                if observation['context_type']!='LIVE_OBSERVATION':raise ValueError('current_observation_required')
                report['sources'].append(dict(path=str(source.resolve()),sha256=hashlib.sha256(raw).hexdigest()))
                intro=dict(role='user',content=[copy.deepcopy(original['messages'][1]['content'][0]),*copy.deepcopy(examples)])
                for condition in (('with_history','current_only') if index%2==0 else ('current_only','with_history')):
                    body=copy.deepcopy(original)
                    body['messages'] = ([body['messages'][0],copy.deepcopy(intro),*body['messages'][2:]]
                        if condition=='with_history' else [body['messages'][0],copy.deepcopy(intro),copy.deepcopy(current)])
                    body['chat_template_kwargs']=dict(body.get('chat_template_kwargs') or {},enable_thinking=False)
                    body['max_tokens']=512
                    directory=output/f'{index:04d}'/condition
                    atomic_json(directory/'request.json',body)
                    row=dict(condition=condition,source_round=index+12,observation_id=observation['observation_id'],
                             images=image_count(body['messages']))
                    print(f'Round {index+12}, {condition}, {row["images"]} images',flush=True)
                    started=time.monotonic()
                    try:
                        response=control.call(request,base,'/chat/completions',body,timeout=180)
                        row['request_s']=time.monotonic()-started
                        atomic_json(directory/'response.json',dict(response=response,elapsed_s=row['request_s']))
                        row['usage']=response.get('usage')
                        row['finish_reason']=response['choices'][0]['finish_reason']
                        row['reasoning_tokens']=(response.get('usage',{}).get('completion_tokens_details') or {}).get('reasoning_tokens')
                        proposal=parse_proposal(response,observation,session)
                        row['proposal']=proposal;row['valid_action']=True
                        atomic_json(directory/'proposal.json',proposal)
                    except Exception as exc:
                        row.update(error=str(exc),valid_action=False)
                        atomic_json(directory/'error.json',dict(error_type=type(exc).__name__,error=str(exc)))
                        if isinstance(exc,InterruptedError):raise
                    finally:
                        row['elapsed_s']=time.monotonic()-started
                        report['results'].append(row);save()
                        print(json.dumps(row,ensure_ascii=False),flush=True)
    finally:
        report['source_requests_unchanged']=all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
                                               for x in report['sources'])
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--demo',default=str(ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047'))
    p.add_argument('--source',default=str(ROOT/'outputs/perception_validation_20261101/direct'))
    p.add_argument('--session',default=str(ROOT/'outputs/agent_web/20261101_211122_639d06d8/agent/session.json'))
    p.add_argument('--model-base')
    p.add_argument('--thor-host',default='Thor');p.add_argument('--port',type=int,default=18004)
    run(p.parse_args())
