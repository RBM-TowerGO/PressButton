#!/usr/bin/env python3
"""Saved-image perception/action comparison. Model calls only, no robot client."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pressb.agent_loop.common import atomic_json
from pressb.agent_loop.client import propose
from pressb.agent_loop.conversation import Conversation, live_message, image_count
from pressb.agent_loop.demonstration import Demonstration
from pressb.agent_loop.failure_case import FailureCase
from pressb.agent_loop.model_context import text
from pressb.agent_loop.perception import ACTION_GUIDANCE, build_request, inspect
from s1b_agent import StopControl, tunnel


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    source=Path(args.recording)/'agent'
    hashes={}
    def read(path):
        data=path.read_bytes();hashes[str(path.resolve())]=hashlib.sha256(data).hexdigest();return data
    def data(path):return json.loads(read(path))
    def observation(oid):
        folder=source/'observations'/oid
        return data(folder/'manifest.json'),read(folder/'head.jpg'),read(folder/'right.jpg')
    def round_data(index,name):return data(source/'rounds'/f'{index:04d}'/name)
    demo=Demonstration(args.demo);failure=FailureCase(args.failure_case)
    demo.save(output);failure.save(output)
    session=data(source/'session.json')
    goal='按下电梯外呼向下按钮并撤回'
    prompt=(ROOT/'docs/astribot/planar_control/agent_system_prompt.txt').read_text()
    first=round_data(0,'decision_context.json')['observation_id']
    ob,head,wrist=observation(first)
    seed=Conversation(goal,ob,head,wrist,session,prompt,demo,failure)
    # Seed authentic prior actions, without fabricating independent assessments for old rounds.
    for n in range(args.start_round):
        proposal=round_data(n,'proposal.json')
        assistant=round_data(n,'assistant.json')
        seed.accept(dict(choices=[dict(message=assistant)]))
        result=round_data(n,'result.json')
        after,h,w=observation(result['capture']['observation_id'])
        seed.append_result(proposal,result,after,h,w)
    def configure(mode):
        c=copy.deepcopy(seed)
        c.decision_mode=mode
        if mode=='separated':
            c.format='pressb-conversation-v2'
            c.fixed[0]['content']+='\n'+ACTION_GUIDANCE
            intro=json.loads(c.fixed[1]['content'][0]['text']);intro['input_format']=c.format
            c.fixed[1]['content'][0]=text(intro)
        c.estimated_tokens=None
        return c
    conversations={mode:configure(mode) for mode in ('direct','separated')}
    report=dict(robot_connected=False,robot_commands_sent=0,results=[],
        limitations=['Saved subsequent observations are NOT outcomes of replay proposals.',
                     'Prior action history is authentic direct-mode history, without invented perception.',
                     'Shared service caches were not reset; small sample, not a task-success benchmark.'])
    def save():atomic_json(output/'report.json',report)
    def decision(base,control,c,rd,index,label):
        ob,_,_=observation(c.current_id)
        row=dict(condition=label,source_round=index+1,observation_id=c.current_id)
        started=time.monotonic()
        try:
            body=control.call(c.prepare_decision,base,rd,control.cancel)
            proposal=control.call(propose,base,body,ob,session,rd,cancel=control.cancel,conversation=c)
            reply=data(rd/('response-1.json' if (rd/'response-1.json').exists() else 'response-0.json'))
            row.update(proposal=proposal,action_images=image_count(body['messages']),
                       action_usage=reply['response'].get('usage'),action_last_request_s=reply['elapsed_s'])
            if (rd/'perception/assessment.json').exists():
                row['assessment']=data(rd/'perception/assessment.json')
                row['perception_metrics']=data(rd/'perception/metrics.json')
            row['action_attempts']=len(list(rd.glob('response-*.json')))
            return proposal,row
        except Exception as exc:
            row['error']=str(exc)
            raise
        finally:
            row['wall_s']=time.monotonic()-started
            report['results'].append(row);save()
            print(json.dumps(row,ensure_ascii=False),flush=True)
    try:
        with tunnel(None if args.model_base else args.thor_host,args.port),StopControl() as control:
            base=args.model_base or f'http://127.0.0.1:{args.port}/v1'
            for offset in range(0 if args.cases_only else 3):
                index=args.start_round+offset
                recorded=round_data(index,'result.json')
                after,h,w=observation(recorded['capture']['observation_id'])
                for mode in (('direct','separated') if offset%2==0 else ('separated','direct')):
                    c=conversations[mode]
                    proposal,row=decision(base,control,c,output/mode/f'{offset:04d}',index,mode)
                    result=dict(status='not_executed_replay',request=dict(command_id=f'replay-{mode}-{offset}',
                        type=proposal['name'],**{k:v for k,v in proposal['arguments'].items() if k not in ('reason','evidence')}),
                        motion=dict(action_completed=False),capture=dict(observation_id=after['observation_id']),error=None,
                        replay=dict(note='Proposal NOT executed. Next observation comes from the saved actual command.',
                                    recorded_actual_command={k:v for k,v in recorded['request'].items()
                                                             if k in ('type','xyz_mm','head_deg','key')}))
                    atomic_json(output/mode/f'{offset:04d}'/'replay_result.json',result)
                    c.append_result(proposal,result,after,h,w)
            for n in range(0 if args.cases_only else 2):
                decision(base,control,configure('separated'),output/f'repeat_{n+1}',args.start_round,f'repeat_{n+1}')
            root=Path(args.demo)/'operations'
            def demo_message(step,phase):
                folder=root/f'{step:06d}'/phase
                # Keyboard targets are not measured state. Do not manufacture SDK measurements.
                return live_message(dict(observation_id=f'saved-demo-{step}-{phase}',state={},depth={}),
                                    read(folder/'head.jpg'),read(folder/'right.jpg'))
            before=demo_message(10,'before');after=demo_message(10,'after')
            operation=data(root/'000010/operation.json')
            body=build_request(goal,after,before,dict(command=dict(type=operation['kind'],**operation['command']),
                                                    status=operation['status'],sends_completed=True))
            result=control.call(inspect,base,body,'saved-demo-10-after','saved-demo-10-before',output/'success_step10',control.cancel)
            report['success_step10']=dict(assessment=result,metrics=data(output/'success_step10/metrics.json'))
            operation=data(root/'000011/operation.json')
            if operation['after']['status']!='missing':raise ValueError('expected_missing_retreat_boundary')
            before=demo_message(11,'before')
            result=control.call(inspect,base,build_request(goal,before),'saved-demo-11-before',None,
                                output/'interrupted_step11',control.cancel)
            report['interrupted_step11']=dict(assessment=result,after_missing=True,
                withdrawal_verification='unknown',metrics=data(output/'interrupted_step11/metrics.json'))
    finally:
        # Only input files are integrity evidence; output artifacts are intentionally newly written.
        inputs={k:v for k,v in hashes.items() if not Path(k).is_relative_to(output.resolve())}
        report['source_integrity']=dict(sha256=inputs,unchanged=all(
            hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in inputs.items()))
        save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--recording',default=str(ROOT/'outputs/agent_web/20261101_211122_639d06d8'))
    parser.add_argument('--demo',default=str(ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047'))
    parser.add_argument('--failure-case',default=str(ROOT/'outputs/agent_web/20261101_184824_dcb23038'))
    parser.add_argument('--start-round',type=int,default=11,help='zero-based first decision round')
    parser.add_argument('--thor-host',default='Thor');parser.add_argument('--port',type=int,default=18002)
    parser.add_argument('--model-base')
    parser.add_argument('--cases-only',action='store_true',help='Only run the two independent success/missing-boundary inspections')
    run(parser.parse_args())
