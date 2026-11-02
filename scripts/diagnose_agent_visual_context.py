#!/usr/bin/env python3
"""Read-only paired visual/action probes of a saved request. No robot connection."""
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
from pressb.agent_loop.conversation import image_count
from s1b_agent import tunnel

PROBE = dict(type='function', function=dict(name='inspect_scene', description='Read-only image interpretation; no execution.',
    parameters=dict(type='object', properties=dict(
        head_button_visible=dict(type='string',enum=['yes','no','uncertain']),
        head_button_relative_to_gripper=dict(type='string',enum=['left','center','right','unknown']),
        head_button_bbox_0_1000=dict(type='array',items=dict(type='number'),description='[left,top,right,bottom], normalized image coordinates 0..1000; [] if not localizable.'),
        wrist_button_visible=dict(type='string',enum=['yes','no','uncertain']),
        wrist_button_location=dict(type='string'),
        alignment=dict(type='string',enum=['aligned','misaligned','uncertain']),
        next_axis=dict(type='string',enum=['+X','-X','+Y','-Y','+Z','-Z','observe','other']),
        visual_evidence=dict(type='string')),
    required=['head_button_visible','head_button_relative_to_gripper','head_button_bbox_0_1000',
              'wrist_button_visible','wrist_button_location','alignment','next_axis','visual_evidence'], additionalProperties=False)))
DIAGNOSTIC = '''Read-only diagnostic, not an execution request. Inspect ONLY the most recent LIVE head and wrist images.
Determine whether the target down-call button is visible in each view; locate it relative to the gripper in the head image.
Assess alignment from the actual images, then state the next adjustment axis if justified. Use chassis +X forward, +Y robot left, +Z up.
Do not assume a button exists in a view; unknown is allowed. Historical images and past reasoning are not current visual evidence.
Call inspect_scene exactly once, without surrounding text. No command will be executed.'''


GROUNDED_ACTION = """Before the next single action, independently inspect the FINAL two LIVE images.
Past reasons are hypotheses, not visual facts; the scene may differ from a demonstration.
In the existing reason, state: target location/visibility in HEAD; target visibility in WRIST;
actual intended contacting finger and alignment uncertainty; desired ROBOT movement direction;
then verify that the command's sign matches that robot direction.
Distinguish object motion in the camera image from robot hand motion. Chassis robot LEFT is +Y,
robot RIGHT is -Y, FORWARD is +X. Head angles use a different actuator and must not substitute
for hand translation when hand alignment is needed. Choose the appropriate action from current
evidence; do not restart a historical sequence merely because it is shown. Unknown is allowed.
Return exactly one existing tool call, no other text. This is a read-only test; it will not execute."""


def variants(original):
    messages=original['messages']
    if len(messages)<3 or messages[0]['role']!='system':
        raise ValueError('saved_conversation_required')
    current=messages[-1]
    live=json.loads(current['content'][0]['text'])
    if live.get('context_type')!='LIVE_OBSERVATION':
        raise ValueError('latest_live_observation_required')
    intro=messages[1]
    # Preserve task/session but remove historical case blocks and images in the minimal condition.
    minimal_intro=dict(role='user',content=[copy.deepcopy(intro['content'][0])])
    return dict(current_only=[messages[0],minimal_intro,current],
                cases_current=[messages[0],intro,current],
                full_history=messages)


def perception_message(diagnostic, observation_id):
    fields = ('head_button_visible','head_button_relative_to_gripper','head_button_bbox_0_1000',
              'wrist_button_visible','wrist_button_location','alignment')
    evidence = {key: diagnostic[key] for key in fields}
    return dict(role='user', content=(
        'Independent read-only perception from the SAME current observation '+observation_id+': '+
        json.dumps(evidence,ensure_ascii=False)+
        '\nThis is a separate model assessment, not human verification. Reconcile it with the final LIVE images '
        'before choosing one next action. Do not inherit an earlier alignment claim without checking it. '
        'Robot RIGHT is -Y and LEFT is +Y; image/object movement differs from robot-hand movement. '
        'Your reason must correctly state current wrist target visibility and explain the chosen robot direction. '
        'Exactly one existing tool call. No proposed action will be executed.'))


def run(args):
    original=json.loads(Path(args.saved_request).read_text())
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    groups=variants(original)
    report=dict(source_request=str(Path(args.saved_request).resolve()),
        source_sha256=hashlib.sha256(Path(args.saved_request).read_bytes()).hexdigest(),
        observation_id=json.loads(original['messages'][-1]['content'][0]['text'])['observation_id'],
        robot_connected=False,robot_commands_sent=0,conditions={},results=[],
        caveat='One sample per condition at temperature 0; shared server caches retained. Diagnostic prompts and action prompts are separate comparisons.')
    for name,messages in groups.items():
        report['conditions'][name]=dict(images=image_count(messages),messages=len(messages))
    atomic_json(output/'report.json',report)
    with tunnel(args.thor_host,args.port):
        base=f'http://127.0.0.1:{args.port}/v1'
        # All three diagnostics share the identical question and tool. All actions retain original prompts/tools.
        for kind,order in [('diagnostic',['current_only','full_history','cases_current']),
                           ('action',['cases_current','current_only','full_history']),
                           ('grounded_action',['full_history']),
                           ('perception_then_action',['full_history','current_only'])]:
            for name in order:
                body=copy.deepcopy(original);body['messages']=copy.deepcopy(groups[name])
                if kind=='diagnostic':
                    body['messages'][0]['content']+='\n\n'+DIAGNOSTIC
                    body['tools']=[PROBE]
                    body['messages'].append(dict(role='user',content=DIAGNOSTIC))
                if kind=='grounded_action':
                    body['messages'].append(dict(role='user',content=GROUNDED_ACTION))
                if kind=='perception_then_action':
                    diagnostic=next(r for r in report['results'] if r['kind']=='diagnostic' and r['condition']=='full_history')['calls'][0]['arguments']
                    body['messages'].append(perception_message(diagnostic,report['observation_id']))
                directory=output/f'{kind}_{name}'
                atomic_json(directory/'request.json',body)
                started=time.monotonic()
                try:
                    response=request(base,'/chat/completions',body,timeout=180)
                    elapsed=time.monotonic()-started
                    atomic_json(directory/'response.json',dict(elapsed_s=elapsed,response=response))
                    message=response['choices'][0]['message']
                    calls=message.get('tool_calls') or []
                    parsed=[]
                    for call in calls:
                        parsed.append(dict(name=call['function']['name'],arguments=json.loads(call['function']['arguments'])))
                    row=dict(kind=kind,condition=name,elapsed_s=elapsed,usage=response.get('usage'),
                             finish_reason=response['choices'][0]['finish_reason'],calls=parsed,text=message.get('content'))
                except Exception as exc:
                    row=dict(kind=kind,condition=name,error=str(exc),elapsed_s=time.monotonic()-started)
                    atomic_json(directory/'error.json',row)
                report['results'].append(row)
                atomic_json(output/'report.json',report)
                print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--saved-request',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--thor-host',default='Thor');p.add_argument('--port',type=int,default=18001)
    run(p.parse_args())
