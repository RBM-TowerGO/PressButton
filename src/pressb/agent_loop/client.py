"""Model request construction, strict proposals and artifact downloads."""
import json
import time
import uuid
from pathlib import Path
from .common import atomic_json, request, sha256

from s1b_keyboard_control import validate_command
from .common import SESSION_SCHEMA
from .demonstration import image_content

TOOLS = []
for name, fields in (
    ('propose_move', {'xyz_mm': dict(type='array', items=dict(type='number'), minItems=3, maxItems=3), 'reason': dict(type='string')}),
    ('propose_head', {'head_deg': dict(type='array', items=dict(type='number'), minItems=2, maxItems=2), 'reason': dict(type='string')}),
    ('propose_gripper', {'key': dict(type='string', enum=['c','o']), 'reason': dict(type='string')}),
    ('request_observation', {'reason': dict(type='string')}),
    ('report_done', {'evidence': dict(type='string')}),
    ('report_unable', {'reason': dict(type='string')})):
    TOOLS.append(dict(type='function', function=dict(name=name, description=name.replace('_',' '),
        parameters=dict(type='object', properties=fields, required=list(fields), additionalProperties=False))))


def build_request(goal, obs, wrist_bytes, session, history, prompt, head_bytes, demo=None, failure_case=None):
    if session.get('schema') != SESSION_SCHEMA:
        raise ValueError('keyboard_session_schema_required')
    context = dict(goal=goal, context_type='LIVE_OBSERVATION', observation_id=obs['observation_id'],
        state=obs['state'], fixed_point=obs['selection'], depth=obs['depth'],
        session=session, recent_history=history[-5:])
    content = list(demo.content) if demo else []
    if failure_case:
        content.extend(failure_case.content)
    content.extend([dict(type='text',text=json.dumps(context,ensure_ascii=False,allow_nan=False)),
        dict(type='text',text='LIVE HEAD camera; current observation, not a demonstration.'), image_content(head_bytes),
        dict(type='text',text='LIVE RIGHT WRIST camera; original image. The depth sample pixel is metadata, not a marked contact point.'),
        image_content(wrist_bytes)])
    return dict(model='qwen38',temperature=0,max_tokens=1024,stream=False,
        chat_template_kwargs=dict(enable_thinking=False),tools=TOOLS,tool_choice='required',
        messages=[dict(role='system',content=prompt),dict(role='user',content=content)])


def parse_proposal(response, observation, session):
    if session.get('schema') != SESSION_SCHEMA:
        raise ValueError('keyboard_session_schema_required')
    choices=response.get('choices',[])
    if len(choices)!=1 or choices[0].get('finish_reason')!='tool_calls':raise ValueError('one_complete_tool_required')
    msg=choices[0]['message'];calls=msg.get('tool_calls',[])
    if len(calls)!=1 or (msg.get('content') or '').strip():raise ValueError('one_tool_without_prose_required')
    call=calls[0]['function'];name=call['name']
    schema=next((t['function']['parameters'] for t in TOOLS if t['function']['name']==name),None)
    if schema is None:raise ValueError('unknown_tool')
    args=json.loads(call['arguments'])
    if not isinstance(args,dict) or set(args)!=set(schema['required']):raise ValueError('invalid_tool_arguments')
    reason=args.get('reason',args.get('evidence'))
    if not isinstance(reason,str) or not reason.strip():raise ValueError('reason_required')
    if name.startswith('propose_'):
        validate_command(dict(type=name.removeprefix('propose_'), **{k:v for k,v in args.items() if k!='reason'}))
    return dict(name=name,arguments=args)


def propose(model_base, body, obs, session, directory, timeout=120, cancel=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/'request.json',body)
    for attempt in range(2):
        if cancel is not None and cancel.is_set():raise InterruptedError('operator_stopped')
        started=time.monotonic()
        try:
            response=request(model_base,'/chat/completions',body,timeout=timeout)
        except Exception as exc:
            atomic_json(directory/'error.json',dict(stage='model_request',attempt=attempt,
                error_type=type(exc).__name__,error=str(exc),elapsed_s=time.monotonic()-started))
            raise
        atomic_json(directory/f'response-{attempt}.json',dict(response=response,elapsed_s=time.monotonic()-started))
        try:
            proposal=parse_proposal(response,obs,session)
            atomic_json(directory/'proposal.json',proposal);return proposal
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            if attempt:
                atomic_json(directory/'error.json',dict(stage='parse_proposal',attempt=attempt,
                    error_type=type(exc).__name__,error=str(exc)))
                raise ValueError('invalid_model_proposal:'+str(exc)) from exc
            body=dict(body,messages=body['messages']+[dict(role='user',content=
                'Your previous response was invalid: '+str(exc)+'. Return exactly one valid tool call for the original observation.')])
            atomic_json(directory/'repair-request.json',body)


class RobotClient:
    def __init__(self, base, output, cancel=None):
        self.base=base;self.output=Path(output);self.cancel=cancel
        self.session=request(base,'/v2/session')
        if self.session.get('schema') != SESSION_SCHEMA:
            raise ValueError('keyboard_session_schema_required')

    def check_running(self):
        if self.cancel is not None and self.cancel.is_set():
            raise InterruptedError('operator_stopped')

    def wait(self, receipt):
        while True:
            self.check_running()
            task=request(self.base,'/v2/tasks/'+receipt['task_id'])
            if task['status'] in ('completed','failed','interrupted','rejected'):
                atomic_json(self.output/'tasks'/(receipt['task_id']+'.json'),task)
                if task['status']!='completed':raise ValueError('task_failed:'+str(task.get('error')))
                return task
            time.sleep(.1)

    def submit(self, route, body):
        # Reuse exact ID on ambiguous transport failure, never create a new action.
        for attempt in range(2):
            try:
                self.check_running()
                return request(self.base,route,body)
            except (OSError,TimeoutError):
                if attempt:raise

    def observe(self):
        body=dict(session_id=self.session['session_id'],command_id=uuid.uuid4().hex)
        task=self.wait(self.submit('/v2/observations',body))
        return self.download(task['capture']['observation_id'])

    def download(self, oid):
        obs=request(self.base,'/v2/observations/'+oid)
        folder=self.output/'observations'/oid;folder.mkdir(parents=True,exist_ok=True)
        for name,meta in obs['artifacts'].items():
            if Path(name).name!=name:raise ValueError('invalid_artifact_name')
            data=None
            for attempt in range(3):
                try:
                    data=request(self.base,meta['url'],binary=True,timeout=30)
                    if len(data)!=meta['bytes'] or sha256(data)!=meta['sha256']:raise ValueError('artifact_hash_mismatch')
                    break
                except (OSError,ValueError):
                    if attempt==2:raise
            tmp=folder/(name+'.tmp');tmp.write_bytes(data);tmp.replace(folder/name)
        atomic_json(folder/'manifest.json',obs)
        return obs,folder

    def execute(self, proposal, obs):
        self.check_running()
        args=proposal['arguments']
        body=dict(session_id=self.session['session_id'],command_id=uuid.uuid4().hex,
            observation_id=obs['observation_id'],type=proposal['name'].removeprefix('propose_'))
        for key in ('xyz_mm','head_deg','key'):
            if key in args:body[key]=args[key]
        atomic_json(self.output/'commands'/(body['command_id']+'.json'),body)
        task=self.wait(self.submit('/v2/actions',body))
        return task,self.download(task['capture']['observation_id'])

    def stop(self):return request(self.base,'/v2/stop',{})
