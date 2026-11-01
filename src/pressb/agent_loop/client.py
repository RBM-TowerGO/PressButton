"""Model request construction, strict proposals and artifact downloads."""
import base64
import io
import json
import math
import time
import uuid
from pathlib import Path
from PIL import Image, ImageDraw
from .common import atomic_json, depth_estimate, number, request, sha256

TOOLS = []
for name, fields in (
    ('propose_move', {'right_mm':'number','up_mm':'number','reason':'string'}),
    ('propose_press', {'forward_mm':'number','reason':'string'}),
    ('request_observation', {'reason':'string'}),
    ('report_done', {'evidence':'string'}),
    ('report_unable', {'reason':'string'})):
    TOOLS.append(dict(type='function', function=dict(name=name,description=name.replace('_',' '),
        parameters=dict(type='object',properties={k:dict(type=v) for k,v in fields.items()},
                        required=list(fields),additionalProperties=False))))


def build_request(goal, obs, wrist_bytes, session, history, prompt):
    image=Image.open(io.BytesIO(wrist_bytes)).convert('RGB')
    w,h=image.size;point=obs['selection']['point_normalized']
    x,y=(w//2,h//2) if point is None else (int(point[0]*(w-1)+.5),int(point[1]*(h-1)+.5))
    draw=ImageDraw.Draw(image)
    draw.line((x-10,y,x+10,y),fill=(0,255,0),width=2)
    draw.line((x,y-10,x,y+10),fill=(0,255,0),width=2)
    data=io.BytesIO();image.save(data,format='JPEG',quality=95)
    context=dict(goal=goal,observation_id=obs['observation_id'],state=obs['state'],
        fixed_point=obs['selection'],depth=obs['depth'],forward_estimate=depth_estimate(obs),
        limits=session,recent_history=history[-5:],image_label='Right-wrist camera only. Green cross: fixed depth sample.')
    return dict(model='qwen38',temperature=0,max_tokens=512,stream=False,
        chat_template_kwargs=dict(enable_thinking=False),tools=TOOLS,tool_choice='auto',
        messages=[dict(role='system',content=prompt),dict(role='user',content=[
            dict(type='text',text=json.dumps(context,ensure_ascii=False,allow_nan=False)),
            dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(data.getvalue()).decode()))])])


def parse_proposal(response, observation, session):
    choices=response.get('choices',[])
    if len(choices)!=1 or choices[0].get('finish_reason')!='tool_calls':raise ValueError('one_complete_tool_required')
    msg=choices[0]['message'];calls=msg.get('tool_calls',[])
    if len(calls)!=1 or (msg.get('content') or '').strip():raise ValueError('one_tool_without_prose_required')
    call=calls[0]['function'];name=call['name']
    schema=next((t['function']['parameters'] for t in TOOLS if t['function']['name']==name),None)
    if schema is None:raise ValueError('unknown_tool')
    args=json.loads(call['arguments'])
    if not isinstance(args,dict) or set(args)!=set(schema['required']):raise ValueError('invalid_tool_arguments')
    for key,spec in schema['properties'].items():
        if spec['type']=='number':number(args[key])
        elif not isinstance(args[key],str) or not args[key].strip():raise ValueError('reason_required')
    if name=='propose_move':
        if not 0<math.hypot(args['right_mm'],args['up_mm'])<=session['max_move_mm']:raise ValueError('move_out_of_range')
    if name=='propose_press':
        expected=depth_estimate(observation)['forward_mm']
        if expected is None or not 0<args['forward_mm']<=session['max_forward_mm'] or abs(args['forward_mm']-expected)>.1:
            raise ValueError('forward_must_equal_depth_minus_150_within_limits')
    return dict(name=name,arguments=args)


def propose(model_base, body, obs, session, directory, timeout=120):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/'request.json',body)
    for attempt in range(2):
        started=time.monotonic()
        response=request(model_base,'/chat/completions',body,timeout=timeout)
        atomic_json(directory/f'response-{attempt}.json',dict(response=response,elapsed_s=time.monotonic()-started))
        try:
            proposal=parse_proposal(response,obs,session)
            atomic_json(directory/'proposal.json',proposal);return proposal
        except (ValueError,KeyError,TypeError) as exc:
            if attempt:raise ValueError('invalid_model_proposal:'+str(exc)) from exc
            body=dict(body,messages=body['messages']+[dict(role='user',content=
                'Your previous response was invalid: '+str(exc)+'. Return exactly one valid tool call for the original observation.')])
            atomic_json(directory/'repair-request.json',body)


class RobotClient:
    def __init__(self, base, output):
        self.base=base;self.output=Path(output)
        self.session=request(base,'/v2/session')

    def wait(self, receipt):
        while True:
            task=request(self.base,'/v2/tasks/'+receipt['task_id'])
            if task['status'] in ('completed','failed','interrupted','rejected'):
                atomic_json(self.output/'tasks'/(receipt['task_id']+'.json'),task)
                if task['status']!='completed':raise ValueError('task_failed:'+str(task.get('error')))
                return task
            time.sleep(.1)

    def submit(self, route, body):
        # Reuse exact ID on ambiguous transport failure, never create a new action.
        for attempt in range(2):
            try:return request(self.base,route,body)
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
        args=proposal['arguments']
        body=dict(session_id=self.session['session_id'],command_id=uuid.uuid4().hex,
            observation_id=obs['observation_id'],type='move' if proposal['name']=='propose_move' else 'press')
        for key in ('right_mm','up_mm','forward_mm'):
            if key in args:body[key]=args[key]
        atomic_json(self.output/'commands'/(body['command_id']+'.json'),body)
        task=self.wait(self.submit('/v2/actions',body))
        return task,self.download(task['capture']['observation_id'])

    def stop(self):return request(self.base,'/v2/stop',{})


def proposal_still_valid(old,new,proposal):
    if old['session_id']!=new['session_id']:return False
    if old['selection']['revision']!=new['selection']['revision']:return False
    if proposal['name']=='propose_press':
        value=depth_estimate(new)['forward_mm']
        if value is None or abs(proposal['arguments']['forward_mm']-value)>2.:return False
    return True
