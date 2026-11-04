"""Model request construction, strict proposals and artifact downloads."""
import json
import copy
import itertools
import time
import uuid
from pathlib import Path
from .common import atomic_json, request, sha256

from s1b_keyboard_control import validate_command
from .common import SESSION_SCHEMA

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

for tool in TOOLS:
    function = tool['function']
    fields = function['parameters']['properties']
    if function['name'] == 'propose_move':
        function['description'] = ('Translate hand by chassis-frame delta; preserve orientation. No automatic return. '
            'Any positive X, including an XYZ combination, requires an independent CURRENT wrist check '
            'that the button is horizontally between both visible finger ends. Unknown does not pass. '
            'A rejected forward proposal is not executed or clipped; choose a new command.')
        fields['xyz_mm']['description'] = ('[X,Y,Z] millimeters; +X forward, -X retract, +Y robot LEFT, '
            '-Y robot RIGHT, +Z up. Added to previous commanded target, not measured pose. '
            'Image directions are not chassis directions. XYZ may move together.')
    elif function['name'] == 'propose_head':
        function['description'] = 'Set absolute head joint angles; changes head view, not hand or wrist pose.'
        fields['head_deg']['description'] = ('[yaw,pitch] absolute degrees, not increments. This setup has a demonstrated '
            'look-down pitch near +30 degrees; it is a reference, not a required value or an additional 30-degree turn. '
            'Choose angles from the current head image to see the full panel; keep an already adequate view. '
            'Head movement does not reposition the wrist camera or hand.')
    elif function['name'] == 'propose_gripper':
        function['description'] = ('c closes, o opens the gripper. With a calibrated merged end, close only after '
            'the open-gripper wrist view shows the button on the aim pixel and the press-phase range is frozen; '
            'closing hides the button at that pixel.')
        fields['key']['description'] = 'Keyboard c/o; startup preserves gripper state.'
    elif function['name'] == 'request_observation':
        function['description'] = ('Capture fresh images at the current camera poses, without head or hand movement. '
            'CURRENT dual images are already supplied each round. Refresh can resolve transient uncertainty, '
            'but cannot reveal a panel cropped out of the head view; choose propose_head to change that framing.')


def build_request(goal, obs, wrist_bytes, session, history, prompt, head_bytes, demo=None, failure_case=None):
    # Compatibility entry for single-observation callers; the live runner owns a Conversation.
    from .conversation import Conversation
    if history:
        raise ValueError('use_conversation_for_history')
    return Conversation(goal, obs, head_bytes, wrist_bytes, session, prompt, demo, failure_case).body()


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


def is_forward(proposal):
    return (proposal.get('name') == 'propose_move'
            and proposal['arguments']['xyz_mm'][0] > 0)


def _press_phase_press():
    """True once a trusted open-gripper range is frozen and the gripper is closed."""
    from . import press_phase
    return press_phase.active() and press_phase.phase() == press_phase.PRESS


def require_forward_permission(proposal, obs, decision, policy='wrist_horizontal'):
    """Check the same observation and exact command again immediately before HTTP.

    This Agent-side rule does not change the keyboard executor or HTTP schema.
    """
    if not is_forward(proposal):
        return
    if policy == 'wrist_tip_overlap':
        from . import wrist_overlap_guard
        wrist_overlap_guard.validate_permission(proposal, obs, decision)
        return
    if policy != 'wrist_horizontal':
        raise ValueError('invalid_forward_guard_policy:'+str(policy))
    from . import forward_guard as guard
    d = decision if isinstance(decision, dict) else {}
    command = dict(type='move', xyz_mm=proposal['arguments']['xyz_mm'])
    if (d.get('format') != guard.GATE_FORMAT or d.get('policy') != guard.POLICY
            or d.get('source') != 'program_rule_on_model_judgment' or d.get('allowed') is not True
            or d.get('observation_id') != obs['observation_id']
            or d.get('assessment_observation_id') != obs['observation_id']
            or d.get('authorized_command') != command
            or any(d.get('basis', {}).get(k) != 'yes' for k in guard.FIELDS)):
        raise ValueError('positive_X_requires_current_wrist_between_fingers_permission')


def propose(model_base, body, obs, session, directory, timeout=120, cancel=None, conversation=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/'request.json',body)
    initial_body = copy.deepcopy(body)
    metrics=dict(elapsed_s=0.,attempts=0,prompt_tokens=0,completion_tokens=0,
                 validation_failures=0, status='requesting')
    guard_assessment = None
    if conversation is not None:
        conversation.forward_guard_decision = None

    def correction_status(status):
        path = directory/'correction.json'
        if path.exists():
            value = json.loads(path.read_text())
            value['status'] = status
            atomic_json(path, value)

    def check():
        if cancel is not None and cancel.is_set():
            metrics['status'] = 'interrupted'
            atomic_json(directory/'action_metrics.json', metrics)
            correction_status('interrupted')
            atomic_json(directory/'error.json', dict(stage='model_request', error='operator_stopped'))
            raise InterruptedError('operator_stopped')
        if conversation is not None:
            conversation.check_observation(obs['observation_id'])

    def stage(value):
        path = directory/'decision_context.json'
        meta = json.loads(path.read_text()) if path.exists() else dict(observation_id=obs['observation_id'])
        if conversation is not None and hasattr(conversation, 'decision_context'):
            meta.update(conversation.decision_context(value))
        meta.update(stage=value, forward_guard_policy=getattr(conversation, 'forward_guard_policy', 'wrist_horizontal'))
        atomic_json(path, meta)

    def forward_check(proposal, attempt):
        nonlocal guard_assessment
        from . import forward_guard as guard
        if not is_forward(proposal):
            return None
        if conversation is not None and getattr(conversation, 'forward_guard_policy', None) == 'none':
            return None
        if conversation is None:
            raise ValueError('positive_X_requires_bound_current_observation')
        policy = getattr(conversation, 'forward_guard_policy', 'wrist_horizontal')
        if policy == 'wrist_tip_overlap':
            from . import wrist_overlap_guard as guard
        elif policy != 'wrist_horizontal':
            raise ValueError('invalid_forward_guard_policy:'+str(policy))
        check()
        if policy == 'wrist_tip_overlap' and _press_phase_press():
            # The gripper is closed over the calibrated aim pixel, so the model box is
            # unreliable there. Live depth at that pixel, capped by the frozen
            # range, bounds the stroke instead.
            from . import wrist_overlap_guard as press_guard
            decision = press_guard.press_gate(obs['observation_id'], proposal['arguments']['xyz_mm'][0])
            if decision['allowed']:
                decision['authorized_command'] = dict(type='move',
                    xyz_mm=copy.deepcopy(proposal['arguments']['xyz_mm']))
            atomic_json(directory/'forward_guard'/f'candidate-{attempt}.json', dict(proposal=proposal, decision=decision))
            atomic_json(directory/'forward_guard'/'decision.json', decision)
            if hasattr(conversation, 'attach_press_check'):
                conversation.attach_press_check(decision)
            stage('action_request')
            if not decision['allowed']:
                raise ValueError('forward_blocked_press_stroke_exceeds_press_limit:'+
                    json.dumps(decision['press'], ensure_ascii=False)+
                    '; This entire command was NOT executed; it was not clipped. Reduce the forward '
                    'stroke to at most remaining_m, withdraw with negative X, verify the target from '
                    'the head view, or report unable. remaining_m is the smaller of the live depth '
                    'at the merged end minus the press target and the frozen open-gripper range.')
            return decision
        if guard_assessment is None:
            stage('forward_guard')
            try:
                current = conversation.observations[obs['observation_id']]
                guard_body = guard.build_request(conversation.goal, current)
                guard_assessment = guard.inspect(model_base, guard_body, obs['observation_id'],
                                                 directory/'forward_guard', cancel)
                check()
            except Exception as exc:
                stage('interrupted' if isinstance(exc, InterruptedError) else 'failed')
                atomic_json(directory/'error.json', dict(stage='forward_guard', error_type=type(exc).__name__, error=str(exc)))
                if isinstance(exc, InterruptedError):
                    raise
                # A failed visual request is not an action-format repair opportunity.
                raise RuntimeError('forward_guard_failed:'+str(exc)) from exc
        check()
        decision = guard.evaluate(guard_assessment, obs['observation_id'])
        if decision['allowed']:
            decision['authorized_command'] = dict(type='move', xyz_mm=copy.deepcopy(proposal['arguments']['xyz_mm']))
        atomic_json(directory/'forward_guard'/f'candidate-{attempt}.json', dict(proposal=proposal, decision=decision))
        atomic_json(directory/'forward_guard'/'decision.json', decision)
        if policy == 'wrist_tip_overlap' and hasattr(conversation, 'attach_forward_check'):
            conversation.attach_forward_check(guard_assessment, decision)
        stage('action_request')
        if not decision['allowed']:
            if policy == 'wrist_tip_overlap':
                raise ValueError('forward_blocked_wrist_tip_not_inside_button_box:'+
                    json.dumps(decision['basis'], ensure_ascii=False)+
                    '; This entire command was NOT executed. Align in Y/Z with X=0, improve the view, '
                    'refresh observations, withdraw with negative X, or report unable. '
                    'Do not repeat positive X on this same rejected observation. The marked physical finger '
                    'tip must be inside the CURRENT wrist button face box in BOTH image axes. '
                    'Unknown does not pass. No gripper closure, depth validity or displacement cap is required.')
            raise ValueError('forward_blocked_button_not_confirmed_between_wrist_fingers:'+
                json.dumps(decision['basis'], ensure_ascii=False)+
                '; This command was NOT executed. Choose a new zero/negative-X move, head/gripper adjustment, '
                'observation, or report unable. Do not repeat positive X on this same observation. '
                'Only horizontal wrist alignment is required; there is no height or depth threshold.')
        require_forward_permission(proposal, obs, decision, policy=policy)
        return decision

    for attempt in itertools.count():
        check()
        metrics.update(attempts=attempt+1, status='requesting')
        atomic_json(directory/'action_metrics.json',metrics)
        started=time.monotonic()
        try:
            response=request(model_base,'/chat/completions',body,timeout=timeout)
        except Exception as exc:
            metrics['elapsed_s'] += time.monotonic()-started
            metrics['status'] = 'failed'
            correction_status('failed')
            atomic_json(directory/'action_metrics.json',metrics)
            atomic_json(directory/'error.json',dict(stage='model_request',attempt=attempt,
                error_type=type(exc).__name__,error=str(exc),elapsed_s=time.monotonic()-started))
            raise
        atomic_json(directory/f'response-{attempt}.json',dict(response=response,elapsed_s=time.monotonic()-started))
        metrics['elapsed_s'] += time.monotonic()-started
        for field in ('prompt_tokens','completion_tokens'):
            metrics[field] += (response.get('usage') or {}).get(field,0)
        atomic_json(directory/'action_metrics.json',metrics)
        check()
        try:
            proposal=(conversation.parse_response(response,obs,session,directory)
                      if conversation is not None and hasattr(conversation,'parse_response')
                      else parse_proposal(response,obs,session))
            if conversation is not None and getattr(conversation, 'max_press_attempts', None):
                # A missed press is not the end of the run while attempts remain.
                from . import press_retry
                press_retry.check(conversation.turns, proposal, conversation.max_press_attempts)
            if conversation is not None and hasattr(conversation, 'review_proposal'):
                conversation.review_proposal(model_base, proposal, directory, cancel)
            permission = forward_check(proposal, attempt)
            check()
            if conversation is not None:
                conversation.check_observation(obs['observation_id'])
                conversation.accept(response)
                conversation.forward_guard_decision = permission
            atomic_json(directory/'assistant.json', response['choices'][0]['message'])
            metrics['status'] = 'validated'
            atomic_json(directory/'action_metrics.json', metrics)
            if attempt:
                correction = json.loads((directory/'correction.json').read_text())
                correction.update(status='accepted', accepted_attempt=attempt,
                                  corrections=metrics['validation_failures'])
                atomic_json(directory/'correction.json', correction)
            atomic_json(directory/'proposal.json',proposal);return proposal
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            metrics.update(status='repairing', validation_failures=metrics['validation_failures']+1)
            atomic_json(directory/'action_metrics.json', metrics)
            correction = dict(stage='parse_proposal', status='retrying', attempt=attempt,
                              next_attempt=attempt+1, observation_id=obs['observation_id'],
                              error_type=type(exc).__name__, error=str(exc))
            atomic_json(directory/f'validation-{attempt}.json', correction)
            atomic_json(directory/'correction.json', correction)
            repair='Your previous response was invalid: '+str(exc)+'. Return exactly one valid tool call for the original observation.'
            if conversation is not None and getattr(conversation,'decision_mode',None)=='grounded':
                repair += (' Match the CURRENT alignment assessment and actuator effect. Do not invent a substitute execution result. '
                           'Previous invalid response: '+json.dumps(response.get('choices'),ensure_ascii=False))
            if conversation is not None and (getattr(conversation, 'monitor_progress', False)
                    or getattr(conversation, 'decision_mode', None) == 'localized'):
                # A newly triggered review is appended to this observation before
                # retrying; the rejected candidate never enters accepted history.
                body = conversation.body()
            else:
                body = copy.deepcopy(initial_body)
            body=dict(body,messages=body['messages']+[dict(role='user',content=repair)])
            atomic_json(directory/f'repair-request-{attempt+1}.json',body)
            atomic_json(directory/'repair-request.json',body)
        except InterruptedError:
            metrics['status'] = 'interrupted'
            atomic_json(directory/'action_metrics.json', metrics)
            correction_status('interrupted')
            raise
        except Exception:
            metrics['status'] = 'failed'
            atomic_json(directory/'action_metrics.json', metrics)
            correction_status('failed')
            raise


CAPTURE_RECOVERY_ATTEMPTS = 3


class TaskFailed(ValueError):
    """A robot task ended in a non-completed status; carries the task record."""
    def __init__(self, task):
        super().__init__('task_failed:'+str(task.get('error')))
        self.task = task


def capture_only_failure(task):
    """True when only the photo failed: motion done (or none), service not stopped.

    The service marks such a task 'failed' (not 'interrupted') and waits for a new
    observation; a real motion failure or a stop sets 'interrupted' and halts it.
    """
    motion = task.get('motion') or {}
    kind = (task.get('request') or {}).get('type')
    return (task.get('status') == 'failed' and task.get('phase') == 'CAPTURING'
            and (kind is None or motion.get('action_completed') is True))


class RobotClient:
    def __init__(self, base, output, cancel=None, forward_guard_policy='wrist_horizontal'):
        if forward_guard_policy not in ('none', 'wrist_horizontal', 'wrist_tip_overlap'):
            raise ValueError('invalid_forward_guard_policy')
        self.forward_guard_policy = forward_guard_policy
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
                if task['status']!='completed':raise TaskFailed(task)
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
        for attempt in range(CAPTURE_RECOVERY_ATTEMPTS):
            body=dict(session_id=self.session['session_id'],command_id=uuid.uuid4().hex)
            try:
                task=self.wait(self.submit('/v2/observations',body))
            except TaskFailed as exc:
                # A failed photo leaves the service waiting for a new observation; try again.
                if not capture_only_failure(exc.task) or attempt == CAPTURE_RECOVERY_ATTEMPTS-1:
                    raise
                continue
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

    def execute(self, proposal, obs, forward_check=None):
        self.check_running()
        if getattr(self, 'forward_guard_policy', 'wrist_horizontal') != 'none':
            require_forward_permission(proposal, obs, forward_check,
                                       policy=getattr(self, 'forward_guard_policy', 'wrist_horizontal'))
        args=proposal['arguments']
        body=dict(session_id=self.session['session_id'],command_id=uuid.uuid4().hex,
            observation_id=obs['observation_id'],type=proposal['name'].removeprefix('propose_'))
        for key in ('xyz_mm','head_deg','key'):
            if key in args:body[key]=args[key]
        atomic_json(self.output/'commands'/(body['command_id']+'.json'),body)
        try:
            task=self.wait(self.submit('/v2/actions',body))
        except TaskFailed as exc:
            if not capture_only_failure(exc.task):
                raise
            # The motion was executed; only the photo after it failed. Take a new
            # observation instead of ending the run, and keep the original failure on record.
            obs_folder=self.observe()
            task=copy.deepcopy(exc.task)
            task['capture_recovery']=dict(original_status=task['status'], original_error=task.get('error'),
                                          recovered_observation_id=obs_folder[0]['observation_id'])
            task.update(status='completed', capture=dict(status='complete_after_retry',
                        observation_id=obs_folder[0]['observation_id']))
            atomic_json(self.output/'tasks'/(task['task_id']+'.json'),task)
            return task,obs_folder
        return task,self.download(task['capture']['observation_id'])

    def stop(self):return request(self.base,'/v2/stop',{})
