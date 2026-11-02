"""Bounded visual control context; full decisions and observations stay archived."""
import copy
import json
from pathlib import Path

from .common import SESSION_SCHEMA, atomic_json
from .client import TOOLS, parse_proposal
from .conversation import image_count, live_message, request_body, tokenize
from .model_context import execution, session as session_view, text
from .grounding import DEFAULT_CONTACT
from .decision_validation import reason_conflicts, motion_reason_conflicts

FORMAT = 'pressb-conversation-v3'
BASIS_FIELDS = ('head_visible', 'wrist_visible', 'contact_finger', 'head_horizontal', 'head_vertical')
EFFECTS = {
    'propose_move': ['reduce_horizontal_error', 'reduce_vertical_error', 'reduce_both_errors', 'approach', 'press', 'retract', 'improve_wrist_view'],
    'propose_head': ['improve_head_view'], 'propose_gripper': ['change_gripper'],
    'request_observation': ['observe'], 'report_done': ['finish'], 'report_unable': ['unable'],
}
GUIDANCE = '''
CURRENT CONTROL POLICY
Use the CURRENT_ALIGNMENT_MODEL_JUDGMENT and control_plan for this observation.
They are model-derived estimates, never human verification or calibrated contact.
visual_basis must copy the five named current fields exactly. With v2 perception,
HEAD directions and error sizes are calculated from the model's button box and
selected fingertip point: image x increases right, y increases down. Button above
finger means finger below button. Geometry and image locations can still be wrong.

Follow control_plan.stage:
- get_view: obtain a view of the panel/button and selected contact end. Compare
  head_target_info with current measured angles. Repeating an already-sent absolute
  head target does not request an additional tilt. Choose a DIFFERENT appropriate
  pose when that view remains inadequate. About +30 degrees pitch is an initial
  look-down example, not a fixed target to repeat. Inspect the new image.
  The default next choice here is propose_head to reveal the missing geometry.
  A coarse wrist_relation alone is not enough to continue hand micro-adjustments
  while the selected HEAD contact end remains unidentified. If choosing a
  different action, state the specific current visual evidence for that exception.
- align: address control_plan.focus. Use the stated error size and the suggested
  step as a starting choice, then inspect the result. Size is measured in apparent
  button widths/heights, NOT millimeters. Step hints 5/10/20 mm are configurable
  heuristics, not calibrated compensation, guaranteed safe distances or limits.
  Explain a different chosen step when needed; do not repeat 10 mm by habit.
- recheck: the image-based error changed unexpectedly. Reassess the target/finger
  localization with a fresh observation, or improve the head view if cropped.
  Do not blindly reverse the hand or increase the step from one disputed judgment.
- approach: choose an explicit approach/press intent and evaluate current views.
  Program approval of CURRENT wrist horizontal containment is still required.
  Projected center/level alone is not proof that a finger can contact the button.
- withdraw: this run has light-on evidence; choose retreat and inspect separation.
- complete: report completion only with the recorded light-on and later withdrawal IDs.

Purpose must describe the actual command. reduce_horizontal_error,
reduce_vertical_error and reduce_both_errors are Y/Z corrections with X=0; they
cannot conceal forward motion. An explicit approach/press may be selected with a
current visual justification and must pass the existing wrist-only forward guard.
Stage suggestions do not add a height or depth gate. The program never clips,
splits, substitutes or replays the numerical command to enforce these rules.

Default contact is the RIGHT-side finger in HEAD; keep that preference if its
current identity is unknown. Never assume the same image-side identity in WRIST.
For this setup's ordinary upright HEAD view, a button above the finger suggests
raising the hand, below suggests lowering. Treat these as hypotheses to check in
the next image, not a calibrated camera transform. +Z raises, -Z lowers; robot
right is -Y, robot left +Y. Head movement changes HEAD framing, not wrist/hand pose.
Use actual recorded commands and geometry_progress to assess effects. Unknown
progress or unchanged coarse categories does not prove the hand failed to move.

Every X>0 candidate, including XYZ combinations, must pass the independent CURRENT
WRIST horizontal-between-fingers check. Denied/unknown candidates are not sent;
choose a new non-forward action on the one allowed repair. Open single-finger use,
invalid depth and unconfirmed light remain permitted; none replaces the wrist rule.
Keep reason to one short sentence stating view, chosen finger, deviation and purpose.
No prior successful demo, sent command, depth threshold or missing frame proves
activation. report_done must cite this run's light-on and later verified retreat.
'''


def grounded_tools():
    tools = copy.deepcopy(TOOLS)
    fields = dict(
        head_visible=dict(type='string', enum=['yes', 'no', 'unknown']),
        wrist_visible=dict(type='string', enum=['yes', 'no', 'unknown']),
        contact_finger=dict(type='string', enum=['head_image_left', 'head_image_right', 'unknown']),
        head_horizontal=dict(type='string', enum=['left', 'center', 'right', 'unknown']),
        head_vertical=dict(type='string', enum=['above', 'level', 'below', 'unknown']))
    for tool in tools:
        fn = tool['function']; params = fn['parameters']
        params['properties']['visual_basis'] = dict(type='object', properties=copy.deepcopy(fields),
            required=list(fields), additionalProperties=False,
            description='Copy these five fields from CURRENT_ALIGNMENT_MODEL_JUDGMENT exactly. Never copy old scene assumptions.')
        params['properties']['expected_effect'] = dict(type='string', enum=EFFECTS[fn['name']])
        params['required'] += ['visual_basis', 'expected_effect']
    return tools


def actual_command(feedback):
    replay = (feedback or {}).get('replay')
    command = (replay.get('recorded_actual_command') if replay else (feedback or {}).get('command')) or {}
    return {k: copy.deepcopy(command[k]) for k in ('type', 'xyz_mm', 'head_deg', 'key') if k in command}


class ShortConversation:
    """One run with persistent factual memory and a freshly rendered bounded input."""
    def __init__(self, goal, obs, head, wrist, session, prompt, demo=None, failure_case=None,
                 budget=65536, decision_mode='grounded'):
        if session.get('schema') != SESSION_SCHEMA:
            raise ValueError('keyboard_session_schema_required')
        if budget <= 0 or decision_mode != 'grounded':
            raise ValueError('invalid_grounded_configuration')
        self.goal, self.session, self.budget = goal, session, budget
        self.decision_mode, self.format, self.generation = decision_mode, FORMAT, 0
        intro = [text(dict(goal=goal, session=session_view(session), input_format=FORMAT,
            operator_preferences=dict(contact_finger=DEFAULT_CONTACT,
                head_view=dict(reference_pitch_deg=30.0, absolute_example_only=True,
                    adapt_to_current_view=True, purpose='See the complete panel and selected finger in HEAD.'))))]
        for case in (demo, failure_case):
            if case:
                intro.extend(copy.deepcopy(case.content))
        self.fixed = [dict(role='system', content=prompt+'\n'+GUIDANCE), dict(role='user', content=intro)]
        self.current_id = obs['observation_id']; self.previous_id = None
        self.observations = {self.current_id: live_message(obs, head, wrist)}
        self.assessments, self.turns = {}, []
        self.pending = None; self.candidate_basis = None
        self.last_usage = {}; self.estimated_tokens = None
        self.memory = dict(source='recorded_execution_and_model_judgments_kept_separate',
            preferred_contact=DEFAULT_CONTACT, selected_contact=DEFAULT_CONTACT, latest_result=None, previous_alignment=None,
            alignment_progress=None, alignment_control=None, control_plan=None,
            evidence=dict(initial_lamp=None, last_off=None, light_on=None, withdrawal=None))
        self.processed = set()

    def check_observation(self, observation_id):
        if observation_id != self.current_id:
            raise ValueError('decision_observation_mismatch')
        if observation_id not in self.assessments or observation_id not in self.processed:
            raise ValueError('current_perception_required')

    def decision_context(self, stage):
        from .localization import FORMAT as PERCEPTION_FORMAT
        return dict(format=FORMAT, decision_mode='grounded', observation_id=self.current_id,
            previous_observation_id=self.previous_id, stage=stage, perception_format=PERCEPTION_FORMAT,
            alignment_control_version='pressb-alignment-control-v1',
            control_policy_version='pressb-control-plan-v1',
            memory_mode='bounded_current_state', compression_generation=0)

    def messages(self):
        # Keep legacy categorical counters in the archive, but expose only the
        # current geometry-based policy as an action priority. Coarse categories
        # and axis-neglect counts must not compete with the new size estimate.
        memory = {k: v for k, v in self.memory.items()
                  if k not in ('alignment_control', 'alignment_progress')}
        result = self.fixed+[dict(role='user', content=[text(dict(
            context_type='CONTROL_MEMORY_THIS_RUN_NOT_CURRENT_VISUAL_FACT', **memory))]),
            self.observations[self.current_id]]
        if self.current_id in self.assessments:
            result.append(dict(role='user', content=[text(dict(
                context_type='CURRENT_ALIGNMENT_MODEL_JUDGMENT', **self.assessments[self.current_id]))]))
        return result

    def body(self):
        return request_body(self.messages(), grounded_tools())

    def _state(self, oid):
        if oid not in self.observations:
            return None
        return json.loads(self.observations[oid]['content'][0]['text'])['state']

    def _last_head_target(self):
        for turn in reversed(self.turns):
            command = actual_command(turn['result'])
            if command.get('type') == 'head':
                oid = turn['after_observation_id']
                return dict(head_deg=copy.deepcopy(command['head_deg']), observation_id=oid,
                    measured_head_deg=(self._state(oid) or {}).get('head_deg'),
                    command_id=turn['result'].get('command_id'))
        return None

    def _alignment_control(self, current):
        """Explain subject direction and unattended axes; never synthesize a command.

        Counts describe recorded commands and categorical model judgments. They
        neither measure image error nor assume a camera-to-chassis transform.
        Recompute from records so repeated preparation cannot inflate a count.
        """
        inverse = dict(left='right', right='left', center='center', above='below',
                       below='above', level='level', unknown='unknown')
        result = dict(format='pressb-alignment-control-v1', observation_id=self.current_id,
            source='derived_from_model_judgments_and_recorded_commands_not_new_visual_evidence',
            contact_finger=current['contact_finger'], axes={}, attention_axis=None,
            limitation='Image relations only. Axis activity is not calibrated correction or proof of improvement.')
        for axis, field, index, centered in (('horizontal', 'head_horizontal', 1, 'center'),
                                              ('vertical', 'head_vertical', 2, 'level')):
            relation = current[field]
            unresolved = (current['head_visible'] == 'yes' and current['contact_finger'] != 'unknown'
                          and relation not in ('unknown', centered))
            entry = dict(button_relative_to_finger=relation,
                finger_relative_to_button=inverse[relation],
                unresolved_observations=int(unresolved),
                transitions_without_axis_command=0, chassis_axis='YZ'[index-1],
                recent_nonzero_axis_commands=[], axis_direction_reversals=0)
            result['axes'][axis] = entry
            if relation != 'unknown':
                entry['relation_statement'] = (
                    f'In HEAD, BUTTON is {relation} relative to selected FINGER; '
                    f'the same FINGER is {inverse[relation]} relative to BUTTON. '
                    'This describes image projection only.')
            if not unresolved:
                continue
            newer = self.current_id
            seen_axis = False
            last_sign = None
            for turn in reversed(self.turns):
                older = turn['before_observation_id']
                prior = self.assessments.get(older)
                if (turn['after_observation_id'] != newer or not prior
                        or prior['head_visible'] != 'yes'
                        or prior['contact_finger'] != current['contact_finger']
                        or prior[field] in ('unknown', centered)):
                    break
                command = actual_command(turn['result'])
                if command.get('type') in ('head', 'gripper'):
                    break
                if not command and turn['result'].get('status') != 'observed':
                    break  # Missing execution data is not evidence that an axis stayed still.
                heads = [json.loads(self.observations[oid]['content'][0]['text'])['state'].get('head_deg')
                         for oid in (older, newer)]
                if (any(h is None or len(h) != 2 or any(v is None for v in h) for h in heads)
                        or max(abs(a-b) for a, b in zip(*heads)) > 1.0):
                    break
                entry['unresolved_observations'] += 1
                delta = command.get('xyz_mm', [0, 0, 0])[index] if command.get('type') == 'move' else 0
                if delta:
                    seen_axis = True
                    sign = 1 if delta > 0 else -1
                    if last_sign is not None and sign != last_sign:
                        entry['axis_direction_reversals'] += 1
                    last_sign = sign
                    if len(entry['recent_nonzero_axis_commands']) < 3:
                        entry['recent_nonzero_axis_commands'].append(dict(delta_mm=delta,
                            before_observation_id=older, after_observation_id=newer,
                            result_status=turn['result'].get('status')))
                elif not seen_axis:
                    entry['transitions_without_axis_command'] += 1
                newer = older
        pending = [(name, entry) for name, entry in result['axes'].items() if entry['unresolved_observations']]
        if pending:
            # Attention is a prompt priority, not a direction, distance or execution gate.
            result['attention_axis'] = max(pending, key=lambda item: (
                item[1]['transitions_without_axis_command'], item[1]['unresolved_observations'],
                item[0] == 'vertical'))[0]
        return result

    def _progress(self, current):
        prior = self.assessments.get(self.previous_id)
        command = actual_command(self.memory['latest_result'])
        result = dict(source='categorical_model_comparison_not_metric', status='unknown',
            previous_observation_id=self.previous_id, observation_id=self.current_id, previous_command=command or None,
            limitation='Coarse image categories only; unchanged categories do not prove no physical improvement.')
        if not prior or not command or command.get('type') != 'move':
            return result
        before_state = json.loads(self.observations[self.previous_id]['content'][0]['text'])['state']
        after_state = json.loads(self.observations[self.current_id]['content'][0]['text'])['state']
        heads = (before_state.get('head_deg'), after_state.get('head_deg'))
        if (any(h is None or len(h) != 2 or any(x is None for x in h) for h in heads)
                or max(abs(a-b) for a, b in zip(*heads)) > 1.0
                or current['head_visible'] != 'yes' or prior['head_visible'] != 'yes'
                or current['contact_finger'] == 'unknown' or current['contact_finger'] != prior['contact_finger']):
            return result
        pairs = [(prior[k], current[k], center) for k, center in
                 (('head_horizontal', 'center'), ('head_vertical', 'level'))]
        if any('unknown' in (a, b) for a, b, _ in pairs):
            return result
        improved = any(a != c and b == c for a, b, c in pairs)
        worsened = any(a == c and b != c for a, b, c in pairs)
        # Crossing past center is unresolved, not numerical error reduction.
        if any(a != b and a != c and b != c for a, b, c in pairs):
            return result
        result['status'] = ('unknown' if improved and worsened else 'improved' if improved else
                            'worsened' if worsened else 'unchanged_or_unresolved')
        return result

    def _update_events(self, current):
        evidence = self.memory['evidence']
        entry = dict(observation_id=self.current_id, lamp=current['lamp'], view=current['lamp_view'],
                     source='model_judgment_not_human_verification')
        if evidence['initial_lamp'] is None:
            evidence['initial_lamp'] = entry
        if current['lamp'] == 'off':
            evidence['last_off'] = entry
        if current['lamp'] == 'on' and evidence['last_off'] and evidence['light_on'] is None:
            evidence['light_on'] = dict(previous_observation_id=evidence['last_off']['observation_id'],
                observation_id=self.current_id, source='model_observed_off_then_on_not_human_verification',
                view=current['lamp_view'])

    def prepare_decision(self, base, directory, cancel=None):
        from .localization import build_request, inspect
        from .grounding import verify_withdrawal, verify_activation
        from .control_policy import build_plan
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        def check():
            if cancel is not None and cancel.is_set():
                raise InterruptedError('operator_stopped')
        oid = self.current_id
        meta = self.decision_context('perception')
        atomic_json(directory/'decision_context.json', meta)
        try:
            check()
            if oid not in self.assessments:
                body = build_request(self.goal, self.observations[oid], self.memory['preferred_contact'])
                assessment = inspect(base, body, oid, directory/'perception', cancel)
                check()
                if oid != self.current_id:
                    raise ValueError('decision_observation_mismatch')
                self.assessments[oid] = copy.deepcopy(assessment)
            current = self.assessments[oid]
            if oid not in self.processed:
                progress = self._progress(current)
                # An off-to-on first noticed after retreat cannot retrospectively authorize that retreat as post-activation evidence.
                prior_light = copy.deepcopy(self.memory['evidence']['light_on'])
                self._update_events(current)
                self.memory['alignment_progress'] = progress
                self.memory['alignment_control'] = self._alignment_control(current)
                command = actual_command(self.memory['latest_result'])
                if (current['lamp'] == 'on' and self.previous_id
                        and self.memory['evidence']['light_on'] is None):
                    meta['stage'] = 'activation_verification'
                    atomic_json(directory/'decision_context.json', meta)
                    checked = verify_activation(base, self.observations[self.previous_id],
                        self.observations[oid], command, directory/'activation', cancel)
                    check()
                    self.memory['evidence']['activation_check'] = copy.deepcopy(checked)
                    if checked['light_on'] == 'yes':
                        self.memory['evidence']['light_on'] = dict(checked,
                            previous_observation_id=self.previous_id, observation_id=oid,
                            source='model_visual_off_to_on_verification_not_human_confirmation')
                if (prior_light and self.previous_id and command.get('type') == 'move'
                        and command.get('xyz_mm', [0])[0] < 0):
                    meta['stage'] = 'withdrawal_verification'
                    atomic_json(directory/'decision_context.json', meta)
                    checked = verify_withdrawal(base, self.observations[self.previous_id],
                        self.observations[oid], command, directory/'withdrawal', cancel)
                    check()
                    if checked['withdrawal'] == 'yes':
                        self.memory['evidence']['withdrawal'] = dict(checked,
                            previous_observation_id=self.previous_id, observation_id=oid,
                            source='model_visual_verification_not_human_confirmation')
                check()
                self.memory['control_plan'] = build_plan(current, self._state(oid),
                    previous_assessment=self.assessments.get(self.previous_id),
                    previous_state=self._state(self.previous_id),
                    latest_command=command or None, evidence=self.memory['evidence'],
                    last_head_target=self._last_head_target())
                self.processed.add(oid)
            atomic_json(directory/'alignment_progress.json', self.memory['alignment_progress'])
            atomic_json(directory/'control_plan.json', self.memory['control_plan'])
            atomic_json(directory/'control_memory.json', self.memory)
            self.check_observation(oid)
            return self.prepare(base, directory, cancel)
        except Exception as exc:
            meta['stage'] = 'interrupted' if isinstance(exc, InterruptedError) else 'failed'
            atomic_json(directory/'decision_context.json', meta)
            atomic_json(directory/'error.json', dict(stage='decision_prepare', error_type=type(exc).__name__, error=str(exc)))
            raise

    def prepare(self, base, directory, cancel=None):
        def check():
            if cancel is not None and cancel.is_set():
                raise InterruptedError('operator_stopped')
        check(); self.check_observation(self.current_id)
        body = self.body()
        # A bounded input has no accumulated image history. Count initially; a conservative
        # byte bound on changing text decides when another exact tokenization is needed.
        # Exclude immutable image bytes from this text bound.
        variable_messages = copy.deepcopy(body['messages'][2:])
        for m in variable_messages:
            if isinstance(m.get('content'), list):
                m['content'] = [p for p in m['content'] if p.get('type') != 'image_url']
        bound = len(json.dumps(variable_messages, ensure_ascii=False).encode())
        exact = None
        if self.estimated_tokens is None or self.estimated_tokens+bound >= self.budget:
            exact = tokenize(base, body); check()
            self.estimated_tokens = exact
            if exact > self.budget:
                raise ValueError('context_budget_exceeded_for_bounded_input')
        meta = dict(self.decision_context('action_request'), image_count=image_count(body['messages']),
                    compression_status='bounded_not_needed', input_tokens_exact=exact,
                    input_tokens_estimate=self.estimated_tokens+bound if exact is None else exact)
        atomic_json(Path(directory)/'decision_context.json', meta)
        atomic_json(Path(directory)/'context_metrics.json', meta)
        return body

    def parse_response(self, response, observation, session, directory):
        self.check_observation(observation['observation_id'])
        transformed = copy.deepcopy(response)
        message = transformed['choices'][0]['message']; calls = message.get('tool_calls') or []
        if len(calls) != 1:
            raise ValueError('one_tool_required')
        call = calls[0]['function']; args = json.loads(call['arguments'])
        if not isinstance(args, dict):
            raise ValueError('invalid_tool_arguments')
        basis = args.pop('visual_basis', None); effect = args.pop('expected_effect', None)
        assessment = self.assessments[self.current_id]
        expected = {k: assessment[k] for k in BASIS_FIELDS}
        if basis != expected:
            raise ValueError('visual_basis_must_match_CURRENT_assessment:'+text(expected)['text'])
        if effect not in EFFECTS.get(call['name'], []):
            raise ValueError('expected_effect_incompatible_with_actuator')
        reason_conflicts(args.get('reason', args.get('evidence', '')), assessment)
        call['arguments'] = json.dumps(args, ensure_ascii=False)
        proposal = parse_proposal(transformed, observation, session)
        if effect == 'retract' and args['xyz_mm'][0] >= 0:
            raise ValueError('retract_effect_requires_negative_X')
        if proposal['name'] == 'propose_move':
            motion_reason_conflicts(args['reason'], args['xyz_mm'])
            if effect.startswith('reduce_'):
                x, y, z = args['xyz_mm']
                if x != 0:
                    raise ValueError('alignment_correction_requires_X_zero: choose explicit approach/press for forward motion; wrist guard still required')
                if effect in ('reduce_horizontal_error', 'reduce_both_errors') and y == 0:
                    raise ValueError('horizontal_correction_requires_nonzero_Y')
                if effect in ('reduce_vertical_error', 'reduce_both_errors') and z == 0:
                    raise ValueError('vertical_correction_requires_nonzero_Z')
        if proposal['name'] == 'propose_head' and (self.memory.get('control_plan') or {}).get('stage') == 'get_view':
            last = self._last_head_target()
            current_head = (self._state(self.current_id) or {}).get('head_deg')
            prior_head = last.get('measured_head_deg') if last else None
            if (last and args['head_deg'] == last['head_deg'] and current_head and prior_head
                    and len(current_head) == len(prior_head) == 2
                    and all(v is not None for v in current_head+prior_head)
                    and max(abs(a-b) for a,b in zip(current_head, prior_head)) <= 1.0):
                raise ValueError('same_absolute_head_target_does_not_improve_view: choose a different suitable head pose or observe; no extra 30-degree increment is applied')
        if proposal['name'] == 'report_done':
            evidence = self.memory['evidence']; statement = args['evidence']
            if not evidence['light_on'] or not evidence['withdrawal']:
                raise ValueError('this_run_light_on_and_later_verified_withdrawal_required')
            if any(evidence[k]['observation_id'] not in statement for k in ('light_on', 'withdrawal')):
                raise ValueError('report_done_must_cite_light_on_and_withdrawal_observation_ids')
        self.candidate_basis = dict(observation_id=self.current_id, visual_basis=basis,
            expected_effect=effect, source='model_decision_checked_against_current_assessment',
            alignment_control_version='pressb-alignment-control-v1',
            control_policy_version='pressb-control-plan-v1',
            check_scope='Structured agreement, explicit scene relations and stated chassis motion signs; not general semantic proof or calibrated geometry.')
        atomic_json(Path(directory)/'decision_basis.json', self.candidate_basis)
        return proposal

    def accept(self, response):
        self.check_observation(self.current_id)
        if self.pending is not None:
            raise ValueError('unresolved_tool_call')
        message = response['choices'][0]['message']; call = message['tool_calls'][0]
        if not call.get('id') or not isinstance(call['id'], str):
            raise ValueError('tool_call_id_required')
        if not self.candidate_basis or self.candidate_basis['observation_id'] != self.current_id:
            raise ValueError('decision_basis_required')
        self.pending = copy.deepcopy(message)
        self.last_usage = response.get('usage') or {}

    def append_result(self, proposal, result, obs, head, wrist):
        if self.pending is None:
            raise ValueError('model_response_not_accepted')
        original = self.pending['tool_calls'][0]['function']
        args = json.loads(original['arguments']); args.pop('visual_basis'); args.pop('expected_effect')
        if original['name'] != proposal['name'] or args != proposal['arguments']:
            raise ValueError('tool_result_proposal_mismatch')
        if obs['observation_id'] in self.observations:
            raise ValueError('new_observation_required')
        feedback = execution(result) if result is not None else dict(status='observed', observation_id=obs['observation_id'])
        # Only actual recorded commands are used to evaluate a saved replay's state changes.
        memory_feedback = copy.deepcopy(feedback)
        memory_feedback['command'] = actual_command(feedback) or None
        if feedback.get('replay'):
            memory_feedback['command_source'] = 'saved_recording_not_replay_proposal'
        command = actual_command(feedback)
        if command.get('type') == 'move' and any(command.get('xyz_mm', [])):
            if self.memory['evidence']['withdrawal']:
                self.memory['evidence']['withdrawal_invalidated_by'] = dict(
                    command_id=feedback.get('command_id'), command=command,
                    previous_withdrawal_observation_id=self.memory['evidence']['withdrawal']['observation_id'],
                    reason='New hand movement requires a fresh visual withdrawal check before completion.')
            self.memory['evidence']['withdrawal'] = None
        self.turns.append(dict(assistant=self.pending, proposal=copy.deepcopy(proposal),
            tool=dict(role='tool', tool_call_id=self.pending['tool_calls'][0]['id'], content=text(feedback)['text']),
            before_observation_id=self.current_id, after_observation_id=obs['observation_id'], result=feedback))
        self.memory['latest_result'] = memory_feedback
        self.memory['previous_alignment'] = copy.deepcopy(self.assessments[self.current_id])
        self.previous_id, self.current_id = self.current_id, obs['observation_id']
        self.observations[self.current_id] = live_message(obs, head, wrist)
        self.pending = None; self.candidate_basis = None

