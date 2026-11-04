"""Append-only task conversation, compact model views and evidence-preserving compaction."""
import copy
import json
import time
from pathlib import Path

from .common import SESSION_SCHEMA, atomic_json, request
from .demonstration import image_content
from .model_context import FORMAT, text, observation, session as session_view, execution


def image_count(messages):
    return sum(item.get('type') == 'image_url' for message in messages
               if isinstance(message.get('content'), list) for item in message['content'])


def tokenize(base, body, return_tokens=False):
    from . import model_provider
    if not model_provider.supports_tokenize():
        # Hosted APIs have no /tokenize; budget on a conservative estimate instead.
        count = model_provider.estimate_tokens(body)
        return dict(count=count, estimated=True) if return_tokens else count
    payload = {k: body[k] for k in ('model', 'messages', 'tools', 'chat_template_kwargs') if k in body}
    payload['add_generation_prompt'] = True
    result = request(base.rstrip('/').removesuffix('/v1'), '/tokenize', payload, timeout=120)
    return result if return_tokens else result['count']


def live_message(obs, head, wrist):
    return dict(role='user', content=[text(observation(obs)),
        dict(type='text', text='LIVE HEAD camera; current observation, not a demonstration.'), image_content(head),
        dict(type='text', text='LIVE RIGHT WRIST camera; original image; depth pixel is not a contact point.'), image_content(wrist)])


def request_body(messages, tools=None, max_tokens=512):
    from .client import TOOLS
    from . import model_provider
    return dict(**model_provider.body_fields(), temperature=0, max_tokens=max_tokens, stream=False,
                tools=TOOLS if tools is None else tools, tool_choice='required', messages=messages)


EVIDENCE_FIELDS = ('initial_light', 'light_on', 'withdrawal')
VIEW_ACTIONS = ('propose_head', 'request_observation', 'report_unable')
SUMMARY_TOOL = dict(type='function', function=dict(name='summarize_context',
    description='Read-only summary of this run; no robot command.', parameters=dict(type='object',
    properties=dict(progress=dict(type='string'), uncertainties=dict(type='string'), **{
        name: dict(type='object', properties=dict(observation_id=dict(type=['string', 'null']),
                   assessment=dict(type='string')), required=['observation_id', 'assessment'], additionalProperties=False)
        for name in EVIDENCE_FIELDS}), required=['progress', 'uncertainties', *EVIDENCE_FIELDS], additionalProperties=False)))


class Conversation:
    def __init__(self, goal, obs, head, wrist, session, prompt, demo=None, failure_case=None, budget=None,
                 decision_mode='direct', monitor_progress=False, acquire_view=False):
        if session.get('schema') != SESSION_SCHEMA:
            raise ValueError('keyboard_session_schema_required')
        # budget=None: no context limit; the full history is sent and never compacted.
        if budget is not None and budget <= 0:
            raise ValueError('positive_context_budget_required')
        if decision_mode not in ('direct', 'separated'):
            raise ValueError('invalid_decision_mode')
        self.decision_mode = decision_mode
        self.monitor_progress = bool(monitor_progress and decision_mode == 'direct')
        if acquire_view and not self.monitor_progress:
            raise ValueError('view_acquisition_requires_direct_progress_monitor')
        self.acquire_view = bool(acquire_view)
        self.view_phase = 'get_view' if self.acquire_view else 'ready'
        self.view_reason = 'initial_head_framing_unverified' if self.acquire_view else None
        self.view_assessments = {}
        self.view_contexts = {}
        self.recovery_contexts = {}
        self.view_conflict_id = None
        self.view_validated_id = None
        self.progress_comparison_start = 0
        self.progress_records = {}
        self.progress_reviews = {}
        self.progress_contexts = {}
        self.candidate_progress = None
        self.last_review_round = 0
        self.progress_unresolved = {}
        self.progress_reference = ('white_arrow' if 'white arrow' in prompt.lower() else
                                   'red_marker' if 'red-marked' in prompt.lower() else 'selected_fingertip')
        if self.monitor_progress:
            prompt += ('\nPROGRESS RECORD: Include progress in every tool call, bound to the CURRENT observation ID. '
                       'Report the button relative to the selected physical contact end in one identifiable view; '
                       'unknown remains unknown. last_effect describes the last actual operation, not its intent. '
                       'A DIRECT_PROGRESS_REVIEW is an independent model judgment of an actual image pair. '
                       'Use it to reassess the subgoal; do not repeat a direction it finds unchanged, worsened, '
                       'crossed, reached or unknown. Neither model is ground truth: a disagreement enters '
                       'recover_view, where only head adjustment, observation or report_unable is available. '
                       'Do not merely change your stated relation to agree with the reviewer. '
                       'Same-height image projection is not 3D contact or forward permission.')
        if self.acquire_view:
            prompt += ('\nVIEW ACQUISITION: Before hand motion, CURRENT_VIEW_ACQUISITION must establish '
                       'a usable HEAD view of the panel, target button and identifiable contact finger. '
                       'In get_view or recover_view, use only propose_head, request_observation or report_unable. '
                       'Choose an absolute head pose; pitch +30 degrees is only a look-down reference. '
                       'A completed head command is not proof of adequate framing; inspect the resulting image. '
                       'Refreshing a cropped view without changing framing cannot recover missing objects. '
                       'Localization and derived image relations are model evidence, not calibrated contact '
                       'or a pixel-to-robot command. Current images override historical examples.')
        self.format = 'pressb-conversation-v2' if decision_mode == 'separated' else FORMAT
        self.goal = goal
        self.assessments = {}
        if decision_mode == 'separated':
            from .perception import ACTION_GUIDANCE
            prompt += '\n'+ACTION_GUIDANCE
        content = [text(dict(goal=goal, session=session_view(session), input_format=self.format))]
        if demo:
            content.extend(demo.content)
        if failure_case:
            content.extend(failure_case.content)
        self.fixed = [dict(role='system', content=prompt), dict(role='user', content=content)]
        self.initial = live_message(obs, head, wrist)
        self.observations = {obs['observation_id']: self.initial}
        self.current_id = obs['observation_id']
        self.initial_id = self.current_id
        self.turns = []
        self.start = 0
        self.memory = []
        self.generation = 0
        self.pending = None
        self.budget = budget
        self.estimated_tokens = None
        self.last_usage = {}

    def messages(self):
        initial = self.initial if self.start == 0 else self.turns[self.start-1]['after']
        result = self.fixed + self.memory + [initial]
        def assessment(message):
            oid = json.loads(message['content'][0]['text'])['observation_id']
            if oid in self.assessments:
                result.append(dict(role='user', content=[text(dict(
                    context_type='INDEPENDENT_PERCEPTION_MODEL_ASSESSMENT', **self.assessments[oid]))]))
            if oid in self.view_contexts:
                result.append(copy.deepcopy(self.view_contexts[oid]))
            if oid in self.progress_contexts:
                result.append(copy.deepcopy(self.progress_contexts[oid]))
            if oid in self.recovery_contexts:
                result.append(copy.deepcopy(self.recovery_contexts[oid]))
        assessment(initial)
        for turn in self.turns[self.start:]:
            result.extend([turn['assistant'], turn['tool'], turn['after']])
            assessment(turn['after'])
        return result

    @property
    def previous_id(self):
        return self.turns[-1]['ledger']['before_observation_id'] if self.turns else None

    def check_observation(self, observation_id):
        if observation_id != self.current_id:
            raise ValueError('decision_observation_mismatch')
        if self.decision_mode == 'separated' and observation_id not in self.assessments:
            raise ValueError('current_perception_required')

    def attach_assessment(self, assessment):
        oid = assessment['observation_id']
        if oid != self.current_id or assessment['previous_observation_id'] != self.previous_id:
            raise ValueError('perception_observation_mismatch')
        if self.pending is not None or oid in self.assessments:
            raise ValueError('perception_already_committed')
        self.assessments[oid] = copy.deepcopy(assessment)
        if self.estimated_tokens is not None:
            self.estimated_tokens += len(text(assessment)['text'].encode('utf-8')) + 256

    def decision_context(self, stage):
        return dict(format=self.format, decision_mode=self.decision_mode, observation_id=self.current_id,
                    previous_observation_id=self.previous_id, stage=stage,
                    perception_format='pressb-perception-v1' if self.decision_mode == 'separated' else None,
                    compression_generation=self.generation,
                    progress_monitor_format='pressb-progress-v1' if self.monitor_progress else None,
                    view_acquisition_format='pressb-view-acquisition-v1' if self.acquire_view or self.view_conflict_id else None,
                    view_phase=self.view_phase if self.monitor_progress else None)

    def prepare_decision(self, base, directory, cancel=None):
        """Write round identity before any network request; inspection precedes action."""
        from .perception import build_request, inspect
        directory = Path(directory)
        meta = self.decision_context('perception' if self.decision_mode == 'separated' else 'context_prepare')
        atomic_json(directory/'decision_context.json', meta)
        oid = self.current_id
        try:
            if cancel is not None and cancel.is_set():
                raise InterruptedError('operator_stopped')
            if self.decision_mode == 'separated' and oid not in self.assessments:
                body = build_request(self.goal, self.observations[oid], self.observations.get(self.previous_id),
                                     self.turns[-1]['ledger']['result'] if self.turns else None)
                result = inspect(base, body, oid, self.previous_id, directory/'perception', cancel)
                if cancel is not None and cancel.is_set():
                    raise InterruptedError('operator_stopped')
                self.attach_assessment(result)
            if self.monitor_progress and self.view_phase != 'ready':
                self.prepare_view(base, directory, cancel)
            if self.monitor_progress and self.view_phase == 'ready':
                self.prepare_progress(base, directory, cancel)
            if self.acquire_view or self.view_conflict_id:
                self._save_view_state(directory)
            self.check_observation(oid)
            return self.prepare(base, directory, cancel)
        except Exception as exc:
            meta = self.decision_context('interrupted' if isinstance(exc, InterruptedError) else 'failed')
            atomic_json(directory/'decision_context.json', meta)
            if not (directory/'error.json').exists():
                atomic_json(directory/'error.json', dict(stage='decision_prepare', error_type=type(exc).__name__, error=str(exc)))
            raise

    def body(self):
        if self.monitor_progress:
            from .client import TOOLS
            from .progress import extend_tools
            tools = extend_tools(TOOLS)
            if self.view_phase != 'ready':
                tools = [tool for tool in tools if tool['function']['name'] in VIEW_ACTIONS]
            return request_body(self.messages(), tools)
        return request_body(self.messages())

    def view_state(self):
        return dict(format='pressb-view-state-v1', source='program_phase_on_model_localization',
                    phase=self.view_phase, observation_id=self.current_id,
                    assessment_observation_id=self.view_validated_id,
                    conflict_observation_id=self.view_conflict_id, reason=self.view_reason,
                    allowed_tools=list(VIEW_ACTIONS) if self.view_phase != 'ready' else None)

    def _save_view_state(self, directory):
        atomic_json(Path(directory)/'view_state.json', self.view_state())

    def _check_view_action(self, proposal):
        if self.view_phase != 'ready' and proposal['name'] not in VIEW_ACTIONS:
            raise ValueError('view_acquisition_required:'+self.view_phase+
                '; The candidate was NOT executed. Only propose_head, request_observation or report_unable '
                'is available until a NEW observation provides a usable HEAD view. Choose a head pose '
                'to reveal the panel, button and contact finger. Do not claim agreement to bypass recovery.')

    def prepare_view(self, base, directory, cancel=None):
        from . import view_acquisition
        directory = Path(directory)
        oid = self.current_id
        self._save_view_state(directory)
        # A contradiction cannot be cleared by another answer on the identical snapshot.
        if oid == self.view_conflict_id:
            return
        if oid not in self.view_assessments:
            atomic_json(directory/'decision_context.json', self.decision_context('view_acquisition'))
            try:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError('operator_stopped')
                body = view_acquisition.build_request(self.goal, self.observations[oid], self.progress_reference)
                assessment = view_acquisition.inspect(base, body, oid, directory/'view_acquisition', cancel)
                if cancel is not None and cancel.is_set():
                    raise InterruptedError('operator_stopped')
                self.check_observation(oid)
                if (assessment.get('observation_id') != oid or assessment.get('format') != view_acquisition.FORMAT
                        or assessment.get('source') != view_acquisition.SOURCE):
                    raise ValueError('view_acquisition_observation_mismatch')
            except Exception as exc:
                atomic_json(directory/'error.json', dict(stage='view_acquisition', error_type=type(exc).__name__, error=str(exc)))
                if isinstance(exc, InterruptedError):
                    raise
                raise RuntimeError('view_acquisition_failed:'+str(exc)) from exc
            self.view_assessments[oid] = copy.deepcopy(assessment)
            self.view_validated_id = oid
            recovering = self.view_phase == 'recover_view'
            if assessment['usable_head']:
                self.view_phase = 'ready'
                self.view_reason = 'fresh_localized_view_after_conflict' if recovering else 'head_view_localized'
                if recovering:
                    # Only a fresh, independently localized usable view establishes
                    # a new comparison baseline. An ordinary refresh does not reset it.
                    self.progress_comparison_start = len(self.turns)
                    self.last_review_round = len(self.turns)
                    self.progress_unresolved = {}
            else:
                self.view_reason = 'head_panel_button_or_contact_not_fully_identified'
            context = dict(context_type='CURRENT_VIEW_ACQUISITION', assessment=assessment,
                           view_state=self.view_state(),
                           instruction='Model localization only. Image coordinates do not generate robot moves; '
                           'obtain a usable HEAD view before hand motion. Earlier alignment claims and reviews '
                           'are not current facts. After conflict recovery this fresh localization establishes '
                           'a new comparison baseline; use current evidence, not agreement with an old reviewer.')
            self.view_contexts[oid] = dict(role='user', content=[text(context)])
            if self.estimated_tokens is not None:
                self.estimated_tokens += len(text(context)['text'].encode('utf-8')) + 256
        self._save_view_state(directory)

    def _enter_view_recovery(self, directory, record, review, conflicts):
        oid = self.current_id
        self.view_phase = 'recover_view'
        self.view_reason = 'action_review_disagreement'
        self.view_conflict_id = oid
        conflict = dict(source='model_disagreement_not_visual_fact', observation_id=oid,
                        model_claim=copy.deepcopy(record), review_claim=copy.deepcopy(review),
                        conflicting_axes=conflicts)
        atomic_json(Path(directory)/'progress_review'/'conflict.json', conflict)
        context = dict(context_type='CURRENT_VISUAL_CONFLICT_RECOVERY', conflict=conflict,
                       view_state=self.view_state(), instruction='Neither judgment is ground truth. '
                       'The rejected motion was NOT executed. Do not change your relation simply to agree. '
                       'Obtain a new observation through head adjustment or observation; only head, observation '
                       'or report_unable is available until independent localization establishes a usable HEAD view.')
        if oid not in self.recovery_contexts:
            self.recovery_contexts[oid] = dict(role='user', content=[text(context)])
            if self.estimated_tokens is not None:
                self.estimated_tokens += len(text(context)['text'].encode('utf-8')) + 256
        self._save_view_state(directory)
        atomic_json(Path(directory)/'decision_context.json', self.decision_context('action_request'))

    def parse_response(self, response, obs, session, directory):
        from .client import parse_proposal
        if not self.monitor_progress:
            return parse_proposal(response, obs, session)
        from .progress import validate_progress
        self.candidate_progress = None
        clean = copy.deepcopy(response)
        call = clean['choices'][0]['message']['tool_calls'][0]['function']
        arguments = json.loads(call['arguments'])
        if not isinstance(arguments, dict):
            raise ValueError('invalid_tool_arguments')
        progress = arguments.pop('progress', None)
        record = validate_progress(progress, obs['observation_id'], bool(self.turns))
        if (self.turns and record['last_effect'] == 'no_motion'
                and self.turns[-1]['ledger']['result'].get('status') != 'observed'):
            raise ValueError('progress_no_motion_requires_observation_only_previous_turn')
        call['arguments'] = json.dumps(arguments)
        proposal = parse_proposal(clean, obs, session)
        self.candidate_progress = record
        return proposal

    def _last_progress_move(self):
        # Fresh observations do not erase the last actual motion. Head/gripper changes
        # end this comparison interval; no skipped motion is hidden in a review pair.
        for turn in reversed(self.turns[self.progress_comparison_start:]):
            ledger = turn['ledger']
            result = ledger['result']
            if result.get('status') == 'observed':
                continue
            if (result.get('status') == 'completed' and result.get('sends_completed') is True
                    and result.get('command', {}).get('type') == 'move'):
                if any(result['command']['xyz_mm']):
                    return ledger
                continue
            return None
        return None

    def _progress_snapshot(self, directory, proposal=None):
        from .progress import monitor
        value = monitor(self.turns[self.progress_comparison_start:], self.current_id, self.candidate_progress if proposal else None,
                        proposal, since_round=self.last_review_round)
        if self.progress_unresolved and self._last_progress_move() is not None:
            value['unresolved_directions'] = copy.deepcopy(self.progress_unresolved)
            value['reasons'].append('unresolved_progress_requires_fresh_review')
            value['review_required'] = True
        record = dict(format='pressb-progress-v1', observation_id=self.current_id,
                      candidate_progress=copy.deepcopy(self.candidate_progress) if proposal else None,
                      monitor=value)
        trigger = Path(directory)/'progress_review'/'trigger.json'
        if trigger.exists():
            record['review_trigger'] = json.loads(trigger.read_text())
        atomic_json(Path(directory)/'progress.json', record)
        return value

    def _review_progress(self, base, directory, cancel):
        from . import progress
        oid = self.current_id
        if oid in self.progress_reviews:
            return self.progress_reviews[oid]
        last = self._last_progress_move()
        if last is None:
            return None
        directory = Path(directory)
        snapshot_path = directory/'progress.json'
        if snapshot_path.exists():
            atomic_json(directory/'progress_review'/'trigger.json', json.loads(snapshot_path.read_text()))
        atomic_json(directory/'decision_context.json', self.decision_context('progress_review'))
        try:
            body = progress.build_review_request(self.goal, self.observations[last['before_observation_id']],
                self.observations[oid], last['result'], contact_reference=self.progress_reference)
            review = progress.inspect(base, body, oid, last['before_observation_id'],
                                      directory/'progress_review', cancel)
            if cancel is not None and cancel.is_set():
                raise InterruptedError('operator_stopped')
            self.check_observation(oid)
            if (review.get('observation_id') != oid
                    or review.get('previous_observation_id') != last['before_observation_id']
                    or review.get('evidence_observation_ids') != [last['before_observation_id'], oid]):
                raise ValueError('progress_review_observation_mismatch')
        except Exception as exc:
            atomic_json(directory/'decision_context.json', self.decision_context(
                'interrupted' if isinstance(exc, InterruptedError) else 'failed'))
            atomic_json(directory/'error.json', dict(stage='progress_review', error_type=type(exc).__name__, error=str(exc)))
            if isinstance(exc, InterruptedError):
                raise
            # A review transport/format failure must not become an action repair.
            raise RuntimeError('progress_review_failed:'+str(exc)) from exc
        self.progress_reviews[oid] = copy.deepcopy(review)
        self.progress_unresolved = {}
        command = last['result']['command']
        for index, axis in enumerate(('X', 'Y', 'Z')):
            value = command['xyz_mm'][index]
            if value:
                delta = [0, 0, 0]
                delta[index] = value
                candidate = dict(name='propose_move', arguments=dict(xyz_mm=delta))
                if progress.continuation_errors(review, candidate, command):
                    self.progress_unresolved[axis] = 1 if value > 0 else -1
        context = dict(context_type='DIRECT_PROGRESS_REVIEW', assessment=review,
                       last_actual_command=last['result']['command'],
                       instruction='Reassess this subgoal using the CURRENT view. No automatic motion or direction reversal.')
        self.progress_contexts[oid] = dict(role='user', content=[text(context)])
        self.last_review_round = len(self.turns)
        if self.estimated_tokens is not None:
            self.estimated_tokens += len(text(context)['text'].encode('utf-8')) + 256
        atomic_json(directory/'decision_context.json', self.decision_context('action_request'))
        return review

    def prepare_progress(self, base, directory, cancel=None):
        signal = self._progress_snapshot(directory)
        if signal['review_required']:
            self._review_progress(base, directory, cancel)

    def review_proposal(self, base, proposal, directory, cancel=None):
        if not self.monitor_progress:
            return
        if self.view_phase != 'ready':
            self._save_view_state(directory)
            self._check_view_action(proposal)
            if self.current_id in self.progress_reviews:
                atomic_json(Path(directory)/'progress_review'/'decision.json', dict(
                    observation_id=self.current_id, proposal=proposal, allowed=True, reasons=[],
                    requires_new_decision=False, phase=self.view_phase,
                    source='program_rule_on_model_judgment'))
            return
        signal = self._progress_snapshot(directory, proposal)
        fresh_review = self.current_id not in self.progress_reviews
        if signal['review_required']:
            self._review_progress(base, directory, cancel)
        review = self.progress_reviews.get(self.current_id)
        if review is None:
            return
        from .progress import continuation_errors
        last = self._last_progress_move()
        if (review.get('observation_id') != self.current_id or last is None
                or review.get('previous_observation_id') != last['before_observation_id']):
            raise ValueError('progress_review_observation_mismatch')
        errors = continuation_errors(review, proposal, last['result']['command'] if last else {})
        record = self.candidate_progress
        conflicts = []
        if record and record['view'] in ('head', 'wrist'):
            for field in ('vertical', 'horizontal'):
                expected = review[record['view']+'_'+field]
                claimed = record['button_'+field]
                if expected != 'unknown' and claimed not in (expected, 'unknown'):
                    conflicts.append(field)
        if conflicts:
            self._enter_view_recovery(directory, record, review, conflicts)
            errors.append('visual_conflict_requires_new_view:'+','.join(conflicts))
        atomic_json(Path(directory)/'progress_review'/'decision.json', dict(
            observation_id=self.current_id, proposal=proposal, allowed=not errors and not fresh_review,
            reasons=errors, requires_new_decision=fresh_review, source='program_rule_on_model_judgment'))
        if errors or fresh_review:
            raise ValueError('progress_reassessment_required: '+('; '.join(errors) if errors else 'new independent evidence')+
                '. The candidate was NOT executed. A visual conflict requires head adjustment or a new observation, '
                'not agreement with the reviewer. Consume DIRECT_PROGRESS_REVIEW and choose a supported new action, '
                'improve the view, reobserve, or report unable. Do not repeat an unsupported same-axis direction.')

    def check_execution(self, proposal, observation_id):
        self.check_observation(observation_id)
        if not self.monitor_progress:
            return
        self._check_view_action(proposal)
        if self.pending is None or observation_id not in self.progress_records:
            raise ValueError('accepted_current_progress_required')
        raw = self.pending['tool_calls'][0]['function']
        arguments = json.loads(raw['arguments'])
        arguments.pop('progress')
        if proposal != dict(name=raw['name'], arguments=arguments):
            raise ValueError('accepted_progress_proposal_mismatch')
        if observation_id in self.progress_reviews:
            from .progress import continuation_errors
            last = self._last_progress_move()
            if (self.progress_reviews[observation_id].get('observation_id') != observation_id
                    or last is None or self.progress_reviews[observation_id].get('previous_observation_id') != last['before_observation_id']):
                raise ValueError('progress_review_observation_mismatch')
            errors = continuation_errors(self.progress_reviews[observation_id], proposal,
                                          last['result']['command'] if last else {})
            if errors:
                raise ValueError('progress_permission_invalid:'+ '; '.join(errors))

    def accept(self, response):
        self.check_observation(self.current_id)
        if self.pending is not None:
            raise ValueError('unresolved_tool_call')
        message = response['choices'][0]['message']
        calls = message.get('tool_calls') or []
        if len(calls) != 1 or not isinstance(calls[0].get('id'), str) or not calls[0]['id']:
            raise ValueError('tool_call_id_required')
        if self.monitor_progress:
            from .progress import validate_progress
            args = json.loads(calls[0]['function']['arguments'])
            self.progress_records[self.current_id] = validate_progress(args.get('progress'), self.current_id, bool(self.turns))
        # Preserve the model's actual call id, argument string and message (including optional fields).
        self.pending = copy.deepcopy(message)
        self.pending['role'] = 'assistant'
        self.last_usage = response.get('usage') or {}

    def action_arguments(self, arguments, name=None):
        """Remove conversation-only metadata without changing actuator parameters."""
        arguments = copy.deepcopy(arguments)
        if self.monitor_progress:
            arguments.pop('progress')
        return arguments

    def append_result(self, proposal, result, obs, head, wrist):
        if self.pending is None:
            raise ValueError('model_response_not_accepted')
        call = self.pending['tool_calls'][0]
        arguments = self.action_arguments(json.loads(call['function']['arguments']), call['function']['name'])
        if call['function']['name'] != proposal['name'] or arguments != proposal['arguments']:
            raise ValueError('tool_result_proposal_mismatch')
        if obs['observation_id'] in self.observations:
            raise ValueError('new_observation_required')
        after = live_message(obs, head, wrist)
        feedback = execution(result) if result is not None else dict(status='observed', observation_id=obs['observation_id'])
        tool = dict(role='tool', tool_call_id=call['id'], content=text(feedback)['text'])
        ledger = dict(round=len(self.turns)+1, before_observation_id=self.current_id,
                      after_observation_id=obs['observation_id'], proposal=copy.deepcopy(proposal), result=feedback)
        if self.monitor_progress:
            ledger['progress'] = copy.deepcopy(self.progress_records[self.current_id])
            if self.current_id in self.progress_reviews:
                ledger['progress_review'] = copy.deepcopy(self.progress_reviews[self.current_id])
            ledger['unresolved_progress_directions'] = copy.deepcopy(self.progress_unresolved)
            if self.acquire_view or self.view_conflict_id:
                ledger['view_state'] = self.view_state()
                if self.current_id in self.view_assessments:
                    ledger['view_acquisition'] = copy.deepcopy(self.view_assessments[self.current_id])
                if self.current_id in self.recovery_contexts:
                    ledger['visual_conflict'] = json.loads(self.recovery_contexts[self.current_id]['content'][0]['text'])['conflict']
            command = feedback.get('command', {})
            if (feedback.get('sends_completed') is True and
                    (command.get('type') in ('head', 'gripper') or any(command.get('xyz_mm', [])))):
                self.progress_unresolved = {}
            if (feedback.get('sends_completed') is True and command.get('type') == 'head'
                    and (self.acquire_view or self.view_conflict_id)):
                if self.view_phase != 'recover_view':
                    self.view_phase = 'get_view'
                self.view_reason = 'head_command_requires_new_framing_check'
                self.view_validated_id = None
        if self.current_id in self.assessments:
            assessment = self.assessments[self.current_id]
            ledger['perception_summary'] = {k: copy.deepcopy(assessment[k]) for k in (
                'source', 'observation_id', 'previous_observation_id', 'head_visibility', 'wrist_visibility',
                'head_relation', 'wrist_relation', 'head_light', 'wrist_light', 'alignment_change',
                'light_on', 'withdrawal', 'evidence_observation_ids', 'note')}
        self.turns.append(dict(assistant=self.pending, tool=tool, after=after, ledger=ledger))
        self.pending = None
        self.current_id = obs['observation_id']
        self.observations[self.current_id] = after
        # Conservative screen-sized image allowance; exact count is checked near the budget.
        from PIL import Image
        import io
        import math
        image_bound = sum(max(4096, math.ceil(Image.open(io.BytesIO(data)).width/14) *
                              math.ceil(Image.open(io.BytesIO(data)).height/14)*4) for data in (head, wrist))
        added_text = len((tool['content'] + after['content'][0]['text']).encode('utf-8')) + 1024
        previous = self.last_usage.get('prompt_tokens', self.estimated_tokens)
        self.estimated_tokens = (previous + self.last_usage.get('completion_tokens', 512) + image_bound + added_text
                                 if previous is not None else None)

    def prepare(self, base, directory, cancel=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        def check():
            if cancel is not None and cancel.is_set():
                raise InterruptedError('operator_stopped')
        check()
        self.check_observation(self.current_id)
        body = self.body()
        meta = dict(self.decision_context('context_prepare'),
                    image_count=image_count(body['messages']), compression_status='not_needed')
        atomic_json(directory/'decision_context.json', meta)
        atomic_json(directory/'context_metrics.json', meta)
        try:
            exact = None
            if self.budget is None:
                # No context limit: no counting, no compaction, no budget exit.
                meta.update(compression_generation=self.generation, image_count=image_count(body['messages']),
                            input_tokens_exact=None, input_tokens_estimate=None, stage='action_request')
                atomic_json(directory/'decision_context.json', meta)
                atomic_json(directory/'context_metrics.json', meta)
                return body
            if self.estimated_tokens is None or self.estimated_tokens >= self.budget-4096:
                exact = tokenize(base, body)
                check()
                self.estimated_tokens = exact
            if self.estimated_tokens > self.budget:
                meta['compression_status'] = 'running'
                atomic_json(directory/'context_metrics.json', meta)
                self.compact(base, directory/'compaction', cancel)
                check()
                body = self.body()
                exact = tokenize(base, body)
                check()
                self.estimated_tokens = exact
                if exact > self.budget:
                    raise ValueError('context_budget_exceeded_after_compaction')
                meta['compression_status'] = 'completed'
            meta.update(compression_generation=self.generation, image_count=image_count(body['messages']),
                        input_tokens_exact=exact, input_tokens_estimate=self.estimated_tokens, stage='action_request')
            atomic_json(directory/'decision_context.json', meta)
            atomic_json(directory/'context_metrics.json', meta)
            return body
        except Exception as exc:
            atomic_json(directory/'request.json', body)
            meta.update(compression_generation=self.generation, image_count=image_count(body['messages']))
            meta['compression_status'] = 'interrupted' if isinstance(exc, InterruptedError) else 'failed'
            atomic_json(directory/'context_metrics.json', meta)
            atomic_json(directory/'error.json', dict(stage='context_prepare', error_type=type(exc).__name__, error=str(exc)))
            raise

    def compact(self, base, directory, cancel=None):
        start = len(self.turns)-5
        if start <= self.start:
            raise ValueError('context_budget_too_small_for_cases_and_recent_rounds')
        directory = Path(directory)
        ledger = [turn['ledger'] for turn in self.turns[:start]]
        messages = [dict(role='system', content=(
            'Summarize this robot run only. Do not execute any action. Use summarize_context once, no prose. '
            'All assessments are MODEL JUDGMENTS, not human verification. Keep uncertainty explicit. '
            'initial_light means initial lamp state; light_on means observed activation in THIS run; '
            'withdrawal means subsequent observed retreat. Unknown evidence uses null observation_id. '
            'Select at most one actual live observation ID for each field; never cite historical demonstrations. '
            'Keep progress and uncertainties concise; do not promote prior model claims into visual facts.'))]
        messages += self.messages()[1:]
        messages += [dict(role='user', content='Summarize progress and unresolved alignment; identify initial lamp, light-on and withdrawal evidence. Unknown remains unknown.')]
        body = request_body(messages, [SUMMARY_TOOL], max_tokens=1024)
        initial_body = copy.deepcopy(body)
        body = copy.deepcopy(initial_body)
        atomic_json(directory/'request.json', body)
        started = time.monotonic()
        metrics = dict(attempts=0, validation_failures=0, prompt_tokens=0, completion_tokens=0,
                       image_count=image_count(body['messages']), status='requesting')
        def check():
            if cancel is not None and cancel.is_set():
                raise InterruptedError('operator_stopped')
        def save_metrics():
            metrics['elapsed_s'] = time.monotonic()-started
            atomic_json(directory/'metrics.json', metrics)
        try:
            attempt = 0
            while True:
                check()
                metrics.update(attempts=attempt+1, status='requesting')
                save_metrics()
                response = request(base, '/chat/completions', body, timeout=120)
                atomic_json(directory/f'response-{attempt}.json', response)
                usage = (response.get('usage') or {}) if isinstance(response, dict) else {}
                if not isinstance(usage, dict):
                    usage = {}
                metrics['usage'] = usage
                for field in ('prompt_tokens', 'completion_tokens'):
                    metrics[field] += usage.get(field, 0)
                save_metrics()
                check()
                try:
                    summary = self.validate_summary(response)
                except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                    check()
                    atomic_json(directory/f'validation-{attempt}.json', dict(attempt=attempt, valid=False,
                                error_type=type(exc).__name__, error=str(exc)))
                    metrics.update(status='repairing', validation_failures=metrics['validation_failures']+1,
                                   last_validation_error=str(exc))
                    save_metrics()
                    repair = ('Invalid summary: '+str(exc)+'. Correct the summary tool call for the original history. '
                              'Previous invalid response: '+json.dumps(
                                  response.get('choices') if isinstance(response, dict) else response, ensure_ascii=False))
                    body = dict(initial_body, messages=copy.deepcopy(initial_body['messages'])+
                                [dict(role='user', content=repair)])
                    attempt += 1
                    atomic_json(directory/f'repair-request-{attempt}.json', body)
                    atomic_json(directory/'repair-request.json', body)
                    continue
                check()
                atomic_json(directory/f'validation-{attempt}.json', dict(attempt=attempt, valid=True))
                metrics['status'] = 'validated'
                save_metrics()
                break
            retained = {self.turns[start-1]['ledger']['after_observation_id']}
            retained.update(t['ledger']['after_observation_id'] for t in self.turns[start:])
            pin_ids = list(dict.fromkeys(summary[k]['observation_id'] for k in EVIDENCE_FIELDS
                                        if summary[k]['observation_id'] and summary[k]['observation_id'] not in retained))
            memory = [dict(role='user', content=[text(dict(context_type='MODEL_MEMORY_NOT_HUMAN_VERIFICATION',
                        summary=summary, earlier_action_ledger=ledger))])]
            for oid in pin_ids:
                pinned = copy.deepcopy(self.observations[oid])
                data = json.loads(pinned['content'][0]['text'])
                data['context_type'] = 'EARLIER_OBSERVATION_THIS_RUN_NOT_CURRENT'
                pinned['content'][0] = text(data)
                pinned['content'][1]['text'] = 'EARLIER THIS RUN: HEAD; not current.'
                pinned['content'][3]['text'] = 'EARLIER THIS RUN: RIGHT WRIST; not current.'
                memory.append(pinned)
                if oid in self.assessments:
                    memory.append(dict(role='user', content=[text(dict(
                        context_type='EARLIER_PERCEPTION_MODEL_ASSESSMENT_NOT_CURRENT', **self.assessments[oid]))]))
            check()
            self.memory, self.start = memory, start
            self.generation += 1
            atomic_json(directory/'result.json', dict(summary=summary, pinned_observation_ids=pin_ids,
                earlier_action_ledger=ledger, generation=self.generation, retained_recent_rounds=5,
                elapsed_s=time.monotonic()-started))
        except Exception as exc:
            metrics['status'] = 'interrupted' if isinstance(exc, InterruptedError) else 'failed'
            atomic_json(directory/'error.json', dict(error_type=type(exc).__name__, error=str(exc), elapsed_s=time.monotonic()-started))
            raise
        finally:
            save_metrics()

    def validate_summary(self, response):
        choices = response['choices']
        if len(choices) != 1 or choices[0]['finish_reason'] != 'tool_calls':
            raise ValueError('one_summary_tool_required')
        message = choices[0]['message']
        calls = message['tool_calls']
        if len(calls) != 1 or (message.get('content') or '').strip() or calls[0]['function']['name'] != 'summarize_context':
            raise ValueError('one_summary_tool_required')
        summary = json.loads(calls[0]['function']['arguments'])
        if not isinstance(summary, dict) or set(summary) != {'progress', 'uncertainties', *EVIDENCE_FIELDS}:
            raise ValueError('invalid_summary_fields')
        if any(not isinstance(summary[k], str) or not summary[k].strip() for k in ('progress', 'uncertainties')):
            raise ValueError('summary_text_required')
        for key in EVIDENCE_FIELDS:
            item = summary[key]
            if not isinstance(item, dict) or set(item) != {'observation_id', 'assessment'} or not isinstance(item['assessment'], str) or not item['assessment'].strip():
                raise ValueError('invalid_evidence_fields')
            if item['observation_id'] is not None and (not isinstance(item['observation_id'], str) or item['observation_id'] not in self.observations):
                raise ValueError('evidence_must_reference_this_run')
        if summary['initial_light']['observation_id'] not in (None, self.initial_id):
            raise ValueError('initial_light_must_reference_initial_observation')
        order = list(self.observations)
        light, withdrawal = summary['light_on']['observation_id'], summary['withdrawal']['observation_id']
        if light and withdrawal and order.index(withdrawal) <= order.index(light):
            raise ValueError('withdrawal_must_follow_light_on')
        return summary
