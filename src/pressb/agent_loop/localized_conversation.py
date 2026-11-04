"""Persistent decisions with canonical localization and triggered hypothesis review."""
import copy
import json
from pathlib import Path

from .common import atomic_json, SESSION_SCHEMA
from .conversation import Conversation, EVIDENCE_FIELDS, request_body
from .client import TOOLS, validate_command, require_forward_permission
from .model_context import text
from . import model_localization as locator
from . import localized_reassessment as reassess
from . import wrist_overlap_guard as overlap
from . import head_view_advice
from . import press_phase
from . import press_retry
from . import calibrated_tip
from . import wrist_direction_rules as direction_rules

FORMAT = 'pressb-conversation-v5'
PROGRESS_FORMAT = 'pressb-localized-progress-v3'
GUIDANCE = '''
RULE-GUIDED MODEL DECISION
The current pictures, localization estimates and raw coordinate differences are
evidence for YOUR judgment. Follow the current visual subgoal: obtain a useful
view, improve marked-end alignment, approach/press, or verify/withdraw. For actions
that continue the run, choose an expected image change; a visible target alone does not
make forward motion the default. Image centering is not the contact objective.
Before positive-X motion, an independent CURRENT WRIST check must confirm the
marked physical finger tip lies inside the button-face box in BOTH image axes.
Until it passes, align in Y/Z (X=0) or improve the view. Negative-X withdrawal is
always available. Permission never carries to a new observation. A returned
box may describe only a visible portion; its center is not automatically the
physical button center. Reconcile estimates with the actual images.
For this fixed-orientation wrist setup, use this explicit direction policy:
button face left of marked physical tip -> +Y; right -> -Y;
button face above the tip -> +Z; below -> -Z.
These mean the entire face interval lies outside the tip on that image axis.
Already inside that interval -> zero correction on that axis. Button left AND
above -> [0,+a,+b], with a,b positive millimeter magnitudes YOU choose, coarse to
fine. Positive Y moves the button right in the wrist image; positive Z moves it
down. Use the actual button face and physical end, not panel/sticker centers.
This mapping is specific to the present wrist mounting and preserved hand
orientation, not a general image-to-robot transform or a rule for head images.
Do not invent a pixel-to-millimeter scale. If actual motion contradicts the
mapping, reacquire/reassess instead of repeatedly trying an unsupported direction.
If +Z improves the gap and the button remains above, continue +Z; do not alternate
back to -Z after restoring the view. Reverse only after crossing or resolving
contradictory feedback. Apply the same logic to Y. Changing button-box size can
shrink error_button_units without real progress: compare consistent objects in
the images and pixel offsets. You may adjust the head,
refresh observations, move the hand or gripper, or report unable as appropriate.
Repeated actual motion with unknown, unchanged or worsening projection, and
repeated forward motion, triggers an independent MOTION_HYPOTHESIS_REVIEW before
the next decision. It tests the assumption behind repetition, not just the wording
of your reason. After review include the required reassessment: current evidence,
expected visual change, and which assumption to reconsider if that change fails.
You may disagree only with specific current image evidence. A visible target alone
does not support a distance-only explanation or override the wrist overlap check.
Use larger Y/Z corrections for clear large deviations with a supported direction,
then smaller corrections near overlap; combine Y/Z where useful. There is no fixed
step size, positive-X displacement cap, automatic XYZ correction or contact guarantee.
Keep one short reason with each action. Check the command signs: robot right=-Y,
left=+Y, up=+Z, down=-Z. Follow the task-specific wrist table for alignment signs;
choose the magnitudes, not a new sign convention. Unknown objects require reacquisition.
Read CURRENT_WRIST_ALIGNMENT_DIRECTION_RULE for this observation's explicit Y/Z
signs. If its objects match the current wrist image, choose nonzero magnitudes
for the indicated nonzero corrections. An all-zero move does not realign or
request a new image; use request_observation when a fresh image is needed.
Report done only after seeing this run's button light up and subsequent withdrawal,
with the light STILL on in a HEAD image taken after the withdrawal; describe the
visual evidence. A press that reaches the contact depth without the light is a
miss: withdraw, open, realign, re-range, close and press again; do not keep pushing. Completion remains a model judgment, not human confirmation.
'''


def press_status(state):
    """Tell the model which stage of align-open → close → press it is in. Advisory text only."""
    state = state or {}
    phase, gripper = state.get('phase'), state.get('gripper_state')
    budget, consumed = state.get('budget_m') or 0.0, state.get('consumed_m') or 0.0
    remaining = state.get('remaining_m')
    if remaining is None:
        remaining = max(0.0, budget-consumed)
    target = state.get('target_m') or 0.0
    limits = state.get('limits') or {}
    band = [round(limits[k]*1000, 1) if limits.get(k) is not None else None
            for k in ('range_min_m', 'range_max_m')]
    if phase == press_phase.ALIGN:
        stage = 'align_open'
        instruction = ('Order: with the gripper OPEN finish alignment and ranging, then close, then press by '
                       'distance. Keep the gripper OPEN now. Align the button face onto the fixed merged-end aim '
                       'point with Y/Z, and approach with positive X only when the wrist overlap check passes. '
                       f'Size the approach so the next open-gripper depth lands inside the close band '
                       f'{band[0]}-{band[1]} mm, not at the contact plane; the open gripper must not touch the '
                       'panel. The range is frozen automatically once a trusted open-gripper depth at the aim '
                       'pixel lies within that band; until then do not close. Use the HEAD view to estimate the '
                       'remaining distance and whether the hand is in front of the target button.')
    elif gripper != 'closed':
        stage = 'close'
        instruction = ('The open-gripper range is frozen. If the wrist view still shows the button on the aim '
                       'point, close the gripper now (gripper c), then press by distance. Do not move forward '
                       'while open.')
    elif remaining <= 0.0005:
        stage = 'press_reached'
        instruction = ('The live depth at the merged end has reached the press target (contact plus overpress) '
                       'or the frozen range is used up. Do not push further. Judge the target light from the '
                       'HEAD view, then withdraw with negative X and check the light is still on after withdrawal. '
                       'If the light is not on, the closed end most likely missed the button: do not push '
                       'again; withdraw to the close band, open (o), realign onto the aim pixel, re-range, '
                       'close and press.')
    else:
        stage = 'press'
        instruction = ('The gripper is closed. Press with positive X of at most '
                       f'{remaining*1000:.1f} mm, keep Y/Z at 0. This limit comes from the live wrist depth at '
                       f'the merged end down to {target*1000:.0f} mm, capped by the frozen range. Use the HEAD '
                       'view to check the closed end is moving onto the target button and to confirm the light, '
                       'then withdraw with negative X. If an executed press already brought the live depth to '
                       f'within a few mm of {target*1000:.0f} mm and the light is not on, treat it as a miss: '
                       'withdraw, open (o), realign, re-range, close and press again instead of pushing.')
    live = state.get('live_m')
    return dict(format=press_phase.FORMAT, stage=stage, gripper_state=gripper, z_b_m=state.get('z_b_m'),
                frozen_from=state.get('frozen_from'), budget_mm=round(budget*1000, 1),
                consumed_mm=round(consumed*1000, 1), remaining_mm=round(remaining*1000, 1),
                live_depth_mm=round(live*1000, 1) if live is not None else None,
                target_depth_mm=round(target*1000, 1), bound=state.get('bound'),
                tip_plane_mm=round((state.get('tip_plane_m') or 0)*1000, 1), close_band_mm=band,
                instruction=instruction, source='program_rule_on_recorded_depth')


def _actual(result):
    result = result or {}
    replay = result.get('replay') or {}
    command = replay.get('recorded_actual_command') if replay else result.get('command')
    return {k: copy.deepcopy(v) for k, v in (command or {}).items()
            if k in ('type', 'xyz_mm', 'head_deg', 'key')}


class LocalizedConversation(Conversation):
    execution_policy = 'yz_alignment_before_forward'
    forward_guard_policy = overlap.POLICY
    max_press_attempts = press_retry.DEFAULT_LIMIT

    def __init__(self, goal, obs, head, wrist, session, prompt, demo=None,
                 failure_case=None, budget=None, decision_mode='localized'):
        if decision_mode != 'localized':
            raise ValueError('localized_mode_required')
        super().__init__(goal, obs, head, wrist, session, prompt+'\n'+GUIDANCE,
                         demo, failure_case, budget, decision_mode='direct')
        self.decision_mode, self.format = 'localized', FORMAT
        self.session = copy.deepcopy(session)
        intro = json.loads(self.fixed[1]['content'][0]['text'])
        intro.update(input_format=FORMAT, execution_policy=self.execution_policy)
        self.fixed[1]['content'][0] = text(intro)
        self.localizations, self.localization_messages, self.localized_progress = {}, {}, {}
        self.processed_localizations = set()
        self.hypothesis_reviews, self.review_signals, self.reassessments = {}, {}, {}
        self.review_phase = 'not_required'
        self.forward_checks = {}
        self.press_checks = {}
        self.press_status = {}
        self.range_evidence = {}
        # Raw manifests (depth, selection) per observation; images live in self.observations.
        self.manifests = {obs['observation_id']: copy.deepcopy(obs)}
        self.forward_guard_decision = None
        self.direction_hints = {}
        self.head_advice = {}

    def _check(self, oid, cancel=None):
        if cancel is not None and cancel.is_set():
            raise InterruptedError('operator_stopped')
        if oid != self.current_id:
            raise ValueError('decision_observation_mismatch')

    def check_observation(self, observation_id):
        self._check(observation_id)
        if observation_id not in self.processed_localizations:
            raise ValueError('current_observation_not_prepared')
        if self.review_signals.get(observation_id, {}).get('review_required') and observation_id not in self.hypothesis_reviews:
            raise ValueError('current_hypothesis_review_not_completed')

    def _state(self, oid):
        return json.loads(self.observations[oid]['content'][0]['text']).get('state') or {}

    def _context(self, oid, kind, **data):
        message = dict(role='user', content=[text(dict(context_type=kind, **copy.deepcopy(data)))])
        self.localization_messages.setdefault(oid, []).append(message)
        if self.estimated_tokens is not None:
            self.estimated_tokens += len(message['content'][0]['text'].encode('utf-8')) + 256

    def messages(self):
        result = []
        for message in super().messages():
            result.append(message)
            content = message.get('content')
            if message['role'] != 'user' or not isinstance(content, list) or not content:
                continue
            try:
                data = json.loads(content[0].get('text', ''))
            except (ValueError, TypeError):
                continue
            oid = data.get('observation_id') if isinstance(data, dict) else None
            if oid in self.localization_messages:
                contexts = copy.deepcopy(self.localization_messages[oid])
                if data.get('context_type') == 'EARLIER_OBSERVATION_THIS_RUN_NOT_CURRENT':
                    for item in contexts:
                        archived = json.loads(item['content'][0]['text'])
                        archived['context_type'] = 'EARLIER_LOCALIZATION_THIS_RUN_NOT_CURRENT'
                        item['content'][0] = text(archived)
                result.extend(contexts)
        return result

    def body(self):
        tools = copy.deepcopy(TOOLS)
        move = next(t['function'] for t in tools if t['function']['name'] == 'propose_move')
        move['description'] = ('Move the hand by chassis-frame XYZ increments, preserving orientation. '
                               'First align in Y/Z (X=0), coarse to fine: in the current fixed wrist setup, '
                               'button face left/right of the marked tip -> +Y/-Y; above/below -> +Z/-Z. '
                               'Already within the button-face interval on one axis -> zero on that axis. '
                               'Choose millimeter magnitudes from feedback. ANY positive X, including combined XYZ, '
                               'requires an independent CURRENT wrist check that the marked physical fingertip '
                               'lies inside the button-face box horizontally AND vertically. Unknown does not pass. '
                               'Negative X remains available. No displacement cap, automatic return or clipping.')
        return request_body(self.messages(), reassess.extend_tools(tools, self.hypothesis_reviews.get(self.current_id)))

    def decision_context(self, stage):
        current = self.localizations.get(self.current_id, {})
        return dict(format=FORMAT, decision_mode='localized', observation_id=self.current_id,
                    previous_observation_id=self.previous_id, stage=stage,
                    execution_policy=self.execution_policy, forward_guard_policy=self.forward_guard_policy,
                    forward_guard_format=overlap.FORMAT,
                    direction_rule_format=direction_rules.FORMAT,
                    localization_format=locator.FORMAT, localization_revision=current.get('revision'),
                    progress_monitor_format=PROGRESS_FORMAT, view_phase='model_decides',
                    hypothesis_review_format=reassess.FORMAT, review_phase=self.review_phase,
                    forward_alignment_phase=('overlap_confirmed' if self.forward_checks.get(self.current_id, {}).get('decision', {}).get('allowed')
                        else 'align_yz'),
                    review_observation_id=(self.current_id if self.current_id in self.hypothesis_reviews else None),
                    compression_generation=self.generation)

    def prepare_decision(self, base, directory, cancel=None):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        oid = self.current_id
        atomic_json(directory/'decision_context.json', self.decision_context('localization'))
        try:
            self._check(oid, cancel)
            if oid not in self.localizations:
                current = locator.inspect(base, locator.build_request(self.goal, self.observations[oid]),
                                          oid, directory/'localization', cancel, revision=1)
                self._check(oid, cancel)
                self.localizations[oid] = copy.deepcopy(current)
                self._context(oid, 'CURRENT_LOCALIZATION_AND_RAW_OFFSETS', **locator.model_view(current))
            if oid not in self.processed_localizations:
                previous = self.localizations.get(self.previous_id)
                actual = _actual(self.turns[-1]['ledger']['result']) if self.turns else {}
                progress = locator.compare(previous, self.localizations[oid],
                    self._state(self.previous_id) if self.previous_id else {}, self._state(oid),
                    [actual] if actual else [])
                progress['format'] = PROGRESS_FORMAT
                self.localized_progress[oid] = progress
                self._context(oid, 'RAW_OBSERVATION_CHANGE_MODEL_TO_INTERPRET', **progress)
                self._check(oid, cancel)
                self.processed_localizations.add(oid)
            atomic_json(directory/'localization_current.json', self.localizations[oid])
            atomic_json(directory/'localized_progress.json', self.localized_progress[oid])
            if oid not in self.head_advice:
                advice = head_view_advice.build_advice(self.localizations[oid], self._state(oid))
                self.head_advice[oid] = advice
                self._context(oid, 'CURRENT_HEAD_VIEW_ADVICE', **advice)
            atomic_json(directory/'head_view_advice.json', self.head_advice[oid])
            if press_phase.active() and oid not in self.range_evidence:
                wrist = self.localizations[oid].get('wrist') or {}
                box, tip = wrist.get('button_box'), wrist.get('contact_tip')
                aligned = (wrist.get('button_visibility') == 'full' and box is not None and tip is not None
                           and box[0] <= tip[0] <= box[2] and box[1] <= tip[1] <= box[3])
                evidence = press_phase.observe(self.manifests.get(oid), calibrated_tip.aim_pixel(),
                                               calibrated_tip.image_size(), aligned=aligned)
                evidence['phase_after'] = press_phase.phase()
                self.range_evidence[oid] = evidence
            if oid in self.range_evidence:
                atomic_json(directory/'range_evidence.json', self.range_evidence[oid])
            if press_phase.active() and oid not in self.press_status:
                status = press_status(press_phase.describe(oid))
                status['this_observation_range'] = {k: self.range_evidence[oid].get(k)
                    for k in ('median_m', 'trusted', 'reasons', 'froze', 'role')}
                self.press_status[oid] = status
                self._context(oid, 'CURRENT_PRESS_PHASE', **status)
            if oid in self.press_status:
                atomic_json(directory/'press_phase_status.json', self.press_status[oid])
            self.prepare_hypothesis_review(base, directory, cancel)
            self._check(oid, cancel)
            if oid not in self.direction_hints:
                hint = direction_rules.build_hint(self.localizations[oid])
                self.direction_hints[oid] = hint
                self._context(oid, 'CURRENT_WRIST_ALIGNMENT_DIRECTION_RULE', **hint)
            atomic_json(directory/'wrist_direction_rule.json', self.direction_hints[oid])
            return self.prepare(base, directory, cancel)
        except Exception as exc:
            atomic_json(directory/'decision_context.json', self.decision_context(
                'interrupted' if isinstance(exc, InterruptedError) else 'failed'))
            atomic_json(directory/'error.json', dict(stage='decision_prepare', error_type=type(exc).__name__, error=str(exc)))
            raise

    def attach_press_check(self, decision):
        """Attach press-phase evidence; unlike the alignment gate this uses no model judgment."""
        oid = self.current_id
        self._check(decision.get('observation_id'))
        if decision.get('policy') != overlap.PRESS_POLICY:
            raise ValueError('forward_check_policy_mismatch')
        if oid not in self.press_checks:
            self._context(oid, 'CURRENT_PRESS_PHASE_CHECK',
                observation_id=oid,
                decision={k: copy.deepcopy(v) for k, v in decision.items()
                          if k not in ('authorized_command',)},
                instruction='The gripper is closed, so the button is not boxed at the merged end any '
                    'more. This phase bounds the stroke by live depth at that pixel down to the press '
                    'target, capped by the frozen pre-close range. Keep positive X within remaining_m; '
                    'the program does not clip a rejected stroke.')
        self.press_checks[oid] = copy.deepcopy(decision)

    def attach_forward_check(self, assessment, decision):
        """Attach fresh gate evidence without rewriting previous observation messages."""
        oid = self.current_id
        self._check(assessment.get('observation_id'))
        if decision.get('observation_id') != oid or decision.get('policy') != self.forward_guard_policy:
            raise ValueError('forward_check_observation_mismatch')
        if oid not in self.forward_checks:
            # Raw thousand-unit coordinates remain in the separate gate archive.
            self._context(oid, 'CURRENT_WRIST_OVERLAP_FORWARD_CHECK',
                observation_id=oid, assessment={k: copy.deepcopy(v) for k,v in assessment.items()
                    if k not in ('raw_output', 'raw_locations', 'images', 'coordinate_evidence')},
                decision={k: copy.deepcopy(v) for k,v in decision.items()
                    if k not in ('coordinate_evidence', 'authorized_command')},
                instruction='This applies only to this observation. If overlap is not confirmed, '
                    'positive X is unavailable: choose Y/Z alignment with X=0, improve the view, '
                    'observe again or withdraw. Use coarse-to-fine Y/Z; no displacement cap is imposed.')
        self.forward_checks[oid] = dict(assessment=copy.deepcopy(assessment), decision=copy.deepcopy(decision))

    def prepare_hypothesis_review(self, base, directory, cancel=None):
        oid = self.current_id
        self._check(oid, cancel)
        signal = reassess.monitor(self.turns, self.localizations, oid)
        self.review_signals[oid] = signal
        atomic_json(directory/'hypothesis_review'/'trigger.json', signal)
        if not signal['review_required']:
            self.review_phase = 'not_required'
            return
        if oid in self.hypothesis_reviews:
            return
        self.review_phase = 'reviewing'
        atomic_json(directory/'decision_context.json', self.decision_context('hypothesis_review'))
        baseline = signal['baseline_observation_id']
        body = reassess.build_request(self.goal, self.observations[baseline], self.observations[oid],
                                      signal, self.localizations)
        review = reassess.inspect(base, body, oid, baseline, directory/'hypothesis_review', cancel)
        self._check(oid, cancel)
        if (review.get('observation_id') != oid or review.get('previous_observation_id') != baseline
                or review.get('evidence_observation_ids') != [baseline, oid]):
            raise ValueError('hypothesis_review_observation_mismatch')
        self.hypothesis_reviews[oid] = copy.deepcopy(review)
        self._context(oid, 'MOTION_HYPOTHESIS_REVIEW', assessment=review, trigger=signal,
            instruction='Reassess the cause, not merely the reason wording. Include reassessment with current '
            'evidence, expected_visual_change and if_no_change. Apply the task-specific wrist direction table; '
            'choose magnitudes, not new axis signs. Do not reverse a demonstrated improving direction '
            'while the target remains on the same side. Changing box size alone is not improvement.')
        self.review_phase = 'decision_required'

    def action_arguments(self, arguments, name=None):
        """Decode SDK fields; optional narration/old metadata never veto motion."""
        if not isinstance(arguments, dict):
            raise ValueError('command_arguments_must_be_object')
        fields = {'propose_move': ('xyz_mm',), 'propose_head': ('head_deg',),
                  'propose_gripper': ('key',), 'request_observation': (),
                  'report_unable': (), 'report_done': ()}
        if name not in fields:
            raise ValueError('unknown_tool')
        result = {key: copy.deepcopy(arguments[key]) for key in fields[name]}
        for key in ('xyz_mm', 'head_deg'):
            if isinstance(result.get(key), str):
                result[key] = json.loads(result[key])
        narrative = 'evidence' if name == 'report_done' else 'reason'
        value = arguments.get(narrative, '')
        result[narrative] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return result

    def parse_response(self, response, obs, session, directory):
        self.check_observation(obs['observation_id'])
        if session.get('schema') != SESSION_SCHEMA:
            raise ValueError('keyboard_session_schema_required')
        choices = response.get('choices') or []
        if len(choices) != 1 or choices[0].get('finish_reason') != 'tool_calls':
            raise ValueError('one_complete_tool_required')
        calls = choices[0]['message'].get('tool_calls') or []
        if len(calls) != 1:
            raise ValueError('one_command_required')
        fn = calls[0]['function']
        raw = json.loads(fn['arguments'])
        review = self.hypothesis_reviews.get(self.current_id)
        if review and fn['name'] not in ('report_done', 'report_unable'):
            reassess.validate_reassessment(raw.get('reassessment'), review)
        args = self.action_arguments(raw, fn['name'])
        if fn['name'].startswith('propose_'):
            validate_command(dict(type=fn['name'].removeprefix('propose_'),
                                  **{k:v for k,v in args.items() if k != 'reason'}))
        if review and fn['name'] not in ('report_done', 'report_unable'):
            atomic_json(Path(directory)/'hypothesis_review'/'decision.json', dict(
                **raw['reassessment'], source='model_not_human', status='validated_proposal_not_execution',
                proposal=dict(name=fn['name'], arguments=args)))
        return dict(name=fn['name'], arguments=args)

    def accept(self, response):
        review = self.hypothesis_reviews.get(self.current_id)
        fn = response['choices'][0]['message']['tool_calls'][0]['function']
        if review and fn['name'] not in ('report_done', 'report_unable'):
            self.reassessments[self.current_id] = reassess.validate_reassessment(
                json.loads(fn['arguments']).get('reassessment'), review)
        super().accept(response)

    def check_execution(self, proposal, observation_id):
        self.check_observation(observation_id)
        if self.pending is None:
            raise ValueError('accepted_action_required')
        fn = self.pending['tool_calls'][0]['function']
        review = self.hypothesis_reviews.get(observation_id)
        if review and fn['name'] not in ('report_done', 'report_unable'):
            reassess.validate_reassessment(json.loads(fn['arguments']).get('reassessment'), review)
        if proposal != dict(name=fn['name'], arguments=self.action_arguments(json.loads(fn['arguments']), fn['name'])):
            raise ValueError('accepted_action_mismatch')
        require_forward_permission(proposal, {'observation_id': observation_id}, self.forward_guard_decision,
                                   policy=self.forward_guard_policy)

    def append_result(self, proposal, result, obs, head, wrist):
        oid = self.current_id
        super().append_result(proposal, result, obs, head, wrist)
        self.manifests[obs['observation_id']] = copy.deepcopy(obs)
        ledger = self.turns[-1]['ledger']
        ledger['localization_summary'] = locator.model_view(self.localizations[oid])
        ledger['localized_progress'] = copy.deepcopy(self.localized_progress[oid])
        ledger['wrist_direction_rule'] = copy.deepcopy(self.direction_hints.get(oid))
        ledger['head_view_advice'] = copy.deepcopy(self.head_advice.get(oid))
        if oid in self.press_status:
            ledger['press_phase'] = copy.deepcopy(self.press_status[oid])
        if oid in self.press_checks:
            ledger['press_check'] = {k: copy.deepcopy(v) for k, v in self.press_checks[oid].items()
                                     if k != 'authorized_command'}
        if oid in self.hypothesis_reviews:
            ledger['hypothesis_review'] = copy.deepcopy(self.hypothesis_reviews[oid])
            ledger['reassessment'] = copy.deepcopy(self.reassessments.get(oid))
        if oid in self.forward_checks:
            check = self.forward_checks[oid]
            ledger['wrist_overlap_forward_check'] = {k: copy.deepcopy(v) for k,v in check['decision'].items()
                if k not in ('coordinate_evidence',)}
        self.forward_guard_decision = None
        self.review_phase = 'not_required'

    def validate_summary(self, response):
        """Use one advisory summary, without a second semantic/format review.

        Keep the model's original output. Only references to images actually in
        this run are bound for the parent's optional snapshot retention; an
        unbound reference is preserved as a claim, never replaced with another ID.
        """
        choices = response.get('choices') if isinstance(response, dict) else None
        outputs, candidates, prose = [], [], []
        for choice in choices if isinstance(choices, list) else []:
            message = choice.get('message') if isinstance(choice, dict) else None
            if not isinstance(message, dict):
                continue
            outputs.append(message)
            calls = message.get('tool_calls')
            for call in calls if isinstance(calls, list) else []:
                function = call.get('function') if isinstance(call, dict) else None
                value = locator._object(function.get('arguments')) if isinstance(function, dict) else None
                if value is not None:
                    candidates.append(value)
            content = message.get('content')
            value = locator._object(content)
            if value is not None:
                candidates.append(value)
            elif isinstance(content, str):
                prose.append(content)
        fields = ('progress', 'uncertainties', *EVIDENCE_FIELDS)
        data = next((value for value in candidates if any(key in value for key in fields)),
                    candidates[0] if candidates else {})
        notes = [] if candidates else ['No readable summary object; original model output retained.']
        summary = dict(source='model_not_human', raw_summary=locator._json_safe(data),
                       raw_output=locator._json_safe(outputs if outputs else response))
        for field in ('progress', 'uncertainties'):
            value = data.get(field)
            if isinstance(value, str):
                summary[field] = value
            else:
                summary[field] = '\n'.join(prose) if field == 'progress' and prose else 'unknown'
                if field in data:
                    notes.append(field+': non-text value retained in raw_summary.')
        for field in EVIDENCE_FIELDS:
            item = data.get(field)
            item = item if isinstance(item, dict) else {}
            raw_id = item.get('observation_id')
            bound = isinstance(raw_id, str) and raw_id in self.observations
            assessment = item.get('assessment')
            summary[field] = dict(observation_id=raw_id if bound else None,
                assessment=assessment if isinstance(assessment, str) else 'unknown',
                model_observation_id=locator._json_safe(raw_id),
                reference_status='bound_this_run' if bound else 'not_supplied' if raw_id is None else 'unbound')
            if raw_id is not None and not bound:
                notes.append(field+': model reference is not an available observation ID in this run; '
                             'preserved without loading or substituting an image.')
            if field in data and not isinstance(data[field], dict):
                notes.append(field+': non-object evidence retained in raw_summary; no image reference bound.')
        summary['parse_notes'] = notes
        return summary
