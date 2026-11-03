"""Persistent actions grounded in independently located, per-camera image points.

Only the action model proposes movement. Coordinates are model estimates; derived
relationships never calibrate millimeters, prove contact, or authorize forward X.
"""
import copy
import json
import re
from pathlib import Path

from .common import atomic_json
from .conversation import Conversation, request_body
from .client import TOOLS, parse_proposal
from .model_context import text
from . import spatial_localization as locator
from . import spatial_progress

FORMAT = 'pressb-conversation-v4'
PROGRESS_FORMAT = 'pressb-localized-progress-v1'
VIEW_ACTIONS = {'propose_head', 'request_observation', 'report_unable', 'report_done'}
BASIS = dict(type='object', properties=dict(
    observation_id=dict(type='string'), localization_revision=dict(type='integer', minimum=1),
    view=dict(type='string', enum=['head', 'wrist', 'both', 'unknown'])),
    required=['observation_id', 'localization_revision', 'view'], additionalProperties=False)
GUIDANCE = '''
CURRENT LOCALIZATION CONTRACT
Use CURRENT_LOCALIZATION_AND_PROGRAM_RELATIONS for the current observation and
its latest revision. Coordinates are MODEL estimates; image relations are PROGRAM
calculations from those coordinates, not measured robot offsets or 3D contact.
Each action's visual_basis cites observation_id, localization_revision and view.
Use head/wrist/both only when those views are in usable_views. Unknown is valid
for taking a new view or reporting inability; never invent a coordinate.
An available WRIST view permits hand adjustment even when HEAD is incomplete.
When neither view is available, only head, observe, unable and evidence-verified
done are available. About +30 degrees pitch is a framing reference, not a required
step. Do not repeat an already-sent head target as if it were an increment.
Do not re-estimate left/right/above/below or report your own progress fields.
If you dispute a localization, state concrete visible evidence and request a NEW
observation. A review revision supersedes only that observation's earlier estimate;
conflicting views remain unusable until a fresh observation. Other views may work.
LOCALIZED_PROGRESS is projected-error comparison, not contact or actual arrival.
Reassess after no improvement/crossing. Never convert pixels directly to XYZ or
blindly reverse motion. No fixed millimeter step is supplied by this program.
Positive X still requires an independent CURRENT wrist two-fingertip horizontal
check. No extra height, depth, closure or X-progress condition is introduced.
report_done evidence must name exactly two live observation IDs: the light-on
observation, then the later withdrawal observation. They are independently checked
against saved images; a sent command or historical demo is insufficient.
'''


def _actual(result):
    result = result or {}
    replay = result.get('replay') or {}
    command = replay.get('recorded_actual_command') if replay else result.get('command')
    return {k: copy.deepcopy(v) for k, v in (command or {}).items()
            if k in ('type', 'xyz_mm', 'head_deg', 'key')}


class LocalizedConversation(Conversation):
    def __init__(self, goal, obs, head, wrist, session, prompt, demo=None,
                 failure_case=None, budget=65536, decision_mode='localized'):
        if decision_mode != 'localized':
            raise ValueError('localized_mode_required')
        super().__init__(goal, obs, head, wrist, session, prompt+'\n'+GUIDANCE,
                         demo, failure_case, budget, decision_mode='direct')
        self.decision_mode, self.format = 'localized', FORMAT
        self.session = copy.deepcopy(session)
        intro = json.loads(self.fixed[1]['content'][0]['text'])
        intro['input_format'] = FORMAT
        self.fixed[1]['content'][0] = text(intro)
        self.localizations, self.localization_messages, self.localized_progress = {}, {}, {}
        self.reviewed_pairs, self.reviews = set(), {}
        self.last_local_review_round = 0
        self.processed_localizations = set()
        self.completion_checks = {}
        self.candidate_basis = None

    def _check(self, oid, cancel=None):
        if cancel is not None and cancel.is_set():
            raise InterruptedError('operator_stopped')
        if oid != self.current_id:
            raise ValueError('decision_observation_mismatch')

    def check_observation(self, observation_id):
        self._check(observation_id)
        if observation_id not in self.localizations or observation_id not in self.processed_localizations:
            raise ValueError('current_localization_required')

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
        usable = self.localizations.get(self.current_id, {}).get('usable_views', [])
        tools = copy.deepcopy(TOOLS)
        for tool in tools:
            schema = tool['function']['parameters']
            schema['properties']['visual_basis'] = copy.deepcopy(BASIS)
            schema['required'].append('visual_basis')
        if not usable:
            tools = [t for t in tools if t['function']['name'] in VIEW_ACTIONS]
        return request_body(self.messages(), tools)

    def decision_context(self, stage):
        current = self.localizations.get(self.current_id, {})
        return dict(format=FORMAT, decision_mode='localized', observation_id=self.current_id,
                    previous_observation_id=self.previous_id, stage=stage,
                    localization_format=locator.FORMAT, localization_revision=current.get('revision'),
                    progress_monitor_format=PROGRESS_FORMAT,
                    view_phase='ready' if current.get('usable_views') else 'get_view',
                    compression_generation=self.generation)

    def _comparison(self):
        """Last real movement, including subsequent refresh/head changes, not intent."""
        commands = []
        for turn in reversed(self.turns):
            ledger, result = turn['ledger'], turn['ledger']['result']
            if result.get('status') == 'observed':
                continue
            if result.get('status') != 'completed' or result.get('sends_completed') is not True:
                return None, []
            command = _actual(result)
            if not command:
                return None, []
            commands.insert(0, command)
            if command['type'] == 'gripper':
                return ledger['before_observation_id'], commands
            if command['type'] == 'move' and any(command.get('xyz_mm', [])):
                return ledger['before_observation_id'], commands
        return None, []

    def _compare(self, before, after, commands):
        return spatial_progress.compare(before, after,
            self._state(before['observation_id']) if before else {}, self._state(self.current_id), commands)

    @staticmethod
    def _conflict(initial, reviewed, view):
        old, new = initial[view], reviewed[view]
        if (view in initial['usable_views']) != (view in reviewed['usable_views']):
            return True
        if view not in initial['usable_views']:
            return False
        a, b = old['derived_relation'], new['derived_relation']
        if any({a[k], b[k]} in ({'left', 'right'}, {'above', 'below'}) for k in ('horizontal', 'vertical')):
            return True
        box = old['button_box']
        size = [box[2]-box[0], box[3]-box[1]]
        # A large relocation of either entity can hide behind an unchanged offset.
        old_center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
        nb = new['button_box']; new_center = [(nb[0]+nb[2])/2, (nb[1]+nb[3])/2]
        return any(abs(x-y)/s > .5 for points in ((old_center, new_center),
                       (old['contact_tip'], new['contact_tip'])) for x, y, s in zip(*points, size))

    def _review(self, base, before_id, commands, directory, cancel):
        oid = self.current_id
        pair = before_id, oid
        if pair in self.reviewed_pairs:
            return
        rd = directory/'localization_review'
        atomic_json(directory/'decision_context.json', self.decision_context('localization_review'))
        # No case, previous result, reasoning, or command is sent to either locator.
        before = locator.inspect(base, locator.build_request(self.goal, self.observations[before_id]),
                                 before_id, rd/'before', cancel,
                                 revision=self.localizations[before_id]['revision']+1)
        self._check(oid, cancel)
        initial = self.localizations[oid]
        current = locator.inspect(base, locator.build_request(self.goal, self.observations[oid]),
                                  oid, rd/'current', cancel, revision=initial['revision']+1)
        self._check(oid, cancel)
        conflicts = [view for view in ('head', 'wrist')
                     if self._conflict(initial, current, view)
                     or self._conflict(self.localizations[before_id], before, view)]
        progress = self._compare(before, current, commands)
        current = copy.deepcopy(current)
        current['recovery_views'] = conflicts
        current['usable_views'] = [v for v in current['usable_views'] if v not in conflicts]
        current['revision_semantics'] = 'Reinspection, not human confirmation; conflicting views need a new observation.'
        review = dict(format='pressb-localization-review-v1',
                      source='model_relocalization_and_program_comparison',
                      previous_observation_id=before_id, observation_id=oid,
                      conflicting_views=conflicts, progress=progress, revision=current['revision'])
        atomic_json(rd/'assessment.json', review)
        self.localizations[oid] = current
        self.localized_progress[oid] = progress
        self.reviews[oid] = review
        self.reviewed_pairs.add(pair)
        self.last_local_review_round = len(self.turns)
        self._context(oid, 'CURRENT_LOCALIZATION_AND_PROGRAM_RELATIONS', **current)
        self._context(oid, 'LOCALIZATION_REVIEW', **review)

    def prepare_decision(self, base, directory, cancel=None):
        from .progress import monitor
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        oid = self.current_id
        stage = 'localization'
        atomic_json(directory/'decision_context.json', self.decision_context(stage))
        try:
            self._check(oid, cancel)
            if oid not in self.localizations:
                current = locator.inspect(base, locator.build_request(self.goal, self.observations[oid]),
                                          oid, directory/'localization', cancel, revision=1)
                self._check(oid, cancel)
                if current['observation_id'] != oid or current['format'] != locator.FORMAT:
                    raise ValueError('localization_observation_mismatch')
                self.localizations[oid] = copy.deepcopy(current)
                self._context(oid, 'CURRENT_LOCALIZATION_AND_PROGRAM_RELATIONS', **current)
            if oid not in self.processed_localizations:
                before_id, commands = self._comparison()
                previous = self.localizations.get(before_id)
                progress = self._compare(previous, self.localizations[oid], commands)
                monitored = copy.deepcopy(self.turns)
                for turn in monitored:
                    result = turn['ledger']['result']
                    result['command'] = _actual(result)
                signal = monitor(monitored, oid, since_round=self.last_local_review_round)
                progress.update(format=PROGRESS_FORMAT, monitor=signal)
                self.localized_progress[oid] = progress
                self._context(oid, 'LOCALIZED_PROGRESS', **progress)
                if previous and (progress.get('needs_recheck') or signal['review_required']):
                    stage = 'localization_review'
                    self._review(base, before_id, commands, directory, cancel)
                self._check(oid, cancel)
                self.processed_localizations.add(oid)
            atomic_json(directory/'localization_current.json', self.localizations[oid])
            atomic_json(directory/'localized_progress.json', self.localized_progress[oid])
            stage = 'context_prepare'
            return self.prepare(base, directory, cancel)
        except Exception as exc:
            atomic_json(directory/'decision_context.json', self.decision_context(
                'interrupted' if isinstance(exc, InterruptedError) else 'failed'))
            atomic_json(directory/'error.json', dict(stage=stage, error_type=type(exc).__name__, error=str(exc)))
            raise

    def _validate_basis(self, basis, name):
        current = self.localizations[self.current_id]
        if (not isinstance(basis, dict) or set(basis) != set(BASIS['required'])
                or basis['observation_id'] != self.current_id
                or type(basis['localization_revision']) is not int
                or basis['localization_revision'] != current['revision']
                or basis['view'] not in ('head', 'wrist', 'both', 'unknown')):
            raise ValueError('visual_basis_requires_current_observation_and_localization_revision')
        views = ['head', 'wrist'] if basis['view'] == 'both' else [basis['view']]
        if basis['view'] != 'unknown' and any(v not in current['usable_views'] for v in views):
            raise ValueError('visual_basis_view_not_usable:'+str(current['usable_views']))
        if name not in VIEW_ACTIONS and (not current['usable_views'] or basis['view'] == 'unknown'):
            raise ValueError('hand_action_requires_one_usable_view; wrist_alone_is_allowed')

    def parse_response(self, response, obs, session, directory):
        self.check_observation(obs['observation_id'])
        clean = copy.deepcopy(response)
        fn = clean['choices'][0]['message']['tool_calls'][0]['function']
        args = json.loads(fn['arguments'])
        basis = args.pop('visual_basis', None)
        self._validate_basis(basis, fn['name'])
        fn['arguments'] = json.dumps(args, ensure_ascii=False)
        proposal = parse_proposal(clean, obs, session)
        self.candidate_basis = copy.deepcopy(basis)
        atomic_json(Path(directory)/'decision_basis.json', basis)
        return proposal

    def _check_alignment_continuation(self, proposal):
        review = self.reviews.get(self.current_id)
        if not review or proposal['name'] != 'propose_move':
            return
        xyz = proposal['arguments']['xyz_mm']
        if xyz[0] != 0:
            return  # Projection error is not a condition on approaching/retracting X.
        _, commands = self._comparison()
        prior = next((c for c in reversed(commands) if c.get('type') == 'move'), {})
        old = prior.get('xyz_mm', [0, 0, 0])
        basis = self.candidate_basis or {}
        selected = ['head', 'wrist'] if basis.get('view') == 'both' else [basis.get('view')]
        sign = lambda x: (x > 0)-(x < 0)
        if old[0] != 0 or not any(xyz[1:]) or [sign(x) for x in xyz[1:]] != [sign(x) for x in old[1:]]:
            return
        assessments = [review['progress']['views'].get(v, {}) for v in selected]
        # No assumption that chassis Y maps to image horizontal or Z to vertical.
        # Either projected component improving is enough to avoid this veto.
        if assessments and all(a.get('comparable') and all(
                a.get('axes', {}).get(axis, {}).get('trend') in ('unchanged', 'worsened')
                or a.get('axes', {}).get(axis, {}).get('large_sign_reversal')
                for axis in ('horizontal', 'vertical')) for a in assessments):
            raise ValueError('reviewed_alignment_without_progress: reassess, use a new observation or a supported different action; no automatic reversal')

    def _completion_ids(self, evidence):
        ids = [oid for oid in self.observations
               if re.search(r'(?<![A-Za-z0-9_-])'+re.escape(oid)+r'(?![A-Za-z0-9_-])', evidence)]
        if len(ids) != 2:
            raise ValueError('report_done_requires_exactly_two_live_evidence_ids_light_then_withdrawal')
        order = list(self.observations)
        light, retreat = ids
        if order.index(light) == 0 or order.index(retreat) <= order.index(light):
            raise ValueError('light_requires_before_frame_and_withdrawal_must_follow')
        if evidence.index(light) > evidence.index(retreat):
            raise ValueError('cite_light_observation_before_withdrawal_observation')
        later = [t['ledger'] for t in self.turns
                 if order.index(t['ledger']['after_observation_id']) > order.index(retreat)]
        if any(_actual(t['result']).get('type') == 'move' and
               any(_actual(t['result']).get('xyz_mm', [])) for t in later):
            raise ValueError('new_hand_motion_requires_new_withdrawal_evidence')
        return order[order.index(light)-1], light, retreat

    def review_proposal(self, base, proposal, directory, cancel=None):
        self._check_alignment_continuation(proposal)
        if proposal['name'] != 'report_done':
            return
        from .grounding import verify_activation, verify_withdrawal
        directory = Path(directory)
        before, light, retreat = self._completion_ids(proposal['arguments']['evidence'])
        key = before, light, retreat
        self._check(self.current_id, cancel)
        if key not in self.completion_checks:
            def command_at(oid):
                return next((_actual(t['ledger']['result']) for t in self.turns
                             if t['ledger']['after_observation_id'] == oid), {})
            atomic_json(directory/'decision_context.json', self.decision_context('activation_verification'))
            activated = verify_activation(base, self.observations[before], self.observations[light],
                                          command_at(light), directory/'activation', cancel)
            self._check(self.current_id, cancel)
            atomic_json(directory/'decision_context.json', self.decision_context('withdrawal_verification'))
            withdrawn = verify_withdrawal(base, self.observations[light], self.observations[retreat],
                                         command_at(retreat), directory/'withdrawal', cancel)
            self._check(self.current_id, cancel)
            self.completion_checks[key] = dict(activation=activated, withdrawal=withdrawn)
            self._context(self.current_id, 'TASK_EVIDENCE_MODEL_VERIFICATION',
                          **self.completion_checks[key])
        checked = self.completion_checks[key]
        atomic_json(directory/'decision_context.json', self.decision_context('action_request'))
        if checked['activation']['light_on'] != 'yes' or checked['withdrawal']['withdrawal'] != 'yes':
            raise ValueError('light_on_and_later_withdrawal_not_verified; use new observations, never repeat evidence until agreement')

    def action_arguments(self, arguments):
        arguments = copy.deepcopy(arguments)
        arguments.pop('visual_basis')
        return arguments

    def accept(self, response):
        fn = response['choices'][0]['message']['tool_calls'][0]['function']
        self._validate_basis(json.loads(fn['arguments']).get('visual_basis'), fn['name'])
        super().accept(response)

    def check_execution(self, proposal, observation_id):
        self.check_observation(observation_id)
        if self.pending is None:
            raise ValueError('accepted_localized_action_required')
        fn = self.pending['tool_calls'][0]['function']
        args = json.loads(fn['arguments'])
        self._validate_basis(args.get('visual_basis'), fn['name'])
        if proposal != dict(name=fn['name'], arguments=self.action_arguments(args)):
            raise ValueError('accepted_localized_action_mismatch')
        self._check_alignment_continuation(proposal)
        if proposal['name'] == 'report_done':
            key = self._completion_ids(proposal['arguments']['evidence'])
            checked = self.completion_checks.get(key, {})
            if (checked.get('activation', {}).get('light_on') != 'yes'
                    or checked.get('withdrawal', {}).get('withdrawal') != 'yes'):
                raise ValueError('completion_evidence_required')

    def append_result(self, proposal, result, obs, head, wrist):
        oid = self.current_id
        super().append_result(proposal, result, obs, head, wrist)
        ledger = self.turns[-1]['ledger']
        ledger['localization_summary'] = copy.deepcopy(self.localizations[oid])
        ledger['localized_progress'] = copy.deepcopy(self.localized_progress[oid])
        if oid in self.reviews:
            ledger['localization_review'] = copy.deepcopy(self.reviews[oid])
        self.candidate_basis = None
