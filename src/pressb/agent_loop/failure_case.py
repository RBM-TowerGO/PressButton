"""Reviewed failure evidence, kept separate from demonstrations and live state."""
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .common import SESSION_SCHEMA, atomic_json, sha256
from .demonstration import image_content


DECISION_FORMATS = ('pressb-conversation-v1', 'pressb-conversation-v2', 'pressb-conversation-v3')
PROGRESS_FORMAT = 'pressb-progress-v1'
PROGRESS_REVIEW_FORMAT = 'pressb-progress-review-v1'
VIEW_FORMAT = 'pressb-view-acquisition-v1'
VIEW_STATE_FORMAT = 'pressb-view-state-v1'
VIEW_PHASES = ('get_view', 'ready', 'recover_view')
INSPECTION_FORMATS = {'pressb-conversation-v2': ('pressb-perception-v1',),
                      'pressb-conversation-v3': ('pressb-grounding-v1', 'pressb-grounding-v2')}


def _check_observation_pair(record, decision, label):
    """Accept both explicit pair naming conventions, but never contradictory IDs."""
    current = [record[k] for k in ('observation_id', 'current_observation_id') if k in record]
    previous = [record[k] for k in ('previous_observation_id', 'before_observation_id') if k in record]
    if (not current or any(oid != decision['observation_id'] for oid in current)
            or not previous or any(oid != decision.get('previous_observation_id') for oid in previous)):
        raise ValueError('failure_'+label+'_observation_mismatch')


def compact_state(state):
    """Keep measured/command poses distinct; full joint traces stay in source logs."""
    fields = ('pose_m_xyzw', 'command_pose_m_xyzw', 'frame_id', 'reference_point',
              'receipt_ns', 'head_deg', 'gripper_command', 'rgb_state_skew_ms')
    return {key: state[key] for key in fields if key in state}


class FailureCase:
    def __init__(self, directory):
        self.root = Path(directory).expanduser().resolve()
        self.run = self.root/'agent' if (self.root/'agent').is_dir() else self.root
        self.source_hashes = {}
        self.review = self._read(self.root/'case_review.json')
        if (self.review.get('schema') != 'pressb-case-review-v1'
                or self.review.get('outcome') != 'failure'
                or not self.review.get('analysis_version')):
            raise ValueError('reviewed_failure_case_required')
        session = self._read(self.run/'session.json')
        if session.get('schema') != SESSION_SCHEMA:
            raise ValueError('keyboard_session_schema_required')
        if self.review['source_session_id'] != session['session_id']:
            raise ValueError('failure_review_session_mismatch')
        outcome = self._read(self.run/'result.json')
        self.summary = dict(context_type='HISTORICAL_FAILURE_NOT_LIVE', source=str(self.root),
            review=self.review, recorded_result=outcome, rounds=[],
            coordinates='chassis: +X forward, +Y robot left, +Z up; deltas mm; poses m / xyzw',
            semantics='Failure actions are NOT a successful recipe. Command completion only means sends finished. '
                      'Measured pose and command pose are separate. Missing evidence is unknown.')
        self.observations = {}
        self.images = []
        self.content = []
        completed = {}
        for folder in sorted((self.run/'rounds').iterdir()):
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            number = int(folder.name)+1
            decision_path = folder/'decision_context.json'
            decision = self._read(decision_path) if decision_path.exists() else None
            request_path = folder/'request.json'
            body = self._read(request_path) if request_path.exists() else None
            monitored = bool(decision and decision.get('format') == 'pressb-conversation-v1'
                             and decision.get('progress_monitor_format') == PROGRESS_FORMAT)
            view_enabled = bool(decision and decision.get('view_acquisition_format') == VIEW_FORMAT)
            if decision and decision.get('progress_monitor_format') is not None:
                if (decision.get('progress_monitor_format') != PROGRESS_FORMAT
                        or decision.get('format') != 'pressb-conversation-v1'
                        or decision.get('decision_mode', 'direct') != 'direct'):
                    raise ValueError('unsupported_failure_progress_format')
            if decision and decision.get('view_acquisition_format') is not None:
                if (not view_enabled or not monitored or decision.get('format') != 'pressb-conversation-v1'
                        or decision.get('decision_mode', 'direct') != 'direct'
                        or decision.get('view_phase') not in VIEW_PHASES):
                    raise ValueError('unsupported_failure_view_format')
            if body is None and (not decision or decision.get('format') not in INSPECTION_FORMATS) and not monitored and not view_enabled:
                raise ValueError('failure_round_missing_action_request')
            contexts = []
            for message in body['messages'] if body else []:
                if not isinstance(message.get('content'), list):
                    continue
                for item in message['content']:
                    if item.get('type') != 'text' or not item.get('text', '').startswith('{'):
                        continue
                    context = json.loads(item['text'])
                    if context.get('context_type') == 'LIVE_OBSERVATION':
                        contexts.append(context)
            if decision:
                if decision.get('format') not in DECISION_FORMATS:
                    raise ValueError('unsupported_decision_context')
                oid = decision['observation_id']
                if body is not None and (not contexts or contexts[-1]['observation_id'] != oid):
                    raise ValueError('failure_decision_observation_mismatch')
            else:
                if len(contexts) != 1:
                    raise ValueError('failure_round_requires_one_recorded_observation')
                oid = contexts[0]['observation_id']
            before = self._observation(oid, session['session_id'])
            proposal = self._read(folder/'proposal.json') if (folder/'proposal.json').exists() else None
            row = dict(round=number, proposal=proposal, before=before, after=None,
                       executed=False, status='no_execution_record')
            if monitored:
                row['decision_stage'] = decision.get('stage')
                self._progress_records(folder, decision, row, session['session_id'])
            if view_enabled:
                row['decision_stage'] = decision.get('stage')
                self._view_records(folder, decision, row, session['session_id'])
            if decision and decision.get('format') in INSPECTION_FORMATS:
                row['decision_stage'] = decision.get('stage')
                assessment_path = folder/'perception/assessment.json'
                if assessment_path.exists():
                    assessment = self._read(assessment_path)
                    if (assessment.get('format') not in INSPECTION_FORMATS[decision['format']]
                            or assessment.get('observation_id') != oid
                            or (decision['format'] == 'pressb-conversation-v2'
                                and assessment.get('previous_observation_id') != decision.get('previous_observation_id'))):
                        raise ValueError('failure_perception_observation_mismatch')
                    row['perception'] = assessment
                if decision['format'] == 'pressb-conversation-v3':
                    for name, relative in (('alignment_progress', 'alignment_progress.json'),
                                           ('activation', 'activation/assessment.json'),
                                           ('withdrawal', 'withdrawal/assessment.json'),
                                           ('decision_basis', 'decision_basis.json'),
                                           ('control_plan', 'control_plan.json'),
                                           ('control_memory', 'control_memory.json')):
                        path = folder/relative
                        if not path.exists():
                            continue
                        record = self._read(path)
                        if name in ('alignment_progress', 'activation', 'withdrawal'):
                            _check_observation_pair(record, decision, name)
                        elif name == 'decision_basis' and record.get('observation_id', oid) != oid:
                            raise ValueError('failure_decision_basis_observation_mismatch')
                        elif name == 'control_plan':
                            if record.get('observation_id') != oid:
                                raise ValueError('failure_control_plan_observation_mismatch')
                            if (record.get('format') != 'pressb-control-plan-v1' or record.get('source') !=
                                    'program_advice_from_model_judgments_not_human_confirmation'):
                                raise ValueError('failure_control_plan_model_provenance_required')
                            if record.get('geometry_progress') is not None:
                                _check_observation_pair(record['geometry_progress'], decision,
                                                        'control_plan_geometry_progress')
                        # These remain model judgments, never executed actions or human confirmation.
                        row[name] = record
            if body is None:
                if proposal or (folder/'execution.json').exists() or (folder/'result.json').exists():
                    raise ValueError('failure_execution_without_action_request')
                error_path = folder/'error.json'
                for relative in ('view_acquisition/error.json', 'progress_review/error.json', 'withdrawal/error.json',
                                 'activation/error.json', 'perception/error.json'):
                    if not error_path.exists():
                        error_path = folder/relative
                stage = ('view_acquisition' if view_enabled and ((folder/'view_acquisition/request.json').exists()
                             or decision.get('stage') == 'view_acquisition'
                             or decision['view_phase'] != 'ready' and not (folder/'progress_review/request.json').exists()) else
                         'progress_review' if monitored else
                         'withdrawal_verification' if (folder/'withdrawal/request.json').exists()
                         else 'activation_verification' if (folder/'activation/request.json').exists()
                         else 'perception')
                row.update(status=stage+('_interrupted' if outcome['status'] == 'operator_stopped'
                                        else '_failed'), error=self._read(error_path) if error_path.exists() else None)
                self.summary['rounds'].append(row)
                continue
            if (folder/'execution.json').exists():
                execution = self._read(folder/'execution.json')
                result = self._read(folder/'result.json')
                command = result['request']
                if (execution['decision_observation_id'] != before['observation_id']
                        or execution['execution_observation_id'] != before['observation_id']
                        or command['observation_id'] != before['observation_id']
                        or command['session_id'] != session['session_id']):
                    raise ValueError('failure_action_observation_mismatch')
                if not proposal or proposal['name'] != 'propose_'+command['type']:
                    raise ValueError('failure_action_proposal_mismatch')
                for key in ('xyz_mm', 'head_deg', 'key'):
                    if proposal['arguments'].get(key) != command.get(key):
                        raise ValueError('failure_action_proposal_mismatch')
                if result['status'] != 'completed' or result['capture']['status'] != 'complete':
                    raise ValueError('failure_case_requires_complete_recorded_action_boundaries')
                after = self._observation(result['capture']['observation_id'], session['session_id'])
                motion = {k: v for k, v in result['motion'].items() if k != 'state_trace'}
                if 'final_state' in motion:
                    motion['final_state'] = compact_state(motion['final_state'])
                row.update(executed=True, status=result['status'], after=after,
                           command=command, result=dict(task_id=result['task_id'], motion=motion,
                                                       capture=result['capture'], error=result['error']))
                completed[number] = row
            elif (folder/'observation_result.json').exists():
                if not proposal or proposal['name'] != 'request_observation':
                    raise ValueError('failure_observation_proposal_mismatch')
                capture = self._read(folder/'observation_result.json')
                row.update(status='observed', after=self._observation(capture['observation_id'], session['session_id']))
            elif (folder/'result.json').exists():
                raise ValueError('failure_result_missing_execution_record')
            elif (folder/'error.json').exists():
                error = self._read(folder/'error.json')
                failed_stage = ('view_acquisition' if view_enabled and (
                    error.get('stage') in ('view_acquisition', 'inspect_view')
                    or (folder/'view_acquisition/error.json').exists()) else
                    'progress_review' if monitored and (
                    error.get('stage') in ('progress_review', 'review_progress')
                    or (folder/'progress_review/error.json').exists()) else 'request')
                row.update(status=failed_stage+('_interrupted' if outcome['status'] == 'operator_stopped'
                           else '_failed'), error=error)
            self.summary['rounds'].append(row)
        if len(completed) != self.review['completed_commands'] or len(completed) != outcome['completed_motion_commands']:
            raise ValueError('failure_completed_command_count_mismatch')
        self.summary['completed_commands'] = len(completed)
        selected = self.review['selected_rounds']
        if len(selected) != 3 or len(set(selected)) != 3 or any(n not in completed for n in selected):
            raise ValueError('failure_case_requires_three_completed_key_rounds')
        for number in selected:
            row = completed[number]
            data = self._collage(row)
            self.images.append((f'round_{number:04d}.jpg', data))
            self.content.extend([dict(type='text', text=f'HISTORICAL FAILURE round {number}: '
                'top BEFORE, bottom AFTER; left HEAD, right WRIST. Not live; not an action to imitate. '
                f"Observation IDs: {row['before']['observation_id']} -> {row['after']['observation_id']}"),
                image_content(data)])
        self.summary['source_sha256'] = self.source_hashes
        from .model_context import failure, text
        self.model_summary = failure(self.summary)
        self.content.insert(0, text(self.model_summary))

    def _view_records(self, folder, decision, row, session_id):
        """Keep framing and model disagreement separate from executed commands."""
        oid = decision['observation_id']
        request_path = folder/'view_acquisition/request.json'
        if request_path.exists():
            body = self._read(request_path)
            frames, images = [], []
            for message in body.get('messages', []):
                if not isinstance(message.get('content'), list):
                    continue
                for part in message['content']:
                    if part.get('type') == 'image_url':
                        images.append(part)
                    elif part.get('type') == 'text':
                        try:
                            value = json.loads(part.get('text', ''))
                        except (ValueError, TypeError):
                            continue
                        if isinstance(value, dict) and 'frame' in value:
                            frames.append(value)
            if frames != [dict(frame='CURRENT', observation_id=oid)] or len(images) != 2:
                raise ValueError('failure_view_request_observation_mismatch')
        assessment_path = folder/'view_acquisition/assessment.json'
        if assessment_path.exists():
            from . import view_acquisition as view
            record = self._read(assessment_path)
            if (not isinstance(record, dict) or record.get('format') != VIEW_FORMAT
                    or record.get('source') != 'model_not_human'):
                raise ValueError('failure_view_model_provenance_required')
            if record.get('observation_id') != oid:
                raise ValueError('failure_view_observation_mismatch')
            if not request_path.exists():
                raise ValueError('failure_view_assessment_requires_request')
            # Recompute derived image relations from the original model locations.
            # This verifies the archive, not the visual correctness of those points.
            try:
                data = dict(observation_id=oid, **{camera: {key: record[camera][key]
                    for key in view.VIEW_PROPERTIES} for camera in ('head', 'wrist')})
                response = dict(choices=[dict(finish_reason='tool_calls', message=dict(content=None,
                    tool_calls=[dict(function=dict(name='inspect_view', arguments=json.dumps(data)))]))])
                verified = view.validate(response, oid)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError('failure_view_invalid_localization') from exc
            if any(record.get(key) != value for key, value in verified.items()):
                raise ValueError('failure_view_derived_relation_mismatch')
            row['view_assessment'] = record
        for name in ('metrics', 'error'):
            path = folder/'view_acquisition'/(name+'.json')
            if path.exists():
                record = self._read(path)
                if (not isinstance(record, dict) or record.get('observation_id', oid) != oid
                        or name == 'metrics' and record.get('tool', 'inspect_view') != 'inspect_view'):
                    raise ValueError('failure_view_'+name+'_observation_mismatch')
                row['view_'+name] = record
        conflict_path = folder/'progress_review/conflict.json'
        if conflict_path.exists():
            conflict = self._read(conflict_path)
            if (not isinstance(conflict, dict)
                    or conflict.get('source') != 'model_disagreement_not_visual_fact'):
                raise ValueError('failure_view_conflict_provenance_required')
            if conflict.get('observation_id') != oid:
                raise ValueError('failure_view_conflict_observation_mismatch')
            model, review = conflict.get('model_claim'), conflict.get('review_claim')
            if (not isinstance(model, dict) or model.get('format') != PROGRESS_FORMAT
                    or model.get('source') != 'model_not_human'
                    or not isinstance(review, dict) or review.get('format') != PROGRESS_REVIEW_FORMAT
                    or review.get('source') != 'model_not_human'):
                raise ValueError('failure_view_conflict_claim_provenance_required')
            if model.get('observation_id') != oid or review.get('observation_id') != oid:
                raise ValueError('failure_view_conflict_observation_mismatch')
            from .progress import PROGRESS_PROPERTIES, validate_progress
            try:
                validate_progress({key: model[key] for key in PROGRESS_PROPERTIES}, oid, True)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('failure_view_conflict_invalid_model_claim') from exc
            if review != row.get('progress_review'):
                raise ValueError('failure_view_conflict_requires_matching_review')
            axes = conflict.get('conflicting_axes')
            camera = model.get('view')
            if (not isinstance(axes, list) or not axes or any(axis not in ('vertical', 'horizontal') for axis in axes)
                    or len(set(axes)) != len(axes) or camera not in ('head', 'wrist')):
                raise ValueError('failure_view_invalid_conflicting_axes')
            for axis in axes:
                claimed, other = model.get('button_'+axis), review.get(camera+'_'+axis)
                if claimed in (None, 'unknown', other) or other in (None, 'unknown'):
                    raise ValueError('failure_view_conflict_requires_disagreement')
            row['progress_conflict'] = conflict
        state_path = folder/'view_state.json'
        if state_path.exists():
            state = self._read(state_path)
            if (not isinstance(state, dict) or state.get('format') != VIEW_STATE_FORMAT
                    or state.get('source') != 'program_phase_on_model_localization'):
                raise ValueError('failure_view_state_provenance_required')
            if state.get('observation_id') != oid:
                raise ValueError('failure_view_state_observation_mismatch')
            if state.get('phase') not in VIEW_PHASES or state['phase'] != decision['view_phase']:
                raise ValueError('failure_view_phase_mismatch')
            assessment_oid = state.get('assessment_observation_id')
            if state['phase'] == 'ready' and assessment_oid is None:
                raise ValueError('failure_view_ready_requires_usable_assessment')
            if assessment_oid is not None:
                self._observation(assessment_oid, session_id)
                source = next((r['view_assessment'] for r in [row]+list(reversed(self.summary['rounds']))
                    if r.get('view_assessment', {}).get('observation_id') == assessment_oid), None)
                if source is None:
                    raise ValueError('failure_view_state_requires_recorded_assessment')
                if state['phase'] == 'ready' and source['usable_head'] is not True:
                    raise ValueError('failure_view_ready_requires_usable_assessment')
            conflict_oid = state.get('conflict_observation_id')
            if conflict_oid is not None:
                self._observation(conflict_oid, session_id)
                source = next((r['progress_conflict'] for r in [row]+list(reversed(self.summary['rounds']))
                    if r.get('progress_conflict', {}).get('observation_id') == conflict_oid), None)
                if source is None:
                    raise ValueError('failure_view_state_requires_recorded_conflict')
            row['view_state'] = state
        if any(key in row for key in ('view_assessment', 'view_state', 'progress_conflict', 'view_error')):
            summary = dict(source='model_localization_and_program_phase_not_human_confirmation', observation_id=oid)
            if row.get('view_state'):
                summary.update({key: row['view_state'].get(key) for key in
                    ('phase', 'reason', 'assessment_observation_id', 'conflict_observation_id')})
            if row.get('view_assessment'):
                assessment = row['view_assessment']
                summary['usable_head'] = assessment['usable_head']
                summary['localization'] = {camera: {key: assessment[camera][key] for key in
                    ('button_visibility', 'contact_visibility', 'derived_relation', 'note')}
                    for camera in ('head', 'wrist')}
            if row.get('progress_conflict'):
                conflict = row['progress_conflict']
                summary['conflict'] = dict(source=conflict['source'], conflicting_axes=conflict['conflicting_axes'],
                    meaning='Disagreement between model judgments; neither establishes a visual fact.')
            if row.get('view_error'):
                summary['error'] = row['view_error']
            row['view_summary'] = summary

    def _progress_records(self, folder, decision, row, session_id):
        """Keep direct progress/review provenance without treating it as execution."""
        oid = decision['observation_id']
        progress_path = folder/'progress.json'
        if progress_path.exists():
            record = self._read(progress_path)
            self._check_progress_snapshot(record, oid)
            row['progress'] = record
        assessment_path = folder/'progress_review/assessment.json'
        assessment = None
        if assessment_path.exists():
            assessment = self._read(assessment_path)
            if (not isinstance(assessment, dict) or assessment.get('format') != PROGRESS_REVIEW_FORMAT
                    or assessment.get('source') != 'model_not_human'):
                raise ValueError('failure_progress_review_model_provenance_required')
            if assessment.get('observation_id') != oid:
                raise ValueError('failure_progress_review_observation_mismatch')
            previous = assessment.get('previous_observation_id')
            if previous == oid or assessment.get('evidence_observation_ids') != [previous, oid]:
                raise ValueError('failure_progress_review_observation_mismatch')
            # The comparison can span observation refreshes. Do not compare its
            # BEFORE ID with decision.previous_observation_id, which is only the
            # immediately preceding decision boundary.
            self._observation(previous, session_id)
            actual = self._progress_review_interval(previous, oid)
            request_path = folder/'progress_review/request.json'
            if request_path.exists():
                self._check_progress_review_request(self._read(request_path), previous, oid, actual)
            row['progress_review'] = assessment
        decision_path = folder/'progress_review/decision.json'
        if decision_path.exists():
            record = self._read(decision_path)
            if not isinstance(record, dict) or record.get('source') != 'program_rule_on_model_judgment':
                raise ValueError('failure_progress_review_decision_provenance_required')
            if record.get('observation_id') != oid:
                raise ValueError('failure_progress_review_decision_observation_mismatch')
            if assessment is None:
                raise ValueError('failure_progress_review_decision_requires_assessment')
            row['progress_review_decision'] = record
        if any(name in row for name in ('progress', 'progress_review', 'progress_review_decision')):
            record = row.get('progress') or {}
            candidate = record.get('candidate_progress') or {}
            monitor = record.get('monitor') or {}
            summary = dict(source='model_judgments_and_program_checks_not_human_confirmation',
                           observation_id=oid)
            if candidate:
                summary['reported_progress'] = {k: candidate.get(k) for k in (
                    'source', 'subgoal', 'view', 'button_vertical', 'button_horizontal', 'last_effect')}
            if monitor:
                summary['review_trigger_reasons'] = monitor.get('reasons', [])
            if assessment:
                summary['independent_review'] = {k: assessment.get(k) for k in (
                    'source', 'observation_id', 'previous_observation_id', 'comparison_view',
                    'head_vertical', 'head_horizontal', 'wrist_vertical', 'wrist_horizontal',
                    'vertical_change', 'horizontal_change', 'approach_change', 'note',
                    'evidence_observation_ids')}
            if row.get('progress_review_decision'):
                summary['program_decision'] = {k: row['progress_review_decision'].get(k) for k in (
                    'source', 'allowed', 'reasons', 'requires_new_decision')}
            row['progress_summary'] = summary

    @staticmethod
    def _check_progress_snapshot(record, oid):
        if not isinstance(record, dict) or record.get('format') != PROGRESS_FORMAT or record.get('observation_id') != oid:
            raise ValueError('failure_progress_observation_mismatch')
        candidate = record.get('candidate_progress')
        if candidate is not None:
            if (not isinstance(candidate, dict) or candidate.get('format') != PROGRESS_FORMAT or candidate.get('source') != 'model_not_human'):
                raise ValueError('failure_progress_model_provenance_required')
            if candidate.get('observation_id') != oid:
                raise ValueError('failure_progress_observation_mismatch')
        monitor = record.get('monitor')
        if monitor is not None:
            if (not isinstance(monitor, dict) or monitor.get('format') != PROGRESS_FORMAT
                    or monitor.get('source') != 'program_execution_events_and_model_judgments'):
                raise ValueError('failure_progress_monitor_provenance_required')
            if monitor.get('observation_id') != oid:
                raise ValueError('failure_progress_observation_mismatch')
        trigger = record.get('review_trigger')
        if trigger is not None:
            # Trigger snapshots do not contain recursively nested triggers.
            if not isinstance(trigger, dict) or 'review_trigger' in trigger:
                raise ValueError('failure_invalid_progress_trigger')
            FailureCase._check_progress_snapshot(trigger, oid)

    def _progress_review_interval(self, previous, current):
        """Prove the pair covers the last actual movement plus refresh-only turns."""
        cursor = current
        for row in reversed(self.summary['rounds']):
            after = row.get('after')
            before = row['before']['observation_id']
            if after is None:
                if before != cursor:
                    raise ValueError('failure_progress_review_interval_mismatch')
                continue
            if after['observation_id'] != cursor:
                raise ValueError('failure_progress_review_interval_mismatch')
            if row['status'] == 'observed':
                cursor = before
                continue
            if row.get('executed'):
                command = row['command']
                if command.get('type') != 'move':
                    raise ValueError('failure_progress_review_interval_contains_nonmove')
                if not any(command.get('xyz_mm', [])):
                    cursor = before
                    continue
                if before != previous:
                    raise ValueError('failure_progress_review_interval_mismatch')
                return {k: command[k] for k in ('type', 'xyz_mm', 'head_deg', 'key') if k in command}
            raise ValueError('failure_progress_review_interval_mismatch')
        raise ValueError('failure_progress_review_requires_prior_completed_move')

    @staticmethod
    def _check_progress_review_request(body, previous, current, actual):
        frames, commands = [], []
        for message in body.get('messages', []):
            if not isinstance(message.get('content'), list):
                continue
            for part in message['content']:
                if part.get('type') != 'text':
                    continue
                try:
                    value = json.loads(part.get('text', ''))
                except (ValueError, TypeError):
                    continue
                if not isinstance(value, dict):
                    continue
                if 'frame' in value:
                    frames.append(value)
                if 'command' in value:
                    commands.append(value['command'])
        if frames != [dict(frame='BEFORE', observation_id=previous),
                      dict(frame='CURRENT', observation_id=current)] or commands != [actual]:
            raise ValueError('failure_progress_review_request_mismatch')

    def _bytes(self, path):
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ValueError('failure_case_missing_artifact: '+str(path)) from exc
        self.source_hashes[str(path.relative_to(self.root))] = sha256(data)
        return data

    def _read(self, path):
        return json.loads(self._bytes(path))

    def _observation(self, oid, session_id):
        if not isinstance(oid, str) or not oid or Path(oid).name != oid:
            raise ValueError('invalid_failure_observation_id')
        if oid not in self.observations:
            obs = self._read(self.run/'observations'/oid/'manifest.json')
            if obs['observation_id'] != oid or obs['session_id'] != session_id:
                raise ValueError('failure_observation_session_mismatch')
            self.observations[oid] = obs
        obs = self.observations[oid]
        return dict(observation_id=oid, depth=obs['depth'], fixed_point=obs['selection'],
                    state=compact_state(obs['state']))

    def _collage(self, row):
        canvas = Image.new('RGB', (1280, 840), 'white')
        draw = ImageDraw.Draw(canvas)
        command = {k: v for k, v in row['command'].items() if k in ('type', 'xyz_mm', 'head_deg', 'key')}
        draw.text((8, 8), f"HISTORICAL FAILURE / NOT LIVE / round {row['round']} / "
                  +json.dumps(command), fill='darkred')
        for i, phase in enumerate(('before', 'after')):
            oid = row[phase]['observation_id']
            y = 36+i*400
            draw.text((8, y), f'{phase.upper()} observation_id={oid}', fill='black')
            for j, view in enumerate(('head', 'right')):
                x = j*640
                draw.text((x+8, y+16), 'HISTORICAL FAILURE / '+view.upper(), fill='darkred')
                path = self.run/'observations'/oid/(view+'.jpg')
                data = self._bytes(path)
                artifact = self.observations[oid].get('artifacts', {}).get(view+'.jpg')
                if artifact and sha256(data) != artifact['sha256']:
                    raise ValueError('failure_image_hash_mismatch: '+str(path))
                with Image.open(io.BytesIO(data)) as image:
                    canvas.paste(ImageOps.contain(image.convert('RGB'), (640, 360)), (x, y+34))
        buf = io.BytesIO()
        canvas.save(buf, format='JPEG', quality=95)
        return buf.getvalue()

    def save(self, output):
        folder = Path(output)/'failure_case'
        folder.mkdir(parents=True, exist_ok=False)
        atomic_json(folder/'reference.json', self.summary)
        atomic_json(folder/'model_reference.json', self.model_summary)
        atomic_json(folder/'case_review.json', self.review)
        for name, data in self.images:
            (folder/name).write_bytes(data)
