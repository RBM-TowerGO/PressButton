"""Reviewed failure evidence, kept separate from demonstrations and live state."""
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .common import SESSION_SCHEMA, atomic_json, sha256
from .demonstration import image_content


DECISION_FORMATS = ('pressb-conversation-v1', 'pressb-conversation-v2', 'pressb-conversation-v3', 'pressb-conversation-v4',
                    'pressb-conversation-v5')
LOCALIZATION_FORMAT = 'pressb-localization-v1'
MODEL_LOCALIZATION_FORMATS = ('pressb-localization-v2', 'pressb-localization-v3')
HYPOTHESIS_REVIEW_FORMAT = 'pressb-hypothesis-review-v1'
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
            model_localized = bool(decision and decision.get('format') == 'pressb-conversation-v5')
            localized = bool(decision and decision.get('format') in ('pressb-conversation-v4', 'pressb-conversation-v5'))
            if localized and not model_localized and (decision.get('decision_mode') != 'localized'
                              or decision.get('progress_monitor_format') not in (None, 'pressb-localized-progress-v1')
                              or any(decision.get(key) not in (None, LOCALIZATION_FORMAT)
                                     for key in ('localization_format', 'perception_format'))):
                raise ValueError('unsupported_failure_localization_format')
            if model_localized and (decision.get('decision_mode') != 'localized'
                                    or decision.get('localization_format') not in MODEL_LOCALIZATION_FORMATS
                                    or decision.get('progress_monitor_format') not in (None, 'pressb-localized-progress-v2', 'pressb-localized-progress-v3')):
                raise ValueError('unsupported_failure_localization_format')
            if decision and decision.get('progress_monitor_format') is not None and not localized:
                if (decision.get('progress_monitor_format') != PROGRESS_FORMAT
                        or decision.get('format') != 'pressb-conversation-v1'
                        or decision.get('decision_mode', 'direct') != 'direct'):
                    raise ValueError('unsupported_failure_progress_format')
            if decision and decision.get('view_acquisition_format') is not None:
                if (not view_enabled or not monitored or decision.get('format') != 'pressb-conversation-v1'
                        or decision.get('decision_mode', 'direct') != 'direct'
                        or decision.get('view_phase') not in VIEW_PHASES):
                    raise ValueError('unsupported_failure_view_format')
            if body is None and (not decision or decision.get('format') not in INSPECTION_FORMATS) and not monitored and not view_enabled and not localized:
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
            if localized:
                row['decision_stage'] = decision.get('stage')
                if model_localized:
                    self._model_localized_records(folder, decision, row, session['session_id'])
                else:
                    self._localized_records(folder, decision, row, session['session_id'])
                    self._localized_evidence_records(folder, decision, row, session['session_id'])
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
                for relative in ('hypothesis_review/error.json', 'localization_review/error.json', 'localization_review/current/error.json',
                                 'localization_review/before/error.json', 'localization/error.json',
                                 'forward_guard/error.json', 'view_acquisition/error.json', 'progress_review/error.json', 'withdrawal/error.json',
                                 'activation/error.json', 'perception/error.json'):
                    if not error_path.exists():
                        error_path = folder/relative
                stage = (self._localized_failed_stage(folder, decision,
                             self._read(error_path) if error_path.exists() else None) if localized else
                         'view_acquisition' if view_enabled and ((folder/'view_acquisition/request.json').exists()
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
            elif (folder/'error.json').exists() or (localized and (folder/'forward_guard/error.json').exists()):
                error = self._read(folder/'error.json' if (folder/'error.json').exists() else folder/'forward_guard/error.json')
                failed_stage = (self._localized_failed_stage(folder, decision, error) if localized else
                    'view_acquisition' if view_enabled and (
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

    @staticmethod
    def _localized_failed_stage(folder, decision, error=None):
        """No action request means localization work remains an unexecuted round."""
        stage = (error or {}).get('stage', decision.get('stage'))
        if stage in ('localization', 'localization_review', 'hypothesis_review', 'forward_guard',
                     'activation_verification', 'withdrawal_verification'):
            return stage
        if (folder/'forward_guard/error.json').exists():
            return 'forward_guard'
        if (folder/'hypothesis_review/request.json').exists() and not (folder/'request.json').exists():
            return 'hypothesis_review'
        review = folder/'localization_review'
        if (review/'assessment.json').exists() or any((review/name/'request.json').exists()
                                                      for name in ('before', 'current')):
            return 'localization_review'
        if not (folder/'request.json').exists() or (folder/'localization/error.json').exists():
            return 'localization'
        return 'request'

    def _model_localized_records(self, folder, decision, row, session_id):
        """Read v5 advisory locations without applying historical visual gates.

        Observation IDs and image hashes bind the archive to the original input.
        Model visibility and malformed location fields never reject a case: the
        original response remains archived and display geometry is recomputed.
        """
        from . import model_localization as localization
        from .spatial_localization import image_metadata
        oid = decision['observation_id']
        directory = folder/'localization'
        request_path, assessment_path = directory/'request.json', directory/'assessment.json'
        stage, pictures = {}, []
        if request_path.exists():
            body = self._read(request_path)
            frames = []
            for message in body.get('messages', []):
                for part in message.get('content', []) if isinstance(message.get('content'), list) else []:
                    if part.get('type') == 'image_url':
                        pictures.append(part)
                    elif part.get('type') == 'text':
                        try:
                            frame = json.loads(part.get('text', ''))
                        except (ValueError, TypeError):
                            continue
                        if isinstance(frame, dict) and 'frame' in frame:
                            frames.append(frame)
            if frames != [dict(frame='CURRENT', observation_id=oid)] or len(pictures) != 2:
                raise ValueError('failure_localization_request_observation_mismatch')
        active = display = None
        if assessment_path.exists():
            active = self._read(assessment_path)
            if (not isinstance(active, dict) or active.get('format') not in MODEL_LOCALIZATION_FORMATS
                    or active.get('source') != 'model_not_human'):
                raise ValueError('failure_localization_model_provenance_required')
            if active.get('observation_id') != oid:
                raise ValueError('failure_localization_observation_mismatch')
            if not request_path.exists():
                raise ValueError('failure_localization_assessment_requires_request')
            self._observation(oid, session_id)
            sizes, image_records = {}, {}
            for (view, filename), picture in zip((('head', 'head.jpg'), ('wrist', 'right.jpg')), pictures):
                image_bytes = self._bytes(self.run/'observations'/oid/filename)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    sizes[view] = list(image.size)
                image_records[view] = image_metadata(picture)
                artifact = self.observations[oid].get('artifacts', {}).get(filename)
                if (image_records[view]['sha256'] != sha256(image_bytes)
                        or image_records[view]['image_size'] != sizes[view]
                        or artifact and artifact.get('sha256') != sha256(image_bytes)):
                    raise ValueError('failure_localization_image_mismatch')
            if active.get('images', image_records) != image_records:
                raise ValueError('failure_localization_image_mismatch')
            stage['assessment'] = active
            current_path = folder/'localization_current.json'
            if current_path.exists():
                current = self._read(current_path)
                if current != active:
                    raise ValueError('failure_current_localization_version_mismatch')
            if decision.get('localization_revision', active.get('revision')) != active.get('revision'):
                raise ValueError('failure_current_localization_version_mismatch')
            row['localization_current'] = active
            raw = active.get('raw_locations', {view: active.get(view) for view in ('head', 'wrist')})
            display = localization.normalize(raw, oid, sizes, revision=active.get('revision', 1),
                contact_reference=active.get('contact_reference', 'white_arrow'),
                raw_output=active.get('raw_output'), parse_notes=active.get('parse_notes'),
                # v2 declared normalized coordinates but had no explicit unit field.
                # Respect that historical contract; never guess a 1000-scale repair.
                source_units='normalized_01' if active['format'] == 'pressb-localization-v2' else None)
        # Response parsing notes are data, never a retry or execution rejection.
        for path in sorted(directory.glob('response-*.json')):
            self._read(path)
        for filename in ('metrics', 'error'):
            path = directory/(filename+'.json')
            if path.exists():
                record = self._read(path)
                if not isinstance(record, dict) or record.get('observation_id', oid) != oid:
                    raise ValueError('failure_localization_'+filename+'_observation_mismatch')
                stage[filename] = record
        if stage:
            row['localization'] = stage
        progress_path = folder/'localized_progress.json'
        if progress_path.exists():
            progress = self._read(progress_path)
            if progress.get('observation_id', progress.get('current_observation_id')) != oid:
                raise ValueError('failure_localized_progress_observation_mismatch')
            if progress.get('previous_observation_id') is not None:
                self._observation(progress['previous_observation_id'], session_id)
            row['localized_progress'] = progress
        summary = dict(source='model_localizations_and_program_geometry_not_human_confirmation',
                       observation_id=oid, decision_stage=decision.get('stage'),
                       execution_policy=decision.get('execution_policy', 'model_decides'),
                       forward_guard_policy=decision.get('forward_guard_policy', 'none'))
        if display:
            summary.update(revision=display['revision'], coordinate_units=display.get('coordinate_units'),
                coordinate_conversion=display.get('coordinate_conversion'),
                views={view: {key: display[view].get(key) for key in ('button_visibility', 'contact_visibility',
                    'button_box', 'arrow_box', 'contact_tip', 'note', 'derived_relation')} for view in ('head', 'wrist')})
            if display.get('parse_notes'):
                summary.update(parse_notes=display['parse_notes'])
        if row.get('localized_progress'):
            summary['progress'] = row['localized_progress']
        self._hypothesis_review_records(folder, decision, row, session_id)
        if row.get('hypothesis_review'):
            summary['hypothesis_review'] = {k: row['hypothesis_review'][k] for k in ('trigger', 'assessment')
                                            if k in row['hypothesis_review']}
        self._wrist_overlap_records(folder, decision, row, session_id)
        if row.get('forward_guard'):
            guard = row['forward_guard']
            summary['forward_guard'] = {key: guard[key] for key in ('error',) if key in guard}
            if guard.get('decision'):
                summary['forward_guard']['decision'] = {key: guard['decision'].get(key) for key in
                    ('format', 'source', 'policy', 'observation_id', 'assessment_observation_id',
                     'allowed', 'basis', 'reasons', 'authorized_command')}
            if guard.get('assessment'):
                summary['forward_guard']['assessment'] = {key: guard['assessment'].get(key) for key in
                    ('format', 'source', 'observation_id', 'button_visibility', 'contact_visibility',
                     'button_box', 'contact_tip', 'tip_inside_button', 'note', 'geometry')}
        row['localization_summary'] = summary

    def _wrist_overlap_records(self, folder, decision, row, session_id):
        """Archive current-frame gate evidence; a candidate is never an executed action."""
        directory = folder/'forward_guard'
        if decision.get('forward_guard_policy') != 'wrist_tip_overlap' or not directory.exists():
            return
        from .spatial_localization import image_metadata
        oid = decision['observation_id']
        archive = {}
        for path in sorted(directory.glob('*.json')):
            record = self._read(path)
            if path.stem in ('assessment', 'decision', 'metrics', 'error'):
                archive[path.stem] = record
            elif path.stem.startswith('candidate-'):
                archive.setdefault('candidates', []).append(record)
        assessment = archive.get('assessment')
        request_path = directory/'request.json'
        if request_path.exists():
            body = self._read(request_path)
            frames, pictures = [], []
            for message in body.get('messages', []):
                for part in message.get('content', []) if isinstance(message.get('content'), list) else []:
                    if part.get('type') == 'image_url':
                        pictures.append(part)
                    elif part.get('type') == 'text':
                        try:
                            frame = json.loads(part.get('text', ''))
                        except (TypeError, ValueError):
                            continue
                        if isinstance(frame, dict) and 'frame' in frame:
                            frames.append(frame)
            if frames != [dict(frame='CURRENT', observation_id=oid)] or len(pictures) != 1:
                raise ValueError('failure_wrist_overlap_request_observation_mismatch')
            self._observation(oid, session_id)
            image_bytes = self._bytes(self.run/'observations'/oid/'right.jpg')
            image_record = image_metadata(pictures[0])
            if image_record['sha256'] != sha256(image_bytes):
                raise ValueError('failure_wrist_overlap_image_mismatch')
            if assessment and assessment.get('images') != {'wrist': image_record}:
                raise ValueError('failure_wrist_overlap_image_mismatch')
        if assessment:
            if (assessment.get('format') != 'pressb-wrist-tip-overlap-v1'
                    or assessment.get('source') != 'model_not_human'):
                raise ValueError('failure_wrist_overlap_model_provenance_required')
            if assessment.get('observation_id') != oid:
                raise ValueError('failure_wrist_overlap_observation_mismatch')
            if not request_path.exists():
                raise ValueError('failure_wrist_overlap_assessment_requires_request')
        for key in ('decision', 'metrics', 'error'):
            record = archive.get(key)
            if record and record.get('observation_id', oid) != oid:
                raise ValueError('failure_wrist_overlap_observation_mismatch')
        gates = [archive['decision']] if archive.get('decision') else []
        gates.extend(item['decision'] for item in archive.get('candidates', [])
                     if isinstance(item, dict) and item.get('decision'))
        for gate in gates:
            if gate.get('observation_id') != oid:
                raise ValueError('failure_wrist_overlap_observation_mismatch')
            if (gate.get('policy') != 'wrist_tip_overlap'
                    or gate.get('source') != 'program_rule_on_model_judgment'):
                raise ValueError('failure_wrist_overlap_decision_provenance_required')
            if not assessment or gate.get('assessment_observation_id') != oid:
                raise ValueError('failure_wrist_overlap_decision_requires_current_assessment')
        if archive:
            row['forward_guard'] = archive

    def _hypothesis_review_records(self, folder, decision, row, session_id):
        """Keep conditional read-only reviews, including interruptions before action."""
        directory = folder/'hypothesis_review'
        if not directory.exists():
            return
        archive = {}
        for path in sorted(directory.glob('*.json')):
            record = self._read(path)
            if path.stem in ('trigger', 'assessment', 'metrics', 'error', 'decision'):
                archive[path.stem] = record
        oid = decision['observation_id']
        trigger = archive.get('trigger') or {}
        if trigger.get('observation_id', oid) != oid:
            raise ValueError('failure_hypothesis_review_observation_mismatch')
        baseline = trigger.get('baseline_observation_id')
        assessment = archive.get('assessment')
        if assessment:
            if (assessment.get('format') != HYPOTHESIS_REVIEW_FORMAT
                    or assessment.get('source') != 'model_not_human'):
                raise ValueError('failure_hypothesis_review_model_provenance_required')
            if (assessment.get('observation_id') != oid
                    or baseline is not None and assessment.get('previous_observation_id') != baseline):
                raise ValueError('failure_hypothesis_review_observation_mismatch')
            baseline = assessment.get('previous_observation_id')
            if (not baseline or assessment.get('evidence_observation_ids') != [baseline, oid]):
                raise ValueError('failure_hypothesis_review_evidence_mismatch')
        if baseline is not None:
            self._observation(baseline, session_id)
        if archive:
            row['hypothesis_review'] = archive

    def _localized_records(self, folder, decision, row, session_id):
        """Archive explicit per-frame positions and deterministic geometry without inventing motion."""
        from . import spatial_localization as localization
        oid = decision['observation_id']
        versions = {}

        def load_stage(directory, expected_oid, name):
            request_path = directory/'request.json'
            assessment_path = directory/'assessment.json'
            value = {}
            pictures = []
            if request_path.exists():
                body = self._read(request_path)
                frames, images = [], 0
                for message in body.get('messages', []):
                    if not isinstance(message.get('content'), list):
                        continue
                    for part in message['content']:
                        if part.get('type') == 'image_url':
                            pictures.append(part)
                            images += 1
                        if part.get('type') == 'text':
                            try:
                                text = json.loads(part.get('text', ''))
                            except (ValueError, TypeError):
                                continue
                            if isinstance(text, dict) and 'frame' in text:
                                frames.append(text)
                if frames != [dict(frame='CURRENT', observation_id=expected_oid)] or images != 2:
                    raise ValueError('failure_localization_request_observation_mismatch')
            if assessment_path.exists():
                record = self._read(assessment_path)
                if (not isinstance(record, dict) or record.get('format') != LOCALIZATION_FORMAT
                        or record.get('source') != 'model_not_human'):
                    raise ValueError('failure_localization_model_provenance_required')
                if record.get('observation_id') != expected_oid:
                    raise ValueError('failure_localization_observation_mismatch')
                if not request_path.exists():
                    raise ValueError('failure_localization_assessment_requires_request')
                self._observation(expected_oid, session_id)
                sizes, image_records = {}, {}
                for (view, filename), picture in zip((('head', 'head.jpg'), ('wrist', 'right.jpg')), pictures):
                    image_bytes = self._bytes(self.run/'observations'/expected_oid/filename)
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        sizes[view] = list(image.size)
                    image_records[view] = localization.image_metadata(picture)
                    artifact = self.observations[expected_oid].get('artifacts', {}).get(filename)
                    if (image_records[view]['sha256'] != sha256(image_bytes)
                            or image_records[view]['image_size'] != sizes[view]
                            or artifact and artifact.get('sha256') != sha256(image_bytes)):
                        raise ValueError('failure_localization_image_mismatch')
                if record.get('images', image_records) != image_records:
                    raise ValueError('failure_localization_image_mismatch')
                try:
                    raw = dict(observation_id=expected_oid, **{view: {key: record[view][key]
                        for key in ('button_visibility', 'contact_visibility', 'button_box', 'arrow_box', 'contact_tip', 'note')}
                        for view in ('head', 'wrist')})
                    verified = localization.validate_raw(raw, expected_oid, sizes, revision=record['revision'],
                        contact_reference=record.get('contact_reference', 'white_arrow'))
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError('failure_localization_invalid_positions') from exc
                if any(record.get(key) != expected for key, expected in verified.items()):
                    raise ValueError('failure_localization_derived_relation_mismatch')
                value['assessment'] = record
                if expected_oid == oid:
                    if record['revision'] in versions and record != versions[record['revision']]:
                        raise ValueError('failure_localization_revision_reused')
                    versions[record['revision']] = record
            for filename in ('metrics', 'error'):
                path = directory/(filename+'.json')
                if path.exists():
                    record = self._read(path)
                    if not isinstance(record, dict) or record.get('observation_id', expected_oid) != expected_oid:
                        raise ValueError('failure_localization_'+filename+'_observation_mismatch')
                    value[filename] = record
            if value:
                row[name] = value
            return value

        load_stage(folder/'localization', oid, 'localization')
        review_root = folder/'localization_review'
        review_path = review_root/'assessment.json'
        review = self._read(review_path) if review_path.exists() else None
        before_path = review_root/'before/assessment.json'
        before_request = review_root/'before/request.json'
        previous_oid = (review or {}).get('previous_observation_id')
        if previous_oid is None and before_path.exists():
            previous_oid = self._read(before_path).get('observation_id')
        if previous_oid is None and before_request.exists():
            # A review may stop before its first response. The explicit frame in
            # that review request identifies BEFORE; do not choose from action history.
            body = self._read(before_request)
            for message in body.get('messages', []):
                for part in message.get('content', []) if isinstance(message.get('content'), list) else []:
                    if part.get('type') != 'text':
                        continue
                    try:
                        frame = json.loads(part.get('text', ''))
                    except (ValueError, TypeError):
                        continue
                    if isinstance(frame, dict) and frame.get('frame') == 'CURRENT':
                        previous_oid = frame.get('observation_id')
        if previous_oid is not None:
            earlier_ids = {record[boundary]['observation_id'] for record in self.summary['rounds']
                           for boundary in ('before', 'after') if record.get(boundary)}
            if previous_oid == oid or previous_oid not in earlier_ids:
                raise ValueError('failure_localization_review_observation_mismatch')
            self._observation(previous_oid, session_id)
            load_stage(review_root/'before', previous_oid, 'localization_review_before')
        load_stage(review_root/'current', oid, 'localization_review_current')
        if review is not None:
            if (not isinstance(review, dict) or review.get('observation_id') != oid
                    or review.get('previous_observation_id') != previous_oid):
                raise ValueError('failure_localization_review_observation_mismatch')
            if (review.get('format') != 'pressb-localization-review-v1'
                    or review.get('source') != 'model_relocalization_and_program_comparison'):
                raise ValueError('failure_localization_review_provenance_required')
            row['localization_review'] = review
        active_path = folder/'localization_current.json'
        active = self._read(active_path) if active_path.exists() else versions.get(max(versions)) if versions else None
        if active is not None:
            source = versions.get(active.get('revision'))
            if (not source or active.get('observation_id') != oid or active.get('source') != 'model_not_human'
                    or any(active.get(k) != source[k] for k in ('format', 'head', 'wrist', 'contact_reference'))
                    or not set(active.get('usable_views', [])).issubset(source['usable_views'])
                    or decision.get('localization_revision', active['revision']) != active['revision']):
                raise ValueError('failure_current_localization_version_mismatch')
            row['localization_current'] = active
        progress_path = folder/'localized_progress.json'
        if progress_path.exists():
            progress = self._read(progress_path)
            current = progress.get('observation_id', progress.get('current_observation_id'))
            if current != oid:
                raise ValueError('failure_localized_progress_observation_mismatch')
            previous = progress.get('previous_observation_id')
            if previous is not None:
                self._observation(previous, session_id)
            row['localized_progress'] = progress
        summary = dict(source='model_localizations_and_program_geometry_not_human_confirmation',
                       observation_id=oid, decision_stage=decision.get('stage'))
        if active:
            summary.update(revision=active['revision'], usable_views=active['usable_views'],
                views={view: {key: active[view][key] for key in ('button_visibility', 'contact_visibility',
                    'button_box', 'contact_tip', 'derived_relation')} for view in ('head', 'wrist')})
        if review is not None:
            summary['review'] = {k: review[k] for k in ('source', 'previous_observation_id',
                'observation_id', 'conflicting_views', 'before_conflicting_views', 'conflicts', 'usable_views') if k in review}
        if row.get('localized_progress'):
            summary['progress'] = row['localized_progress']
        row['localization_summary'] = summary

    def _localized_evidence_records(self, folder, decision, row, session_id):
        """Completion reviews may cite earlier live frames, not the decision's current image."""
        from .spatial_localization import image_metadata
        seen = []
        for record in self.summary['rounds']+[row]:
            for boundary in ('before', 'after'):
                item = record.get(boundary)
                if item and item['observation_id'] not in seen:
                    seen.append(item['observation_id'])
        for name, format_name, field in (
                ('activation', 'pressb-activation-v1', 'light_on'),
                ('withdrawal', 'pressb-withdrawal-v1', 'withdrawal')):
            directory = folder/name
            request_path, assessment_path = directory/'request.json', directory/'assessment.json'
            frames, pictures = [], []
            if request_path.exists():
                body = self._read(request_path)
                for message in body.get('messages', []):
                    if not isinstance(message.get('content'), list):
                        continue
                    for part in message['content']:
                        if part.get('type') == 'image_url':
                            pictures.append(part)
                        if part.get('type') == 'text':
                            try:
                                value = json.loads(part.get('text', ''))
                            except (ValueError, TypeError):
                                continue
                            if isinstance(value, dict) and 'frame' in value:
                                frames.append(value)
                if (len(frames) != 2 or len(pictures) != 4
                        or [frame.get('frame') for frame in frames] != ['BEFORE', 'CURRENT']):
                    raise ValueError('failure_localized_evidence_request_mismatch')
                ids = [frame.get('observation_id') for frame in frames]
                if (any(oid not in seen for oid in ids) or seen.index(ids[0]) >= seen.index(ids[1])):
                    raise ValueError('failure_localized_evidence_requires_ordered_live_observations')
                for oid, pair in zip(ids, (pictures[:2], pictures[2:])):
                    self._observation(oid, session_id)
                    for filename, picture in zip(('head.jpg', 'right.jpg'), pair):
                        data = self._bytes(self.run/'observations'/oid/filename)
                        if image_metadata(picture)['sha256'] != sha256(data):
                            raise ValueError('failure_localized_evidence_image_mismatch')
            if assessment_path.exists():
                record = self._read(assessment_path)
                if (not isinstance(record, dict) or record.get('format') != format_name
                        or record.get('source') != 'model_not_human'
                        or record.get(field) not in ('yes', 'no', 'unknown')):
                    raise ValueError('failure_localized_evidence_model_provenance_required')
                expected = [record.get('previous_observation_id'), record.get('observation_id')]
                if (not request_path.exists() or expected != ids
                        or record.get('evidence_observation_ids') != expected):
                    raise ValueError('failure_localized_evidence_observation_mismatch')
                row[name] = record
            for file in ('metrics', 'error'):
                path = directory/(file+'.json')
                if path.exists():
                    row[name+'_'+file] = self._read(path)
        if (row.get('activation') and row.get('withdrawal')
                and row['activation']['observation_id'] != row['withdrawal']['previous_observation_id']):
            raise ValueError('failure_localized_withdrawal_must_follow_reviewed_light')

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
