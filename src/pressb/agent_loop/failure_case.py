"""Reviewed failure evidence, kept separate from demonstrations and live state."""
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .common import SESSION_SCHEMA, atomic_json, sha256
from .demonstration import image_content


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
            body = self._read(folder/'request.json')
            contexts = []
            for message in body['messages']:
                if not isinstance(message.get('content'), list):
                    continue
                for item in message['content']:
                    if item.get('type') != 'text' or not item.get('text', '').startswith('{'):
                        continue
                    context = json.loads(item['text'])
                    if context.get('context_type') == 'LIVE_OBSERVATION':
                        contexts.append(context)
            if len(contexts) != 1:
                raise ValueError('failure_round_requires_one_recorded_observation')
            before = self._observation(contexts[0]['observation_id'], session['session_id'])
            proposal = self._read(folder/'proposal.json') if (folder/'proposal.json').exists() else None
            row = dict(round=number, proposal=proposal, before=before, after=None,
                       executed=False, status='no_execution_record')
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
            elif (folder/'result.json').exists():
                raise ValueError('failure_result_missing_execution_record')
            elif (folder/'error.json').exists():
                row.update(status='request_interrupted' if outcome['status'] == 'operator_stopped'
                           else 'request_failed', error=self._read(folder/'error.json'))
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
        self.content.insert(0, dict(type='text', text=json.dumps(self.summary, ensure_ascii=False, allow_nan=False)))

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
        atomic_json(folder/'case_review.json', self.review)
        for name, data in self.images:
            (folder/name).write_bytes(data)
