"""Load an original keyboard recording into reusable, explicitly historical context."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def image_content(data):
    return dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' +
                                                base64.b64encode(data).decode()))


class Demonstration:
    def __init__(self, directory):
        self.root = Path(directory).expanduser().resolve()
        self.source_sha256 = {}
        def read(path):
            data = path.read_bytes()
            self.source_sha256[str(path.relative_to(self.root))] = hashlib.sha256(data).hexdigest()
            return data
        session = json.loads(read(self.root / 'session.json'))
        if session.get('schema') != 'pressb-keyboard-record-v1':
            raise ValueError('keyboard_demo_schema_required')
        if session['settings'].get('frame') != 'chassis' or session['settings'].get('units') != 'mm':
            raise ValueError('unsupported_demo_coordinates')
        self.summary = dict(source=str(self.root), session=session, operations=[],
            context_type='HISTORICAL_DEMONSTRATION_NOT_LIVE',
            coordinates='chassis: +X forward, +Y robot left, +Z up; deltas mm; pose translation m; quaternion xyzw',
            state_semantics='Recorded poses are COMMAND poses, not measured states.',
            outcome='Operation completed means commands sent, not verified contact or button activation.')
        review_path = self.root/'case_review.json'
        if review_path.exists():
            review = json.loads(read(review_path))
            if review.get('schema') != 'pressb-case-review-v1':
                raise ValueError('unsupported_demo_review_schema')
            self.summary['case_review'] = review
        self.images = []
        self.content = []
        for path in sorted((self.root / 'operations').glob('*/operation.json')):
            operation = json.loads(read(path))
            row = dict(operation=operation, observations={})
            canvas = Image.new('RGB', (1280, 792), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), 'DEMO ' + operation['operation_id'] + ' ' + operation['kind'] +
                      ' status=' + operation.get('status', 'unknown') +
                      ' ' + json.dumps(operation['command'], ensure_ascii=True), fill='black')
            for i, phase in enumerate(('before', 'after')):
                folder = path.parent / phase
                metadata_path = folder / 'metadata.json'
                metadata = json.loads(read(metadata_path)) if metadata_path.exists() else None
                row['observations'][phase] = dict(metadata=metadata, missing_images=[])
                for j, view in enumerate(('head', 'right')):
                    x, y = j * 640, 28 + i * 382
                    draw.text((x+8, y), 'DEMO ' + phase.upper() + ' / ' + view.upper(), fill='black')
                    image_path = folder / (view+'.jpg')
                    complete = operation.get(phase, {}).get('status') == 'complete'
                    if complete and not image_path.exists():
                        raise ValueError('demo_missing_required_image: '+str(image_path))
                    if complete and image_path.exists():
                        with Image.open(io.BytesIO(read(image_path))) as image:
                            tile = ImageOps.contain(image.convert('RGB'), (640, 360))
                            canvas.paste(tile, (x, y+20))
                    else:
                        draw.text((x+20, y+80), 'MISSING - DO NOT INFER CONTENT', fill='red')
                        row['observations'][phase]['missing_images'].append(view)
            buf = io.BytesIO()
            canvas.save(buf, format='JPEG', quality=95)
            data = buf.getvalue()
            self.images.append((operation['operation_id']+'.jpg', data))
            self.content.extend([dict(type='text', text='HISTORICAL DEMO step '+operation['operation_id']+
                                      ' status='+operation.get('status', 'unknown')+
                                      ': top BEFORE, bottom AFTER; left HEAD, right WRIST. '
                                      'Interrupted commands and missing AFTER images do not prove completed motion.'), image_content(data)])
            self.summary['operations'].append(row)
        if not self.images:
            raise ValueError('demo_has_no_operations')
        from .model_context import demonstration, text
        self.model_summary = demonstration(self.summary)
        self.content.insert(0, text(self.model_summary))
        self._all_images = tuple(self.images)
        self._all_content = copy.deepcopy(self.content)
        self._full_model_summary = copy.deepcopy(self.model_summary)
        self.selection = dict(mode='all', selected_steps=[r['step'] for r in self.model_summary['operations']],
                              model_image_count=len(self.images), source_operation_count=len(self.images))

    def select_activation_tail(self, count):
        """Select a reviewed activation tail for model input; retain the full source archive."""
        if type(count) is not int or count <= 0:
            raise ValueError('positive_demo_last_steps_required')
        review = self.summary.get('case_review') or {}
        if review.get('button_activation_verified') is not True:
            raise ValueError('reviewed_demo_activation_required')
        ends = set()
        for evidence in review.get('evidence', []):
            parts = evidence.split('/') if isinstance(evidence, str) else []
            if (len(parts) == 4 and parts[0] == 'operations' and parts[2] == 'after'
                    and parts[3] in ('head.jpg', 'right.jpg')):
                ends.add(parts[1])
        if len(ends) != 1:
            raise ValueError('unique_reviewed_demo_activation_boundary_required')
        activation_step = next(iter(ends))
        rows = self._full_model_summary['operations']
        matches = [i for i, row in enumerate(rows) if row['step'] == activation_step]
        if len(matches) != 1:
            raise ValueError('reviewed_demo_activation_step_missing')
        end = matches[0]
        if end + 1 < count:
            raise ValueError('not_enough_completed_demo_activation_steps')
        indices = list(range(end-count+1, end+1))
        selected = [rows[i] for i in indices]
        for row in selected:
            if (row['status'] != 'completed' or any(row[phase]['status'] != 'complete'
                    or row[phase]['missing_images'] for phase in ('before', 'after'))):
                raise ValueError('demo_activation_tail_requires_complete_boundaries:'+row['step'])
        self.selection = dict(mode='reviewed_activation_tail', requested_last_steps=count,
            selected_steps=[row['step'] for row in selected], activation_step=activation_step,
            activation_boundary='after', model_image_count=len(selected), source_operation_count=len(rows),
            analysis_version=review.get('analysis_version'),
            limitations='Selection ends at reviewed activation; interrupted later withdrawal and missing final remain unverified.')
        summary = copy.deepcopy(self._full_model_summary)
        summary['operations'] = copy.deepcopy(selected)
        # Avoid reintroducing unselected actions through the review's full lessons list.
        summary['case_review'] = {key: copy.deepcopy(review[key]) for key in (
            'case_id', 'analysis_version', 'outcome', 'outcome_source', 'outcome_scope',
            'button_activation_verified', 'withdrawal_verified', 'image_review', 'limitations',
            'measurement_point') if key in review}
        summary['selection'] = copy.deepcopy(self.selection)
        summary['reviewed_activation_depth_references'] = [r for r in summary.get(
            'reviewed_activation_depth_references', []) if r['step'] in self.selection['selected_steps']]
        from .model_context import text
        self.model_summary = summary
        self.images = [self._all_images[i] for i in indices]
        self.content = [text(summary)]
        for i in indices:
            self.content.extend(copy.deepcopy(self._all_content[1+2*i:3+2*i]))
        return self

    def save(self, output):
        from .common import atomic_json
        for name, expected in self.source_sha256.items():
            if hashlib.sha256((self.root/name).read_bytes()).hexdigest() != expected:
                raise ValueError('demo_source_changed:'+name)
        folder = Path(output) / 'demonstration'
        folder.mkdir(parents=True)
        atomic_json(folder / 'reference.json', self.summary)
        atomic_json(folder / 'model_reference.json', self.model_summary)
        atomic_json(folder / 'selection.json', self.selection)
        atomic_json(folder / 'source_sha256.json', self.source_sha256)
        for name, data in self.images:
            (folder / name).write_bytes(data)
