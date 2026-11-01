"""Load an original keyboard recording into reusable, explicitly historical context."""
import base64
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
        session = json.loads((self.root / 'session.json').read_text())
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
            review = json.loads(review_path.read_text())
            if review.get('schema') != 'pressb-case-review-v1':
                raise ValueError('unsupported_demo_review_schema')
            self.summary['case_review'] = review
        self.images = []
        self.content = []
        for path in sorted((self.root / 'operations').glob('*/operation.json')):
            operation = json.loads(path.read_text())
            row = dict(operation=operation, observations={})
            canvas = Image.new('RGB', (1280, 792), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), 'DEMO ' + operation['operation_id'] + ' ' + operation['kind'] +
                      ' status=' + operation.get('status', 'unknown') +
                      ' ' + json.dumps(operation['command'], ensure_ascii=True), fill='black')
            for i, phase in enumerate(('before', 'after')):
                folder = path.parent / phase
                metadata_path = folder / 'metadata.json'
                metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
                row['observations'][phase] = dict(metadata=metadata, missing_images=[])
                for j, view in enumerate(('head', 'right')):
                    x, y = j * 640, 28 + i * 382
                    draw.text((x+8, y), 'DEMO ' + phase.upper() + ' / ' + view.upper(), fill='black')
                    image_path = folder / (view+'.jpg')
                    complete = operation.get(phase, {}).get('status') == 'complete'
                    if complete and image_path.exists():
                        with Image.open(image_path) as image:
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
        self.content.insert(0, dict(type='text', text=json.dumps(self.summary, ensure_ascii=False, allow_nan=False)))

    def save(self, output):
        from .common import atomic_json
        folder = Path(output) / 'demonstration'
        folder.mkdir(parents=True)
        atomic_json(folder / 'reference.json', self.summary)
        for name, data in self.images:
            (folder / name).write_bytes(data)
