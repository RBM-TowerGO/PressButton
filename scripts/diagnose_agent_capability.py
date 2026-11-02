#!/usr/bin/env python3
"""Small saved-image capability probes. Model HTTP only; no robot execution."""
import argparse
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from pressb.agent_loop.common import atomic_json, request
from s1b_agent import tunnel

FRAMES = {
    'A': dict(run='20261102_120741_0ca36af5', round=8,
              observation_id='f31cc38545bf422b980a9196c7a4ebbd',
              target_xy=[420, 117], physical_tip_xy=[470, 280], sticker_center_xy=[509, 315],
              crop_box=[345, 75, 590, 350]),
    'B': dict(run='20261102_142505_13668097', round=5,
              observation_id='981630409f1349a2a93a53a23eb19399',
              target_xy=[164, 60], physical_tip_xy=[470, 280], sticker_center_xy=[509, 315],
              crop_box=[115, 20, 565, 350]),
}
VERTICAL = ['above', 'level', 'below', 'unknown']
HORIZONTAL = ['left', 'center', 'right', 'unknown']
RELATION = dict(vertical=dict(type='string', enum=VERTICAL),
                horizontal=dict(type='string', enum=HORIZONTAL))
VISIBILITY = {name: dict(type='string', enum=['yes', 'no', 'uncertain'])
              for name in ('target_visible', 'reference_visible')}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def image_bytes(image):
    output = io.BytesIO()
    image.convert('RGB').save(output, format='JPEG', quality=95)
    return output.getvalue()


def reference(kind):
    if kind == 'sticker':
        return ('Target T: the elevator DOWN-CALL button face. Reference P: the entire '
                'WHITE ARROW STICKER attached to a finger, using the sticker center. '
                'The button symbol is not this sticker.')
    if kind == 'tip':
        return ('Target T: the elevator DOWN-CALL button face. Reference P: the physical '
                'finger front end indicated by the WHITE ARROW STICKER. Use that physical '
                'end, not the sticker center or the other finger. The button symbol is not this sticker.')
    return ('Target T: the center of the small colored point labelled T. Reference P: '
            'the center of the small colored point labelled P. Compare these two annotated '
            'points only; do not infer contact or robot movement.')


def body(raw=None, kind='tip', points=None):
    coordinate = points is not None
    props = RELATION if coordinate else dict(VISIBILITY, **RELATION)
    system = ('Compare the target T to reference P in a 2D image coordinate system: x increases '
              'toward image RIGHT, y increases toward image BOTTOM. Return inspect_relation once, '
              'without prose. The relations describe T relative to P, never robot motion or 3D contact. ')
    if coordinate:
        system += 'Use only the supplied numeric point coordinates; this is a coordinate-comparison question, not image interpretation.'
        content = json.dumps(points)
    else:
        system += ('Identify each referent in the supplied image. If a referent cannot be identified, '
                   'report its visibility uncertain or no and keep unsupported relations unknown. '
                   'These are offline saved or diagnostic images, never a live control request.')
        content = [dict(type='text', text=reference(kind)),
                   dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,'+
                                                        base64.b64encode(raw).decode()))]
    return dict(model='qwen38', temperature=0, max_tokens=256, stream=False,
                chat_template_kwargs=dict(enable_thinking=False), tool_choice='required',
                tools=[dict(type='function', function=dict(name='inspect_relation',
                    description='Read-only 2D relation comparison, no action.', parameters=dict(
                        type='object', properties=copy.deepcopy(props), required=list(props), additionalProperties=False)))],
                messages=[dict(role='system', content=system), dict(role='user', content=content)])


def expected(target, point):
    return dict(vertical='above' if target[1] < point[1] else 'below' if target[1] > point[1] else 'level',
                horizontal='left' if target[0] < point[0] else 'right' if target[0] > point[0] else 'center')


def parse(answer, coordinate=False):
    choices = answer['choices']
    if len(choices) != 1 or choices[0]['finish_reason'] != 'tool_calls':
        raise ValueError('one_complete_tool_call_required')
    msg = choices[0]['message']
    calls = msg.get('tool_calls', [])
    if len(calls) != 1 or (msg.get('content') or '').strip():
        raise ValueError('one_tool_without_prose_required')
    fn = calls[0]['function']
    if fn['name'] != 'inspect_relation':
        raise ValueError('wrong_tool')
    data = json.loads(fn['arguments'])
    props = RELATION if coordinate else dict(VISIBILITY, **RELATION)
    if not isinstance(data, dict) or set(data) != set(props):
        raise ValueError('invalid_fields')
    if any(data[key] not in value['enum'] for key, value in props.items()):
        raise ValueError('invalid_enum')
    return data


def archive_request(payload, folder, output):
    readable = copy.deepcopy(payload)
    for message in readable['messages']:
        if not isinstance(message['content'], list):
            continue
        for part in message['content']:
            if part.get('type') == 'image_url':
                url = part['image_url']['url']
                prefix, encoded = url.split(',', 1)
                if prefix != 'data:image/jpeg;base64':
                    raise ValueError('jpeg_archive_required')
                raw = base64.b64decode(encoded, validate=True)
                path = output/'images'/(sha(raw)+'.jpg')
                path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    path.write_bytes(raw)
                part['image_url']['url'] = '../../images/'+path.name
    restored = copy.deepcopy(readable)
    for message in restored['messages']:
        if isinstance(message['content'], list):
            for part in message['content']:
                if part.get('type') == 'image_url':
                    raw = (folder/part['image_url']['url']).read_bytes()
                    part['image_url']['url'] = 'data:image/jpeg;base64,'+base64.b64encode(raw).decode()
    if restored != payload:
        raise ValueError('request_archive_roundtrip_failed')
    atomic_json(folder/'request.readable.json', readable)


def prepare(output):
    sources, jobs, images = {}, [], {}
    for name, meta in FRAMES.items():
        run = ROOT/'outputs/agent_web'/meta['run']/'agent'
        decision = json.loads((run/'rounds'/f'{meta["round"]-1:04d}'/'decision_context.json').read_text())
        if decision['observation_id'] != meta['observation_id']:
            raise ValueError('source_observation_changed')
        path = run/'observations'/meta['observation_id']/'right.jpg'
        raw = path.read_bytes()
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        images[name] = raw
        sources[name] = dict(meta, source=str(path), sha256=sha(raw), size=list(img.size),
            annotation_source='assistant_visual_review_before_model_calls_not_human_operator_or_calibration',
            annotation_uncertainty='Approximate 2D centers/front end; not a calibrated physical contact. Offsets are large relative to a few pixels.')
        def add(case, raw_image, kind, target, point, transform):
            jobs.append(dict(name=case, frame=name, kind=kind, transform=transform,
                             expected=expected(target, point), payload=body(raw_image, kind)))
        add(name+'_raw_sticker', raw, 'sticker', meta['target_xy'], meta['sticker_center_xy'], 'none')
        add(name+'_raw_tip', raw, 'tip', meta['target_xy'], meta['physical_tip_xy'], 'none')
        if name == 'A':
            flipped = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            def flip(point):
                return [point[0], img.height-1-point[1]]
            flipped_bytes = image_bytes(flipped)
            add('A_flip_sticker', flipped_bytes, 'sticker', flip(meta['target_xy']),
                flip(meta['sticker_center_xy']), 'vertical_flip_offline_only')
            add('A_flip_tip', flipped_bytes, 'tip', flip(meta['target_xy']),
                flip(meta['physical_tip_xy']), 'vertical_flip_offline_only')
            for suffix, picture, target, point in (
                    ('raw', img.copy(), meta['target_xy'], meta['physical_tip_xy']),
                    ('flip', flipped.copy(), flip(meta['target_xy']), flip(meta['physical_tip_xy']))):
                draw = ImageDraw.Draw(picture)
                for label, xy, color in (('T', target, '#ff00ff'), ('P', point, '#00ffff')):
                    x, y = xy
                    draw.ellipse((x-4, y-4, x+4, y+4), fill=color, outline='black', width=1)
                    draw.text((x+7, y-7), label, fill=color, stroke_width=1, stroke_fill='black')
                add('A_'+suffix+'_marked', image_bytes(picture), 'marked', target, point,
                    'assistant_T_P_annotations'+('_vertical_flip' if suffix == 'flip' else ''))
        left, top, right, bottom = meta['crop_box']
        cropped = img.crop((left, top, right, bottom))
        cropped = cropped.resize((cropped.width*2, cropped.height*2), Image.Resampling.LANCZOS)
        target = [(meta['target_xy'][0]-left)*2, (meta['target_xy'][1]-top)*2]
        point = [(meta['physical_tip_xy'][0]-left)*2, (meta['physical_tip_xy'][1]-top)*2]
        add(name+'_crop2x_tip', image_bytes(cropped), 'tip', target, point,
            dict(crop_box=meta['crop_box'], scale=2, caveat='Changes resolution AND surrounding context; cannot attribute a difference to one alone.'))
    # Numeric comparisons isolate coordinate ordering; they do not test visual recognition.
    for name, target, point in (
            ('above_left', [420, 117], [470, 280]),
            ('below_left', [420, 280], [470, 117]),
            ('above_right', [470, 117], [420, 280]),
            ('below_right', [470, 280], [420, 117])):
        jobs.append(dict(name='coordinates_'+name, frame=None, kind='coordinates', transform='text_only',
            expected=expected(target, point), payload=body(points=dict(T=target, P=point,
                units='image pixels; x right, y down'))))
    for name in ('A', 'B'):
        original = next(j for j in jobs if j['name'] == name+'_raw_tip')
        duplicate = copy.deepcopy(original)
        duplicate['name'] = name+'_raw_tip_repeat'
        duplicate['repeat_of'] = original['name']
        jobs.append(duplicate)
    assert len(jobs) == 16
    plan = dict(sources=sources, conditions=[{k:v for k,v in job.items() if k!='payload'} for job in jobs],
        predeclared_before_requests=True, robot_connected=False, robot_commands_sent=0,
        design_limits=[
            'Two source images, one observation per condition; selected difficult frames, not a population accuracy estimate.',
            'Vertical flips and T/P overlays are diagnostic stimuli, not real robot observations or movement outcomes.',
            'Visibility fields report object presence, not precise localization accuracy.',
            'T/P marks remove referent-identification burden but add salient cues; improvement does not uniquely identify an internal model layer.',
            'Point coordinates are assistant annotations, not calibrated contact geometry.',
            'No model correction, selection of best output, live robot execution, precision comparison or thinking ablation.',
        ])
    atomic_json(output/'plan.json', plan)
    for job in jobs:
        folder = output/'cases'/job['name']
        folder.mkdir(parents=True)
        archive_request(job['payload'], folder, output)
    return plan, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/report/20261102_capability_bottleneck/probes')
    parser.add_argument('--model-base')
    parser.add_argument('--thor-host', default='Thor')
    parser.add_argument('--thor-port', type=int, default=18023)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan, jobs = prepare(output)
    (output/'script_used.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(output/'source_hashes.json', dict(script_sha256=sha(Path(__file__).read_bytes()),
        image_sha256={k:v['sha256'] for k,v in plan['sources'].items()}))
    if args.prepare_only:
        print('Prepared '+str(output), flush=True)
        return
    base = args.model_base or f'http://127.0.0.1:{args.thor_port}/v1'
    report = dict(robot_connected=False, robot_commands_sent=0, model_only=True,
                  thinking=False, max_tokens=256, results=[], status='running')
    atomic_json(output/'summary.json', report)
    try:
        with tunnel(None if args.model_base else args.thor_host, args.thor_port):
            atomic_json(output/'models.json', request(base, '/models', timeout=5))
            for index, job in enumerate(jobs):
                folder = output/'cases'/job['name']
                row = {k:v for k,v in job.items() if k!='payload'}
                started = time.monotonic()
                print(f'{index+1}/16 '+job['name'], flush=True)
                try:
                    answer = request(base, '/chat/completions', job['payload'], timeout=120)
                    row['elapsed_s'] = time.monotonic()-started
                    atomic_json(folder/'response.json', answer)
                    row['usage'] = answer.get('usage', {})
                    row['finish_reason'] = answer.get('choices', [{}])[0].get('finish_reason')
                    parsed = parse(answer, coordinate=job['kind']=='coordinates')
                    row['parsed'] = parsed
                    row['format_valid'] = True
                    row['relation_correct'] = all(parsed[k] == value for k,value in job['expected'].items())
                    row['vertical_correct'] = parsed['vertical'] == job['expected']['vertical']
                    row['horizontal_correct'] = parsed['horizontal'] == job['expected']['horizontal']
                    row['relation_unknown'] = any(parsed[k]=='unknown' for k in RELATION)
                    if job['kind'] != 'coordinates':
                        row['presence_report_correct'] = all(parsed[k]=='yes' for k in VISIBILITY)
                    if job.get('repeat_of'):
                        previous = next(r for r in report['results'] if r['name']==job['repeat_of'])
                        row['parsed_equal_to_first'] = parsed == previous.get('parsed')
                except Exception as exc:
                    row.setdefault('elapsed_s', time.monotonic()-started)
                    row['format_valid'] = False
                    row['error'] = dict(type=type(exc).__name__, message=str(exc))
                atomic_json(folder/'result.json', row)
                report['results'].append(row)
                atomic_json(output/'summary.json', report)
                print(json.dumps({k:row[k] for k in ('name','format_valid','parsed','relation_correct','elapsed_s','error') if k in row},
                                 ensure_ascii=False), flush=True)
            report['status'] = 'completed'
    finally:
        report['totals'] = dict(requests=len(report['results']),
            prompt_tokens=sum(r.get('usage',{}).get('prompt_tokens',0) for r in report['results']),
            completion_tokens=sum(r.get('usage',{}).get('completion_tokens',0) for r in report['results']),
            elapsed_s=sum(r['elapsed_s'] for r in report['results']),
            format_valid=sum(r['format_valid'] for r in report['results']),
            relation_correct=sum(r.get('relation_correct',False) for r in report['results']))
        atomic_json(output/'summary.json', report)
        print('Report: '+str(output/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
