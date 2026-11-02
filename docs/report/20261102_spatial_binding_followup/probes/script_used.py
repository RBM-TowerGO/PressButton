#!/usr/bin/env python3
"""Saved-image spatial binding probes; model HTTP only, never robot execution."""
import argparse
import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from pressb.agent_loop.common import atomic_json, request
from s1b_agent import tunnel

FRAMES = {
    'A': dict(run='20261102_120741_0ca36af5', round=8,
              observation_id='f31cc38545bf422b980a9196c7a4ebbd',
              target_xy=[420, 117], physical_tip_xy=[470, 280],
              crop_box=[345, 75, 590, 350]),
    'B': dict(run='20261102_142505_13668097', round=5,
              observation_id='981630409f1349a2a93a53a23eb19399',
              target_xy=[164, 60], physical_tip_xy=[470, 280],
              crop_box=[115, 20, 565, 350]),
}
RELATION = dict(vertical=dict(type='string', enum=['above', 'level', 'below', 'unknown']),
                horizontal=dict(type='string', enum=['left', 'center', 'right', 'unknown']))
POINT = dict(anyOf=[dict(type='null'), dict(type='object', properties={
    axis: dict(type='number', minimum=0, maximum=1000) for axis in ('x', 'y')},
    required=['x', 'y'], additionalProperties=False)])
SCHEMAS = {
    'point': dict(point=POINT),
    'relation': RELATION,
    'combined': dict(target_point=POINT, reference_point=POINT, **RELATION),
    'ocr': {color + '_label': dict(anyOf=[dict(type='null'), dict(type='string')])
            for color in ('magenta', 'cyan')},
}
TOOL = 'inspect_saved_image'
TIP = ('the physical finger front end indicated by the WHITE ARROW STICKER; '
       'use that physical end, not the sticker center and not the other finger')
BUTTON = 'the center of the circular elevator button face'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def png_bytes(image):
    output = io.BytesIO()
    image.convert('RGB').save(output, format='PNG')
    return output.getvalue()


def relation(target, reference):
    return dict(vertical='above' if target[1] < reference[1] else 'below' if target[1] > reference[1] else 'level',
                horizontal='left' if target[0] < reference[0] else 'right' if target[0] > reference[0] else 'center')


def body(mode, prompt, raw=None, mime='image/png'):
    if mode not in SCHEMAS:
        raise ValueError('unknown_probe_mode')
    system = ('This is an offline read-only image inspection experiment, never a robot control request. '
              'Image coordinates: x increases toward image RIGHT; y increases toward image BOTTOM. '
              'For point localization, use normalized x and y in [0,1000], with (0,0) at the top-left '
              'and (1000,1000) at the bottom-right. If the requested point cannot be identified, return null. '
              'Relations always describe the requested TARGET relative to the requested REFERENCE. '
              'Do not infer robot motion or 3D contact. Return the specified tool exactly once without prose. '
              'Report unknown when an image relation cannot be established.')
    content = [dict(type='text', text=prompt)]
    if raw is not None:
        content.append(dict(type='image_url', image_url=dict(
            url=f'data:{mime};base64,' + base64.b64encode(raw).decode())))
    props = copy.deepcopy(SCHEMAS[mode])
    return dict(model='qwen38', temperature=0, max_tokens=384, stream=False,
                chat_template_kwargs=dict(enable_thinking=False), tool_choice='required',
                tools=[dict(type='function', function=dict(name=TOOL,
                    description='Read-only image localization, relation or label inspection; no action.',
                    parameters=dict(type='object', properties=props, required=list(props), additionalProperties=False)))],
                messages=[dict(role='system', content=system), dict(role='user', content=content)])


def valid_point(point):
    if point is None:
        return True
    return (isinstance(point, dict) and set(point) == {'x', 'y'} and
            all(type(point[a]) in (int, float) and math.isfinite(point[a]) and 0 <= point[a] <= 1000
                for a in ('x', 'y')))


def parse(answer, mode):
    choices = answer['choices']
    if len(choices) != 1 or choices[0]['finish_reason'] != 'tool_calls':
        raise ValueError('one_complete_tool_call_required')
    msg = choices[0]['message']
    calls = msg.get('tool_calls', [])
    if len(calls) != 1 or (msg.get('content') or '').strip():
        raise ValueError('one_tool_without_prose_required')
    fn = calls[0]['function']
    if fn['name'] != TOOL:
        raise ValueError('wrong_tool')
    data = json.loads(fn['arguments'])
    if not isinstance(data, dict) or set(data) != set(SCHEMAS[mode]):
        raise ValueError('invalid_fields')
    for key, value in data.items():
        if key in ('point', 'target_point', 'reference_point'):
            if not valid_point(value):
                raise ValueError('invalid_normalized_point')
        elif key in RELATION:
            if value not in RELATION[key]['enum']:
                raise ValueError('invalid_relation_enum')
        elif key.endswith('_label') and value is not None and not isinstance(value, str):
            raise ValueError('invalid_ocr_label')
    return data


def archive_request(payload, folder, output):
    """Store small readable requests, retaining exact image bytes and round-trip equality."""
    readable = copy.deepcopy(payload)
    image_refs = []
    for message in readable['messages']:
        if not isinstance(message['content'], list):
            continue
        for part in message['content']:
            if part.get('type') != 'image_url':
                continue
            prefix, encoded = part['image_url']['url'].split(',', 1)
            ext = {'data:image/jpeg;base64': '.jpg', 'data:image/png;base64': '.png'}.get(prefix)
            if ext is None:
                raise ValueError('unsupported_archive_image_mime')
            raw = base64.b64decode(encoded, validate=True)
            path = output / 'images' / (sha(raw) + ext)
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                path.write_bytes(raw)
            part['image_url']['url'] = '../../images/' + path.name
            image_refs.append(dict(path=part['image_url']['url'], prefix=prefix, sha256=sha(raw), bytes=len(raw)))
    restored = copy.deepcopy(readable)
    index = 0
    for message in restored['messages']:
        if isinstance(message['content'], list):
            for part in message['content']:
                if part.get('type') == 'image_url':
                    raw = (folder / part['image_url']['url']).read_bytes()
                    part['image_url']['url'] = image_refs[index]['prefix'] + ',' + base64.b64encode(raw).decode()
                    index += 1
    if restored != payload:
        raise ValueError('request_archive_roundtrip_failed')
    atomic_json(folder / 'request.readable.json', readable)
    atomic_json(folder / 'request_images.json', dict(images=image_refs, exact_payload_restoration_checked=True))


def load_sources():
    sources, pictures = {}, {}
    for name, meta in FRAMES.items():
        run = ROOT / 'outputs/agent_web' / meta['run'] / 'agent'
        decision = json.loads((run / 'rounds' / f'{meta["round"] - 1:04d}' / 'decision_context.json').read_text())
        if decision['observation_id'] != meta['observation_id']:
            raise ValueError('source_observation_changed')
        path = run / 'observations' / meta['observation_id'] / 'right.jpg'
        raw = path.read_bytes()
        pictures[name] = Image.open(io.BytesIO(raw)).convert('RGB')
        sources[name] = dict(meta, source=str(path), sha256=sha(raw), size=list(pictures[name].size),
            annotation_source='assistant_visual_review_before_requests_not_operator_label_or_calibration',
            annotation_uncertainty='Approximate 2D button center and physical finger front end; not calibrated contact coordinates.')
    return sources, pictures


def build_jobs(output, pictures):
    """Twenty-five fixed conditions, including one declared dependency on earlier point outputs."""
    jobs, stimuli = [], {}
    figures = output / 'figures'
    figures.mkdir()
    for name, picture in pictures.items():
        stimuli[name + '_full1x'] = picture
    a = pictures['A']
    roi = FRAMES['A']['crop_box']
    cropped = a.crop(roi)
    stimuli['A_full2x'] = a.resize((a.width * 2, a.height * 2), Image.Resampling.LANCZOS)
    stimuli['A_crop1x'] = cropped
    stimuli['A_crop2x'] = cropped.resize((cropped.width * 2, cropped.height * 2), Image.Resampling.LANCZOS)
    gray = Image.new('RGB', a.size, (128, 128, 128))
    gray.paste(cropped, (roi[0], roi[1]))
    stimuli['A_gray_outside_ROI'] = gray
    translated = Image.new('RGB', a.size, (128, 128, 128))
    translated.paste(cropped, (10, roi[1]))
    stimuli['A_translated_ROI'] = translated
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    font = ImageFont.truetype(str(font_path), 28)
    t, p = FRAMES['A']['target_xy'], FRAMES['A']['physical_tip_xy']
    for key, background, labels in (
            ('large_photo', a, ('T', 'P')),
            ('swapped_photo', a, ('P', 'T')),
            ('blank_large', Image.new('RGB', a.size, (128, 128, 128)), ('T', 'P'))):
        marked = background.copy()
        draw = ImageDraw.Draw(marked)
        for xy, color, label in zip((t, p), ('#ff00ff', '#00ffff'), labels):
            x, y = xy
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline='black', width=1)
            draw.text((x + 13, y - 17), label, font=font, fill=color, stroke_width=2, stroke_fill='black')
        stimuli[key] = marked
    for name, stimulus in stimuli.items():
        (figures / (name + '.png')).write_bytes(png_bytes(stimulus))
    original_marked_folder = ROOT / 'docs/report/20261102_capability_bottleneck/probes/cases/A_raw_marked'
    original_marked_request = json.loads((original_marked_folder / 'request.readable.json').read_text())
    original_part = next(part for message in original_marked_request['messages'] if isinstance(message['content'], list)
                         for part in message['content'] if part.get('type') == 'image_url')
    original_marked_path = (original_marked_folder / original_part['image_url']['url']).resolve()
    original_marked_bytes = original_marked_path.read_bytes()
    (figures / 'prior_small_TP.jpg').write_bytes(original_marked_bytes)

    natural = f'TARGET T: {BUTTON}. REFERENCE P: {TIP}. '
    compare = 'Compare TARGET relative to REFERENCE in the image plane. Output only vertical and horizontal relations.'
    marked = ('TARGET T: center of the colored dot labelled T. REFERENCE P: center of the colored dot labelled P. '
              'Use the dot centers, not the letter centers; compare only these annotated points. ')
    ocr_prompt = ('Read the letter next to the MAGENTA dot and the letter next to the CYAN dot. '
                  'Return magenta_label and cyan_label. Do not report positions or object names. If unreadable, use null.')
    point_boxes = {'A_button': [410, 107, 430, 127], 'B_button': [154, 50, 174, 70],
                   'A_tip': [460, 260, 490, 297], 'B_tip': [460, 260, 490, 297]}

    def add(name, mode, stimulus, prompt, expected, frame='A', scale=1, transform=None, **extra):
        image = stimuli[stimulus]
        jobs.append(dict(name=name, mode=mode, stimulus=stimulus, frame=frame, scale=scale,
                         image_size=list(image.size), transform=transform or 'none', expected=expected,
                         prompt=prompt, payload=body(mode, prompt, png_bytes(image)), **extra))

    for frame, object_name in (('A', 'button'), ('A', 'tip'), ('B', 'button'), ('B', 'tip')):
        point = FRAMES[frame]['target_xy' if object_name == 'button' else 'physical_tip_xy']
        referent = BUTTON if object_name == 'button' else TIP
        add(frame + '_' + object_name + '_point', 'point', frame + '_full1x',
            'Locate ' + referent + '. Return only its normalized point, or null if uncertain.',
            dict(point=point, point_box=point_boxes[frame + '_' + object_name]), frame=frame)
    add('A_relation', 'relation', 'A_full1x', natural + compare, relation(t, p))
    reverse = (f'TARGET: {TIP}. REFERENCE: {BUTTON}. ' + compare)
    add('A_reverse_relation', 'relation', 'A_full1x', reverse, relation(p, t))
    combined = natural + ('Return target_point and reference_point as normalized coordinates, '
                          'and vertical and horizontal relations describing TARGET relative to REFERENCE.')
    for frame in ('A', 'B'):
        target, reference = FRAMES[frame]['target_xy'], FRAMES[frame]['physical_tip_xy']
        add(frame + '_combined_points_relation', 'combined', frame + '_full1x', combined,
            dict(target_point=target, reference_point=reference,
                 target_point_box=point_boxes[frame + '_button'], reference_point_box=point_boxes[frame + '_tip'],
                 **relation(target, reference)), frame=frame)
    for stimulus, scale, transform in (
            ('A_full2x', 2, dict(scale=2, crop=None)),
            ('A_crop1x', 1, dict(scale=1, crop=roi)),
            ('A_crop2x', 2, dict(scale=2, crop=roi)),
            ('A_gray_outside_ROI', 1, dict(outside_ROI='RGB128', ROI=roi, original_ROI_pixels_and_positions=True))):
        add(stimulus + '_relation', 'relation', stimulus, natural + compare, relation(t, p), scale=scale, transform=transform)
    add('large_photo_ocr', 'ocr', 'large_photo', ocr_prompt, dict(magenta_label='T', cyan_label='P'))
    for label, point in (('T', t), ('P', p)):
        add('large_photo_' + label + '_point', 'point', 'large_photo',
            f'Locate the center of the colored dot labelled {label}. Use the dot center, not the letter center. '
            'Return only its normalized point, or null if uncertain.', dict(point=point))
    add('large_photo_relation', 'relation', 'large_photo', marked + compare, relation(t, p))
    marked_reverse = ('TARGET: center of the colored dot labelled P. REFERENCE: center of the colored dot labelled T. '
                      'Use the dot centers, not the letter centers; compare only these annotated points. ')
    add('large_photo_reverse', 'relation', 'large_photo', marked_reverse + compare, relation(p, t))
    add('swapped_photo_relation', 'relation', 'swapped_photo', marked + compare, relation(p, t),
        transform='only_letters_T_and_P_exchanged; colors_and_points_unchanged')
    add('blank_large_relation', 'relation', 'blank_large', marked + compare, relation(t, p),
        transform='same_markers_and_letters_on_RGB128_background')
    jobs.append(dict(name='numeric_from_A_predicted_points', mode='relation', frame='A',
                     stimulus=None, expected={}, payload=None,
                     dependencies=['A_button_point', 'A_tip_point'],
                     adaptive_rule='Use the exact first-response normalized points from the two dependencies; skip without a request if either is null or invalid.',
                     prompt_template='TARGET T={A_button_point}; REFERENCE P={A_tip_point}; coordinates x right, y down. ' + compare,
                     transform='text_only_declared_dependency'))
    add('swapped_photo_ocr', 'ocr', 'swapped_photo', ocr_prompt, dict(magenta_label='P', cyan_label='T'))
    for label, point in (('T', p), ('P', t)):
        add('swapped_photo_' + label + '_point', 'point', 'swapped_photo',
            f'Locate the center of the colored dot labelled {label}. Use the dot center, not the letter center. '
            'Return only its normalized point, or null if uncertain.', dict(point=point))
    jobs.append(dict(name='prior_small_TP_ocr', mode='ocr', stimulus='prior_small_TP', frame='A',
                     image_size=list(a.size), scale=1, transform='exact_previous_stimulus_bytes',
                     source=str(original_marked_path), sha256=sha(original_marked_bytes),
                     expected=dict(magenta_label='T', cyan_label='P'), prompt=ocr_prompt,
                     payload=body('ocr', ocr_prompt, original_marked_bytes, mime='image/jpeg')))
    add('A_translated_ROI_relation', 'relation', 'A_translated_ROI', natural + compare, relation(t, p),
        transform=dict(source_ROI=roi, destination_left_top=[10, roi[1]],
                       delta=[10 - roi[0], 0], canvas_size=list(a.size), original_ROI_pixels_preserved=True))
    assert len(jobs) == 25
    atomic_json(output / 'stimulus_manifest.json', dict(
        font=dict(path=str(font_path), sha256=sha(font_path.read_bytes()), size_px=28),
        marker_radius_px=8, label_offset_px=[13, -17],
        images={name: dict(size=list(image.size), sha256=sha(png_bytes(image))) for name, image in stimuli.items()},
        point_acceptance_boxes=point_boxes,
        normalization='x = source_x / (width - 1) * 1000; y = source_y / (height - 1) * 1000',
        small_label_source=dict(path=str(original_marked_path), sha256=sha(original_marked_bytes))))
    return jobs


def score(job, parsed):
    result = {}
    wanted = job.get('expected', {})
    if 'vertical' in wanted and 'vertical' in parsed:
        for axis in ('vertical', 'horizontal'):
            result[axis + '_correct'] = parsed[axis] == wanted[axis]
        result['relation_correct'] = all(result[a + '_correct'] for a in ('vertical', 'horizontal'))
    if 'magenta_label' in wanted:
        for color in ('magenta', 'cyan'):
            result[color + '_label_correct'] = parsed[color + '_label'] == wanted[color + '_label']
        result['labels_correct'] = result['magenta_label_correct'] and result['cyan_label_correct']
    for key in ('point', 'target_point', 'reference_point'):
        if key not in parsed or key not in wanted:
            continue
        if parsed[key] is None:
            result[key + '_error_px'] = None
            continue
        width, height = job['image_size']
        xy = [parsed[key]['x'] / 1000 * (width - 1), parsed[key]['y'] / 1000 * (height - 1)]
        result[key + '_pixel_xy'] = xy
        result[key + '_signed_error_xy_px'] = [xy[i] - wanted[key][i] for i in (0, 1)]
        result[key + '_error_px'] = math.dist(xy, wanted[key])
        # Equivalent-source error makes upsampling and crop conditions comparable.
        result[key + '_source_error_px'] = result[key + '_error_px'] / job.get('scale', 1)
        if key + '_box' in wanted:
            left, top, right, bottom = wanted[key + '_box']
            result[key + '_in_assistant_annotation_box'] = left <= xy[0] <= right and top <= xy[1] <= bottom
    if job['mode'] == 'combined':
        tp, rp = parsed['target_point'], parsed['reference_point']
        if tp is not None and rp is not None:
            point_relation = relation([tp['x'], tp['y']], [rp['x'], rp['y']])
            result['relation_from_own_points'] = point_relation
            result['relation_agrees_with_own_points'] = all(parsed[a] == point_relation[a] for a in RELATION)
        else:
            result['relation_from_own_points'] = None
            result['relation_agrees_with_own_points'] = None
    return result


def prepare(output):
    sources, pictures = load_sources()
    (output / 'source_images').mkdir()
    for name, meta in sources.items():
        (output / 'source_images' / (name + '.jpg')).write_bytes(Path(meta['source']).read_bytes())
    jobs = build_jobs(output, pictures)
    if not jobs or len({j['name'] for j in jobs}) != len(jobs):
        raise ValueError('missing_or_duplicate_conditions')
    plan = dict(sources=sources, conditions=[{k: v for k, v in job.items() if k != 'payload'} for job in jobs],
                predeclared_before_requests=True, robot_connected=False, robot_commands_sent=0,
                design_limits=[
                    'Selected saved frames and diagnostic stimuli; no population accuracy estimate.',
                    'Manual source coordinates are approximate assistant annotations, not calibrated physical contact.',
                    'Each first response is retained; no correction, best-response selection or silent retries.',
                    'No robot client is constructed and no command is executed.',
                    'Correct coordinate arithmetic does not prove visual identification or 3D understanding.',
                    'PNG derivatives avoid additional JPEG compression but cannot restore lost source information.',
                    'Synthetic marker and label tasks are distinct from natural-object tasks.',
                ])
    atomic_json(output / 'plan.json', plan)
    for job in jobs:
        folder = output / 'cases' / job['name']
        folder.mkdir(parents=True)
        if job['payload'] is not None:
            archive_request(job['payload'], folder, output)
        else:
            atomic_json(folder / 'declared_dependency.json', {k: v for k, v in job.items() if k != 'payload'})
    (output / 'script_used.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(output / 'source_hashes.json', dict(script_sha256=sha(Path(__file__).read_bytes()),
        image_sha256={k: v['sha256'] for k, v in sources.items()}))
    return plan, jobs


def load_prepared(output):
    plan = json.loads((output / 'plan.json').read_text())
    hashes = json.loads((output / 'source_hashes.json').read_text())
    if sha(Path(__file__).read_bytes()) != hashes['script_sha256']:
        raise ValueError('runner_changed_since_preparation')
    if (output / 'summary.json').exists() or list((output / 'cases').glob('*/response.json')):
        raise ValueError('refuse_to_overwrite_prior_run')
    jobs = []
    for condition in plan['conditions']:
        job = copy.deepcopy(condition)
        folder = output / 'cases' / job['name']
        path = folder / 'request.readable.json'
        payload = None
        if path.exists():
            payload = json.loads(path.read_text())
            refs = json.loads((folder / 'request_images.json').read_text())['images']
            index = 0
            for message in payload['messages']:
                if isinstance(message['content'], list):
                    for part in message['content']:
                        if part.get('type') == 'image_url':
                            raw = (folder / part['image_url']['url']).read_bytes()
                            if sha(raw) != refs[index]['sha256']:
                                raise ValueError('prepared_image_hash_changed')
                            part['image_url']['url'] = refs[index]['prefix'] + ',' + base64.b64encode(raw).decode()
                            index += 1
        job['payload'] = payload
        jobs.append(job)
    return plan, jobs


def resolve_dependency(job, results):
    points = []
    for name in job['dependencies']:
        row = next(r for r in results if r['name'] == name)
        point = row.get('parsed', {}).get('point')
        if not row.get('format_valid') or point is None:
            return None
        points.append(point)
    target, reference = points
    prompt = ('Use only these numeric coordinates, no image or object identification: '
              'TARGET T=' + json.dumps(target) + '; REFERENCE P=' + json.dumps(reference) +
              '; both use x increasing right and y increasing down. Compare TARGET relative to REFERENCE. '
              'Output only vertical and horizontal relations.')
    resolved = copy.deepcopy(job)
    resolved['dependency_outputs'] = dict(zip(job['dependencies'], points))
    resolved['prompt'] = prompt
    resolved['expected'] = relation([target['x'], target['y']], [reference['x'], reference['y']])
    resolved['payload'] = body('relation', prompt)
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/report/20261102_spatial_binding_followup/probes')
    parser.add_argument('--model-base')
    parser.add_argument('--thor-host', default='Thor')
    parser.add_argument('--thor-port', type=int, default=18024)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--execute-prepared', action='store_true', help='Run reviewed prepared requests; refuses an existing run or changed script.')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare_only and args.execute_prepared:
        parser.error('choose prepare-only or execute-prepared')
    if args.execute_prepared:
        plan, jobs = load_prepared(output)
    else:
        output.mkdir(parents=True, exist_ok=False)
        plan, jobs = prepare(output)
    if args.prepare_only:
        print('Prepared ' + str(output), flush=True)
        return
    base = args.model_base or f'http://127.0.0.1:{args.thor_port}/v1'
    report = dict(robot_connected=False, robot_commands_sent=0, model_only=True,
                  thinking=False, max_tokens=384, results=[], status='running')
    atomic_json(output / 'summary.json', report)
    try:
        with tunnel(None if args.model_base else args.thor_host, args.thor_port):
            atomic_json(output / 'models.json', request(base, '/models', timeout=5))
            for index, job in enumerate(jobs):
                folder = output / 'cases' / job['name']
                if job['payload'] is None:
                    resolved = resolve_dependency(job, report['results'])
                    if resolved is None:
                        row = {k: v for k, v in job.items() if k != 'payload'}
                        row.update(skipped=True, request_sent=False, format_valid=False, elapsed_s=0,
                                   skip_reason='dependency_point_null_or_invalid; no invented coordinate')
                        atomic_json(folder / 'result.json', row)
                        report['results'].append(row)
                        atomic_json(output / 'summary.json', report)
                        print(f'{index + 1}/{len(jobs)} ' + job['name'] + ' SKIPPED', flush=True)
                        continue
                    job = resolved
                    archive_request(job['payload'], folder, output)
                row = {k: v for k, v in job.items() if k != 'payload'}
                row['request_sent'] = True
                started = time.monotonic()
                print(f'{index + 1}/{len(jobs)} ' + job['name'], flush=True)
                try:
                    answer = request(base, '/chat/completions', job['payload'], timeout=120)
                    row['elapsed_s'] = time.monotonic() - started
                    atomic_json(folder / 'response.json', answer)
                    row['usage'] = answer.get('usage', {})
                    row['finish_reason'] = answer.get('choices', [{}])[0].get('finish_reason')
                    parsed = parse(answer, job['mode'])
                    row['parsed'] = parsed
                    row['format_valid'] = True
                    row.update(score(job, parsed))
                except Exception as exc:
                    row.setdefault('elapsed_s', time.monotonic() - started)
                    row['format_valid'] = False
                    row['error'] = dict(type=type(exc).__name__, message=str(exc))
                atomic_json(folder / 'result.json', row)
                report['results'].append(row)
                atomic_json(output / 'summary.json', report)
                print(json.dumps({k: row[k] for k in ('name', 'format_valid', 'parsed', 'relation_correct', 'elapsed_s', 'error') if k in row},
                                 ensure_ascii=False), flush=True)
            report['status'] = 'completed'
    finally:
        report['totals'] = dict(requests=sum(r.get('request_sent', False) for r in report['results']),
            skipped=sum(r.get('skipped', False) for r in report['results']),
            prompt_tokens=sum(r.get('usage', {}).get('prompt_tokens', 0) for r in report['results']),
            completion_tokens=sum(r.get('usage', {}).get('completion_tokens', 0) for r in report['results']),
            elapsed_s=sum(r['elapsed_s'] for r in report['results']),
            format_valid=sum(r['format_valid'] for r in report['results']))
        atomic_json(output / 'summary.json', report)
        print('Report: ' + str(output / 'summary.json'), flush=True)


if __name__ == '__main__':
    main()
