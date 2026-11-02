#!/usr/bin/env python3
"""Post-result follow-up of saved-image binding probes; model requests only."""
import argparse
import base64
import copy
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont

import diagnose_agent_spatial_binding as probe


def restore_request(folder):
    payload = json.loads((folder / 'request.readable.json').read_text())
    refs = json.loads((folder / 'request_images.json').read_text())['images']
    index = 0
    for message in payload['messages']:
        if isinstance(message['content'], list):
            for part in message['content']:
                if part.get('type') == 'image_url':
                    raw = (folder / part['image_url']['url']).read_bytes()
                    if probe.sha(raw) != refs[index]['sha256']:
                        raise ValueError('request_image_hash_changed')
                    part['image_url']['url'] = refs[index]['prefix'] + ',' + base64.b64encode(raw).decode()
                    index += 1
    return payload


def prepare(output, original):
    jobs = []
    old_plan = json.loads((original / 'plan.json').read_text())
    conditions = {row['name']: row for row in old_plan['conditions']}
    for name, source in (
            ('swapped_photo_relation_repeat1', 'swapped_photo_relation'),
            ('swapped_photo_relation_repeat2', 'swapped_photo_relation'),
            ('large_photo_relation_repeat', 'large_photo_relation'),
            ('A_relation_repeat', 'A_relation')):
        job = copy.deepcopy(conditions[source])
        job.update(name=name, repeat_of=str(original / 'cases' / source),
                   payload=restore_request(original / 'cases' / source))
        jobs.append(job)

    swapped = (original / 'figures/swapped_photo.png').read_bytes()
    t, p = probe.FRAMES['A']['physical_tip_xy'], probe.FRAMES['A']['target_xy']
    marked = ('TARGET T: center of the colored dot labelled T. REFERENCE P: center of the colored dot labelled P. '
              'Use the dot centers, not the letter centers; compare only these annotated points. ')
    combined_prompt = marked + ('Return target_point and reference_point as normalized coordinates, '
                                'and vertical and horizontal relations describing TARGET relative to REFERENCE.')
    jobs.append(dict(name='swapped_photo_combined_points_relation', mode='combined', frame='A',
                     image_size=[640, 360], scale=1, stimulus='existing_swapped_photo',
                     expected=dict(target_point=t, reference_point=p, **probe.relation(t, p)),
                     prompt=combined_prompt, payload=probe.body('combined', combined_prompt, swapped)))
    compare = 'Compare TARGET relative to REFERENCE in the image plane. Output only vertical and horizontal relations.'
    color_prompt = ('TARGET: center of the CYAN dot. REFERENCE: center of the MAGENTA dot. '
                    'Compare only these colored points. ' + compare)
    jobs.append(dict(name='swapped_photo_color_relation', mode='relation', frame='A',
                     image_size=[640, 360], scale=1, stimulus='existing_swapped_photo',
                     expected=probe.relation(t, p), prompt=color_prompt,
                     payload=probe.body('relation', color_prompt, swapped)))

    blank = Image.new('RGB', (640, 360), (128, 128, 128))
    draw = ImageDraw.Draw(blank)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    font = ImageFont.truetype(str(font_path), 28)
    for xy, color, label in ((p, '#ff00ff', 'P'), (t, '#00ffff', 'T')):
        x, y = xy
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline='black', width=1)
        draw.text((x + 13, y - 17), label, font=font, fill=color, stroke_width=2, stroke_fill='black')
    (output / 'figures').mkdir(exist_ok=True)
    (output / 'figures/blank_swapped.png').write_bytes(probe.png_bytes(blank))
    prompt = conditions['swapped_photo_relation']['prompt']
    jobs.append(dict(name='blank_swapped_relation', mode='relation', frame='A',
                     image_size=[640, 360], scale=1, stimulus='blank_swapped',
                     expected=probe.relation(t, p), prompt=prompt,
                     payload=probe.body('relation', prompt, probe.png_bytes(blank))))

    points = {}
    for label in ('T', 'P'):
        path = original / 'cases' / ('swapped_photo_' + label + '_point') / 'result.json'
        row = json.loads(path.read_text())
        point = row.get('parsed', {}).get('point')
        if not row['format_valid'] or point is None:
            raise ValueError('prior_numeric_point_unavailable')
        points[label] = point
    numeric_prompt = ('Use only these numeric coordinates, no image or object identification: '
        'TARGET T=' + json.dumps(points['T']) + '; REFERENCE P=' + json.dumps(points['P']) +
        '; both use x increasing right and y increasing down. Compare TARGET relative to REFERENCE. '
        'Output only vertical and horizontal relations.')
    jobs.append(dict(name='numeric_from_swapped_predicted_points', mode='relation', frame='A',
                     stimulus=None, dependencies=dict(source_run=str(original), prior_predictions=points),
                     expected=probe.relation([points['T']['x'], points['T']['y']], [points['P']['x'], points['P']['y']]),
                     prompt=numeric_prompt, payload=probe.body('relation', numeric_prompt)))
    original_picture = Image.open(original / 'figures/A_full1x.png').convert('RGB')
    for name, box, interpretation in (
            ('A_mask_other_finger_relation', [0, 250, 250, 360], 'mask_left_other_finger'),
            ('A_mask_background_control_relation', [0, 0, 250, 110], 'equal_area_background_mask_keep_both_fingers')):
        picture = original_picture.copy()
        picture.paste((128, 128, 128), tuple(box))
        (output / 'figures' / (name + '.png')).write_bytes(probe.png_bytes(picture))
        payload = restore_request(original / 'cases/A_relation')
        for message in payload['messages']:
            if isinstance(message['content'], list):
                for part in message['content']:
                    if part.get('type') == 'image_url':
                        part['image_url']['url'] = 'data:image/png;base64,' + base64.b64encode(probe.png_bytes(picture)).decode()
        jobs.append(dict(name=name, mode='relation', frame='A', image_size=[640, 360], scale=1,
                         stimulus=name, expected=probe.relation(probe.FRAMES['A']['target_xy'], probe.FRAMES['A']['physical_tip_xy']),
                         prompt=conditions['A_relation']['prompt'], payload=payload,
                         mask_box_exclusive_right_bottom=box, mask_area_px=250 * 110,
                         rationale=interpretation, target_and_arrow_tip_outside_mask=True))
    assert len(jobs) == 10
    plan = dict(post_initial_result_supplement=True, prior_run=str(original),
                rationale='After observing the initial label-swap relation error, test exact repetitions, joint localization, color referents, blank background and arithmetic on the actual predicted points.',
                predeclared_before_supplement_requests=True, robot_connected=False, robot_commands_sent=0,
                conditions=[{k: v for k, v in job.items() if k != 'payload'} for job in jobs],
                limits=['Follow-up chosen after initial results; not an independent preregistered benchmark.',
                        'Repeated exact requests are not independent scenes or a population reliability estimate.',
                        'No correction or best-response selection; all first responses retained.'])
    probe.atomic_json(output / 'plan.json', plan)
    for job in jobs:
        folder = output / 'cases' / job['name']
        folder.mkdir(parents=True, exist_ok=True)
        probe.archive_request(job['payload'], folder, output)
    (output / 'script_used.py').write_bytes(Path(__file__).read_bytes())
    probe.atomic_json(output / 'source_hashes.json', dict(
        script_sha256=probe.sha(Path(__file__).read_bytes()),
        imported_probe_script_sha256=probe.sha(Path(probe.__file__).read_bytes()),
        prior_summary_sha256=probe.sha((original / 'summary.json').read_bytes()),
        swapped_image_sha256=probe.sha(swapped),
        font_sha256=probe.sha(font_path.read_bytes())))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = probe.ROOT / 'docs/report/20261102_spatial_binding_followup'
    parser.add_argument('--original', type=Path, default=base / 'probes')
    parser.add_argument('--output', type=Path, default=base / 'supplement')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--execute-prepared', action='store_true')
    parser.add_argument('--extend-prepared', action='store_true', help='Extend the unexecuted initial eight-condition plan to ten; preserves the original plan.')
    parser.add_argument('--thor-host', default='Thor')
    parser.add_argument('--thor-port', type=int, default=18024)
    parser.add_argument('--model-base')
    args = parser.parse_args()
    if (args.prepare_only or args.extend_prepared) and args.execute_prepared:
        parser.error('choose prepare-only or execute-prepared')
    output, original = args.output.resolve(), args.original.resolve()
    if args.execute_prepared:
        if (output / 'summary.json').exists():
            raise ValueError('refuse_to_overwrite_prior_supplement')
        hashes = json.loads((output / 'source_hashes.json').read_text())
        if (probe.sha(Path(__file__).read_bytes()) != hashes['script_sha256'] or
                probe.sha(Path(probe.__file__).read_bytes()) != hashes['imported_probe_script_sha256']):
            raise ValueError('prepared_scripts_changed')
        jobs = json.loads((output / 'plan.json').read_text())['conditions']
        for job in jobs:
            job['payload'] = restore_request(output / 'cases' / job['name'])
    else:
        if args.extend_prepared:
            previous = (output / 'plan.json').read_bytes()
            if ((output / 'summary.json').exists() or list((output / 'cases').glob('*/response.json')) or
                    len(json.loads(previous)['conditions']) != 8):
                raise ValueError('extension_requires_unexecuted_eight_condition_plan')
            (output / 'initial_8_plan.json').write_bytes(previous)
            probe.atomic_json(output / 'plan_extension.json', dict(
                previous_plan_sha256=probe.sha(previous), extension_before_any_supplement_request=True,
                reason='Review of the initial overlays highlighted that ROI masking also removes the other finger; add equal-area other-finger/background masks before issuing any supplement request.',
                added_conditions=['A_mask_other_finger_relation', 'A_mask_background_control_relation'],
                unix_time=time.time()))
        else:
            output.mkdir(parents=True, exist_ok=False)
        jobs = prepare(output, original)
    if args.prepare_only:
        print('Prepared ' + str(output), flush=True)
        return
    base_url = args.model_base or f'http://127.0.0.1:{args.thor_port}/v1'
    report = dict(post_initial_result_supplement=True, robot_connected=False, robot_commands_sent=0,
                  model_only=True, thinking=False, max_tokens=384, results=[], status='running')
    probe.atomic_json(output / 'summary.json', report)
    try:
        with probe.tunnel(None if args.model_base else args.thor_host, args.thor_port):
            probe.atomic_json(output / 'models.json', probe.request(base_url, '/models', timeout=5))
            for index, job in enumerate(jobs):
                folder = output / 'cases' / job['name']
                row = {k: v for k, v in job.items() if k != 'payload'}
                print(f'{index + 1}/{len(jobs)} ' + job['name'], flush=True)
                started = time.monotonic()
                try:
                    answer = probe.request(base_url, '/chat/completions', job['payload'], timeout=120)
                    row['elapsed_s'] = time.monotonic() - started
                    probe.atomic_json(folder / 'response.json', answer)
                    row['usage'] = answer.get('usage', {})
                    row['finish_reason'] = answer.get('choices', [{}])[0].get('finish_reason')
                    row['parsed'] = probe.parse(answer, job['mode'])
                    row['format_valid'] = True
                    row.update(probe.score(job, row['parsed']))
                except Exception as exc:
                    row.setdefault('elapsed_s', time.monotonic() - started)
                    row['format_valid'] = False
                    row['error'] = dict(type=type(exc).__name__, message=str(exc))
                probe.atomic_json(folder / 'result.json', row)
                report['results'].append(row)
                probe.atomic_json(output / 'summary.json', report)
                print(json.dumps({k: row[k] for k in ('name', 'format_valid', 'parsed', 'elapsed_s', 'error') if k in row}), flush=True)
            report['status'] = 'completed'
    finally:
        report['totals'] = dict(requests=len(report['results']),
            prompt_tokens=sum(r.get('usage', {}).get('prompt_tokens', 0) for r in report['results']),
            completion_tokens=sum(r.get('usage', {}).get('completion_tokens', 0) for r in report['results']),
            elapsed_s=sum(r['elapsed_s'] for r in report['results']),
            format_valid=sum(r['format_valid'] for r in report['results']))
        probe.atomic_json(output / 'summary.json', report)


if __name__ == '__main__':
    main()
