#!/usr/bin/env python3
"""Read-only ablations of saved requests; connects only to the model service.

Labels must be supplied from inspection of each saved observation. They describe
image facts, never an intended robot direction or displacement. No robot client
or executor is instantiated. Requests are archived with lossless image references.
"""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pressb.agent_loop.common import atomic_json, request
from s1b_agent import tunnel

COORDINATES = (
    'Coordinate contract for propose_move.xyz_mm: incremental [X,Y,Z] in '
    'millimeters in the robot chassis frame. Robot hand FORWARD = +X, BACKWARD = -X, '
    'LEFT = +Y, RIGHT = -Y, UP = +Z, DOWN = -Z. These directions describe the '
    'robot hand, not object motion in an image. Image left/right alone does not '
    'define a chassis direction without relating the camera to the robot.'
)
FACT_FIELDS = ('head_button_visible', 'head_button_relative_to_gripper',
               'wrist_button_visible', 'alignment')
CONDITIONS = ('A_original', 'B_coordinates', 'C_visual_facts',
              'D_facts_and_coordinates', 'E_tool_description')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive_request(body, directory, output):
    readable = copy.deepcopy(body)
    images = output / 'images'
    images.mkdir(exist_ok=True)
    for msg in readable['messages']:
        if not isinstance(msg.get('content'), list):
            continue
        for item in msg['content']:
            if item.get('type') != 'image_url':
                continue
            url = item['image_url']['url']
            prefix, encoded = url.split(',', 1)
            if prefix != 'data:image/jpeg;base64':
                raise ValueError('expected_saved_jpeg_data_url')
            raw = base64.b64decode(encoded, validate=True)
            path = images / (digest(raw) + '.jpg')
            if not path.exists():
                path.write_bytes(raw)
            item['image_url']['url'] = '../../images/' + path.name
    atomic_json(directory / 'request.readable.json', readable)
    restored = copy.deepcopy(readable)
    for msg in restored['messages']:
        if isinstance(msg.get('content'), list):
            for item in msg['content']:
                if item.get('type') == 'image_url':
                    path = directory / item['image_url']['url']
                    item['image_url']['url'] = 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()
    if restored != body:
        raise ValueError('request_archive_roundtrip_mismatch')


def latest(body):
    content = body['messages'][-1]['content']
    obs = json.loads(content[0]['text'])
    if obs['context_type'] != 'LIVE_OBSERVATION':
        raise ValueError('latest_live_observation_required')
    hashes = [digest(base64.b64decode(x['image_url']['url'].split(',', 1)[1]))
              for x in content if x.get('type') == 'image_url']
    if len(hashes) != 2:
        raise ValueError('two_current_views_required')
    return obs['observation_id'], hashes


def argument_field_errors(body, calls):
    """Check field presence/exclusivity and top-level enums, not full JSON Schema."""
    schemas = {t['function']['name']: t['function']['parameters'] for t in body['tools']}
    errors = []
    for call in calls:
        schema = schemas.get(call['name'])
        if schema is None:
            errors.append('unknown_tool:' + call['name'])
            continue
        args = call['arguments']
        if not isinstance(args, dict):
            errors.append('arguments_not_object')
            continue
        missing = sorted(set(schema.get('required', [])) - set(args))
        extra = sorted(set(args) - set(schema.get('properties', {})))
        if missing:
            errors.append('missing_required_fields:' + ','.join(missing))
        if extra and schema.get('additionalProperties') is False:
            errors.append('unexpected_fields:' + ','.join(extra))
        for name, value in args.items():
            allowed = schema.get('properties', {}).get(name, {}).get('enum')
            if allowed is not None and value not in allowed:
                errors.append('invalid_enum:' + name)
    return errors


def variant(original, condition, facts, oid):
    body = copy.deepcopy(original)
    suffix = []
    if condition in ('C_visual_facts', 'D_facts_and_coordinates'):
        suffix.append(dict(role='user', content=(
            'Offline image assessment for the SAME current observation ' + oid + ': ' +
            json.dumps(facts, ensure_ascii=False) +
            '. This assessment describes the current images, not historical images.')))
    if condition in ('B_coordinates', 'D_facts_and_coordinates'):
        suffix.append(dict(role='user', content=COORDINATES))
    body['messages'].extend(suffix)
    if condition == 'E_tool_description':
        fn = next(t['function'] for t in body['tools'] if t['function']['name'] == 'propose_move')
        fn['parameters']['properties']['xyz_mm']['description'] = COORDINATES
    # Verify that ablations change only the explicitly selected interface fields.
    restored = copy.deepcopy(body)
    restored['messages'] = restored['messages'][:len(original['messages'])]
    restored['tools'] = original['tools']
    if restored != original or body['messages'][:len(original['messages'])] != original['messages']:
        raise ValueError('unexpected_ablation_change')
    if condition != 'E_tool_description' and body['tools'] != original['tools']:
        raise ValueError('unexpected_tool_change')
    return body


def interface_jobs(original, prefix, oid):
    """Test output semantics separately from supplying external image facts."""
    from diagnose_agent_visual_context import PROBE, DIAGNOSTIC
    minimal = [copy.deepcopy(original['messages'][0]),
               dict(role='user', content=[copy.deepcopy(original['messages'][1]['content'][0])]),
               copy.deepcopy(original['messages'][-1])]
    jobs = []
    for name in ('original_axis', 'explicit_hand_axis', 'semantic_hand_direction'):
        body = copy.deepcopy(original)
        body['messages'] = copy.deepcopy(minimal)
        body['messages'][0]['content'] += '\n\n' + DIAGNOSTIC
        body['messages'].append(dict(role='user', content=DIAGNOSTIC))
        body['tools'] = [copy.deepcopy(PROBE)]
        params = body['tools'][0]['function']['parameters']
        if name == 'explicit_hand_axis':
            params['properties']['next_axis']['description'] = (
                'The desired ROBOT HAND displacement axis, not the current hand location '
                'relative to the target and not the target motion in the camera. '
                'Robot hand right=-Y, left=+Y, forward=+X, backward=-X, up=+Z, down=-Z.')
        elif name == 'semantic_hand_direction':
            props = params['properties']
            props.pop('next_axis')
            props['robot_hand_direction'] = dict(type='string', enum=[
                'forward', 'backward', 'left', 'right', 'up', 'down', 'observe', 'unknown'],
                description='Desired motion of the ROBOT HAND, not its current relative location and not image motion of the target.')
            params['required'] = ['robot_hand_direction' if k == 'next_axis' else k for k in params['required']]
        jobs.append((prefix + '_diagnostic_' + name, body,
                     dict(kind='diagnostic_interface', condition=name, observation_id=oid)))
    body = copy.deepcopy(original)
    fn = next(t['function'] for t in body['tools'] if t['function']['name'] == 'propose_move')
    params = fn['parameters']
    fields = dict(
        head_button_relative_to_gripper=dict(type='string', enum=['left', 'center', 'right', 'unknown'],
            description='Observed image-relative location in the FINAL LIVE head image only.'),
        wrist_button_visible=dict(type='string', enum=['yes', 'no', 'uncertain'],
            description='Observed visibility in the FINAL LIVE wrist image only.'),
        alignment=dict(type='string', enum=['aligned', 'misaligned', 'uncertain']),
        robot_hand_direction=dict(type='string', enum=['forward', 'backward', 'left', 'right', 'up', 'down', 'combined'],
            description='Intended robot hand motion, not image motion or current relative location.'))
    fields.update(params['properties'])
    params['properties'] = fields
    params['required'] = list(fields)
    jobs.append((prefix + '_single_call_structured', body,
                 dict(kind='structured_action_interface', condition='single_call_structured', observation_id=oid)))
    return jobs


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    labels_path = Path(args.labels)
    labels = json.loads(labels_path.read_text())
    atomic_json(output / 'labels.json', labels)
    report = dict(suite=args.suite, robot_connected=False, robot_commands_sent=0, sources=[], results=[],
                  caveats=['One deterministic sample per condition and observation; no cache reset.',
                           'Consecutive frames from one failed run are correlated, not independent trials.',
                           'Analyst image facts are injected only in factorial C/D, never in interface probes; this is not an end-to-end perception test.',
                           'Action proposals are unexecuted; pixel offsets do not establish millimeter distances.'])
    jobs = []
    originals = []
    for index, source in enumerate(args.saved_request):
        path = Path(source).resolve()
        original = json.loads(path.read_text())
        originals.append(original)
        oid, hashes = latest(original)
        entry = labels['observations'][oid]
        facts = {field: entry[field] for field in FACT_FIELDS}
        report['sources'].append(dict(path=str(path), sha256=digest(path.read_bytes()),
                                     observation_id=oid, current_image_hashes=hashes))
        if args.suite == 'interface':
            jobs.extend(interface_jobs(original, path.parent.name, oid))
            continue
        # Rotate order across observations rather than always running the same sequence.
        order = CONDITIONS[index:] + CONDITIONS[:index]
        for condition in order:
            body = variant(original, condition, facts, oid)
            assert latest(dict(body, messages=body['messages'][:len(original['messages'])])) == (oid, hashes)
            jobs.append((f'{path.parent.name}_{condition}', body,
                         dict(kind='visual_action', condition=condition, observation_id=oid)))
    directions = [('right', [0, -10, 0]), ('left', [0, 10, 0]),
                  ('forward', [10, 0, 0]), ('backward', [-10, 0, 0]),
                  ('up', [0, 0, 10]), ('down', [0, 0, -10])]
    for direction, expected in directions if args.suite == 'factorial' else []:
        body = copy.deepcopy(originals[0])
        body['messages'] = [copy.deepcopy(originals[0]['messages'][0]), dict(role='user', content=(
            'Read-only coordinate translation test. The desired robot-hand motion is already '
            f'determined: move the hand 10 mm toward the robot\'s {direction}. '
            'No image interpretation or task planning is needed. Encode exactly this displacement '
            'using propose_move; do not move any other axis. No command will be executed.'))]
        jobs.append((f'coordinate_{direction}', body,
                     dict(kind='coordinate_translation', direction=direction, expected_xyz_mm=expected)))
    atomic_json(output / 'report.json', report)
    with tunnel(None if args.model_base else args.thor_host, args.port):
        base = args.model_base or f'http://127.0.0.1:{args.port}/v1'
        atomic_json(output / 'models.json', request(base, '/models'))
        for name, body, meta in jobs:
            directory = output / 'tests' / name
            directory.mkdir(parents=True)
            archive_request(body, directory, output)
            started = time.monotonic()
            row = dict(name=name, **meta)
            try:
                response = request(base, '/chat/completions', body, timeout=180)
                atomic_json(directory / 'response.json', response)
                choice = response['choices'][0]
                message = choice['message']
                calls = [dict(name=call['function']['name'],
                              arguments=json.loads(call['function']['arguments']))
                         for call in message.get('tool_calls', [])]
                row.update(calls=calls, usage=response.get('usage'), finish_reason=choice['finish_reason'])
                row['argument_field_errors'] = argument_field_errors(body, calls)
                if meta['kind'] == 'coordinate_translation':
                    row['correct'] = (len(calls) == 1 and calls[0]['name'] == 'propose_move'
                                      and calls[0]['arguments'].get('xyz_mm') == meta['expected_xyz_mm'])
                row['single_tool_without_prose'] = (choice['finish_reason'] == 'tool_calls' and
                                                    len(calls) == 1 and not (message.get('content') or '').strip())
            except Exception as exc:
                row['error'] = str(exc)
                atomic_json(directory / 'error.json', row)
            row['elapsed_s'] = time.monotonic() - started
            report['results'].append(row)
            atomic_json(output / 'report.json', report)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    report['validation'] = dict(
        source_unchanged=all(digest(Path(s['path']).read_bytes()) == s['sha256'] for s in report['sources']),
        request_archives_roundtrip=True, current_image_hashes_identical_within_observation=True,
        calls=len(report['results']), errors=sum('error' in r for r in report['results']),
        calls_with_argument_field_errors=sum(bool(r.get('argument_field_errors')) for r in report['results']))
    atomic_json(output / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-request', nargs='+', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--thor-host', default='Thor')
    parser.add_argument('--port', type=int, default=18111)
    parser.add_argument('--model-base')
    parser.add_argument('--suite', choices=['factorial', 'interface'], default='factorial')
    run(parser.parse_args())
