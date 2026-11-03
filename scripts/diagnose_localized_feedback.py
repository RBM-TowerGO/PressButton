#!/usr/bin/env python3
"""Controlled saved-request probes; only the model HTTP service is accessed.

No robot client or executor is imported. Existing run files are read-only.
Coordinate correction is an explicit diagnostic hypothesis (legacy 0..1000),
not a claim that the model's locations are visual ground truth.
"""
import argparse
import ast
import base64
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pressb.agent_loop.common import atomic_json, request

LOCATION = 'CURRENT_LOCALIZATION_AND_RAW_OFFSETS'
CHANGE = 'RAW_OBSERVATION_CHANGE_MODEL_TO_INTERPRET'
SCALED_KEYS = {'button_box', 'arrow_box', 'contact_tip', 'returned_box_center',
               'box_size', 'offset_normalized', 'pixel_delta', 'dx', 'dy'}
CONDITIONS = ('A_archived', 'B_units_corrected', 'C_corrected_reassessment',
              'D_corrected_current_only')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def image_hashes(body):
    return [digest(base64.b64decode(part['image_url']['url'].split(',', 1)[1], validate=True))
            for message in body['messages'] if isinstance(message.get('content'), list)
            for part in message['content'] if part.get('type') == 'image_url']


def structured(message):
    content = message.get('content')
    if isinstance(content, list) and content and content[0].get('type') == 'text':
        try:
            return json.loads(content[0]['text'])
        except (ValueError, TypeError):
            pass
    return None


def scaled(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [scaled(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value / 1000
    raise ValueError('non_numeric_legacy_coordinate:' + repr(value))


def correct_tree(value):
    """Only coordinate dimensions change; commands, ratios and signs do not."""
    if isinstance(value, list):
        return [correct_tree(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in SCALED_KEYS:
            result[key] = scaled(item)
        elif key == 'arguments' and isinstance(item, str):
            # Duplicate raw locate_current_scene tool arguments are inside the
            # location message; transform their representation consistently.
            result[key] = json.dumps(correct_tree(json.loads(item)), ensure_ascii=False)
        else:
            result[key] = correct_tree(item)
    return result


def correct_body(original):
    body = copy.deepcopy(original)
    changed = []
    for index, message in enumerate(body['messages']):
        data = structured(message)
        if isinstance(data, dict) and data.get('context_type') in (LOCATION, CHANGE):
            new = correct_tree(data)
            message['content'][0]['text'] = json.dumps(new, ensure_ascii=False, separators=(',', ':'))
            changed.append(index)
    if not changed:
        raise ValueError('no_legacy_localization_context')
    return body, changed


def feedback_fact_table(corrected):
    rows, executed = [], []
    for message in corrected['messages']:
        data = structured(message)
        if isinstance(data, dict) and data.get('context_type') == LOCATION:
            wrist = data['wrist']
            rows.append(dict(observation_id=data['observation_id'],
                wrist_button_visibility=wrist['button_visibility'],
                wrist_pixel_delta=wrist['derived_relation']['pixel_delta'],
                wrist_relation=[wrist['derived_relation']['horizontal'], wrist['derived_relation']['vertical']]))
        elif message['role'] == 'tool':
            data = json.loads(message['content'])
            executed.append(dict(command_id=data.get('command_id'), command=data.get('command'),
                                 status=data.get('status'), observation_id=data.get('observation_id')))
    streak = []
    for record in reversed(executed):
        command = record.get('command') or {}
        xyz = command.get('xyz_mm')
        if (record['status'] != 'completed' or command.get('type') != 'move' or
                not xyz or xyz[0] <= 0 or xyz[1:] != [0, 0]):
            break
        streak.append(record)
    return dict(context_type='DIAGNOSTIC_FEEDBACK_REASSESSMENT',
        observation_id=rows[-1]['observation_id'],
        source='program_recorded_commands_and_model_location_arithmetic_not_visual_truth',
        consecutive_forward_sends=len(streak), forward_commanded_mm=sum(r['command']['xyz_mm'][0] for r in streak),
        recent_locations=rows[-6:],
        interpretation=('The visible-part button rectangle remains left/above the marked physical end in the '
            'localizer estimates. This is a projection and does not identify a correct robot-axis move. '
            'Completed sends alone are not alignment or contact evidence. No accepted light-on or withdrawal '
            'evidence is recorded in these preceding action replies.'),
        instruction=('Reassess the hypothesis that missing forward distance is the only problem. '
            'Use both current images and these actual effects. Select one next action. If you continue '
            'forward, explain concrete NEW evidence for that hypothesis; otherwise address uncertainty or '
            'alignment. Do not invent an image-to-chassis transform. All original tools remain available.'))


def make_variants(original):
    corrected, changed = correct_body(original)
    feedback = feedback_fact_table(corrected)
    reassessed = copy.deepcopy(corrected)
    reassessed['messages'].append(dict(role='user', content=json.dumps(feedback, ensure_ascii=False)))
    current_only = copy.deepcopy(corrected)
    indices = [i for i,m in enumerate(corrected['messages'])
               if isinstance(structured(m), dict) and structured(m).get('context_type') == 'LIVE_OBSERVATION']
    current_only['messages'] = copy.deepcopy(corrected['messages'][:2] + corrected['messages'][indices[-1]:])
    for body in (corrected, reassessed):
        assert image_hashes(body) == image_hashes(original)
    assert image_hashes(current_only)[-2:] == image_hashes(original)[-2:]
    return dict(zip(CONDITIONS, (copy.deepcopy(original), corrected, reassessed, current_only))), changed, feedback


def archive(body, directory, output):
    readable = copy.deepcopy(body)
    hashes = []
    for message in readable['messages']:
        if not isinstance(message.get('content'), list):
            continue
        for part in message['content']:
            if part.get('type') != 'image_url':
                continue
            prefix, encoded = part['image_url']['url'].split(',', 1)
            if prefix != 'data:image/jpeg;base64':
                raise ValueError('expected_original_jpeg')
            raw = base64.b64decode(encoded, validate=True)
            sha = digest(raw)
            target = output/'images'/f'{sha}.jpg'
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(raw)
            part['image_url']['url'] = '../../images/' + target.name
            hashes.append(sha)
    atomic_json(directory/'request.readable.json', readable)
    restored = copy.deepcopy(readable)
    for message in restored['messages']:
        if isinstance(message.get('content'), list):
            for part in message['content']:
                if part.get('type') == 'image_url':
                    raw = (directory/part['image_url']['url']).read_bytes()
                    part['image_url']['url'] = 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()
    assert restored == body
    atomic_json(directory/'request_evidence.json', dict(image_sha256=hashes,
        exact_request_sha256=digest(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()),
        archive_roundtrip=True))


@contextmanager
def model_tunnel(host, port, base):
    if base:
        yield base
        return
    process = subprocess.Popen(['ssh', '-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
        '-L', f'127.0.0.1:{port}:127.0.0.1:8000', host])
    try:
        endpoint = f'http://127.0.0.1:{port}/v1'
        for _ in range(75):
            if process.poll() is not None:
                raise RuntimeError('model_tunnel_failed')
            try:
                request(endpoint, '/models', timeout=1)
                break
            except OSError:
                time.sleep(.2)
        else:
            raise TimeoutError('model_tunnel_timeout')
        yield endpoint
    finally:
        process.terminate()
        try:
            process.wait(3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run(args):
    source = Path(args.saved_request).resolve()
    raw = source.read_bytes()
    original = json.loads(raw)
    if original.get('temperature') != 0 or original.get('chat_template_kwargs', {}).get('enable_thinking') is not False:
        raise ValueError('expected_temperature_zero_and_thinking_disabled')
    variants, changed, feedback = make_variants(original)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'script_used.py').write_bytes(Path(__file__).read_bytes())
    report = dict(format='pressb-localized-feedback-diagnostic-v1', model_only=True,
        robot_connected=False, robot_commands_sent=0, source_request=str(source), source_sha256=digest(raw),
        source_current_observation_id=feedback['observation_id'], source_image_count=len(image_hashes(original)),
        intervention=dict(legacy_coordinate_units_assumed='normalized_1000', canonical_units='normalized_01',
            all_location_and_progress_messages_changed=changed,
            numerical_fields_scaled_by=.001, ratios_signs_reasons_images_preserved=True,
            raw_model_argument_copies_also_transformed_for_consistent_model_input=True),
        caveats=['No new action is executed. Future images are never attributed to these proposals.',
            'Coordinate normalization does not validate target/endpoint identity or physical contact.',
            'Each condition changes model input; this is a small behavioral diagnostic, not a neural mechanism test.',
            'Deterministic repeats and sequential calls share service cache; timings are descriptive.',
            'D removes action history and old live images together; it cannot isolate these two effects.'], results=[])
    atomic_json(output/'feedback_intervention.json', feedback)
    jobs = []
    for repeat in range(args.repeats):
        order = CONDITIONS if repeat % 2 == 0 else tuple(reversed(CONDITIONS))
        for condition in order:
            jobs.append((f'{condition}_{repeat+1}', variants[condition],
                         dict(kind='saved_visual_action', condition=condition, repeat=repeat+1)))
    for direction, expected in [('right',[0,-10,0]), ('left',[0,10,0]), ('up',[0,0,10]), ('down',[0,0,-10])]:
        body = copy.deepcopy(original)
        body['messages'] = [copy.deepcopy(original['messages'][0]), dict(role='user', content=(
            'Read-only symbolic coordinate test. The desired movement is already determined: '
            f'move the ROBOT HAND exactly 10 mm toward the ROBOT\'S {direction}. '
            'No visual interpretation or alignment planning is needed. Encode only that axis with '
            'propose_move, using the coordinate contract already given. Nothing will be executed.'))]
        jobs.append((f'coordinate_{direction}', body,
                     dict(kind='symbolic_coordinate_translation', expected_xyz_mm=expected)))
    atomic_json(output/'report.json', report)
    with model_tunnel(args.thor_host, args.port, args.model_base) as base:
        atomic_json(output/'models.json', request(base, '/models'))
        for name, body, meta in jobs:
            directory = output/'tests'/name
            directory.mkdir(parents=True)
            archive(body, directory, output)
            started = time.monotonic()
            row = dict(name=name, **meta, image_count=len(image_hashes(body)))
            try:
                response = request(base, '/chat/completions', body, timeout=180)
                atomic_json(directory/'response.json', response)
                choice = response['choices'][0]
                message = choice['message']
                calls = [dict(name=c['function']['name'], arguments=json.loads(c['function']['arguments']))
                         for c in message.get('tool_calls', [])]
                row.update(calls=calls, usage=response.get('usage'), finish_reason=choice.get('finish_reason'),
                    single_tool_without_prose=len(calls)==1 and not (message.get('content') or '').strip())
                if meta['kind'] == 'symbolic_coordinate_translation':
                    row['correct'] = (len(calls)==1 and calls[0]['name']=='propose_move' and
                                      calls[0]['arguments'].get('xyz_mm')==meta['expected_xyz_mm'])
                if len(calls)==1:
                    xyz = calls[0]['arguments'].get('xyz_mm')
                    row['pure_forward'] = (calls[0]['name']=='propose_move' and isinstance(xyz,list)
                                           and len(xyz)==3 and xyz[0]>0 and xyz[1:]==[0,0])
            except Exception as exc:
                row['error'] = str(exc)
                atomic_json(directory/'error.json', row)
            row['elapsed_s'] = time.monotonic()-started
            report['results'].append(row)
            atomic_json(output/'report.json', report)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    report['source_unchanged'] = digest(source.read_bytes()) == report['source_sha256']
    report['completed'] = True
    atomic_json(output/'report.json', report)


def production_smoke(args):
    """Use the current production locator/reviewer on saved immutable evidence."""
    from pressb.agent_loop import model_localization as locator
    from pressb.agent_loop import localized_reassessment as review_module
    source = Path(args.saved_request).resolve()
    original = json.loads(source.read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'script_used.py').write_bytes(Path(__file__).read_bytes())
    # Read the literal prompt constant without importing any execution module.
    module_path = ROOT/'src/pressb/agent_loop/localized_conversation.py'
    tree = ast.parse(module_path.read_text())
    guidance = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id=='GUIDANCE' for target in node.targets))
    prompt_path = ROOT/'docs/astribot/planar_control/agent_system_prompt_localized.txt'
    (output/'prompt_used.txt').write_text(prompt_path.read_text()+'\n'+guidance)
    files = [module_path, prompt_path, Path(locator.__file__), Path(review_module.__file__)]
    atomic_json(output/'source_code_hashes.json', {str(p):digest(p.read_bytes()) for p in files})
    observations, locations, turns, pending, current_id = {}, {}, [], None, None
    for message in original['messages']:
        data = structured(message)
        if isinstance(data, dict) and data.get('context_type')=='LIVE_OBSERVATION':
            current_id = data['observation_id']
            observations[current_id] = copy.deepcopy(message)
        elif isinstance(data, dict) and data.get('context_type')==LOCATION:
            sizes = {view:data[view]['image_size'] for view in locator.VIEWS}
            locations[data['observation_id']] = locator.normalize(data['raw_locations'], data['observation_id'], sizes,
                source_units='normalized_1000', raw_output=data.get('raw_output'))
        elif message['role']=='assistant':
            if pending is not None:
                raise ValueError('unpaired_recorded_tool_call')
            pending = message['tool_calls'][0]['id']
        elif message['role']=='tool':
            if message['tool_call_id'] != pending:
                raise ValueError('recorded_tool_pair_mismatch')
            data = json.loads(message['content'])
            turns.append(dict(ledger=dict(before_observation_id=current_id,
                after_observation_id=data['observation_id'], result=data)))
            pending = None
    if pending is not None:
        raise ValueError('unresolved_recorded_tool_call')
    report = dict(format='pressb-localized-production-smoke-v1', robot_connected=False,
        robot_commands_sent=0, source_request=str(source), source_sha256=digest(source.read_bytes()),
        current_observation_id=current_id, historical_tool_pairs=len(turns), new_action_executed=False)
    atomic_json(output/'report.json', report)
    with model_tunnel(args.thor_host, args.port, args.model_base) as base:
        if args.reuse_localization:
            saved = Path(args.reuse_localization).resolve()
            current = json.loads((saved/'assessment.json').read_text())
            if current['observation_id']!=current_id or current['format']!=locator.FORMAT:
                raise ValueError('reused_localization_binding_mismatch')
            if ([current['images'][view]['sha256'] for view in locator.VIEWS] !=
                    image_hashes(dict(messages=[observations[current_id]]))):
                raise ValueError('reused_localization_image_hash_mismatch')
            for path in saved.iterdir():
                if path.is_file():
                    target = output/'localization'/path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
            report['localization_reused_from'] = str(saved)
        else:
            cancellation = threading.Event()
            timer = threading.Timer(120, cancellation.set)
            timer.start()
            try:
                body = locator.build_request('按下电梯外呼向下按钮并撤回', observations[current_id])
                current = locator.inspect(base, body, current_id, output/'localization', cancellation)
            finally:
                timer.cancel()
        locations[current_id] = current
        report['localization'] = dict(format=current['format'], coordinate_conversion=current['coordinate_conversion'],
            parse_notes=current['parse_notes'], wrist_pixel_delta=current['wrist']['derived_relation']['pixel_delta'],
            metrics=json.loads((output/'localization/metrics.json').read_text()))
        atomic_json(output/'report.json', report)
        signal = review_module.monitor(turns, locations, current_id)
        atomic_json(output/'review_trigger.json', signal)
        if not signal['review_required']:
            raise ValueError('expected_real_repeated_motion_to_trigger_review')
        baseline = signal['baseline_observation_id']
        body = review_module.build_request('按下电梯外呼向下按钮并撤回', observations[baseline],
            observations[current_id], signal, locations)
        assert len(image_hashes(body))==4
        cancellation = threading.Event()
        timer = threading.Timer(120, cancellation.set)
        timer.start()
        try:
            review = review_module.inspect(base, body, current_id, baseline, output/'hypothesis_review', cancellation)
        finally:
            timer.cancel()
        report['review'] = review
        report['review_metrics'] = json.loads((output/'hypothesis_review/metrics.json').read_text())
        atomic_json(output/'report.json', report)
        action, _ = correct_body(original)
        action['messages'][0]['content'] = prompt_path.read_text()+'\n'+guidance
        action['tools'] = review_module.extend_tools(copy.deepcopy(action['tools']), review)
        for message in action['messages']:
            data = structured(message)
            if not isinstance(data, dict):
                continue
            if data.get('context_type')==LOCATION:
                value = dict(context_type=LOCATION, **locator.model_view(locations[data['observation_id']]))
                message['content'][0]['text'] = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
            elif data.get('context_type')==CHANGE and data['observation_id']==current_id:
                previous_id = data['previous_observation_id']
                value = locator.compare(locations[previous_id], current,
                    structured(observations[previous_id])['state'], structured(observations[current_id])['state'],
                    data['actual_commands'])
                message['content'][0]['text'] = json.dumps(dict(context_type=CHANGE, **value), ensure_ascii=False)
        action['messages'].append(dict(role='user', content=[dict(type='text', text=json.dumps(dict(
            context_type='MOTION_HYPOTHESIS_REVIEW', assessment=review, trigger=signal,
            instruction='Reassess the cause, not merely the reason wording. Include reassessment with current '
                'evidence, expected_visual_change and if_no_change. Do not infer robot-axis signs from image signs.'),
            ensure_ascii=False))]))
        assert image_hashes(action)==image_hashes(original)
        directory = output/'tests/action'
        directory.mkdir(parents=True)
        archive(action, directory, output)
        started = time.monotonic()
        response = request(base, '/chat/completions', action, timeout=120)
        atomic_json(directory/'response.json', response)
        choice = response['choices'][0]
        calls = choice['message'].get('tool_calls') or []
        if choice.get('finish_reason')!='tool_calls' or len(calls)!=1:
            raise ValueError('expected_one_action_tool')
        function = calls[0]['function']
        arguments = json.loads(function['arguments'])
        if function['name'] not in ('report_done', 'report_unable'):
            review_module.validate_reassessment(arguments.get('reassessment'), review)
        report['action'] = dict(name=function['name'], arguments=arguments, usage=response.get('usage'),
            elapsed_s=time.monotonic()-started, image_count=len(image_hashes(action)),
            reassessment_valid=True, tool_call_id=calls[0]['id'],
            tool_result=None, execution_status='not_executed_model_only')
    report['completed'] = True
    report['source_unchanged'] = digest(source.read_bytes())==report['source_sha256']
    atomic_json(output/'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-request', default=str(ROOT/'outputs/agent_web/20261102_162741_4885e81e/agent/rounds/0014/request.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--model-base')
    parser.add_argument('--thor-host', default='Thor')
    parser.add_argument('--port', type=int, default=18113)
    parser.add_argument('--production-smoke', action='store_true')
    parser.add_argument('--reuse-localization', help='production smoke only: reuse an explicitly bound saved localization directory')
    args = parser.parse_args()
    (production_smoke if args.production_smoke else run)(args)
