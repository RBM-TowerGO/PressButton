#!/usr/bin/env python3
"""Model-only localization and history lesions on saved, byte-identical images."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pressb.agent_loop.common import atomic_json, request
from pressb.agent_loop.client import parse_proposal
from pressb.agent_loop.conversation import image_count, request_body
from pressb.agent_loop import perception
from diagnose_agent_action_mapping import archive_request
from s1b_agent import StopControl, tunnel


def minimal(current, views=('head', 'wrist')):
    properties = {}
    content = [dict(type='text', text=(
        'Inspect the current elevator down-call button in each supplied image independently. '
        'Report only what is visible in that image. There may be no target in a view. '
        'Image-plane left/right are not robot movement directions. Do not choose an action.'))]
    pictures = [p for p in current['content'] if p.get('type') == 'image_url']
    for view in views:
        properties[view+'_visible'] = dict(type='string', enum=['yes', 'no', 'unknown'])
        properties[view+'_relation'] = dict(type='string', enum=['left', 'center', 'right', 'not_visible', 'unknown'],
            description='Target position relative to visible gripper in THIS image, not motion direction.')
        content += [dict(type='text', text='CURRENT '+view.upper()+' camera original image'),
                    copy.deepcopy(pictures[0 if view == 'head' else 1])]
    tool = dict(type='function', function=dict(name='inspect_current',
        description='Read-only per-image target visibility and relative location.',
        parameters=dict(type='object', properties=properties,
                        required=list(properties), additionalProperties=False)))
    return request_body([dict(role='system', content='Inspect images literally. Unknown is allowed. Use the tool once.'),
                         dict(role='user', content=content)], [tool])


def history_lesion(original, reasons=False, images=False):
    body = copy.deepcopy(original)
    for message in body['messages'][2:-1]:
        if reasons and message['role'] == 'assistant':
            for call in message.get('tool_calls', []):
                args = json.loads(call['function']['arguments'])
                if 'reason' in args:
                    args['reason'] = 'Recorded action. Historical rationale omitted for this diagnostic.'
                    call['function']['arguments'] = json.dumps(args, ensure_ascii=False)
        if images and message['role'] == 'user' and isinstance(message.get('content'), list):
            message['content'] = [p if p.get('type') != 'image_url' else dict(type='text',
                text='Historical image omitted for this diagnostic; observation state and ID retained.')
                for p in message['content']]
    assert body['messages'][:2] == original['messages'][:2]
    assert body['messages'][-1] == original['messages'][-1]
    # All old tool call IDs, physical action parameters, and execution results remain unchanged.
    for before, after in zip(original['messages'], body['messages']):
        if before['role'] == 'tool':
            assert before == after
        if before['role'] == 'assistant':
            for old, new in zip(before.get('tool_calls', []), after.get('tool_calls', [])):
                assert old['id'] == new['id'] and old['function']['name'] == new['function']['name']
                a, b = (json.loads(c['function']['arguments']) for c in (old, new))
                a.pop('reason', None); b.pop('reason', None)
                assert a == b
    return body


def validate_minimal(body, response):
    message = response['choices'][0]['message']
    calls = message.get('tool_calls') or []
    if response['choices'][0]['finish_reason'] != 'tool_calls' or len(calls) != 1 or message.get('content'):
        raise ValueError('one_complete_tool_required')
    if calls[0]['function']['name'] != 'inspect_current':
        raise ValueError('wrong_tool')
    values = json.loads(calls[0]['function']['arguments'])
    fields = body['tools'][0]['function']['parameters']['properties']
    if set(values) != set(fields) or any(values[k] not in f['enum'] for k, f in fields.items()):
        raise ValueError('invalid_fields_or_values')
    return values


def run(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    source = Path(args.source)
    paths = [source/'current_only/request.json', source/'with_history/request.json',
             Path(args.rich_request)]
    current_only, history, rich = [json.loads(p.read_text()) for p in paths]
    current = current_only['messages'][-1]
    obs = json.loads(current['content'][0]['text'])
    session = json.loads(Path(args.session).read_text())
    report = dict(robot_connected=False, robot_commands_sent=0, enable_thinking=False,
                  observation_id=obs['observation_id'], results=[], sources=[
                      dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
                  limitations=['One saved observation; deterministic requests are not independent task trials.',
                      'History lesions are diagnostic inputs, not changes to the append-only production conversation.',
                      'No request supplies target location labels or desired hand direction.',
                      'No robot actions; millimeter magnitudes and physical success are untested.'])
    jobs = [('minimal_dual', minimal(current), 'minimal'),
            ('minimal_wrist', minimal(current, ('wrist',)), 'minimal'),
            ('minimal_head', minimal(current, ('head',)), 'minimal'),
            ('rich_current_only', perception.build_request('按下电梯外呼向下按钮并撤回', current), 'rich_current'),
            ('rich_pair_original', rich, 'rich_pair'),
            ('history_original', history, 'action'),
            ('history_without_reasons', history_lesion(history, reasons=True), 'action'),
            ('history_without_old_images', history_lesion(history, images=True), 'action'),
            ('history_without_both', history_lesion(history, reasons=True, images=True), 'action')]
    with tunnel(None if args.model_base else args.thor_host, args.port), StopControl() as control:
        base = args.model_base or f'http://127.0.0.1:{args.port}/v1'
        atomic_json(output/'models.json', request(base, '/models'))
        index = 0
        while index < len(jobs):
            name, body, kind = jobs[index]; index += 1
            assert body['chat_template_kwargs']['enable_thinking'] is False
            assert body['temperature'] == 0 and body['max_tokens'] == 512
            folder = output/'tests'/name; folder.mkdir(parents=True)
            archive_request(body, folder, output)
            row = dict(condition=name, kind=kind, images=image_count(body['messages']))
            started = time.monotonic()
            print('Starting '+name, flush=True)
            try:
                response = control.call(request, base, '/chat/completions', body, timeout=180)
                row['elapsed_s'] = time.monotonic()-started
                atomic_json(folder/'response.json', dict(response=response, elapsed_s=row['elapsed_s']))
                row['usage'] = response.get('usage'); row['finish_reason'] = response['choices'][0]['finish_reason']
                if kind == 'minimal':
                    parsed = validate_minimal(body, response)
                elif kind.startswith('rich'):
                    previous = None
                    if kind == 'rich_pair':
                        previous = json.loads(history['messages'][-4]['content'][0]['text'])['observation_id']
                    parsed = perception.validate(response, obs['observation_id'], previous)
                else:
                    parsed = parse_proposal(response, obs, session)
                row['parsed'] = parsed; row['format_valid'] = True
                atomic_json(folder/'parsed.json', parsed)
                if name == 'minimal_dual':
                    # Carry its actual output forward regardless of whether the analyst agrees with it.
                    extra = dict(role='user', content='Independent model assessment for current observation '+
                        obs['observation_id']+': '+json.dumps(parsed, ensure_ascii=False)+
                        '. This is model judgment, not human verification. Reconcile it with the current images before choosing a tool.')
                    for label, original in [('minimal_then_action_current', current_only), ('minimal_then_action_history', history)]:
                        updated = copy.deepcopy(original); updated['messages'].append(extra)
                        jobs.append((label, updated, 'action'))
            except Exception as exc:
                row.update(error_type=type(exc).__name__, error=str(exc), format_valid=False)
                atomic_json(folder/'error.json', row)
                if isinstance(exc, InterruptedError):
                    raise
            finally:
                row['total_s'] = time.monotonic()-started
                report['results'].append(row)
                report['sources_unchanged'] = all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
                                                  for x in report['sources'])
                atomic_json(output/'report.json', report)
                print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--source', default=str(ROOT/'outputs/success_tail_validation_20261101/0000'))
    p.add_argument('--rich-request', default=str(ROOT/'outputs/perception_validation_20261101/separated/0000/perception/request.json'))
    p.add_argument('--session', default=str(ROOT/'outputs/agent_web/20261101_211122_639d06d8/agent/session.json'))
    p.add_argument('--model-base'); p.add_argument('--thor-host', default='Thor'); p.add_argument('--port', type=int, default=18005)
    run(p.parse_args())
