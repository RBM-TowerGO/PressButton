#!/usr/bin/env python3
"""Paired direct-policy thinking replay. No robot client or execution calls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pressb.agent_loop.common import atomic_json, request
from pressb.agent_loop.client import parse_proposal
from pressb.agent_loop.conversation import image_count, tokenize
from s1b_agent import StopControl, tunnel


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    session = json.loads(Path(args.session).read_text())
    report = dict(robot_connected=False, robot_commands_sent=0, max_tokens=args.max_tokens,
        results=[], sources=[],
        limitations=['Independent saved direct-mode requests, not a live closed-loop run.',
                     'Same messages, images, tools, temperature and token ceiling in each pair; thinking flag differs.',
                     'Shared model caches not reset. One sample per condition per observation.'])
    def save():atomic_json(output/'report.json', report)
    try:
        with tunnel(None if args.model_base else args.thor_host, args.port), StopControl() as control:
            base = args.model_base or f'http://127.0.0.1:{args.port}/v1'
            atomic_json(output/'models.json', control.call(request, base, '/models'))
            for index in range(args.rounds):
                source = Path(args.source)/f'{index:04d}'/'request.json'
                raw = source.read_bytes()
                original = json.loads(raw)
                live = [json.loads(p['text']) for m in original['messages'] if isinstance(m.get('content'), list)
                        for p in m['content'] if p.get('type') == 'text' and p['text'].startswith('{')
                        and json.loads(p['text']).get('context_type') == 'LIVE_OBSERVATION'][-1]
                report['sources'].append(dict(path=str(source.resolve()),sha256=hashlib.sha256(raw).hexdigest(),
                                               observation_id=live['observation_id']))
                for thinking in ((True, False) if index % 2 == 0 else (False, True)):
                    condition = 'thinking_on' if thinking else 'thinking_off'
                    directory = output/f'{index:04d}'/condition
                    body = copy.deepcopy(original)
                    body['max_tokens'] = args.max_tokens
                    body['chat_template_kwargs'] = dict(body.get('chat_template_kwargs') or {}, enable_thinking=thinking)
                    atomic_json(directory/'request.json', body)
                    row = dict(condition=condition, source_index=index, observation_id=live['observation_id'],
                               images=image_count(body['messages']), tool_choice=body.get('tool_choice'))
                    print(f'Observation {index+1}: {condition}, {row["images"]} images, max_tokens={args.max_tokens}',flush=True)
                    started = time.monotonic()
                    try:
                        tokens = control.call(tokenize, base, body, True)
                        atomic_json(directory/'tokenize_summary.json',dict(count=tokens['count'],
                            tail_tokens=tokens.get('tokens', [])[-32:]))
                        row['tokenize_count'] = tokens['count']
                        request_started = time.monotonic()
                        response = control.call(request, base, '/chat/completions', body, timeout=args.timeout)
                        row['request_s'] = time.monotonic()-request_started
                        atomic_json(directory/'response.json',dict(response=response,elapsed_s=row['request_s']))
                        message = response['choices'][0]['message']
                        reasoning = message.get('reasoning') or message.get('reasoning_content') or ''
                        usage = response.get('usage') or {}
                        row.update(usage=usage, finish_reason=response['choices'][0]['finish_reason'],
                            reasoning_characters=len(reasoning),
                            reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'))
                        row['thinking_observed'] = bool(reasoning) or bool(row['reasoning_tokens'])
                        try:
                            proposal = parse_proposal(response, live, session)
                            atomic_json(directory/'proposal.json',proposal)
                            row.update(valid_action=True,proposal=proposal)
                        except (ValueError, KeyError, TypeError, IndexError) as exc:
                            row.update(valid_action=False,parse_error=str(exc))
                    except Exception as exc:
                        row['error'] = str(exc)
                        atomic_json(directory/'error.json',dict(error_type=type(exc).__name__,error=str(exc)))
                        if isinstance(exc, InterruptedError):raise
                    finally:
                        row['wall_s'] = time.monotonic()-started
                        report['results'].append(row);save()
                        print(json.dumps(row,ensure_ascii=False),flush=True)
    finally:
        report['source_files_unchanged'] = all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
                                             for x in report['sources'])
        save()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--source',default=str(ROOT/'outputs/perception_validation_20261101/direct'))
    p.add_argument('--session',default=str(ROOT/'outputs/agent_web/20261101_211122_639d06d8/agent/session.json'))
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--max-tokens',type=int,default=8192)
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--model-base')
    p.add_argument('--thor-host',default='Thor');p.add_argument('--port',type=int,default=18003)
    run(p.parse_args())
