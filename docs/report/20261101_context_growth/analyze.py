"""Post-hoc context accounting. Tokenization only; no generation or robot API."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from pressb.agent_loop.common import request
from s1b_agent import tunnel

OUT = Path(__file__).resolve().parent
RUN = ROOT / 'outputs/agent_web/20261101_211122_639d06d8/agent'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run():
    task = json.loads((RUN / 'task.json').read_text())
    paths = sorted((RUN / 'rounds').glob('*/request.json'))
    bodies = [json.loads(p.read_text()) for p in paths]
    assert all(d['messages'][:2] == bodies[0]['messages'][:2] and
               d['tools'] == bodies[0]['tools'] for d in bodies)
    assert all(d['messages'] == bodies[-1]['messages'][:len(d['messages'])] for d in bodies)
    report = dict(source_run=str(RUN), application_budget=task['context_budget_tokens'],
                  robot_commands_sent=0, generation_requests=0, rows=[], separated_rows=[],
                  accounting='Prefix token count differences; add_generation_prompt=True on all prefixes. '
                             'System/tools bucket includes an empty user message required by the template. '
                             'Incremental buckets include boundary tokens; these are accounting contributions, not attention weights.')
    with tunnel('Thor', 18113):
        base = 'http://127.0.0.1:18113'
        models = request(base, '/v1/models')
        save('models.json', models)
        report['server_max_model_len'] = next(m['max_model_len'] for m in models['data'] if m['id'] == 'qwen38')
        def count(body, messages):
            payload = {k: body[k] for k in ('model', 'tools', 'chat_template_kwargs') if k in body}
            payload.update(messages=messages, add_generation_prompt=True)
            return request(base, '/tokenize', payload, timeout=120)['count']
        system = count(bodies[0], bodies[0]['messages'][:1] + [dict(role='user', content='')])
        fixed = count(bodies[0], bodies[0]['messages'][:2])
        report['fixed_system_tools_tokens'] = system
        report['fixed_task_cases_tokens'] = fixed - system
        report['fixed_total_tokens'] = fixed
        previous = None
        for path, body in zip(paths, bodies):
            folder = path.parent
            meta = json.loads((folder / 'context_metrics.json').read_text())
            response = folder / 'response-0.json'
            usage = json.loads(response.read_text())['response']['usage'] if response.exists() else None
            actual = usage['prompt_tokens'] if usage else meta['input_tokens_exact']
            verified = count(body, body['messages'])
            if verified != actual:
                raise ValueError(f'tokenizer_count_changed:{folder.name}:{verified}!={actual}')
            before_current = fixed if len(body['messages']) == 3 else count(body, body['messages'][:-1])
            row = dict(round=int(folder.name)+1, prompt_tokens=actual,
                       token_basis='response.usage.prompt_tokens' if usage else 'saved_exact_tokenize_count_no_response',
                       tokenizer_recheck=verified, messages=len(body['messages']),
                       images=sum(x.get('type') == 'image_url' for m in body['messages']
                                  if isinstance(m.get('content'), list) for x in m['content']),
                       system_tools_tokens=system, task_cases_tokens=fixed-system,
                       earlier_history_tokens=before_current-fixed, current_observation_tokens=actual-before_current,
                       increase_tokens=None if previous is None else actual-previous,
                       application_budget_pct=100*actual/report['application_budget'],
                       model_window_pct=100*actual/report['server_max_model_len'],
                       compression_generation=meta['compression_generation'],
                       source_request=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for key in ('system_tools', 'task_cases', 'earlier_history', 'current_observation'):
                row[key+'_pct'] = 100*row[key+'_tokens']/actual
            assert sum(row[k+'_tokens'] for k in ('system_tools','task_cases','earlier_history','current_observation')) == actual
            report['rows'].append(row)
            previous = actual
            save('analysis.json', report)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    # Existing separated replay measurements are a different request series.
    for path in sorted((ROOT / 'outputs/perception_validation_20261101/separated').glob('*/response-0.json')):
        folder = path.parent
        response = json.loads(path.read_text())['response']
        context = json.loads((folder / 'decision_context.json').read_text())
        percept = json.loads((folder / 'perception/metrics.json').read_text())
        report['separated_rows'].append(dict(source_round=int(folder.name)+12,
            action_prompt_tokens=response['usage']['prompt_tokens'],
            action_application_budget_pct=100*response['usage']['prompt_tokens']/report['application_budget'],
            perception_prompt_tokens=percept['prompt_tokens'],
            action_images=context['image_count'], perception_images=percept['image_count']))
    report['mean_increment_tokens'] = sum(r['increase_tokens'] for r in report['rows'][1:])/(len(report['rows'])-1)
    report['cumulative_prompt_tokens_completed_requests'] = sum(r['prompt_tokens'] for r in report['rows']
                                                               if r['token_basis']=='response.usage.prompt_tokens')
    save('analysis.json', report)
    with (OUT/'rounds.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(report['rows'][0]))
        writer.writeheader()
        writer.writerows(report['rows'])
    return report


def add_separated_accounting(report):
    with tunnel('Thor', 18113):
        base = 'http://127.0.0.1:18113'
        def count(body, messages):
            payload = {k: body[k] for k in ('model', 'tools', 'chat_template_kwargs')}
            payload.update(messages=messages, add_generation_prompt=True)
            return request(base, '/tokenize', payload, timeout=120)['count']
        fixed = system = None
        for row in report['separated_rows']:
            path = ROOT / 'outputs/perception_validation_20261101/separated' / f'{row["source_round"]-12:04d}' / 'request.json'
            body = json.loads(path.read_text())
            if fixed is None:
                system = count(body, body['messages'][:1] + [dict(role='user', content='')])
                fixed = count(body, body['messages'][:2])
            assert count(body, body['messages']) == row['action_prompt_tokens']
            before_current = count(body, body['messages'][:-2])
            before_assessment = count(body, body['messages'][:-1])
            row.update(system_tools_tokens=system, task_cases_tokens=fixed-system,
                       earlier_history_tokens=before_current-fixed,
                       current_observation_tokens=before_assessment-before_current,
                       current_assessment_tokens=row['action_prompt_tokens']-before_assessment,
                       source_request=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for key in ('system_tools','task_cases','earlier_history','current_observation','current_assessment'):
                row[key+'_pct'] = 100*row[key+'_tokens']/row['action_prompt_tokens']
            assert sum(row[k+'_tokens'] for k in ('system_tools','task_cases','earlier_history',
                                                  'current_observation','current_assessment')) == row['action_prompt_tokens']
            print(json.dumps(row, ensure_ascii=False), flush=True)
    save('analysis.json', report)


if __name__ == '__main__':
    add_separated_accounting(run())
