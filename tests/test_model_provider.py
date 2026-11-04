"""Backend selection: Thor stays byte-identical, DeepSeek gets its own fields and key."""
import json
from pathlib import Path
import sys

import pytest

from pressb.agent_loop import common, model_provider as provider
from pressb.agent_loop.conversation import request_body, tokenize
from pressb.agent_loop import grounding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    provider.reset()
    yield
    provider.reset()


def _image_body(images=2):
    content = [dict(type='text', text='x'*300)]
    content += [dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,'+'A'*5000))]*images
    return request_body([dict(role='system', content='s'), dict(role='user', content=content)])


def test_default_thor_body_is_unchanged():
    body = request_body([])
    assert body['model'] == 'qwen38'
    assert body['chat_template_kwargs'] == dict(enable_thinking=False)
    assert 'thinking' not in body and body['tool_choice'] == 'required'
    assert grounding._request_body('p', [], dict(type='function'))['model'] == 'qwen38'
    assert provider.auth_headers('http://127.0.0.1:18000/v1/chat/completions') == {}


def test_deepseek_body_disables_thinking_and_drops_vllm_fields(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    described = provider.configure('deepseek')
    assert described['model'] == 'deepseek-flash' and described['base'] == 'https://api.deepseek.com'
    assert 'test-key' not in json.dumps(described)
    for body in (request_body([]), grounding._request_body('p', [], dict(type='function'))):
        assert body['model'] == 'deepseek-flash'
        assert body['thinking'] == dict(type='disabled')
        assert 'chat_template_kwargs' not in body
        assert 'test-key' not in json.dumps(body)


def test_missing_key_fails_before_anything_runs():
    with pytest.raises(ValueError, match='DEEPSEEK_API_KEY'):
        provider.configure('deepseek')


def test_key_is_sent_only_to_the_model_host(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    provider.configure('deepseek')
    assert provider.auth_headers('https://api.deepseek.com/chat/completions') == {'Authorization': 'Bearer test-key'}
    for url in ('http://127.0.0.1:6210/v2/session', 'http://127.0.0.1:8765/api/capture',
                'https://api.deepseek.com-evil.example/chat/completions',
                'https://api.deepseek.com.evil.example/chat/completions',
                'https://api.deepseek.com@evil.example/chat/completions',
                'http://api.deepseek.com/chat/completions'):
        assert provider.auth_headers(url) == {}
    sent = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{}'
    monkeypatch.setattr(common, 'urlopen', lambda req, timeout: sent.append(req) or Response())
    common.request('https://api.deepseek.com', '/chat/completions', {})
    common.request('http://127.0.0.1:6210', '/v2/session')
    assert sent[0].get_header('Authorization') == 'Bearer test-key'
    assert sent[1].get_header('Authorization') is None


def test_hosted_provider_estimates_tokens_without_tokenize(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    provider.configure('deepseek')
    monkeypatch.setattr('pressb.agent_loop.conversation.request',
                        lambda *a, **k: pytest.fail('hosted API has no /tokenize'))
    two, four = tokenize('https://api.deepseek.com', _image_body(2)), tokenize('https://api.deepseek.com', _image_body(4))
    assert four - two == 2*1024           # per-image cap, base64 bytes not counted as text
    assert two > 2*1024
    assert tokenize('x', _image_body(2), return_tokens=True)['estimated'] is True


def test_openai_compatible_requires_model_and_base(monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY', 'k')
    with pytest.raises(ValueError, match='model_name_required'):
        provider.configure('openai_compatible', base='https://example.invalid/v1')
    with pytest.raises(ValueError, match='model_base_required'):
        provider.configure('openai_compatible', model='m')
    described = provider.configure('openai_compatible', model='m', base='https://example.invalid/v1/')
    assert described['base'] == 'https://example.invalid/v1'
    assert request_body([])['model'] == 'm' and 'chat_template_kwargs' not in request_body([])


def test_agent_records_provider_without_key_and_skips_thor_tunnel(tmp_path, monkeypatch):
    import s1b_agent as agent
    from pressb.agent_loop import calibrated_tip, press_phase
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'secret-test-key')
    hosts = []
    def boundary(host, port):
        hosts.append(host)
        raise RuntimeError('offline-boundary')
    monkeypatch.setattr(agent, 'tunnel', boundary)
    monkeypatch.setattr(agent, 'request', lambda *a, **k: pytest.fail('no network'))
    monkeypatch.setattr(agent, 'RobotClient', lambda *a, **k: pytest.fail('no robot'))
    output = tmp_path/'run'
    args = agent.parse_args(['--goal', 'press', '--output', str(output), '--model-provider', 'deepseek',
                             '--replay', str(tmp_path/'unused')])
    try:
        with pytest.raises(RuntimeError, match='offline-boundary'):
            agent.run(args)
    finally:
        calibrated_tip.deactivate()
        press_phase.reset()
    assert hosts == [None]                 # no SSH tunnel to Thor
    task = json.loads((output/'task.json').read_text())
    assert task['model_base'] == 'https://api.deepseek.com'
    assert task['model_provider']['model'] == 'deepseek-flash'
    assert 'secret-test-key' not in (output/'task.json').read_text()


def test_launcher_rejects_missing_key_before_robot_start(monkeypatch, tmp_path):
    import s1b_agent_run as launcher
    from types import SimpleNamespace
    monkeypatch.setattr(launcher.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(launcher, 'check_port_available', lambda port: None)
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **k: pytest.fail('robot must not start'))
    args = launcher.parse_args(['--model-provider', 'deepseek', '--output', str(tmp_path/'run')])
    with pytest.raises(ValueError, match='DEEPSEEK_API_KEY'):
        launcher.launch(args)
    assert not (tmp_path/'run').exists()
