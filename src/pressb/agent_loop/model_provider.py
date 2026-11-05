"""Which chat-completions backend the agent talks to, and the few fields that differ.

All request bodies are still built by their own modules; they take the model
name and backend-specific fields from here. The default ``thor`` provider keeps
the established Qwen/vLLM bodies unchanged.

The API key is read from an environment variable named by the provider and is
attached only to requests whose URL starts with the configured model base, so it
is never sent to the robot or camera services and never written to any record.
"""
import copy
import json
import math
import os
from urllib.parse import urlsplit

PROVIDERS = {
    # Local vLLM on Thor through the SSH tunnel; exact /tokenize available.
    # Qwen3 thinking has no effort levels, only on/off.
    'thor': dict(model='qwen38', base=None, key_env=None, tokenize=True, image_tokens=None,
                 extra=dict(chat_template_kwargs=dict(enable_thinking=False)),
                 thinking_extra=dict(chat_template_kwargs=dict(enable_thinking=True))),
    # DeepSeek V4.1 Flash: OpenAI-compatible, accepts images in user messages,
    # thinking is ON by default and is switched off here like on Thor.
    # Each image is capped at 1024 tokens by the provider.
    # thinking_extra=None keeps thinking off for the whole run: a 2026-11-05 replay
    # put the head button box on the gripper base with thinking on
    # (docs/report/20261105_deepseek_thinking_head_localization).
    'deepseek': dict(model='deepseek-flash', base='https://api.deepseek.com', key_env='DEEPSEEK_API_KEY',
                     tokenize=False, image_tokens=1024, extra=dict(thinking=dict(type='disabled')),
                     thinking_extra=None),
    # Any other OpenAI-compatible vision endpoint; model and base must be given.
    'openai_compatible': dict(model=None, base=None, key_env='MODEL_API_KEY', tokenize=False,
                              image_tokens=1536, extra={}, thinking_extra=dict(reasoning_effort='low')),
}
DEFAULT = 'thor'
_active = None
# Thinking is off for the whole run until the press starts (the first executed
# gripper close), then stays on at low effort (``thinking_extra``) until the run
# ends. Reasoning tokens count against max_tokens, so requests get at least this
# many while it is on.
THINKING_MIN_MAX_TOKENS = 4096
_thinking = False


def configure(name=DEFAULT, model=None, base=None, key_env=None):
    """Select the provider for this process; thinking starts off. Returns a key-free description."""
    global _active, _thinking
    _thinking = False
    if name not in PROVIDERS:
        raise ValueError('unknown_model_provider:'+str(name))
    value = copy.deepcopy(PROVIDERS[name])
    value.update(name=name, model=model or value['model'], base=(base or value['base']),
                 key_env=key_env or value['key_env'])
    if not value['model']:
        raise ValueError('model_name_required_for_provider:'+name)
    if name != 'thor' and not value['base']:
        raise ValueError('model_base_required_for_provider:'+name)
    if value['base']:
        value['base'] = value['base'].rstrip('/')
    if value['key_env'] and not os.environ.get(value['key_env']):
        raise ValueError('model_api_key_missing: set environment variable '+value['key_env'])
    _active = value
    return describe()


def reset():
    global _active, _thinking
    _active = None
    _thinking = False


def enable_thinking():
    """Switch thinking on (low effort) for every later request of this run; never switched off.

    Returns True only on the call that actually turned it on.
    """
    global _thinking
    # A provider without thinking_extra stays off for the whole run.
    if _thinking or _current().get('thinking_extra') is None:
        return False
    _thinking = True
    return True


def thinking_enabled():
    return _thinking


def _current():
    return _active if _active is not None else dict(PROVIDERS[DEFAULT], name=DEFAULT)


def body_fields():
    """Model name plus backend-specific fields for a chat-completions body."""
    current = _current()
    extra = current['thinking_extra'] if _thinking else current['extra']
    return dict(model=current['model'], **copy.deepcopy(extra))


def tool_choice():
    """'required' normally; a provider may need another value while thinking is on."""
    return _current().get('thinking_tool_choice', 'required') if _thinking else 'required'


def max_tokens(value):
    """Completion ceiling for a request; raised while thinking so reasoning cannot crowd out the tool call."""
    return max(value, THINKING_MIN_MAX_TOKENS) if _thinking else value


def base():
    return _current().get('base')


def auth_headers(url):
    """Bearer header only for the configured model host; nothing for robot services."""
    current = _active
    if not current or not current.get('key_env') or not current.get('base'):
        return {}
    # Exact scheme/host/port plus path prefix on a segment boundary; a plain string
    # prefix would also match look-alike hosts such as api.deepseek.com-evil.example.
    target, configured = urlsplit(str(url)), urlsplit(current['base'])
    path = configured.path.rstrip('/')
    if (target.scheme != configured.scheme or target.netloc.lower() != configured.netloc.lower()
            or not (target.path == path or target.path.startswith(path+'/'))):
        return {}
    key = os.environ.get(current['key_env'])
    return {'Authorization': 'Bearer '+key} if key else {}


def supports_tokenize():
    return bool(_current()['tokenize'])


def estimate_tokens(body):
    """Conservative count when the provider has no /tokenize endpoint.

    Text: one token per 3 UTF-8 bytes of the image-free JSON. Images: the
    provider's per-image cap (DeepSeek bills at most 1024 per image).
    """
    per_image = _current()['image_tokens'] or 1536
    images = 0
    stripped = copy.deepcopy(body.get('messages', []))
    for message in stripped:
        content = message.get('content')
        if isinstance(content, list):
            images += sum(1 for part in content if part.get('type') == 'image_url')
            message['content'] = [part for part in content if part.get('type') != 'image_url']
    text = json.dumps(dict(messages=stripped, tools=body.get('tools')), ensure_ascii=False).encode()
    return int(math.ceil(len(text)/3)) + images*per_image


def describe():
    current = _current()
    return dict(provider=current['name'], model=current['model'], base=current.get('base'),
                key_env=current.get('key_env'), token_count='exact_tokenize' if current['tokenize'] else 'estimated',
                extra_fields=sorted(current['extra']),
                thinking=dict(policy='off_until_first_gripper_close_then_low_until_run_end',
                              fields=copy.deepcopy(current['thinking_extra']),
                              min_max_tokens=THINKING_MIN_MAX_TOKENS))
