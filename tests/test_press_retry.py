import io
import json

from PIL import Image
import pytest

from pressb.agent_loop import client, press_retry
from pressb.agent_loop.common import SESSION_SCHEMA
from pressb.agent_loop.end_to_end import EndToEndConversation
from pressb.agent_loop.localized_conversation import LocalizedConversation

UNABLE = dict(name='report_unable', arguments=dict(reason='light never turned on'))


def turn(name, oid, **arguments):
    return dict(ledger=dict(proposal=dict(name=name, arguments=arguments),
                            result=dict(status='completed'), after_observation_id=oid))


# Sequence of run 20261104_212253: align, close, press 47 + 4, withdraw 80 + 60, give up.
RUN = [turn('propose_head', 'a', head_deg=[0, 30]), turn('propose_move', 'b', xyz_mm=[0, 60, 90]),
       turn('propose_move', 'c', xyz_mm=[0, 0, 60]), turn('propose_gripper', 'd', key='c'),
       turn('propose_move', 'e', xyz_mm=[47, 0, 0]), turn('propose_move', 'pressed', xyz_mm=[4, 0, 0]),
       turn('propose_move', 'g', xyz_mm=[-80, 0, 0]), turn('propose_move', 'h', xyz_mm=[-60, 0, 0])]


def test_unable_after_one_missed_press_is_rejected_and_points_at_the_press_images():
    assert press_retry.attempts(RUN) == ['pressed']
    with pytest.raises(ValueError, match='report_unable_rejected_press_attempts_remaining') as exc:
        press_retry.check(RUN, UNABLE, 3)
    assert 'observation pressed' in str(exc.value) and '1 of 3' in str(exc.value)


def test_unable_is_accepted_once_attempts_are_used_or_without_any_press():
    retry = [turn('propose_gripper', 'o', key='o'), turn('propose_move', 'r', xyz_mm=[0, -5, 3]),
             turn('propose_gripper', 'c2', key='c'), turn('propose_move', 'p2', xyz_mm=[45, 0, 0])]
    assert press_retry.attempts(RUN+retry) == ['pressed', 'p2']
    press_retry.check(RUN+retry, UNABLE, 2)
    press_retry.check(RUN[:3], UNABLE, 3)          # never pressed: e.g. button not found
    press_retry.check(RUN, UNABLE, 0)              # 0 disables the rule
    press_retry.check(RUN, dict(name='propose_gripper', arguments=dict(key='o', reason='r')), 3)


def test_failed_or_open_gripper_moves_are_not_attempts():
    failed = turn('propose_move', 'x', xyz_mm=[40, 0, 0])
    failed['ledger']['result']['status'] = 'failed'
    assert press_retry.attempts([turn('propose_move', 'y', xyz_mm=[40, 0, 0])]) == []
    assert press_retry.attempts([turn('propose_gripper', 'd', key='c'), failed]) == []


def test_both_modes_enable_the_rule():
    assert EndToEndConversation.max_press_attempts == LocalizedConversation.max_press_attempts == 3


def test_propose_repairs_a_premature_unable_into_a_new_command(monkeypatch, tmp_path):
    stream = io.BytesIO(); Image.new('RGB', (64, 36)).save(stream, 'JPEG'); jpeg = stream.getvalue()
    obs = dict(observation_id='now', state={}, selection={}, depth={})
    c = EndToEndConversation('press', obs, jpeg, jpeg, dict(schema=SESSION_SCHEMA), 'prompt')
    c.turns = list(RUN)
    replies = iter([('report_unable', dict(reason='light never on')),
                    ('propose_gripper', dict(key='o', reason='closed end landed right of the button; reopen'))])
    bodies = []
    def fake(base, route, body, **kwargs):
        bodies.append(body)
        name, args = next(replies)
        return dict(choices=[dict(finish_reason='tool_calls', message=dict(role='assistant', content=None,
            tool_calls=[dict(id=f'call-{len(bodies)}', type='function',
                             function=dict(name=name, arguments=json.dumps(args)))]))])
    monkeypatch.setattr(client, 'request', fake)
    # Ledger-only fake turns carry no messages, so start from a minimal request body.
    body = dict(model='m', messages=[dict(role='system', content='prompt')])
    proposal = client.propose('fake', body, obs, dict(schema=SESSION_SCHEMA), tmp_path, conversation=c)
    assert proposal['name'] == 'propose_gripper' and len(bodies) == 2
    assert 'observation pressed' in bodies[1]['messages'][-1]['content']
    assert json.loads((tmp_path/'correction.json').read_text())['status'] == 'accepted'
