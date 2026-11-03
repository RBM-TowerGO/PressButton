import copy
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pressb.agent_loop import conversation as conversation_module
from pressb.agent_loop.common import atomic_json, SESSION_SCHEMA
from pressb.agent_loop.conversation import Conversation, image_count
from pressb.agent_loop.demonstration import Demonstration
from test_conversation import advance, jpeg, obs, summary_response


def recording(root):
    atomic_json(root/'session.json', dict(schema='pressb-keyboard-record-v1',
        settings=dict(frame='chassis', units='mm'), final=dict(status='missing'), recording_complete=False))
    atomic_json(root/'case_review.json', dict(schema='pressb-case-review-v1',
        button_activation_verified=True, withdrawal_verified=None, outcome='success',
        outcome_source='operator_feedback', analysis_version='test-v1',
        evidence=['operations/000004/after/head.jpg', 'operations/000004/after/right.jpg'],
        limitations=['step 5 withdrawal interrupted, after and final missing'],
        lessons=['Step 1 command is historical and must not leak into the selected tail.']))
    image = io.BytesIO();Image.new('RGB', (32, 18), 'gray').save(image, 'JPEG')
    for step in range(1, 6):
        folder = root/'operations'/f'{step:06d}'
        atomic_json(folder/'operation.json', dict(operation_id=f'{step:06d}', kind='move',
            command=dict(xyz_mm=[step, 0, 0]), status='completed' if step < 5 else 'interrupted',
            before=dict(status='complete'), after=dict(status='complete' if step < 5 else 'missing')))
        for phase in ('before', 'after') if step < 5 else ('before',):
            atomic_json(folder/phase/'metadata.json', dict(depth=dict(median_m=.13)))
            for view in ('head', 'right'):
                (folder/phase/(view+'.jpg')).write_bytes(image.getvalue())
    return root


def test_tail_selection_preserves_full_archive_and_missing_retreat(tmp_path):
    root = recording(tmp_path/'source')
    before = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*') if p.is_file()}
    demo = Demonstration(root)
    original_collages = dict(demo.images)
    assert demo.select_activation_tail(3) is demo
    assert demo.selection['selected_steps'] == ['000002', '000003', '000004']
    assert demo.selection['model_image_count'] == 3 and demo.selection['source_operation_count'] == 5
    assert [r['step'] for r in demo.model_summary['operations']] == demo.selection['selected_steps']
    assert len(demo.summary['operations']) == 5
    assert demo.model_summary['session']['final']['status'] == 'missing'
    assert demo.model_summary['case_review']['withdrawal_verified'] is None
    assert 'interrupted' in str(demo.model_summary['case_review']['limitations'])
    assert 'lessons' not in demo.model_summary['case_review']
    assert sum(p['type']=='image_url' for p in demo.content) == 3
    assert dict(demo.images) == {step+'.jpg':original_collages[step+'.jpg'] for step in demo.selection['selected_steps']}
    demo.save(tmp_path/'saved')
    archived = tmp_path/'saved/demonstration'
    assert len(json.loads((archived/'reference.json').read_text())['operations']) == 5
    assert len(json.loads((archived/'model_reference.json').read_text())['operations']) == 3
    assert json.loads((archived/'selection.json').read_text()) == demo.selection
    assert json.loads((archived/'source_sha256.json').read_text()) == before
    assert sorted(p.name for p in archived.glob('*.jpg')) == ['000002.jpg', '000003.jpg', '000004.jpg']
    assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==value for name,value in before.items())
    # Re-selection always uses the original complete recording, not a previously shortened slice.
    demo.select_activation_tail(4)
    assert demo.selection['selected_steps'] == ['000001', '000002', '000003', '000004']


@pytest.mark.parametrize('count', [0, -1, True, 1.5])
def test_tail_count_is_positive_integer(tmp_path, count):
    with pytest.raises(ValueError, match='positive_demo_last_steps'):
        Demonstration(recording(tmp_path/'source')).select_activation_tail(count)


@pytest.mark.parametrize('change,match', [
    ({'button_activation_verified':False}, 'reviewed_demo_activation_required'),
    ({'evidence':[]}, 'unique_reviewed_demo_activation_boundary'),
    ({'evidence':['operations/000003/after/head.jpg', 'operations/000004/after/right.jpg']}, 'unique_reviewed_demo_activation_boundary'),
    ({'evidence':['operations/000099/after/head.jpg']}, 'reviewed_demo_activation_step_missing'),
    ({'evidence':['operations/000005/after/head.jpg']}, 'requires_complete_boundaries'),
])
def test_invalid_or_ambiguous_activation_is_not_guessed(tmp_path, change, match):
    root = recording(tmp_path/'source')
    path = root/'case_review.json';review=json.loads(path.read_text());review.update(change);atomic_json(path,review)
    with pytest.raises(ValueError, match=match):
        Demonstration(root).select_activation_tail(3)


def test_missing_selected_image_and_short_recording_fail(tmp_path):
    root = recording(tmp_path/'source')
    with pytest.raises(ValueError, match='not_enough_completed'):
        Demonstration(root).select_activation_tail(5)
    (root/'operations/000003/after/right.jpg').unlink()
    with pytest.raises(ValueError, match='demo_missing_required_image'):
        Demonstration(root).select_activation_tail(3)


def test_changed_source_rejected_before_archive(tmp_path):
    root = recording(tmp_path/'source')
    demo = Demonstration(root).select_activation_tail(3)
    (root/'operations/000004/after/head.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError, match='demo_source_changed'):
        demo.save(tmp_path/'saved')


def test_default_reviewed_case_selects_8_9_10():
    demo = Demonstration(ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047').select_activation_tail(3)
    assert demo.selection['selected_steps'] == ['000008', '000009', '000010']
    assert [r['command']['xyz_mm'] for r in demo.model_summary['operations']] == [[0,-20,0], [30,0,0], [10,0,0]]
    assert demo.summary['operations'][-1]['operation']['status'] == 'interrupted'


@pytest.mark.parametrize('mode', ['direct', 'localized'])
def test_full_reviewed_case_request_keeps_all_steps_and_missing_retreat(jpeg, mode):
    from pressb.agent_loop.localized_conversation import LocalizedConversation

    demo = Demonstration(ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047')
    expected_steps = [f'{step:06d}' for step in range(1, 12)]
    assert demo.selection['mode'] == 'all'
    assert demo.selection['selected_steps'] == expected_steps
    cls = LocalizedConversation if mode == 'localized' else Conversation
    conversation = cls('press', obs(0), jpeg, jpeg, dict(schema=SESSION_SCHEMA), 'prompt', demo)
    messages = conversation.body()['messages']
    assert messages[1]['content'][1:] == demo.content
    assert image_count(messages[:2]) == 11
    assert image_count(messages) == 13  # Eleven demonstration collages plus two current views.
    model_demo = json.loads(messages[1]['content'][1]['text'])
    assert [row['step'] for row in model_demo['operations']] == expected_steps
    assert [row['status'] for row in model_demo['operations']] == ['completed']*10 + ['interrupted']
    retreat = model_demo['operations'][-1]
    assert retreat['command']['xyz_mm'] == [-100, 0, 0]
    assert retreat['before']['status'] == 'complete'
    assert retreat['after']['status'] == 'missing'
    assert retreat['after']['missing_images'] == ['head', 'right']
    assert model_demo['session']['recording_complete'] is False
    assert model_demo['session']['final']['status'] == 'missing'
    assert model_demo['case_review']['button_activation_verified'] is True
    assert model_demo['case_review']['withdrawal_verified'] is None


def test_compaction_preserves_full_reviewed_case_prefix(jpeg, monkeypatch, tmp_path):
    demo = Demonstration(ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047')
    conversation = Conversation('press', obs(0), jpeg, jpeg, dict(schema=SESSION_SCHEMA), 'prompt', demo)
    fixed_before = copy.deepcopy(conversation.body()['messages'][:2])
    for n in range(1, 9):
        advance(conversation, jpeg, n, 'request_observation')
    monkeypatch.setattr(conversation_module, 'request', lambda *args, **kwargs: summary_response())
    conversation.compact('fake', tmp_path/'compaction')
    assert conversation.generation == 1 and conversation.start == 3
    messages = conversation.body()['messages']
    assert messages[:2] == fixed_before
    assert image_count(messages[:2]) == 11
    model_demo = json.loads(messages[1]['content'][1]['text'])
    assert [row['step'] for row in model_demo['operations']] == [f'{step:06d}' for step in range(1, 12)]
    assert len([message for message in messages if message['role'] == 'assistant']) == 5
