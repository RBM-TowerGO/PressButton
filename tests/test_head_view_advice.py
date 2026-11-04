from pressb.agent_loop import head_view_advice as advice


def _loc(wrist, head):
    return dict(observation_id='o1', wrist=dict(button_visibility=wrist), head=dict(button_visibility=head))


def test_head_cannot_see_button_recommends_look_down():
    a = advice.build_advice(_loc('full', 'not_visible'), dict(head_deg=[0.0, 12.5]))
    assert a['situation'] == 'head_cannot_see_button' and a['measured_head_pitch_deg'] == 12.5
    assert 'lower the head' in a['instruction']


def test_wrist_lost_but_head_sees_says_do_not_withdraw_reflexively():
    for wrist in ('partial', 'not_visible', 'unknown'):
        a = advice.build_advice(_loc(wrist, 'full'), dict(head_deg=[0, 30]))
        assert a['situation'] == 'wrist_lost_button_head_sees_it'
        assert 'do not withdraw' in a['instruction']


def test_both_usable_and_missing_data_are_safe():
    assert advice.build_advice(_loc('full', 'partial'), {})['situation'] == 'both_views_usable'
    a = advice.build_advice(None, None)
    assert a['situation'] == 'head_cannot_see_button' and a['measured_head_pitch_deg'] is None
    assert a['source'] == advice.SOURCE and 'gated' in a['instruction']
