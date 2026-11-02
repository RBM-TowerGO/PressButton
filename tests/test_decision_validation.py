import pytest

from pressb.agent_loop.decision_validation import motion_reason_conflicts, reason_conflicts


SCENE = dict(head_visible='yes', wrist_visible='no', contact_finger='head_image_right',
             head_horizontal='right', head_vertical='above')


@pytest.mark.parametrize('reason', [
    '当前头部与右腕画面均未清楚看到目标按钮，需刷新视图以确认按钮位置。',
    '腕图未能清楚看到目标按钮；调整头部。',
    '腕图没有清楚看见按钮；根据头图调整。',
    '腕部相机无法清晰地看到按钮。',
    '腕图按钮不可见。',
    'Wrist cannot clearly see the button; use HEAD.',
    "Wrist can't see the button; use HEAD.",
    'Wrist does not clearly show the button.',
    'Wrist view shows no target button; HEAD shows the button.',
    'Wrist button is not clearly visible.',
    'Wrist cannot see the button and head shows the button.',
    'Wrist no longer shows the target button.',
    'Wrist may show the button, but visibility is uncertain.',
    '腕图可能看到按钮，需要重新检查。',
    '腕图看不到按钮，头图可见按钮。',
    '腕图未见按钮，头图可见按钮在右侧，向右移以纠偏。',
])
def test_negative_visibility_is_not_a_positive_claim(reason):
    reason_conflicts(reason, SCENE)


@pytest.mark.parametrize('reason', [
    'Wrist view shows the button between the fingers; press forward.',
    '腕图可见按钮在两指之间，因此前推。',
    '腕部相机按钮位于两指之间。',
    '腕图按钮清晰可见。',
    'Wrist button is clearly visible.',
    '腕图未清楚看到按钮，但腕图可见按钮在两指之间。',
    'Wrist cannot see the button, but wrist shows the button between fingers.',
    'Wrist cannot see the button and wrist shows the button between fingers.',
    'Wrist shows no target button but wrist sees the button clearly.',
])
def test_positive_visibility_contradiction_survives_other_negative_clause(reason):
    with pytest.raises(ValueError, match='current_wrist_visibility'):
        reason_conflicts(reason, SCENE)


def test_both_views_unknown_accept_recorded_chinese_negative():
    reason_conflicts('当前头部与右腕画面均未清楚看到目标按钮，需调整头部视角以定位按钮、接触手指与对齐偏差。',
                     dict(SCENE, head_visible='no'))


@pytest.mark.parametrize('reason', [
    'HEAD sees the down button at right/below the open right finger; move right and down to align.',
    'HEAD sees the button below the selected finger; move right.',
    'HEAD sees the down button to the right and the right finger is above it; move robot-right.',
    'HEAD sees the down button on the right and the right finger is right/above it; advance.',
    'HEAD sees the down button at right; the right finger is still right of and above it, so move robot-right.',
    '头图按钮位于右侧手指下方，向右移动。',
    '头图右侧手指在按钮上方，向右移动。',
])
def test_actual_vertical_contradictions_are_rejected(reason):
    with pytest.raises(ValueError, match='reason_conflicts_with_current_head_'):
        reason_conflicts(reason, SCENE)


@pytest.mark.parametrize('reason', [
    'HEAD sees the down button on the right and the right finger below it; move forward.',
    'HEAD shows the right finger below the button; move right and down to explore.',
    'HEAD sees the down button at right/above the open right finger; inspect after moving.',
    'HEAD shows the button above the selected finger; move right and down.',
    'HEAD shows the button above the left finger; check the selected right finger.',
    'HEAD cannot tell whether the button is below the selected finger.',
    'HEAD shows the button, which might be below the selected finger.',
    'HEAD shows the button is not below the selected finger.',
    'HEAD shows the button on the right; move the right finger down below the button as a probe.',
    'HEAD shows the button on the right; the finger will move below the button.',
    '头图按钮位于右侧手指上方，向下移动探查。',
    '头图右侧手指在按钮下方，向右移动。',
    '头图不能确定按钮是否在右侧手指下方。',
])
def test_consistent_inverse_uncertain_and_action_language_are_allowed(reason):
    reason_conflicts(reason, SCENE)


def test_horizontal_subject_inverse_and_other_finger_reference():
    reason_conflicts('HEAD shows the right finger left of the button.', SCENE)
    with pytest.raises(ValueError, match='current_head_horizontal'):
        reason_conflicts('HEAD shows the button to the left of the right finger.', SCENE)
    with pytest.raises(ValueError, match='current_head_horizontal'):
        reason_conflicts('HEAD shows the right finger to the right of the button.', SCENE)
    reason_conflicts('HEAD shows the button to the left of the left finger.', SCENE)


def test_uncertain_assessment_does_not_create_a_direction_gate():
    reason_conflicts('HEAD sees the button below the right finger; continue forward.',
                    dict(SCENE, head_vertical='unknown'))
    reason_conflicts('HEAD sees the button below the right finger; continue forward.',
                    dict(SCENE, contact_finger='unknown'))


def test_wrist_relation_is_not_checked_against_head_relation():
    reason_conflicts('HEAD sees the button above the right finger and WRIST sees it below the right finger.',
                    dict(SCENE, wrist_visible='yes'))


def test_wrist_only_relation_is_not_checked_against_head_relation():
    reason_conflicts('WRIST shows the button below the selected finger.', dict(SCENE, wrist_visible='yes'))


@pytest.mark.parametrize('vertical', ['above', 'below'])
def test_nested_finger_object_is_not_reinterpreted_as_subject(vertical):
    reason = ('In HEAD the button is right of the right finger and ' + vertical +
              ' it; move robot right (-Y) to reduce the lateral offset first, keeping height unaddressed.')
    reason_conflicts(reason, dict(SCENE, head_vertical=vertical))
    # The pronoun has no independently established subject/object relation.
    # A partial parser must leave this tail unparsed rather than invert it.
    other = 'below' if vertical == 'above' else 'above'
    reason_conflicts(reason, dict(SCENE, head_vertical=other))


@pytest.mark.parametrize(('reason', 'vertical'), [
    ('HEAD button is above the right finger.', 'above'),
    ('HEAD button is below the right finger.', 'below'),
    ('HEAD right finger is above the button.', 'below'),
    ('HEAD right finger is below the button.', 'above'),
    ('HEAD shows the button; the right finger is above it.', 'below'),
    ('HEAD shows the button; the right finger is below it.', 'above'),
    ('HEAD button is right of the finger and the right finger is above the button.', 'below'),
    ('HEAD button is right of the finger and the right finger is below the button.', 'above'),
    ('HEAD button is right of the finger; the right finger is above it.', 'below'),
    ('HEAD button is right of and above the right finger.', 'above'),
    ('HEAD button is right of and below the right finger.', 'below'),
    ('头图按钮在右侧手指上方。', 'above'),
    ('头图按钮在右侧手指下方。', 'below'),
    ('头图右侧手指在按钮上方。', 'below'),
    ('头图右侧手指在按钮下方。', 'above'),
    ('头图按钮在右侧手指右侧，右侧手指在按钮上方。', 'below'),
    ('头图按钮在右侧手指右侧，右侧手指在按钮下方。', 'above'),
])
def test_explicit_position_subjects_keep_their_direction(reason, vertical):
    reason_conflicts(reason, dict(SCENE, head_vertical=vertical))
    other = 'below' if vertical == 'above' else 'above'
    with pytest.raises(ValueError, match='current_head_vertical'):
        reason_conflicts(reason, dict(SCENE, head_vertical=other))


def test_inverse_pronoun_without_a_button_antecedent_is_not_guessed():
    reason_conflicts('HEAD the right finger is above it.', SCENE)


@pytest.mark.parametrize(('reason', 'xyz', 'axis'), [
    ('HEAD shows the button above the right finger; raise the hand to align.', [0, 0, -10], 'Z'),
    ('Lift the right finger to reduce the height error.', [0, 0, 0], 'Z'),
    ('Move the hand up to align with the button.', [5, 0, -1], 'Z'),
    ('Lower the open right finger.', [0, 0, 10], 'Z'),
    ('Move right and down to align the finger.', [0, -10, 5], 'Z'),
    ('抬高手部来对齐按钮。', [0, 0, -3], 'Z'),
    ('向上移动以调整高度。', [0, 0, 0], 'Z'),
    ('降低夹爪以修正偏差。', [0, 0, 3], 'Z'),
    ('Move robot-right to reduce lateral offset.', [0, 10, 0], 'Y'),
    ('Shift the hand robot-left to probe the mapping.', [0, -10, 0], 'Y'),
    ('向机器人右侧移动以对齐。', [0, 10, 0], 'Y'),
    ('Move forward to approach the panel.', [-5, 0, 0], 'X'),
    ('Retract the hand after activation.', [5, 0, 0], 'X'),
    ('向后移动以撤回手部。', [3, 0, 0], 'X'),
])
def test_explicit_physical_motion_must_match_command_sign(reason, xyz, axis):
    with pytest.raises(ValueError, match='reason_motion_direction_conflicts_with_' + axis):
        motion_reason_conflicts(reason, xyz)


@pytest.mark.parametrize(('reason', 'xyz'), [
    ('Move right and down to align the right finger with the button.', [0, -10, -10]),
    ('Raise the hand to reduce height error.', [0, 0, 0.001]),
    ('向上移动以调整高度。', [0, 0, 1.234567]),
    ('HEAD sees the button above the finger; inspect after approaching.', [5, 0, -2]),
    ('HEAD shows the right finger below the button.', [5, 0, -2]),
    ('Move right to probe image correspondence.', [0, 10, 0]),
    ('The button is on the right; use the selected finger.', [0, 10, 0]),
    ('Do not move forward; raise the hand.', [0, 0, 5]),
    ('不要向前移动，抬高手部。', [0, 0, 5]),
    ('Move the hand up in the head image by probing the projection.', [5, 0, 0]),
    ('Depth unknown; move forward using the head view.', [3.141592, 0, 0]),
    ('Activation unconfirmed; push the open right finger forward.', [5, 0, 0]),
    ('Front finger is down; right finger is up.', [-5, 0, 0]),
    ('尚无撤回证据，抬高手部进行对齐。', [0, 0, 5]),
    ('上一轮前推后没有改善，抬高手部。', [0, 0, 5]),
    ('Before retract, move forward to check the button.', [5, 0, 0]),
    ('Raise the hand before retracting.', [0, 0, 5]),
    ('Raise the hand then retract.', [0, 0, 5]),
    ('Raise the hand; then retract.', [0, 0, 5]),
    ('No need to move forward; lift the hand.', [0, 0, 5]),
    ('抬高手部，然后撤回。', [0, 0, 5]),
    ('不用向前移动，抬高手部。', [0, 0, 5]),
])
def test_motion_checks_do_not_infer_robot_direction_from_scene_or_impose_forward_gate(reason, xyz):
    motion_reason_conflicts(reason, xyz)


@pytest.mark.parametrize(('reason', 'xyz', 'axis'), [
    ('HEAD shows the down button right and below the right finger; '
     'move forward, robot-right and lower to reduce both offsets before pressing.', [10, 0, -10], 'Y'),
    ('Move forward, robot-right and lower.', [10, -10, 0], 'Z'),
    ('Move forward, robot-right and lower.', [0, -10, -10], 'X'),
    ('Shift the gripper forward, robot-left and down.', [10, -10, -10], 'Y'),
    ('Translate the hand robot-right and up.', [0, 0, 10], 'Y'),
    ('Move forward and robot-left.', [10, 0, 0], 'Y'),
])
def test_explicit_same_command_motion_list_checks_each_axis(reason, xyz, axis):
    with pytest.raises(ValueError, match='reason_motion_direction_conflicts_with_' + axis):
        motion_reason_conflicts(reason, xyz)


@pytest.mark.parametrize(('reason', 'xyz'), [
    ('Move forward, robot-right and lower.', [10, -10, -10]),
    ('Shift the gripper forward, robot-left and down.', [10, 10, -10]),
    ('Translate the hand robot-right and up.', [0, -10, 10]),
    ('Do not move forward, robot-right and lower; raise the hand.', [0, 0, 10]),
    ('No need to move forward, robot-right and lower; raise the hand.', [0, 0, 10]),
    ('Previously move forward, robot-right and lower; raise the hand now.', [0, 0, 10]),
    ('Earlier move forward, robot-right and lower; raise the hand now.', [0, 0, 10]),
    ('If we move forward, robot-right and lower, inspect the result.', [0, 0, 0]),
    ('Raise the hand; then move forward, robot-right and lower.', [0, 0, 10]),
    ('Raise the hand, then move forward, robot-right and lower.', [0, 0, 10]),
    ('Move forward, robot-right and lower, then retract.', [10, -10, -10]),
    ('Move forward, then robot-right and lower.', [10, 0, 0]),
    ('HEAD shows the button forward, robot-right and lower.', [0, 0, 0]),
    ('Move forward; the button appears robot-right and lower.', [10, 0, 0]),
    ('Move forward, robot-right and lower in the head image.', [0, 0, 0]),
])
def test_motion_list_retains_shared_scope_and_does_not_invent_intents(reason, xyz):
    motion_reason_conflicts(reason, xyz)
