"""Controlled ablations must not accidentally change robot command semantics."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/diagnose_localized_feedback.py'
spec = importlib.util.spec_from_file_location('feedback_diagnostic', SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class FeedbackDiagnosticTests(unittest.TestCase):
    def test_units_only_preserves_commands_ratios_and_directions(self):
        tree = dict(button_box=[180, 0, 390, 130], contact_tip=[690, 760],
            derived_relation=dict(dx=-405, dy=-695, pixel_delta=[-259200, -250200],
                horizontal='left', vertical='above', error_button_units=[-1.93, -5.35]),
            actual_commands=[dict(type='move', xyz_mm=[5, 0, 0])],
            current_state=dict(head_deg=[-0.7, 25], measured_gripper_sdk_native=[-.1419]))
        original = copy.deepcopy(tree)
        result = diagnostic.correct_tree(tree)
        self.assertEqual(tree, original)
        self.assertEqual(result['button_box'], [.18, 0, .39, .13])
        self.assertEqual(result['contact_tip'], [.69, .76])
        self.assertEqual(result['derived_relation']['pixel_delta'], [-259.2, -250.2])
        for field in ('horizontal', 'vertical', 'error_button_units'):
            self.assertEqual(result['derived_relation'][field], original['derived_relation'][field])
        self.assertEqual(result['actual_commands'], original['actual_commands'])
        self.assertEqual(result['current_state'], original['current_state'])

    def test_duplicate_raw_tool_arguments_are_corrected_consistently(self):
        tree = dict(raw_output=[dict(tool_calls=[dict(function=dict(name='locate_current_scene',
            arguments=json.dumps(dict(wrist=dict(contact_tip=[690, 760])))))])])
        result = diagnostic.correct_tree(tree)
        function = result['raw_output'][0]['tool_calls'][0]['function']
        self.assertEqual(json.loads(function['arguments'])['wrist']['contact_tip'], [.69, .76])

    def test_unknowns_remain_unknown_and_image_size_unchanged(self):
        source = dict(button_box=None, contact_tip=None, image_size=[640, 360],
                      pixel_delta=None, offset_normalized=[None, None],
                      error_button_units=None)
        self.assertEqual(diagnostic.correct_tree(source), source)

    def test_source_round_action_messages_are_untouched(self):
        location = dict(context_type=diagnostic.LOCATION, wrist=dict(contact_tip=[690,760]))
        source = dict(messages=[
            dict(role='user', content=[dict(type='text', text=json.dumps(location))]),
            dict(role='assistant', tool_calls=[dict(function=dict(name='propose_move',
                arguments='{"xyz_mm":[5,0,0],"reason":"advance"}'))]),
            dict(role='tool', content='{"command":{"type":"move","xyz_mm":[5,0,0]}}')])
        result, changed = diagnostic.correct_body(source)
        self.assertEqual(changed, [0])
        self.assertEqual(result['messages'][1:], source['messages'][1:])


if __name__ == '__main__':
    unittest.main()
