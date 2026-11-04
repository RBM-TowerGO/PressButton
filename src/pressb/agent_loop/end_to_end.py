"""End-to-end mode: one model call per round looks at the images and returns the action.

No separate localization, wrist-overlap check, progress review or view-acquisition
request runs before or after the action call. The program only parses the single
tool call, validates its schema and executes it through the shared keyboard
executor. Recorded numbers (measured pose, wrist depth at the fixed pixel,
execution feedback) are still passed as raw observation data, never as a gate.
"""
import copy

from .conversation import Conversation, request_body
from .model_context import text, session as session_view
from . import press_retry

DECISION_MODE = 'end_to_end'
FORMAT = 'pressb-conversation-e2e-v1'
FORWARD_GUARD_POLICY = 'none'


def _tools():
    from .client import TOOLS
    tools = copy.deepcopy(TOOLS)
    for tool in tools:
        function = tool['function']
        if function['name'] == 'propose_move':
            # The shared description names a wrist pre-check that this mode does not run.
            function['description'] = ('Translate hand by chassis-frame delta; preserve orientation. '
                                       'No automatic return. Executed as given: not checked, clipped or replaced.')
        elif function['name'] == 'propose_gripper':
            function['description'] = 'c closes, o opens the gripper.'
    return tools


TOOLS = _tools()


def aim_context():
    """The calibrated merged-end wrist pixel, if one is active; raw reference, not a gate."""
    from . import calibrated_tip
    if not calibrated_tip.active():
        return None
    return dict(context_type='CALIBRATED_MERGED_END_WRIST_PIXEL', source='human_calibrated_fixed_pixel',
                point_normalized=[round(v, 4) for v in calibrated_tip.aim_point()],
                pixel=[round(v) for v in calibrated_tip.aim_pixel()] if calibrated_tip.aim_pixel() else None,
                image_size=calibrated_tip.image_size(),
                semantics='Where the CLOSED gripper end appears in the RIGHT WRIST image; the depth window '
                          'is read at this pixel. Fixed in the image: closing moves only the fingers.')


class EndToEndConversation(Conversation):
    forward_guard_policy = FORWARD_GUARD_POLICY
    max_press_attempts = press_retry.DEFAULT_LIMIT

    def __init__(self, goal, obs, head, wrist, session, prompt, demo=None, failure_case=None,
                 budget=None, decision_mode=DECISION_MODE):
        if decision_mode != DECISION_MODE:
            raise ValueError('end_to_end_mode_required')
        # The base class builds the plain history without perception, progress or view stages.
        super().__init__(goal, obs, head, wrist, session, prompt, demo, failure_case, budget,
                         decision_mode='direct', monitor_progress=False, acquire_view=False)
        self.decision_mode = DECISION_MODE
        self.format = FORMAT
        self.fixed[1]['content'][0] = text(dict(goal=goal, session=session_view(session), input_format=FORMAT))
        aim = aim_context()
        if aim:
            self.fixed[1]['content'].append(text(aim))

    def body(self):
        return request_body(self.messages(), TOOLS)

    def decision_context(self, stage):
        return dict(format=self.format, decision_mode=self.decision_mode, observation_id=self.current_id,
                    previous_observation_id=self.previous_id, stage=stage,
                    compression_generation=self.generation, forward_guard_policy=FORWARD_GUARD_POLICY)
