"""Do not let the model give up after a missed press while attempts remain.

A press attempt is one executed gripper close followed by at least one executed
forward move while closed. If the model reports unable after such an attempt and
fewer than ``limit`` attempts have been made, the report is rejected (not executed)
and the model is pointed back at the images taken at its deepest press, so it plans
a correction and tries again. Program rule on recorded commands only; it does not
judge the images or generate any motion.
"""

DEFAULT_LIMIT = 3


def attempts(turns):
    """Executed press attempts, each with the observation right after its deepest forward move."""
    closed, found = False, []
    for turn in turns:
        ledger = turn['ledger']
        proposal, result = ledger.get('proposal') or {}, ledger.get('result') or {}
        if result.get('status') != 'completed':
            continue
        name, args = proposal.get('name'), proposal.get('arguments') or {}
        if name == 'propose_gripper':
            closed = args.get('key') == 'c'
            if closed:
                found.append(None)
        elif name == 'propose_move' and closed and args.get('xyz_mm', [0])[0] > 0 and found:
            found[-1] = ledger.get('after_observation_id')
    return [oid for oid in found if oid is not None]


def check(turns, proposal, limit=DEFAULT_LIMIT):
    """Raise ValueError (a repairable rejection) for a premature report_unable."""
    if proposal.get('name') != 'report_unable' or not limit:
        return
    made = attempts(turns)
    if not made or len(made) >= limit:
        return
    last = made[-1]
    raise ValueError(
        f'report_unable_rejected_press_attempts_remaining: {len(made)} of {limit} press attempts used. '
        'This report was NOT accepted and the run continues. A press that reaches the panel without lighting '
        'the button usually missed it. Look at the HEAD and WRIST images of observation '
        f'{last} (taken right after the deepest press of the last attempt): where did the closed end land '
        'relative to the target button (left/right, above/below, on the rim)? Plan the correction from that '
        'offset and the current images: withdraw to the close band if needed, open the gripper (o), correct '
        'Y/Z with the WRIST direction table so the button sits on the aim pixel, let a new range be measured, '
        'then close and press again. Return the next command now; state the observed offset in the reason.')
