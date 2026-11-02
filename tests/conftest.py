"""Explicit visual evidence fixtures for tests of the pre-existing control flow."""
import copy

import pytest


@pytest.fixture
def approved_forward_view(monkeypatch):
    """Isolate unrelated tests; guard rejection/HTTP enforcement have separate tests.

    Not autouse: tests must explicitly request an approved CURRENT wrist judgment.
    """
    from pressb.agent_loop import forward_guard as guard

    def assessment(oid):
        return dict(format=guard.FORMAT, source=guard.SOURCE, observation_id=oid,
                    **{key: 'yes' for key in guard.FIELDS})

    monkeypatch.setattr(guard, 'inspect', lambda base, body, oid, directory, cancel=None: assessment(oid))

    def permission(proposal, oid):
        value = guard.evaluate(assessment(oid), oid)
        value['authorized_command'] = dict(type='move', xyz_mm=copy.deepcopy(proposal['arguments']['xyz_mm']))
        return value

    return permission
