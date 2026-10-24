"""Retain valid partial-episode data without hiding new infrastructure errors."""
import json
import pytest
from resume_astribot_b_then_a_job import acknowledged_episodes, sha


def test_only_reviewed_interruption_is_accepted(tmp_path):
    p = tmp_path / 'episodes/episode_000254/report.json'
    p.parent.mkdir(parents=True)
    row = dict(status='infrastructure_error', reason='infrastructure_error', transitions=83,
               control_steps=588, physics_steps=2353, unexpected_contacts=0, timings={'inference_s': 5.7})
    p.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match='Unreviewed'):
        acknowledged_episodes(tmp_path, {})
    ack = {str(p.relative_to(tmp_path)): dict(sha256=sha(p), transitions=83)}
    result = acknowledged_episodes(tmp_path, ack)
    assert result['actual_transitions'] == 83 and result['infrastructure_interrupted_episodes'] == 1
    row['transitions'] = 84
    p.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match='Unreviewed'):
        acknowledged_episodes(tmp_path, ack)
