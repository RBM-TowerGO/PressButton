"""User-selected warning checks preserve execution and normal shutdown."""
import time
import pytest
from pressb.s1b_deployment.executor import ExecutionError
from pressb.s1b_deployment.live_execution import validate_execution_config
from test_s1b_startup import live_backend,session,response


def start(core):
    core.request_start(response(0));core.confirm_start();core.start()


def test_torso_drift_warns_without_faulting_or_stopping_model_commands():
    live,events=live_backend();core=session(live,events)
    try:
        start(core);live.monitor.drift=.01
        for oid in range(1,8):
            assert core.accept(response(oid))['accepted']
            time.sleep(.02)
        assert live.active and live.model_commands>10 and live.backend.stops==0
        assert core.health()['ready'] and not core.health()['fault']
        warnings=[r for r in events if r['event']=='warning']
        assert len(warnings)==1  # One report per second, not one per 250 Hz tick.
        assert warnings[0]['part']=='astribot_torso' and warnings[0]['limit']==.003
        summary=core.stats()
        assert not summary['error'] and not summary['runner_error']
        assert summary['warnings'][0]['occurrences']>1
        assert summary['warnings'][0]['max_drift']==.01
    finally:core.stop();core.close()


@pytest.mark.parametrize('action',['warn','stop'])
def test_unsettled_feedback_warn_mode_exits_without_raising_and_releases_backend(action):
    live,events=live_backend();live.c['stop_feedback_action']=action
    live.c['max_stop_delay_ns']=40_000_000
    snapshot=live.monitor.snapshot
    def moving():
        rows=snapshot()
        for row in rows.values():row['velocity']=[100.]*len(row['position'])
        return rows
    live.monitor.snapshot=moving
    core=session(live,events);start(core);core.stop()
    if action=='stop':
        with pytest.raises(ExecutionError,match='stop_did_not_settle'):core.close()
        assert not core.stats()['warnings']
    else:
        core.close()
        assert not core.error and not core.runner.error
        assert core.stats()['warnings'][0]['check']=='stop_did_not_settle_within_budget'
        assert core.stats()['stop_feedback']['settled'] is False
        assert core.stats()['stop_settled_ns'] is None
    assert live.backend.stops>0 and live.backend.released and live.monitor.closed


def test_warning_configuration_rejects_typo():
    live,_=live_backend()
    try:
        for key in ('uncontrolled_drift_action','stop_feedback_action'):
            c=dict(live.c);c[key]='warning_typo'
            with pytest.raises(ExecutionError,match=key):validate_execution_config(c)
    finally:live.close()
