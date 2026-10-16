import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import pytest
from pressb.astribot_collection.render_controls import setting_matches


def test_render_control_accepts_only_exact_float64_or_float32_representation():
    assert setting_matches(.02,.02)
    assert setting_matches(struct.unpack('f',struct.pack('f',.02))[0],.02)
    for actual in (.019999,.020001,.02000001,True,None):
        assert not setting_matches(actual,.02)
    assert setting_matches(False,False)
    assert not setting_matches(0,False)
    assert setting_matches(2,2)
    assert not setting_matches(3,2)


@pytest.mark.parametrize('fault',['missing','running','error'])
def test_pipeline_rejects_zero_exit_without_finished_result(tmp_path,fault):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_astribot_collection_pipeline import Pipeline
    pipeline=Pipeline.__new__(Pipeline)
    pipeline.output=tmp_path
    pipeline.args=SimpleNamespace(render_layout='homogeneous')
    def command(args,log):
        destination=tmp_path/'worker';destination.mkdir()
        if fault=='running': (destination/'worker_result.json').write_text(json.dumps(dict(status='running')))
        if fault=='error': (destination/'worker_error.json').write_text(json.dumps(dict(error='renderer rejected settings')))
    pipeline.command=command
    with pytest.raises(RuntimeError,match='failed|without a finished result'):
        pipeline.worker('worker',tmp_path/'jobs.json',0,1,'cpu')
