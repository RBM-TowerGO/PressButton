import numpy as np
import pytest
from pressb.s1b_deployment.observation import Sample,TimeMapper,ObservationBuffer,ObservationError

def sample(key,stamp=1_000_000_000,seq=1,receipt=None):
    shapes={'head':(720,1280,3),'right':(360,640,3),'arm':(7,),'gripper':(1,),'head_state':(2,)}
    value=np.zeros(shapes[key],np.uint8 if key in ('head','right') else np.float32)
    return Sample(seq,stamp,stamp,stamp if receipt is None else receipt,{},value)

def populated(capacity=8):
    b=ObservationBuffer(capacity=capacity)
    for k in b.buffers:b.add(k,sample(k))
    return b

def test_epoch_mapping_and_clock_jump_latches():
    mapper=TimeMapper(1_800_000_000_000_000_000,10_000_000_000)
    assert mapper.map(1_800_000_000_000_000_000,1_800_000_000_001_000_000,10_001_000_000)==10_000_000_000
    with pytest.raises(ObservationError,match='unmapped'):mapper.map(100,1_800_000_000_001_000_000,10_001_000_000)
    with pytest.raises(ObservationError,match='clock_jump'):mapper.map(1_800_000_000_000_000_000,1_800_000_000_001_000_000,11_001_000_000)
    assert mapper.failed

def test_nearest_historical_state_and_continuous_gripper():
    b=populated();s=sample('gripper',1_001_000_000,2);s.value[0]=101;b.add('gripper',s)
    head=sample('head',1_002_000_000,2);b.add('head',head)
    selected,state=b.snapshot(1_003_000_000)
    assert state.shape==(10,) and state.dtype==np.float32 and state[7]==101
    assert selected['gripper'].sequence==2
    with pytest.raises(ObservationError,match='no_new_head'):b.snapshot(1_003_000_000,2)

@pytest.mark.parametrize('key,delta,reason',[('right',41_000_000,'skew_right'),('arm',11_000_000,'skew_arm'),('gripper',11_000_000,'skew_gripper'),('head_state',11_000_000,'skew_head_state')])
def test_skew_rejected(key,delta,reason):
    b=ObservationBuffer()
    for k in b.buffers:b.add(k,sample(k,1_000_000_000+delta if k==key else 1_000_000_000))
    with pytest.raises(ObservationError,match=reason):b.snapshot(1_050_000_000)

def test_missing_stale_nonfinite_and_old_packets():
    b=ObservationBuffer()
    with pytest.raises(ObservationError,match='missing'):b.snapshot(1)
    b=populated()
    with pytest.raises(ObservationError,match='stale_head'):b.snapshot(1_081_000_000)
    with pytest.raises(ObservationError,match='nonmonotonic'):b.add('arm',sample('arm'))
    bad=sample('arm',1_001_000_000,2);bad.value[0]=np.nan
    with pytest.raises(ObservationError,match='bad_arm'):b.add('arm',bad)

def test_bounded_buffer_and_latest_head():
    b=populated(capacity=2)
    for i in range(2,10):
        for k in b.buffers:b.add(k,sample(k,1_000_000_000+i*1_000_000,i))
    s,_=b.snapshot(1_010_000_000)
    assert s['head'].sequence==9
    assert all(x==2 for x in b.stats()['occupancy'].values())
    assert all(x==7 for x in b.stats()['overwrites'].values())

def test_bridge_execute_fails_before_loading_ros(tmp_path):
    import subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts/s1b_robot_bridge.py'),'--config',str(root/'configs/s1b_deployment_v1.yaml'),'--mode','execute','--output',str(tmp_path/'unused')],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Unverified execute fields' in result.stderr
    assert 'rclpy' not in result.stderr
    assert not (tmp_path/'unused').exists()

def test_analyzer_verifies_log_coverage_and_observation_clock(tmp_path):
    import importlib.util,json
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('s1b_p2_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_s1b_p2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    summary=dict(callbacks={'head':2},valid_requests=2)
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    sources=[dict(stream='head',receipt_ns=1_000_000_000+i*33_333_333,wall_ns=10_000_000_000+i*33_333_333,header={'sec':10,'nanosec':i*33_333_333},valid=True,source_to_receipt_ns=50_000_000) for i in range(2)]
    (tmp_path/'sources.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sources))
    rows=[dict(valid=True,observation_monotonic_ns=1_000_000_000+i*100_000_000,received_ns=1_100_000_000+i*100_000_000,rtt_ns=90_000_000,source_to_response_ns=150_000_000,send_ns=1_010_000_000+i*100_000_000,inference_seconds=.045,preprocess_seconds=.003) for i in range(2)]
    (tmp_path/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=module.analyze(tmp_path)
    assert result['valid_request_span_seconds']==.1
    assert result['sources']['head']['header_strictly_monotonic']
    assert result['responses_with_any_future_point_fraction']==1
    summary['callbacks']['head']=3;(tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='coverage'):module.analyze(tmp_path)

def test_camera_runtime_isolates_config_and_preserves_vendor_slots(tmp_path):
    import importlib.util,json
    from pathlib import Path
    import yaml
    spec=importlib.util.spec_from_file_location('prepare_cameras',Path(__file__).resolve().parents[1]/'scripts/prepare_s1b_cameras.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'vendor';config=source/'config/camera_config';config.mkdir(parents=True)
    original={'camera_config_list':[{'id':1,'location':'head_stereo_color'},{'id':6,'location':'right_wrist_rgbd_color'},{'id':9,'location':'head_rgbd'}]}
    path=config/'camera_config.yaml';path.write_text(yaml.safe_dump(original))
    (config/'head.yaml').write_text('CameraType: 3\n')
    binary=source/'lib/astribot_camera';binary.mkdir(parents=True)
    for name in ('astribot_camera_main','astribot_camera_sub'):(binary/name).write_bytes(b'original binary')
    output=tmp_path/'private';module.prepare(source,output)
    assert yaml.safe_load(path.read_text())==original
    assert len(yaml.safe_load((output/'config/camera_config/camera_config.yaml').read_text())['camera_config_list'])==2
    assert (output/'config/camera_config/head.yaml').read_text()=='CameraType: 3\n'
    assert (output/'lib/astribot_camera/astribot_camera_main').read_bytes()==b'original binary'
    with pytest.raises(ValueError,match='must not exist'):module.prepare(source,output)
