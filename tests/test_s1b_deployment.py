"""Contract and protocol tests require neither CUDA nor the robot SDK."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from pressb.s1b_deployment.contract import read_config, verify, VERSION, OFFSETS
from pressb.s1b_deployment.protocol import encode,decode,validate_request,Application,ProtocolError
from pressb.s1b_deployment.policy import Policy,sdk_gripper
from pressb.s1b_deployment.recorder import Recorder
ROOT=Path(__file__).resolve().parents[1]

def request():
    return {k:np.asarray(v) for k,v in dict(protocol_version=VERSION,session_id='test',observation_id=np.int64(0),observation_monotonic_ns=np.int64(1),seed=np.int64(42),head=np.zeros((720,1280,3),np.uint8),right=np.zeros((360,640,3),np.uint8),state10=np.zeros(10,np.float32),source_time_ns=np.zeros(3,np.int64),mapped_time_ns=np.zeros(3,np.int64),model_sha256='model',data_contract_sha256='data').items()}

def test_config_execute_all_missing():
    c=read_config(ROOT/'configs/s1b_deployment_v1.yaml')
    assert c['mode']=='shadow'
    with pytest.raises(ValueError,match='sdk.control_way.*constraints.joint_position_lower'):
        read_config(ROOT/'configs/s1b_deployment_v1.yaml','execute')

def test_old_joint8_config(tmp_path):
    text=(ROOT/'configs/s1b_deployment_v1.yaml').read_text().replace('state_dim: 10','state_dim: 8')
    p=tmp_path/'config.yaml';p.write_text(text)
    with pytest.raises(ValueError,match='joint10'):read_config(p)

def test_protocol_roundtrip_and_offsets():
    f=decode(encode(request()));validate_request(f)
    assert OFFSETS.tolist()==[33333333,66666667,100000000,133333333,166666667,200000000,233333333]
    assert f['state10'].dtype==np.float32

@pytest.mark.parametrize('field,value',[('right',np.zeros((480,640,3),np.uint8)),('state10',np.zeros(8,np.float32)),('state10',np.full(10,np.nan,np.float32)),('state10',np.zeros(10,np.float64)),('head',np.zeros((720,1280,3),np.float32)),('seed',np.asarray(-1)),('session_id',np.asarray(''))])
def test_bad_requests(field,value):
    f=request();f[field]=value
    with pytest.raises(ProtocolError):validate_request(f)

def test_no_pickle_and_size_limit():
    out=io.BytesIO();np.savez(out,a=np.array([{}],dtype=object))
    with pytest.raises(ProtocolError):decode(out.getvalue())
    with pytest.raises(ProtocolError):decode(b'x'*(8*1024**2+1))

def test_application_rejects_duplicates_and_identity():
    policy=SimpleNamespace(identity=dict(model_sha256='model',data_contract_sha256='data'),predict=lambda *a:(np.zeros((7,10),np.float32),{}))
    app=Application(policy);f=request();assert app.predict(f)['action_raw'].shape==(7,10)
    with pytest.raises(ProtocolError,match='out_of_order'):app.predict(f)
    f['model_sha256']=np.asarray('other')
    with pytest.raises(ProtocolError,match='identity'):app.predict(f)

def test_continuous_feedback_and_gripper_threshold():
    policy=Policy.__new__(Policy)
    policy.stats=json.loads((ROOT/'models/s1b_real_joint10_8000_v1/data_contract.json').read_text())['normalization']
    raw=np.arange(10,dtype=np.float32);raw[7]=101
    n=policy.normalize_state(raw)
    np.testing.assert_allclose(n[0,0]*np.asarray(policy.stats['state']['scale'],np.float32)+np.asarray(policy.stats['state']['mean'],np.float32),raw,atol=1e-6)
    action=np.zeros((7,10),np.float32);action[:,7]=[-1,0,.4999,.5,.5001,1,2]
    assert sdk_gripper(action).tolist()==[0,0,0,100,100,100,100]
    raw[1]=np.inf
    with pytest.raises(ValueError):policy.normalize_state(raw)

def test_asset_identity_rejects_before_loading(tmp_path):
    run=tmp_path/'run';run.mkdir();(run/'x').write_text('changed')
    lock=tmp_path/'lock.json';lock.write_text(json.dumps(dict(files={'x':'wrong'},processor={})))
    with pytest.raises(ValueError,match='identity'):verify(run,2000,lock)
    with pytest.raises(ValueError,match='checkpoint'):verify(run,8,lock)

def test_recorder_errors_are_visible(tmp_path):
    r=Recorder(tmp_path/'log',{},{});r.append('state',state='SHADOW')
    with pytest.raises(ValueError,match='Missing'):r.append('prediction')
    with pytest.raises(ValueError):r.append('state',value=float('nan'))
    assert len((tmp_path/'log/events.jsonl').read_text().splitlines())==1

@pytest.mark.parametrize('mutation', ['weight', 'processor', 'contract'])
def test_reject_corrupt_assets(tmp_path, mutation):
    from pressb.s1b_deployment.contract import sha,tree
    run=tmp_path/'run';checkpoint=run/'checkpoints/step_002000';checkpoint.mkdir(parents=True)
    (run/'processor').mkdir();(run/'processor/config.json').write_text('{}')
    (run/'data_contract.json').write_text('{}');(checkpoint/'model.pt').write_bytes(b'weights')
    lock=dict(files={'data_contract.json':sha(run/'data_contract.json')},processor=tree(run/'processor'),models={'2000':sha(checkpoint/'model.pt')})
    (checkpoint/'manifest.json').write_text(json.dumps(dict(complete=True,step=2000,model_sha256=sha(checkpoint/'model.pt'))))
    path=tmp_path/'lock.json';path.write_text(json.dumps(lock))
    target={'weight':checkpoint/'model.pt','processor':run/'processor/config.json','contract':run/'data_contract.json'}[mutation]
    target.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='mismatch'):verify(run,2000,path)

def test_http_health_prediction_error_and_logging(tmp_path):
    import http.client
    import threading
    from pressb.s1b_deployment.server import make_server
    policy=SimpleNamespace(device='fake',identity=dict(step=2000,model_sha256='model',data_contract_sha256='data'),predict=lambda *a:(np.zeros((7,10),np.float32),dict(inference_seconds=0.)))
    recorder=Recorder(tmp_path/'http',{},policy.identity)
    server=make_server(policy,0,recorder);thread=threading.Thread(target=server.serve_forever);thread.start()
    client=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=2)
    try:
        client.request('GET','/health');resp=client.getresponse();assert resp.status==200;assert json.loads(resp.read())['ready']
        client.request('POST','/predict',encode(request()));resp=client.getresponse();assert resp.status==200
        result=decode(resp.read());assert result['session_id']=='test';assert result['action_raw'].shape==(7,10)
        client.request('POST','/predict',encode(request()));resp=client.getresponse();assert resp.status==400
        assert 'out_of_order' in json.loads(resp.read())['error']
        row=json.loads((tmp_path/'http/events.jsonl').read_text());assert row['sdk_target'] is None;assert row['gripper_binary']==[0]*7
    finally:
        client.close();server.shutdown();server.server_close();thread.join()

def test_strict_loader_rejects_missing_trained_key(tmp_path):
    import importlib.util
    torch=pytest.importorskip('torch')
    spec=importlib.util.spec_from_file_location('saved_piper_policy',ROOT/'models/s1b_real_joint10_8000_v1/runtime/starVLA/inference/piper_policy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=torch.nn.Module()
    for name in ('qwen_vl_interface','vj_encoder','vj_predictor','action_model'):model.add_module(name,torch.nn.Linear(2,2))
    state=model.state_dict();del state['action_model.bias']
    path=tmp_path/'model.pt';torch.save(state,path)
    with pytest.raises(RuntimeError,match='Missing key'):module.load_trained_state(model,path)

@pytest.mark.parametrize("image_mode", ["resize", "native"])
def test_camera_preprocess_matches_training_without_head_override(image_mode):
    from PIL import Image
    policy=Policy.__new__(Policy)
    policy.stats=json.loads((ROOT/'models/s1b_real_joint10_8000_v1/data_contract.json').read_text())['normalization']
    policy.image_mode=image_mode;policy.resolution=224
    rng=np.random.default_rng(42)
    head=rng.integers(0,256,(720,1280,3),np.uint8)
    right=rng.integers(0,256,(360,640,3),np.uint8)
    state=np.zeros(10,np.float32)
    expected=np.arange(70,dtype=np.float32).reshape(7,10)
    def predict(images, normalized, seed):
        for image,raw in zip(images,(head,right)):
            np.testing.assert_array_equal(np.asarray(image),(raw if image_mode == "native" else np.asarray(Image.fromarray(raw).resize((224,224),Image.Resampling.BILINEAR))))
        assert normalized.shape==(1,1,10);assert seed==42
        return expected,0.
    policy.predict_prepared=predict
    action,_=policy.predict(head,right,state)
    np.testing.assert_array_equal(action,expected)


def test_mixed_config_has_distinct_models_and_task():
    from pressb.s1b_deployment.contract import MIXED_TASK, TASK
    mixed = read_config(ROOT/'configs/s1b_deployment_mixed_v2.yaml')
    old = read_config(ROOT/'configs/s1b_deployment_v1.yaml')
    identity = json.loads(Path(mixed['model']['identity']).read_text())
    assert mixed['model']['task'] == identity['task'] == MIXED_TASK
    assert old['model']['task'] == TASK
    assert mixed['model']['run'] != old['model']['run']
    assert set(identity['models']) == {'4500', '8000'}
    old_identity = json.loads((ROOT/old['model']['identity']).read_text())
    assert identity['models']['8000'] != old_identity['models']['8000']
