"""CPU-only contract, leakage, numerical, artifact and real-socket regression tests."""
import json
from pathlib import Path
import numpy as np
import pytest
from pressb.openpi_s1b.transforms import Inputs, Outputs, fit_stats
from pressb.openpi_s1b.dataset import Dataset
from pressb.openpi_s1b.artifacts import prepare, seal_training, export, verify, write_json
from pressb.s1b_deployment.contract import JOINTS, MIXED_TASK, validate_checkpoint, read_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(tmp_path):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    root=tmp_path/'data'
    (root/'meta/episodes/chunk-000').mkdir(parents=True)
    (root/'data/chunk-000').mkdir(parents=True)
    c=dict(
        names=JOINTS, arm_head_units='rad', action_gripper=dict(encoding='binary',open=0,closed=1),
        time=dict(fps=30,horizon=7,target_offsets_steps=list(range(1,8)),
            single_step_storage='action[k] = command selected for grid[k+1]; omit last grid row without next target',
            training_window='action[k:k+7], no additional one-step shift; require all seven valid'), task=MIXED_TASK)
    write_json(root/'meta/data_contract.json',c)
    write_json(root/'meta/info.json',dict(codebase_version='v3.0',fps=30,total_frames=20,
        features={k:dict(shape=[10], names=JOINTS) for k in ('action','observation.state')}))
    write_json(root/'meta/splits.json',dict(train=[0],validation=[1]))
    write_json(root/'meta/training_anchors.json',dict(train=[0,1,2,3],validation=[10,11,12,13]))
    rows=[]
    for i in range(20):
        state=[float(i)]*10
        action=[float(i+1)]*10;action[7]=i%2
        rows.append(dict(index=i,episode_index=i//10,segment_id=i//10,valid_anchor=int(i%10<4),
            valid_grid_row=1,observation_grid_time=i/30,target_grid_time=(i+1)/30,
            **{'observation.state':state,'action':action}))
    pq.write_table(pa.Table.from_pylist(rows), root/'data/chunk-000/file-000.parquet')
    pq.write_table(pa.Table.from_pylist([dict(episode_index=i,dataset_from_index=i*10,dataset_to_index=(i+1)*10,
        length=10,tasks=[MIXED_TASK]) for i in range(2)]),root/'meta/episodes/chunk-000/file-000.parquet')
    return root


def test_joint_timing_and_train_only_statistics(dataset):
    ds=Dataset(dataset)
    np.testing.assert_array_equal(ds.actions[ds.anchors[0]:ds.anchors[0]+7,0], np.arange(1,8))
    stats=ds.statistics()
    assert stats['state']['center'][0] == pytest.approx(1.5)
    assert stats['actions']['center'][0] == pytest.approx(5.5)
    assert stats['actions']['center'][7] == 0 and stats['actions']['scale'][7] == 1
    with pytest.raises(ValueError,match='train split'):Dataset(dataset,'validation').statistics()


@pytest.mark.parametrize('bad', ['split','order','missing','anchor','gap','time'])
def test_reject_bad_data_contract(dataset,bad):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if bad=='split':write_json(dataset/'meta/splits.json',dict(train=[0,1],validation=[1]))
    elif bad=='order':
        p=dataset/'meta/data_contract.json';c=json.loads(p.read_text());c['names']=list(reversed(JOINTS));write_json(p,c)
    elif bad=='missing':(dataset/'meta/training_anchors.json').unlink()
    elif bad=='anchor':write_json(dataset/'meta/training_anchors.json',dict(train=[0,1,2,4],validation=[10,11,12,13]))
    else:
        p=dataset/'data/chunk-000/file-000.parquet';rows=pq.read_table(p).to_pylist()
        if bad=='gap':rows[6]['segment_id']=999
        else:rows[6]['target_grid_time']+=1/30
        pq.write_table(pa.Table.from_pylist(rows),p)
    with pytest.raises((ValueError,FileNotFoundError)):Dataset(dataset)


def test_shared_transforms_roundtrip_and_masks():
    rng=np.random.default_rng(42)
    state=rng.normal(size=(20,10)).astype(np.float32)
    actions=rng.normal(size=(20,10)).astype(np.float32);actions[:,7]=np.arange(20)%2
    state[:,9]=.25;actions[:,9]=.5
    stats=fit_stats(state,actions)
    row=dict(head=np.full((720,1280,3),10,np.uint8),right=np.full((360,640,3),20,np.uint8),
             state=state[0],actions=actions[:7],prompt=MIXED_TASK)
    mapped=Inputs(stats)(row)
    assert mapped['image_mask']==dict(base_0_rgb=True,left_wrist_0_rgb=False,right_wrist_0_rgb=True)
    assert mapped['image']['base_0_rgb'][0,0,0]==10 and mapped['image']['right_wrist_0_rgb'][0,0,0]==20
    assert not mapped['image']['left_wrist_0_rgb'].any()
    np.testing.assert_allclose(mapped['state']*stats['state']['scale']+stats['state']['center'],state[0],atol=1e-6)
    padded=np.pad(mapped['actions'],((0,0),(0,22)))
    raw=Outputs(stats)({'actions':padded})['actions']
    np.testing.assert_allclose(raw,actions[:7],atol=1e-6)
    np.testing.assert_array_equal(raw[:,7],actions[:7,7])
    for value in (np.zeros((50,32)),np.full((7,32),np.nan)):
        with pytest.raises(ValueError):Outputs(stats)({'actions':value})
    row['right']=np.zeros((3,360,640),np.uint8)
    with pytest.raises(ValueError):Inputs(stats)(row)


def test_artifact_export_tamper_and_backend_rejection(dataset,tmp_path):
    bundle=tmp_path/'bundle';prepare(Dataset(dataset),bundle)
    ckpt=tmp_path/'123';ckpt.mkdir();(ckpt/'model.safetensors').write_bytes(b'unit-test-not-a-real-model')
    seal_training(ckpt,bundle,123,dict(exp_name='test'))
    out=tmp_path/'export';identity=export(ckpt,out)
    assert verify(out,123,out/'identity.json')['step']==123
    assert validate_checkpoint(identity,123,backend='openpi_pi05')=='openpi_pi05'
    with pytest.raises(ValueError):validate_checkpoint(identity,124)
    with pytest.raises(ValueError):validate_checkpoint(identity,123,backend='vlajepa')
    with pytest.raises(ValueError):validate_checkpoint(dict(backend='test_only',models={'123':'x'}),123)
    with pytest.raises(ValueError):validate_checkpoint(dict(models={'123':'x'}),123)
    (out/'normalization.json').write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):verify(out,123,out/'identity.json')


def test_new_template_and_old_default():
    old=read_config(ROOT/'configs/s1b_deployment_v1.yaml')
    new=read_config(ROOT/'configs/s1b_deployment_openpi_pi05.yaml')
    assert old['model'].get('backend','vlajepa')=='vlajepa'
    assert new['model']['backend']=='openpi_pi05'
    assert new['model']['horizon']==7


def test_offline_http_to_synchronous_control():
    from check_openpi_s1b_offline import run
    result=run()
    assert result['shadow_steps']==7 and result['http_npz_roundtrip']
    assert not result['sdk_imported'] and not result['model_inference_verified']


@pytest.mark.parametrize('arches,accepted', [(['sm_80','sm_90'],False), (['sm_90','sm_120'],True)])
def test_blackwell_build_preflight(arches,accepted):
    from types import SimpleNamespace
    from pressb.openpi_s1b.runtime import check_cuda_build
    torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,
        get_device_capability=lambda device:(12,0),get_arch_list=lambda:arches))
    if accepted:check_cuda_build(torch,'cuda:0')
    else:
        with pytest.raises(ValueError,match='Blackwell'):check_cuda_build(torch,'cuda:0')
