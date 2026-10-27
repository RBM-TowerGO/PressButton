"""Reuse vendor drivers by actual streams, never solely by installation path."""
import json
from pathlib import Path
import numpy as np
import pytest
from pressb.s1b_deployment.cameras import CameraReadiness,missing_camera_drivers

ROOT=Path(__file__).resolve().parents[1]
MAPPING=json.loads((ROOT/'configs/s1b_observation_v1.json').read_text())


def test_vendor_compressor_is_reused_and_only_missing_acquisition_is_started():
    existing=[dict(pid=123,path='/opt/astribot_ros/software/astribot_camera/lib/astribot_camera/astribot_camera_sub')]
    assert missing_camera_drivers(existing,False)==[('astribot_camera_main','camera')]
    assert missing_camera_drivers(existing,True)==[]


def test_live_streams_from_other_camera_provider_never_start_duplicate_drivers():
    assert missing_camera_drivers([],True)==[]
    assert len(missing_camera_drivers([],False))==2


def test_duplicate_known_driver_instances_are_not_silently_overwritten():
    processes=[dict(pid=i,path=root+'/astribot_camera_sub') for i,root in enumerate(('/opt/vendor','/private'))]
    with pytest.raises(ValueError,match='multiple_camera_drivers'):missing_camera_drivers(processes,True)


@pytest.fixture
def probe():
    cv2=pytest.importorskip('cv2')
    jpeg={}
    for key,shape in [('head',(720,1280,3)),('right',(360,640,3))]:
        ok,data=cv2.imencode('.jpg',np.zeros(shape,np.uint8));assert ok
        jpeg[key]=data.tobytes()
    return CameraReadiness(MAPPING,10**18,10**9),jpeg


def feed(p,jpeg,key,i,**kw):
    dt=i*33_333_333
    args=dict(source_ns=10**18+dt,receipt_ns=10**9+dt+10_000_000,
              wall_ns=10**18+dt+10_000_000,format='jpeg',payload=jpeg[key])
    args.update(kw);p.receive(key,**args)
    return args['receipt_ns']


def test_both_streams_require_real_decodable_successive_fresh_frames(probe):
    p,jpeg=probe
    for i in range(1,4):now=feed(p,jpeg,'head',i)
    assert not p.report(now)['ready']
    for i in range(1,4):now=feed(p,jpeg,'right',i)
    assert p.report(now)['ready']
    assert p.report(now)['streams']['head']['shape']==[720,1280,3]
    assert not p.report(now+80_000_001)['ready']


@pytest.mark.parametrize('fault',['format','jpeg','shape','timestamp','repeat'])
def test_incompatible_camera_frames_are_rejected_before_sdk_start(probe,fault):
    p,jpeg=probe
    for i in range(1,4):
        for key in jpeg:now=feed(p,jpeg,key,i)
    assert p.report(now)['ready']
    changes={
        'format':dict(format='png'),
        'jpeg':dict(payload=b'broken'),
        'shape':dict(payload=jpeg['right']),
        'timestamp':dict(source_ns=1),
        'repeat':dict(source_ns=10**18+3*33_333_333),
    }
    now=feed(p,jpeg,'head',4,**changes[fault])
    assert not p.report(now)['ready'] and p.report(now)['streams']['head']['error']
