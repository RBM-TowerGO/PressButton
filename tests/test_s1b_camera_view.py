import copy
import json
import threading
import time
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
import pytest
from s1b_camera_view import CenterDepth, Frames, project, rays, serve


def calibration():
    i=dict(width=20,height=16,fx=20.,fy=20.,ppx=10.,ppy=8.,model=0,coeffs=[0.]*5)
    return dict(serial='test',depth=copy.deepcopy(i),color=copy.deepcopy(i),scale_m=.0001,
                rotation_column_major=np.eye(3).ravel(order='F').tolist(),translation_m=[0.,0.,0.])


def test_device_units_and_center_holes():
    m=calibration();measure=CenterDepth(m)
    values=np.full((16,20),1234,dtype='<u2')
    result=measure.measure(values.tobytes(),(16,20,3))
    assert result['center_m']==pytest.approx(.1234)
    assert result['median_m']==pytest.approx(.1234)
    assert result['valid_pixels']==25
    values[8,10]=0
    result=measure.measure(values.tobytes(),(16,20,3))
    assert result['center_m'] is None
    assert result['valid_pixels']==24
    values[:]=0
    assert measure.measure(values.tobytes(),(16,20,3))['median_m'] is None
    with pytest.raises(ValueError,match='size_mismatch'):measure.measure(b'123',(16,20,3))
    with pytest.raises(ValueError,match='resolution'):measure.measure(values.tobytes(),(32,40,3))


@pytest.mark.parametrize('model',[0,2,4])
def test_projection_deprojection_with_factory_distortion(model):
    i=calibration()['color'];i['model']=model
    if model:i['coeffs']=[-.05,.02,.001,.001,-.01]
    points=rays(i)*.25
    pixels=project(points,i)
    y,x=np.indices((16,20))
    np.testing.assert_allclose(pixels,np.column_stack([x.ravel(),y.ravel()]),atol=.002)


def test_color_depth_extrinsic_alignment_changes_center_sample():
    m=calibration();m['translation_m']=[.01,0,0] # 1 pixel shift at z=.2 and fx=20
    image=np.full((16,20),2000,dtype='<u2');image[8,10]=0
    result=CenterDepth(m).measure(image.tobytes(),(16,20,3))
    assert result['center_m']==pytest.approx(.2) # comes from depth pixel x=9, not raw centre


def message(payload,stamp,fmt='jpeg'):
    return SimpleNamespace(data=payload,format=fmt,header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp//10**9,nanosec=stamp%10**9)))


def jpeg():
    import cv2
    ok,data=cv2.imencode('.jpg',np.zeros((16,20,3),np.uint8));assert ok
    return data.tobytes()


def test_timestamp_pairing_stale_and_live_rgb_when_depth_stops():
    frames=Frames();frames.measurement=CenterDepth(calibration());frames.measurement_error=None
    now=time.time_ns();image=jpeg()
    frames.receive('right',message(image,now));frames.receive('head',message(image,now))
    frames.receive('depth',message(np.full((16,20),1000,dtype='<u2').tobytes(),now+10_000_000,'raw'))
    frames.update()
    snapshot=frames.snapshot()
    assert snapshot['depth']['median_m']==pytest.approx(.1)
    assert snapshot['depth']['skew_ms']==10
    frames.depth_receipt_ns=time.monotonic_ns()-2_000_000_000
    frames.receive('right',message(image,now+2_000_000_000))
    snapshot=frames.snapshot()
    assert snapshot['depth']['median_m'] is None
    assert snapshot['frames']['right']['source_ns']==now+2_000_000_000
    assert not snapshot['frames']['right']['stale']


def test_unmatched_frames_never_produce_distance():
    frames=Frames();frames.measurement=CenterDepth(calibration());frames.measurement_error=None
    now=time.time_ns()
    frames.receive('right',message(jpeg(),now))
    frames.receive('depth',message(np.full((16,20),1000,dtype='<u2').tobytes(),now+200_000_000,'raw'))
    frames.update()
    assert frames.snapshot()['depth']['median_m'] is None
    assert '时间差' in frames.snapshot()['depth']['status']


def test_web_has_center_overlay_and_readonly_json():
    frames=Frames();server,thread=serve(frames,0)
    try:
        base='http://127.0.0.1:'+str(server.server_port)
        with urlopen(base) as response:
            html=response.read().decode()
            assert 'ctx.strokeRect(x-2,y-2,5,5)' in html
            assert '选定像素' in html
            assert '保存测距点' in html
        with urlopen(base+'/api/snapshot') as response:
            assert json.load(response)['depth']['median_m'] is None
    finally:
        server.shutdown();server.server_close();thread.join(2)


def test_viewer_health_checks_own_process_identity():
    import os
    from s1b_keyboard_direct import wait_for_viewer
    frames=Frames();server,thread=serve(frames,0)
    try:
        child=SimpleNamespace(poll=lambda:None,pid=os.getpid())
        wait_for_viewer(child,server.server_port,timeout=1.)
        child.pid=-1
        with pytest.raises(RuntimeError,match='超时'):
            wait_for_viewer(child,server.server_port,timeout=.01)
    finally:
        server.shutdown();server.server_close();thread.join(2)


def test_selected_point_uses_selected_depth_and_clips_window():
    from s1b_camera_view import point_pixel
    values=np.full((16,20),3000,dtype='<u2');values[:3,:3]=1000
    depth=CenterDepth(calibration())
    selected=depth.measure(values.tobytes(),(16,20,3),[0,0])
    assert selected['center_px']==[0,0]
    assert selected['center_m']==pytest.approx(.1)
    assert selected['median_m']==pytest.approx(.1)
    assert selected['window_pixels']==9
    assert depth.measure(values.tobytes(),(16,20,3))['center_m']==pytest.approx(.3)
    assert point_pixel([1,1],640,360)==(639,359)
    assert point_pixel(None,640,360)==(320,180)


def test_selection_save_reload_and_unsaved_changes(tmp_path):
    path=tmp_path/'persistent'/'point.json'
    frames=Frames(path)
    selection=frames.select_point([.25,.75])
    assert not selection['saved'] and not path.exists()
    assert frames.save_point(selection['revision'])['saved']
    loaded=Frames(path)
    assert loaded.point==[.25,.75]
    assert loaded.snapshot()['selection']['saved']
    loaded.select_point([.1,.1])
    assert Frames(path).point==[.25,.75] # live selection is not auto-saved
    with pytest.raises(ValueError,match='已变更'):loaded.save_point(0)
    reset=loaded.select_point(None);loaded.save_point(reset['revision'])
    assert Frames(path).point is None


@pytest.mark.parametrize('point',[[float('nan'),0],[1.1,0],[False,0],[-.1,.5],[1],{},'center'])
def test_bad_selection_does_not_change_current_point(point):
    frames=Frames();frames.select_point([.2,.3])
    with pytest.raises((ValueError,TypeError)):frames.select_point(point)
    assert frames.point==[.2,.3]


def test_invalid_saved_config_reports_error(tmp_path):
    path=tmp_path/'point.json';path.write_text('{')
    frames=Frames(path)
    assert frames.point is None and frames.config_error
    selected=frames.select_point([.1,.2]);frames.save_point(selected['revision'])
    assert frames.config_error is None
    assert Frames(path).point==[.1,.2]


def test_point_change_discards_inflight_old_distance():
    frames=Frames();now=time.time_ns()
    class ChangingMeasurement:
        def measure(self,*args):
            frames.select_point([.1,.2])
            return dict(center_m=1,median_m=1)
    frames.measurement=ChangingMeasurement()
    frames.receive('right',message(jpeg(),now))
    frames.receive('depth',message(np.zeros((16,20),dtype='<u2').tobytes(),now,'raw'))
    frames.update()
    assert frames.result is None
    assert frames.snapshot()['depth']['median_m'] is None


def test_http_select_and_save_and_cross_origin_rejected(tmp_path):
    from urllib.request import Request
    from urllib.error import HTTPError
    frames=Frames(tmp_path/'point.json');server,thread=serve(frames,0)
    def post(path,body,origin=None):
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        req=Request('http://127.0.0.1:'+str(server.server_port)+path,
                    data=json.dumps(body).encode(),headers=headers)
        with urlopen(req) as response:return json.load(response)
    try:
        selection=post('/api/selection',dict(point_normalized=[.4,.6]))
        assert not selection['saved']
        assert post('/api/selection/save',dict(revision=selection['revision']))['saved']
        assert Frames(tmp_path/'point.json').point==[.4,.6]
        with pytest.raises(HTTPError) as exc:
            post('/api/selection',dict(point_normalized=[0,0]),origin='http://example.com')
        assert exc.value.code==403
    finally:
        server.shutdown();server.server_close();thread.join(2)


def test_atomic_agent_capture_contains_matching_raw_depth():
    import base64
    frames=Frames();frames.measurement=CenterDepth(calibration());frames.measurement_error=None
    now=time.time_ns();raw=np.full((16,20),1800,dtype='<u2').tobytes()
    frames.receive('right',message(jpeg(),now));frames.receive('head',message(jpeg(),now))
    frames.receive('depth',message(raw,now,'raw'));frames.update()
    bundle=frames.capture()
    assert base64.b64decode(bundle['frames']['depth']['data_base64'])==raw
    assert bundle['depth']['median_m']==pytest.approx(.18)
    frames.select_point([.2,.3])
    with pytest.raises(ValueError,match='not_ready'):frames.capture()


def test_agent_capture_matches_head_history_instead_of_latest_frame():
    frames=Frames();frames.measurement=CenterDepth(calibration());frames.measurement_error=None
    stamp=time.time_ns()-200_000_000
    frames.receive('right',message(jpeg(),stamp))
    frames.receive('depth',message(np.full((16,20),1800,dtype='<u2').tobytes(),stamp,'raw'))
    frames.receive('head',message(jpeg(),stamp+10_000_000))
    frames.update()
    frames.receive('head',message(jpeg(),stamp+200_000_000))
    bundle=frames.capture()
    assert bundle['frames']['head']['source_ns']==stamp+10_000_000
    assert frames.latest['head']['source_ns']==stamp+200_000_000


def test_fingertip_point_save_reload_and_validation(tmp_path):
    import cv2
    path=tmp_path/'tip.json'
    frames=Frames(None,path)
    ok,jpeg=cv2.imencode('.jpg',np.zeros((360,640,3),np.uint8))
    frames.latest['right']=dict(data=jpeg.tobytes(),source_ns=1,receipt_ns=1)
    with pytest.raises(ValueError):frames.select_fingertip('half',[.5,.5])
    with pytest.raises(ValueError):frames.select_fingertip('closed',[1.5,.5])
    info=frames.select_fingertip('closed',[0.5,0.75])
    assert not info['saved']['closed']
    with pytest.raises(ValueError):frames.save_fingertip('closed',info['revision']+1)
    saved=frames.save_fingertip('closed',info['revision'])
    assert saved['saved']['closed']
    data=json.loads(path.read_text())
    assert data['points']['closed']['image_size']==[640,360]
    assert (tmp_path/'tip_closed.jpg').exists()
    again=Frames(None,path)
    assert again.fingertip.points['closed']==[0.5,0.75] and again.fingertip.points['open'] is None
    # saving the other state keeps the first one
    i2=again.select_fingertip('open',[0.2,0.3]);again.latest['right']=frames.latest['right']
    again.save_fingertip('open',i2['revision'])
    assert set(json.loads(path.read_text())['points'])=={'closed','open'}
    assert 'fingertip' in again.snapshot()


def test_web_fingertip_script_is_valid_js(tmp_path):
    import re,shutil,subprocess
    from s1b_camera_view import HTML
    if shutil.which('node') is None:pytest.skip('node missing')
    js=re.search(r'<script>(.*)</script>',HTML,re.S).group(1)
    f=tmp_path/'a.js';f.write_text(js)
    assert subprocess.run(['node','--check',str(f)]).returncode==0
