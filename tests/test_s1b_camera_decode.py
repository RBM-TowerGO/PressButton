"""Deferred camera work must not block or re-date feedback observations."""
import threading
import numpy as np
import pytest
from pressb.s1b_deployment.observation import JpegFrame, ObservationError, selected_rgb
from test_s1b_observation import populated, sample


def test_only_selected_frames_decode_and_feedback_can_arrive(monkeypatch):
    buffer=populated()
    for seq in (2,3):
        for key in buffer.buffers:
            row=sample(key,1_000_000_000+seq*1_000_000,seq)
            if key in ('head','right'):row.value=JpegFrame(bytes([seq]))
            buffer.add(key,row)
    selected,_=buffer.snapshot(1_004_000_000)
    entered=threading.Event();release=threading.Event();calls=[];errors=[]
    def decode(frame,key):
        calls.append((key,frame.data));entered.set()
        assert release.wait(2)
        return sample(key).value
    monkeypatch.setattr(JpegFrame,'rgb',decode)
    def materialize():
        try:selected_rgb(selected)
        except BaseException as exc:errors.append(exc)
    worker=threading.Thread(target=materialize);worker.start()
    try:
        assert entered.wait(1)
        # A slow decoder holds neither the observation lock nor the ROS callback.
        for key in buffer.buffers:buffer.add(key,sample(key,1_005_000_000,4))
        latest,_=buffer.snapshot(1_006_000_000)
        assert latest['arm'].sequence==4
        assert selected['head'].receipt_ns==1_003_000_000
        with pytest.raises(ObservationError,match='stale_head'):
            buffer.snapshot(1_086_000_000)
    finally:
        release.set();worker.join(2)
    assert not errors
    assert calls==[('head',b'\x03'),('right',b'\x03')]


@pytest.mark.parametrize('key,shape',[('head',(720,1280,3)),('right',(360,640,3))])
def test_decoded_rgb_exactly_matches_original_conversion(key,shape):
    cv2=pytest.importorskip('cv2')
    bgr=np.zeros(shape,np.uint8);bgr[:]=[11,80,240]
    ok,encoded=cv2.imencode('.jpg',bgr);assert ok
    frame=JpegFrame(encoded.tobytes())
    expected=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(frame.rgb(key),expected)


def test_bad_jpeg_and_wrong_dimensions_are_rejected():
    cv2=pytest.importorskip('cv2')
    with pytest.raises(ObservationError,match='invalid_jpeg_payload'):JpegFrame(b'')
    with pytest.raises(ObservationError,match='jpeg_decode_failed'):JpegFrame(b'broken').rgb('head')
    ok,encoded=cv2.imencode('.jpg',np.zeros((20,30,3),np.uint8));assert ok
    with pytest.raises(ObservationError,match='bad_head'):JpegFrame(encoded.tobytes()).rgb('head')
    buffer=populated();row=sample('arm',1_001_000_000,2);row.value=JpegFrame(b'bytes')
    with pytest.raises(ObservationError,match='bad_arm'):buffer.add('arm',row)
