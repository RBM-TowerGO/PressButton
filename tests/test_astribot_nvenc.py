"""Explicit hardware integration test: run only with PRESSB_TEST_NVENC=1."""
import os
import subprocess
import numpy as np
import pytest
from pressb.astribot_collection.video import VideoWriter,decode_audit

@pytest.mark.skipif(os.environ.get('PRESSB_TEST_NVENC')!='1',reason='requires real NVENC device')
def test_bounded_queue_and_full_decode_reject_truncation(tmp_path):
    path=tmp_path/'test.mp4'
    writer=VideoWriter(path,(640,480),1,capacity=1)
    for i in range(45):
        frame=np.full((480,640,3),i*5,dtype=np.uint8);writer.write(frame)
    writer.close()
    assert decode_audit(path,45,(640,480))['frames']==45
    decoded=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-filter_threads','1','-i',str(path),
        '-threads','1','-f','rawvideo','-pix_fmt','rgb24','-'])
    means=np.frombuffer(decoded,np.uint8).reshape(45,480,640,3).mean(axis=(1,2,3))
    np.testing.assert_allclose(means,np.arange(45)*5,atol=3)
    with pytest.raises(ValueError):decode_audit(path,44,(640,480))
    data=path.read_bytes();path.write_bytes(data[:len(data)//2])
    with pytest.raises(Exception):decode_audit(path,45,(640,480))
