"""Bounded NVENC writers: reusable buffers, blocking backpressure, full decode audit."""
from __future__ import annotations
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import numpy as np

class VideoWriter:
    def __init__(self,path,size,gpu,capacity=8,pinned=True):
        self.path=Path(path); self.size=tuple(size); self.frames=0; self.wait_s=0.;self.error=None
        self.free=queue.Queue(capacity);self.pending=queue.Queue(capacity);self.closed=False
        self.owners=[]
        for _ in range(capacity):
            if pinned:
                import torch
                owner=torch.empty((size[1],size[0],3),dtype=torch.uint8,pin_memory=True)
                self.owners.append(owner); self.free.put(owner.numpy())
            else: self.free.put(np.empty((size[1],size[0],3),dtype=np.uint8))
        self.log=self.path.with_suffix('.encoder.log').open('wb')
        self.process=subprocess.Popen(['ffmpeg','-nostdin','-y','-v','error','-threads','1','-filter_threads','1',
            '-filter_complex_threads','1','-f','rawvideo','-pixel_format','rgb24',
            '-video_size',f'{size[0]}x{size[1]}','-framerate','30','-i','pipe:0','-an',
            '-c:v','h264_nvenc','-gpu',str(gpu),'-preset','p4','-cq','18','-b:v','0','-bf','0','-g','30',
            '-pix_fmt','yuv420p','-threads','1',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log)
        self.thread=threading.Thread(target=self._run,name=f'encode-{self.path.stem}',daemon=True);self.thread.start()
    def _run(self):
        try:
            while True:
                buf=self.pending.get()
                if buf is None: break
                try: self.process.stdin.write(memoryview(buf).cast('B'))
                finally: self.free.put(buf)
        except BaseException as exc: self.error=exc
    def _check(self):
        if self.error: raise RuntimeError(f'Encoder failed: {self.path}') from self.error
        if self.process.poll() is not None: raise RuntimeError(f'Encoder exited: {self.path}, see encoder log')
    def write(self,rgb):
        if self.closed: raise RuntimeError('Writer closed')
        if rgb.shape!=(self.size[1],self.size[0],3) or rgb.dtype!=np.uint8: raise ValueError('Wrong frame shape/type')
        t=time.perf_counter()
        while True:
            self._check()
            try: buf=self.free.get(timeout=.1);break
            except queue.Empty: pass
        self.wait_s+=time.perf_counter()-t
        np.copyto(buf,rgb);self.pending.put(buf);self.frames+=1
    def close(self):
        if self.closed: return
        self.closed=True
        try:
            while self.thread.is_alive():
                if self.error: break
                try: self.pending.put(None,timeout=.1);break
                except queue.Full: pass
            self.thread.join(timeout=120)
            if self.thread.is_alive(): raise RuntimeError('Encoder drain timeout')
            self.process.stdin.close()
            code=self.process.wait(timeout=120)
            if self.error: raise RuntimeError('Encoder worker failed') from self.error
            if code: raise RuntimeError(f'NVENC failed ({code}), see {self.path.with_suffix(".encoder.log")}')
        finally:
            if self.process.poll() is None: self.process.kill();self.process.wait()
            self.log.close()

def decode_audit(path,expected,size):
    """Decode EVERY frame; ffprobe count alone does not establish integrity."""
    path=Path(path)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))['streams']
    if len(info)!=1: raise ValueError('Expected exactly one video stream')
    stream=info[0]
    if (stream['width'],stream['height'])!=tuple(size) or stream['pix_fmt']!='yuv420p': raise ValueError('Video format mismatch')
    if stream['avg_frame_rate']!='30/1': raise ValueError('Video rate mismatch')
    result=subprocess.run(['ffmpeg','-v','error','-threads','2','-filter_threads','1','-xerror','-i',str(path),
        '-map','0:v:0','-threads','1','-f','framemd5','-'],capture_output=True,check=True,text=True)
    lines=[line for line in result.stdout.splitlines() if line and not line.startswith('#')]
    if len(lines)!=expected: raise ValueError(f'{path}: decoded {len(lines)}, expected {expected}')
    pts=[int(line.split(',')[2]) for line in lines]
    if pts!=list(range(expected)): raise ValueError('Missing/duplicate video timestamps')
    return dict(frames=len(lines),width=size[0],height=size[1],codec=stream['codec_name'],decode='all_frames',bytes=path.stat().st_size)
