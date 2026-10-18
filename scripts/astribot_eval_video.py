"""Observation-rate videos with actual simulation timestamps (no duplicated frames)."""
from fractions import Fraction
import av


class ObservationVideo:
    def __init__(self,path,size,horizon):
        self.container=av.open(str(path),'w')
        self.stream=self.container.add_stream('libx264',rate=Fraction(30,horizon))
        self.stream.width,self.stream.height=size
        self.stream.pix_fmt='yuv420p'
        self.stream.time_base=Fraction(1,120)
        self.stream.codec_context.time_base=Fraction(1,120)
        self.stream.codec_context.thread_count=1
        self.stream.options={'preset':'veryfast','crf':'18','tune':'zerolatency'}
        self.frames=0;self.last_step=None;self.closed=False

    def write(self,rgb,physics_step):
        if self.last_step==physics_step:return
        if self.last_step is not None and physics_step<self.last_step:raise ValueError('Video clock went backwards')
        frame=av.VideoFrame.from_ndarray(rgb,format='rgb24')
        frame.pts=physics_step-1;frame.time_base=Fraction(1,120)
        for packet in self.stream.encode(frame):self.container.mux(packet)
        self.frames+=1;self.last_step=physics_step

    def close(self):
        if self.closed:return
        for packet in self.stream.encode():self.container.mux(packet)
        self.container.close();self.closed=True
