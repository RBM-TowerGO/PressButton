"""Camera reuse decisions and actual frame checks, independent of ROS/SDK."""
from collections import Counter
from pathlib import PurePosixPath
from .observation import JpegFrame,ObservationError,TimeMapper

DRIVERS={'astribot_camera_main':'camera','astribot_camera_sub':'compress_camera'}


def missing_camera_drivers(processes,frames_ready):
    counts=Counter(PurePosixPath(p['path']).name for p in processes)
    duplicates=[name for name in DRIVERS if counts[name]>1]
    if duplicates:raise ValueError('multiple_camera_drivers:'+','.join(duplicates))
    if frames_ready:return []
    return [(name,arg) for name,arg in DRIVERS.items() if not counts[name]]


class CameraReadiness:
    """Require successive real JPEG frames from both configured ROS streams."""
    def __init__(self,mapping,wall_ns,monotonic_ns):
        self.mapping=mapping
        self.mapper=TimeMapper(wall_ns,monotonic_ns,mapping['clock']['max_offset_jump_ns'])
        self.rows={k:dict(valid_frames=0,received=0,source_ns=None,receipt_ns=None,error=None)
                   for k in mapping['cameras']}

    def receive(self,key,source_ns,receipt_ns,wall_ns,format,payload):
        row=self.rows[key];row['received']+=1
        try:
            self.mapper.map(source_ns,wall_ns,receipt_ns)
            if row['source_ns'] is not None and source_ns<=row['source_ns']:
                raise ObservationError('nonmonotonic_'+key)
            if 'jpeg' not in format.lower():raise ObservationError('unsupported_camera_format')
            rgb=JpegFrame(payload).rgb(key)
            row.update(valid_frames=row['valid_frames']+1,source_ns=source_ns,
                       receipt_ns=receipt_ns,error=None,shape=list(rgb.shape))
        except ObservationError as exc:
            row.update(valid_frames=0,error=str(exc))

    def report(self,now_ns):
        ready=all(r['valid_frames']>=3 and r['receipt_ns'] is not None
                  and 0<=now_ns-r['receipt_ns']<=self.mapping['head_age_ns']
                  for r in self.rows.values())
        return dict(ready=ready,streams={k:dict(r,topic=self.mapping['cameras'][k],
            age_ns=now_ns-r['receipt_ns'] if r['receipt_ns'] is not None else None)
            for k,r in self.rows.items()})
