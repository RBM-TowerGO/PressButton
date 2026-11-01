"""Optional keyboard journals and boundary captures; no SDK calls or servo work."""
import base64
import copy
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import urlopen


def stamp():
    return dict(utc_ns=time.time_ns(),monotonic_ns=time.monotonic_ns())


class Recorder:
    def __init__(self, root, camera_base, settings, capture_timeout=2.):
        self.root=Path(root).expanduser()
        if self.root.exists():raise FileExistsError('记录目录已存在：'+str(self.root))
        self.camera_base=camera_base
        self.capture_timeout=capture_timeout
        self.operations=[];self.active=None;self.closed=False
        self.session=dict(schema='pressb-keyboard-record-v1',started=stamp(),status='running',
                          settings=settings,recording_complete=True,recording_errors=[])
        self._write(self.root/'session.json',self.session)
        self.event('session_started')
        print('记录目录：'+str(self.root),flush=True)

    def _error(self, context, exc):
        error=dict(context=context,error=str(exc),**stamp())
        self.session['recording_complete']=False
        self.session['recording_errors'].append(error)
        print('记录不完整：'+context+'：'+str(exc),file=sys.stderr,flush=True)

    def _write(self, path, value):
        temporary=path.with_name(path.name+'.tmp')
        try:
            path.parent.mkdir(parents=True,exist_ok=True)
            with temporary.open('w') as f:
                json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)
                f.write('\n');f.flush();os.fsync(f.fileno())
            temporary.replace(path)
            return True
        except Exception as exc:
            self._error(str(path),exc)
            return False

    def event(self, event, **data):
        row=dict(event=event,**stamp(),**data)
        try:
            self.root.mkdir(parents=True,exist_ok=True)
            with (self.root/'events.jsonl').open('a') as f:
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                f.flush();os.fsync(f.fileno())
        except Exception as exc:self._error('events.jsonl',exc)

    def _fetch(self, timeout):
        with urlopen(self.camera_base+'/api/capture',timeout=timeout) as response:
            return json.load(response)

    def capture(self, relative, boundary=None, cancel=None):
        boundary=boundary or stamp()
        deadline=time.monotonic()+self.capture_timeout
        result=dict(status='missing',boundary=boundary)
        reason='capture_timeout';bundle=None
        while time.monotonic()<deadline:
            if cancel is not None and cancel.is_set():
                reason='operator_stopped';break
            try:
                candidate=self._fetch(max(.001,deadline-time.monotonic()))
                if time.monotonic()>deadline:raise ValueError('capture_timeout')
                frames=candidate['frames']
                for key in ('head','right','depth'):
                    f=frames[key]
                    if f['source_ns']<=boundary['utc_ns'] or f['receipt_ns']<=boundary['monotonic_ns']:
                        raise ValueError('frames_predate_boundary')
                bundle=candidate;break
            except Exception as exc:
                reason=str(exc)
                delay=min(.05,max(0,deadline-time.monotonic()))
                if cancel is not None:cancel.wait(delay)
                else:time.sleep(delay)
        if bundle is not None:
            try:
                for f in bundle['frames'].values():
                    if f['source_ns']<=boundary['utc_ns'] or f['receipt_ns']<=boundary['monotonic_ns']:
                        raise ValueError('frames_predate_boundary')
                import numpy as np
                folder=self.root/relative;folder.mkdir(parents=True,exist_ok=True)
                metadata=copy.deepcopy(bundle)
                raw={k:base64.b64decode(metadata['frames'][k].pop('data_base64'),validate=True)
                     for k in ('head','right','depth')}
                shape=bundle['metadata']['depth']
                depth=np.frombuffer(raw['depth'],dtype='<u2').reshape(shape['height'],shape['width'])
                for key in ('head','right'):
                    with (folder/(key+'.jpg')).open('wb') as f:
                        f.write(raw[key]);f.flush();os.fsync(f.fileno())
                with (folder/'right_depth_raw.npy').open('wb') as f:
                    np.save(f,depth);f.flush();os.fsync(f.fileno())
                metadata.update(boundary=boundary,saved=stamp())
                if not self._write(folder/'metadata.json',metadata):raise OSError('metadata_write_failed')
                result.update(status='complete',frames=metadata['frames'])
            except Exception as exc:reason=str(exc)
        if result['status']!='complete':
            result['error']=reason;self._error(relative,reason)
        self._write(self.root/relative/'capture.json',result)
        self._write(self.root/'session.json',self.session)
        return result

    def begin(self, kind, command, start_pose, target_pose, speed, cancel):
        operation=dict(operation_id=f'{len(self.operations)+1:06d}',kind=kind,command=command,
            start_command_pose_m_xyzw=list(start_pose),target_command_pose_m_xyzw=list(target_pose),
            speed_mm_s=speed,status='prepared',submitted=stamp(),
            completion_basis='command_interpolation' if kind in ('move','head') else 'first_successful_send')
        self.operations.append(operation);self.active=operation
        self._save_operation()
        self.event('operation_submitted',operation_id=operation['operation_id'],kind=kind,command=command)
        operation['before']=self.capture('operations/'+operation['operation_id']+'/before',cancel=cancel)
        self._save_operation()
        return operation

    def started(self, boundary):
        self.active.update(status='running',started=boundary)
        self._save_operation()
        self.event('operation_started',operation_id=self.active['operation_id'],boundary=boundary)

    def finish(self, status, boundary=None, reason=None, cancel=None):
        if self.active is None:return
        operation=self.active
        operation.update(status=status,ended=boundary or stamp(),reason=reason)
        self._save_operation()
        operation['after']=self.capture('operations/'+operation['operation_id']+'/after',
                                       operation['ended'],cancel=cancel)
        self._save_operation()
        self.event('operation_'+status,operation_id=operation['operation_id'],reason=reason)
        self.active=None
        self.summary()

    def _save_operation(self):
        operation=self.active
        self._write(self.root/'operations'/operation['operation_id']/'operation.json',operation)

    def summary(self):
        lines=['# 键盘操作记录','',
               '完成仅表示指令发送完成，不代表实测到位或按钮触发。','',
               '记录完整：'+str(self.session['recording_complete']),'']
        for op in self.operations:
            lines += [f"## {op['operation_id']} {op['kind']} — {op['status']}",
                      '',json.dumps(op['command'],ensure_ascii=False),'']
            for phase,label in (('before','开始'),('after','完成/中断')):
                capture=op.get(phase,{})
                lines += [label+'：'+capture.get('status','pending'),'']
                if capture.get('status')=='complete':
                    for camera in ('head','right'):
                        lines += [f"![{label} {camera}](operations/{op['operation_id']}/{phase}/{camera}.jpg)",'']
        try:
            path=self.root/'summary.md';tmp=path.with_suffix('.tmp')
            tmp.write_text('\n'.join(lines));tmp.replace(path)
        except Exception as exc:self._error('summary.md',exc)

    def close(self, reason='operator_exit'):
        if self.closed:return
        self.closed=True
        if self.active:self.finish('interrupted',reason=reason)
        self.session['final']=self.capture('final')
        self.session.update(status='closed',ended=stamp(),exit_reason=reason)
        self.event('session_ended',reason=reason)
        self.summary()
        self._write(self.root/'session.json',self.session)
