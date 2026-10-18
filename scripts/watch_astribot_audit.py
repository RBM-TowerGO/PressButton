"""Audit each finished trial during this one evaluation job, then exit."""
import json
from pathlib import Path
import time
import traceback
from audit_astribot_closedloop import audit_trial

root=Path('/home/js/robot/VLA/PressButton/PressB/outputs/astribot_closedloop_20261018/test40')
protocol=json.loads((root/'protocol.json').read_text())
completed={}
try:
    while True:
        for candidate in protocol['candidate_ids']:
            if candidate in completed:continue
            path=root/f'candidate_{candidate:06d}'
            if not (path/'report.json').exists():continue
            completed[candidate]=audit_trial(path)
            print(json.dumps(dict(audited=candidate,success=completed[candidate]['success'],total=len(completed))),flush=True)
        if len(completed)==len(protocol['candidate_ids']):break
        if (root/'infrastructure_error.json').exists() or not Path('/proc/3696507').exists():
            raise RuntimeError('Evaluation ended before every declared trial completed')
        time.sleep(5)
    from collections import Counter
    rows=[completed[i] for i in protocol['candidate_ids']]
    result=dict(passed=True,complete=True,requested=len(rows),audited=len(rows),successes=sum(r['success'] for r in rows),
        reasons=dict(Counter(r['reason'] for r in rows)),model_sha256=protocol['model_sha256'],
        all_video_frames_decoded=True,results=rows)
    (root/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('ALL_TRIALS_AUDITED',flush=True)
except BaseException:
    (root/'audit_error.json').write_text(json.dumps(dict(error=traceback.format_exc(),audited=len(completed)),indent=2)+'\n')
    raise
