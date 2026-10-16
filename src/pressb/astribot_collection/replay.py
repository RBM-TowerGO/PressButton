"""Deterministic stratification and reconstruction of exported 30-Hz actions."""
import numpy as np

def reconstruct_targets(raw_targets,training):
    """Replay each next-sample command over four ticks, then the raw safe return.

    The 30-Hz labels specify an absolute target at the next observation time.
    Linear interpolation is explicit here; it is NOT a replacement IK solver.
    The original suffix supplies release/return/reset so the full independent
    physical auditor can still verify the episode after the training prefix.
    """
    targets=np.asarray(raw_targets,dtype=np.float32).copy()
    steps=np.asarray(training['physical_steps'],dtype=int);action=np.asarray(training['action'])
    if targets.ndim!=2 or targets.shape[1]!=32 or action.shape!=(len(steps),32):raise ValueError('Replay schema mismatch')
    if len(steps)<2 or np.any(np.diff(steps)!=4) or steps[-1]>=len(targets):raise ValueError('Replay timing mismatch')
    for i,(start,end) in enumerate(zip(steps[:-1],steps[1:])):
        first=targets[start].copy();last=action[i]
        targets[start:end+1]=first+(last-first)*np.arange(5,dtype=np.float32)[:,None]/4
    return targets

def stratified_replays(selection,candidates):
    by_id={r['candidate_id']:r for r in candidates}
    unused={r['candidate_id']:r for r in selection};chosen=[]
    if len(unused)<20:raise ValueError('Need at least 20 accepted candidates for stratified replay')
    for yaw in (1,3,5,7,9):
        for x in (-.26,-.18):
            for y in (.1,.3):
                def distance(cid):
                    r=by_id[cid]
                    return ((r['yaw_deg']-yaw)/2)**2+((r['x_m']-x)/.08)**2+((r['y_m']-y)/.2)**2,cid
                cid=min(unused,key=distance);r=unused.pop(cid)
                chosen.append({**r,'stratum_center':dict(x_m=x,y_m=y,yaw_deg=yaw),
                               'sample':by_id[cid],'distance':float(distance(cid)[0])})
    return chosen
