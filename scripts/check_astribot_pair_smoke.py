import json
from pathlib import Path
import sys
import numpy as np
from astribot_pair_analysis import classify, transform_points, reference_points, diagnose

assert classify(True,'success',0,0,1,1,1,0)=='成功'
assert classify(False,'invalid_prediction',0,0,1,1,1,0)=='其他终止：invalid_prediction'
assert classify(False,'timeout',0,.03,1,1,1,0)=='接触但未完成按压'
assert classify(False,'timeout',.09,0,1,1,1,0)=='初始姿态附近停滞'
assert classify(False,'timeout',.2,0,1,.01,.01,0)=='进入按钮近区但未有效接触'
assert classify(False,'timeout',.2,0,1,.5,.6,0)=='接近后偏离'
assert classify(False,'timeout',.2,0,1,.5,.5,.02)=='超时时仍在接近'
assert classify(False,'timeout',.2,0,1,.99,.99,0)=='移动但未进入按钮近区'
np.testing.assert_allclose(transform_points(np.array([[1.,2,3]]),np.array([[1.,0,0,0]]),np.array([0.,1,0])),[[1.,3,3]])
np.testing.assert_allclose(transform_points(np.zeros((1,3)),np.array([[0.,0,0,1]]),np.array([1.,0,0])),[[-1.,0,0]],atol=1e-8)
root=Path(sys.argv[1]);audit=json.loads((root/'audit.json').read_text());meta=json.loads((root/'shared/metadata.json').read_text())
points=reference_points(meta)
row=audit['results'][0]
diagnostic=diagnose(root/f'candidate_{row["candidate_id"]:06d}',row,meta,points)
assert 0<=diagnostic['minimum_distance_mm']<3000
assert all(np.isfinite(list(v)).all() for v in points.values())
print(json.dumps(dict(checks_passed=True,diagnostic=diagnostic),ensure_ascii=False,indent=2,allow_nan=False))
