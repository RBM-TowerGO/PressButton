import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from astribot_closedloop_runtime import LiveRuntime
from pressb.astribot_collection.runtime import Runtime


def fake_runtime(tmp_path, monkeypatch, errors, mutate=None):
    module = ModuleType('omni.physxfabric')
    module.get_physx_fabric_interface = lambda: SimpleNamespace(force_update=lambda *a: None)
    monkeypatch.setitem(sys.modules, 'omni.physxfabric', module)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    runtime = LiveRuntime.__new__(LiveRuntime)
    runtime.world = SimpleNamespace(current_time=1.0,
        physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda: None))
    q = np.zeros((1,32), np.float32)
    runtime.view = SimpleNamespace(get_joint_positions=lambda: q.copy())
    runtime.pose_guard = object()
    runtime.gpu, runtime.shared = 0, tmp_path
    runtime.pose_sync_retries, runtime.max_pose_sync_error = 0, 0.0
    runtime.last_camera_transforms = {(0,'wrist_right'): np.eye(4)}
    runtime.captures = 0
    def render(self):
        self.captures += 1
        if mutate:
            mutate(self, q)
        return {0: {'frame': self.captures}}
    monkeypatch.setattr(Runtime, '_render_current', render)
    runtime._camera_pose_errors = lambda: {'native_wrist_right': errors[min(runtime.captures-1,len(errors)-1)]}
    return runtime


def test_stale_capture_is_discarded_and_rechecked_at_original_tolerance(tmp_path, monkeypatch):
    runtime = fake_runtime(tmp_path, monkeypatch, [0.033, 1e-8])
    assert runtime._render_current()[0]['frame'] == 2
    assert runtime.pose_sync_retries == 1
    assert runtime.last_pose_sync['native_wrist_right'] <= 1e-4
    assert runtime.max_pose_sync_error <= 1e-4
    records = [json.loads(line) for line in (tmp_path/'pose_sync_retries.jsonl').read_text().splitlines()]
    assert records[0]['rejected_frame_discarded'] and records[-1]['recovered']


def test_three_invalid_captures_stop_without_accepting_bad_frame(tmp_path, monkeypatch):
    runtime = fake_runtime(tmp_path, monkeypatch, [0.0005])
    with pytest.raises(RuntimeError, match='after 3 frozen captures'):
        runtime._render_current()
    assert runtime.captures == runtime.pose_sync_retries == 3


@pytest.mark.parametrize('what', ['time', 'joints'])
def test_retry_must_not_advance_physics(tmp_path, monkeypatch, what):
    def mutate(runtime, q):
        if what == 'time':runtime.world.current_time += 1/120
        else:q[0,0] += 0.001
    runtime = fake_runtime(tmp_path, monkeypatch, [0.0], mutate)
    with pytest.raises(RuntimeError, match='Physics advanced'):
        runtime._render_current()

