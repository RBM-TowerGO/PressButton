"""Read-only S1B recording audit. Extracts SDK units, NOT model-ready radians.

Run with .venv-data-review/bin/python scripts/audit_s1b_contract.py.
"""
import json
from pathlib import Path

import h5py
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/astribot_s1b_20261025/storage'
OUT = ROOT / 'docs/astribot/s1b_contract_20261025'
# Verified SDK whole_body_names + S1 DOFs; names additionally checked against
# recorded RobotJointController.name. Gripper names require a part namespace.
PARTS = [
    ('astribot_chassis', ['astribot_chassis_x', 'astribot_chassis_y', 'astribot_chassis_z_rot']),
    ('astribot_torso', [f'astribot_torso_joint_{i}' for i in range(1, 5)]),
    ('astribot_arm_left', [f'astribot_arm_left_joint_{i}' for i in range(1, 8)]),
    ('astribot_gripper_left', ['astribot_gripper_joint']),
    ('astribot_arm_right', [f'astribot_arm_right_joint_{i}' for i in range(1, 8)]),
    ('astribot_gripper_right', ['astribot_gripper_joint']),
    ('astribot_head', [f'astribot_head_joint_{i}' for i in range(1, 3)]),
]
NAMES = [f'{part}/{name}' for part, names in PARTS for name in names]
WANTED = [f'astribot_arm_right/astribot_arm_right_joint_{i}' for i in range(1, 8)] + [
    'astribot_gripper_right/astribot_gripper_joint']


def extract(values, names):
    """Name-based extraction; caller must supply an independently verified schema."""
    values = np.asarray(values)
    if len(set(names)) != len(names) or values.shape[-1] != len(names):
        raise ValueError('Invalid or ambiguous joint schema')
    indices = [names.index(name) for name in WANTED]
    result = values[..., indices]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite joint data')
    return result


def stats(values):
    a = np.asarray(values)
    return dict(zip(['min', 'median', 'p95', 'max'], np.percentile(a, [0, 50, 95, 100]).tolist()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted((DATA / 'trans/hdf5_output_elevator1').glob('*.hdf5'))
    keys = ['joints_dict/state_timestamp', 'joints_dict/command_timestamp',
            'images_dict/head/rgb_timestamp', 'images_dict/right/rgb_timestamp']
    offsets = {k: [] for k in keys}
    periods, worst = [], []
    states, commands = [], []
    for p in paths:
        with h5py.File(p) as f:
            t = f['time'][:]
            periods.extend(np.diff(t) * 1000)
            for k in keys:
                offsets[k].extend((f[k][:] - t) * 1000)
            delta = (f[keys[1]][:] - t) * 1000
            i = int(np.argmax(np.abs(delta)))
            worst.append(dict(episode=p.name, row=i, command_minus_image_ms=float(delta[i])))
            states.append(extract(f['joints_dict/joints_position_state'][:], NAMES))
            commands.append(extract(f['joints_dict/joints_position_command'][:], NAMES))
    # Decode recorded names and clocks without requiring ROS or a robot client.
    raw = []
    p = sorted(DATA.rglob('storage_x86*.mcap'))[0]
    seen = set()
    with p.open('rb') as f:
        for _, c, m, d in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
            if 'joint_space' not in c.topic or c.topic in seen:
                continue
            seen.add(c.topic)
            part = c.topic.split('/')[1]
            names = dict(PARTS)[part]
            command = hasattr(d, 'command')
            if command:
                assert list(d.name) == names, (c.topic, d.name, names)
                assert d.mode == 1 and len(d.command) == 3 * len(names)
            else:
                assert len(d.position) == len(names)
            raw.append(dict(topic=c.topic, names=list(d.name), mode=d.mode,
                            header_seconds=d.header.stamp.sec + d.header.stamp.nanosec * 1e-9,
                            log_seconds=m.log_time * 1e-9, publish_seconds=m.publish_time * 1e-9,
                            values=list(d.command if command else d.position)))
    report = dict(episodes=len(paths), frames=sum(map(len, states)),
                  qualified_names25=NAMES, extracted_names8=WANTED,
                  units8=['rad'] * 7 + ['SDK closure scale 0=open,100=closed'],
                  model_ready=False, reason='Gripper SDK-to-model calibration is unverified',
                  frame_period_ms=stats(periods),
                  timestamp_minus_head_image_ms={k: stats(v) for k, v in offsets.items()},
                  worst_command_offsets=sorted(worst, key=lambda x: abs(x['command_minus_image_ms']), reverse=True)[:10],
                  raw_mcap=str(p.relative_to(ROOT)), raw_first_messages=raw,
                  state_gripper_range=stats(np.concatenate(states)[:, -1]),
                  command_gripper_range=stats(np.concatenate(commands)[:, -1]))
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    # Sample artifact deliberately labels the incompatible gripper units.
    np.savez(OUT / 'sample_sdk_joint8.npz', state_sdk8=states[0], command_sdk8=commands[0],
             names=np.asarray(WANTED), units=np.asarray(report['units8']), episode=paths[0].name)
    print(json.dumps({k: report[k] for k in ['episodes', 'frames', 'frame_period_ms',
                                          'timestamp_minus_head_image_ms', 'worst_command_offsets']}, indent=2))


if __name__ == '__main__':
    main()
