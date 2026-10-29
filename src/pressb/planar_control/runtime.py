"""Calibrated live adapter using the existing guarded S1B executor."""
import time
import numpy as np
from .geometry import Plane, Planner, SDKKinematics, pose_matrix, vector, transform
from .controller import Controller
from pressb.s1b_deployment.executor import SDKBackend, GuardedExecutor, PARTS


def validate_config(c):
    if c.get('schema') != 's1b-planar-v1':
        raise ValueError('invalid_planar_schema')
    checks = c['calibration']
    for key in ('coordinates_verified', 'tcp_verified', 'sdk_fk_ik_verified', 'workspace_verified', 'fixed_parts_verified'):
        if checks.get(key) is not True:
            raise ValueError('calibration_required:'+key)
    if not checks.get('evidence'):
        raise ValueError('calibration_evidence_required')
    if not np.isfinite(c['fixed']['max_drift']) or not 0 < c['fixed']['max_drift'] < .1:
        raise ValueError('invalid_fixed_drift')
    for part, value in c['fixed']['positions'].items():
        vector(value, len(value))
    Plane(c['plane'])
    transform(c['tcp_in_wrist'])
    p = c['control']
    for key in ('frequency_hz', 'position_tolerance_m', 'rotation_tolerance_rad', 'max_joint_step_rad',
                'minimum_self_clearance_m', 'max_path_points', 'move_speed_m_s', 'max_move_m', 'air_probe_max_m',
                'observation_max_age_ns', 'settle_timeout_s', 'settle_duration_s',
                'settle_joint_error_rad', 'settle_velocity_rad_s', 'max_servo_lateness_s', 'max_commands_per_session'):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError('invalid_control_parameter:'+key)
    lo, hi = vector(p['joint_lower'], 7), vector(p['joint_upper'], 7)
    if np.any(lo >= hi) or np.any(vector(p['joint_velocity_rad_s'], 7) <= 0):
        raise ValueError('invalid_joint_bounds')
    vector(p['initial_uv_m'], 2)
    forward = p.get('keyboard_forward', {})
    if forward.get('verified'):
        if not forward.get('evidence'):
            raise ValueError('keyboard_forward_evidence_required')
        for key in ('max_distance_m', 'speed_m_s'):
            if not np.isfinite(forward[key]) or forward[key] <= 0:
                raise ValueError('invalid_keyboard_forward_parameter:'+key)
    if p['press']['verified']:
        if not c['torque']['verified'] or not p['press'].get('evidence'):
            raise ValueError('press_requires_torque_and_evidence')
        for k in ('minimum_gap_m', 'maximum_gap_m', 'contact_margin_m', 'overtravel_m',
                  'maximum_extension_m', 'approach_speed_m_s', 'contact_speed_m_s',
                  'retract_speed_m_s', 'dwell_s', 'range_change_tolerance_m'):
            if not np.isfinite(p['press'][k]) or p['press'][k] <= 0:
                raise ValueError('invalid_press_parameter:'+k)
        if p['press']['minimum_gap_m'] >= p['press']['maximum_gap_m']:
            raise ValueError('invalid_press_gap_bounds')
    if c['torque']['verified']:
        vector(c['torque']['baseline_nm'], 7)
        if np.any(vector(c['torque']['max_deviation_nm'], 7) <= 0):
            raise ValueError('invalid_torque_threshold')
    depth = c['depth']
    if depth['verified']:
        transform(depth['camera_in_wrist'])
        if not depth.get('frame_id') or not c['sensors']['depth_topic'] or not c['sensors']['depth_info_topic']:
            raise ValueError('depth_topics_and_frame_required')
        for key in ('depth_min_m', 'depth_max_m', 'minimum_gap_m', 'maximum_gap_m', 'ray_radius_m',
                    'min_points', 'surface_band_m', 'min_inlier_ratio', 'minimum_surface_span_m',
                    'max_panel_tilt_rad', 'max_surface_residual_m'):
            if not np.isfinite(depth[key]) or depth[key] <= 0:
                raise ValueError('invalid_depth_parameter:'+key)
        if (depth['depth_min_m'] >= depth['depth_max_m'] or depth['minimum_gap_m'] >= depth['maximum_gap_m']
                or depth['min_inlier_ratio'] > 1 or depth['max_panel_tilt_rad'] >= np.pi/2):
            raise ValueError('invalid_depth_bounds')


class LiveBackend:
    def __init__(self, config, guard_config, sensors, emit):
        validate_config(config)
        if guard_config.get('command_guard_profile', 'probe') != 'probe':
            raise ValueError('planar_requires_nonclipping_probe_guard')
        if guard_config.get('gripper_override'):
            raise ValueError('planar_disallows_gripper_override')
        self.c, self.sensors, self.emit = config, sensors, emit
        self.sdk = SDKBackend(guard_config['sdk_root'], config['control']['frequency_hz'], control_way='direct')
        self.executor = GuardedExecutor(guard_config, self.sdk, emit)
        self.kin = None
        self.baseline = None
        self.fixed10 = None
        self.planner = None
        self.sensors.monitor.on_fault = self.executor.fault

    def acquire(self, operator_present, estop_ready):
        rows = self.sensors.monitor.snapshot()
        self.baseline = {p:vector(r['position'], len(r['position'])) for p,r in rows.items()}
        initial, stamp = self.sensors.monitor.joint10()
        fixed = self.c['fixed']
        for part, expected in fixed['positions'].items():
            if part not in self.baseline or np.max(np.abs(self.baseline[part]-vector(expected, len(self.baseline[part])))) > fixed['max_drift']:
                raise ValueError('fixed_part_not_at_calibrated_pose:'+part)
        required = {'astribot_chassis', 'astribot_torso', 'astribot_arm_left',
                    'astribot_gripper_left', 'astribot_gripper_right', 'astribot_head'}
        if set(fixed['positions']) != required:
            raise ValueError('all_fixed_parts_must_be_calibrated')
        self.fixed10 = initial.copy()
        self.sdk.torso_target = self.baseline['astribot_torso'].copy()
        # Do not take over a live owner. No handover is attempted by this service.
        if self.sensors.monitor.owners():
            raise ValueError('control_rights_already_owned')
        # Kinematics verification occurs immediately after acquire; no motion is sent.
        def verify():
            self.kin = SDKKinematics(self.sdk.client, self.baseline['astribot_torso'],
                                     self.baseline['astribot_arm_left'], self.c['tcp_in_wrist'])
            self.planner = Planner(Plane(self.c['plane']), self.kin, self.c['control'])
            actual = self.kin.fk(initial[:7])
            reported = self.sdk.client.get_current_cartesian_pose(['astribot_arm_right'], frame='chassis')
            self.planner.check_pose(actual, pose_matrix(reported[0]) @ self.kin.offset)
            self.planner.check_pose(actual, self.planner.plane.target(self.c['control']['initial_uv_m']))
            solved = self.kin.solve(actual, initial[:7])
            if np.max(np.abs(solved-initial[:7])) > self.c['control']['max_joint_step_rad']:
                raise ValueError('initial_ik_branch_mismatch')
            self.planner.check_pose(self.kin.fk(solved), actual)
            if self.kin.clearance(solved) < self.c['control']['minimum_self_clearance_m']:
                raise ValueError('initial_self_clearance')
            lower, upper = self.sdk.client.get_joints_position_limit(['astribot_arm_right'])
            velocity = self.sdk.client.get_joints_velocity_limit(['astribot_arm_right'])
            p = self.c['control']
            if (np.any(np.asarray(p['joint_lower']) < lower[0]) or np.any(np.asarray(p['joint_upper']) > upper[0])
                    or np.any(np.asarray(p['joint_velocity_rad_s']) > velocity[0])):
                raise ValueError('calibration_exceeds_live_joint_limits')
            guard = self.executor.c
            if (np.any(np.asarray(p['joint_lower']) < np.maximum(guard['hardware_lower'], guard['position_lower'])[:7])
                    or np.any(np.asarray(p['joint_upper']) > np.minimum(guard['hardware_upper'], guard['position_upper'])[:7])
                    or np.any(np.asarray(p['joint_velocity_rad_s']) > np.asarray(guard['max_velocity'])[:7])):
                raise ValueError('planner_exceeds_guard_limits')
        self.executor.acquire(initial, stamp, operator_present=operator_present, estop_ready=estop_ready,
                              initial_command=initial, on_acquired=verify)
        return Controller(self.planner, self, self.c['control'], self.emit)

    def snapshot(self):
        if self.executor.state != 'EXECUTING':
            raise ValueError('executor_inactive:'+str(self.executor.fault_reason))
        rows = self.sensors.monitor.snapshot()
        if self.sensors.monitor.error or self.sensors.monitor.driver_fault:
            raise ValueError('robot_state_monitor_failed')
        now = time.monotonic_ns()
        for part, row in rows.items():
            if not 0 <= now-row['receipt_ns'] <= self.executor.c['state_max_age_ns']:
                raise ValueError('stale_part:'+part)
            header = row['header']
            source = header['sec']*10**9+header['nanosec']
            if source <= 0 or abs(time.time_ns()-source) > self.c['sensors']['max_source_age_ns']:
                raise ValueError('stale_part_source:'+part)
            q = vector(row['position'], len(self.baseline[part]))
            if part != 'astribot_arm_right' and np.max(np.abs(q-self.baseline[part])) > self.c['fixed']['max_drift']:
                raise ValueError('fixed_part_drift:'+part)
        q10 = np.concatenate([rows[p]['position'] for p in PARTS])
        arm = rows['astribot_arm_right']
        if arm['names'] and arm['names'] != self.executor.c['joint_names'][:7]:
            raise ValueError('arm_joint_order_mismatch')
        torque = vector(arm['torque'], 7)
        if self.c['torque']['verified']:
            if np.any(np.abs(torque-np.asarray(self.c['torque']['baseline_nm'])) > self.c['torque']['max_deviation_nm']):
                raise ValueError('torque_anomaly')
        tcp = self.kin.fk(q10[:7])
        source = arm['header']['sec']*10**9+arm['header']['nanosec']
        return dict(q=q10[:7], q10=q10, tcp=tcp, torque=torque, velocity=vector(arm['velocity'], 7),
                    source_ns=source, receipt_ns=min(r['receipt_ns'] for r in rows.values()))

    def send(self, q, state):
        target = self.fixed10.copy()
        target[:7] = q
        self.executor.send(target, state['q10'], state['receipt_ns'], source='planar')

    def range(self):
        state = self.snapshot()
        if self.c['depth']['verified']:
            wrist = state['tcp'] @ np.linalg.inv(self.kin.offset)
            return self.sensors.ranging(wrist, state['tcp'][:3, 3], self.planner.plane.basis[:, 2],
                                        self.c['depth'], state['source_ns'])
        manual = self.c['manual_range']
        if manual['verified'] and manual.get('evidence'):
            uv = self.planner.plane.coordinates(state['tcp'])[:2]
            bounds = np.asarray(manual['uv_bounds_m'], float)
            if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(uv < bounds[:, 0]) or np.any(uv > bounds[:, 1]):
                raise ValueError('outside_manual_range_patch')
            gap = float(manual['gap_m'])
            if not np.isfinite(gap) or gap <= 0:
                raise ValueError('invalid_manual_range')
            return dict(source='manual_calibration', gap_m=gap, trusted=True, receipt_ns=state['receipt_ns'])
        raise ValueError('no_verified_depth_or_manual_range')

    def observe(self):
        images = self.sensors.images()
        state = self.snapshot()
        if any(abs(f['source_ns']-state['source_ns']) > self.c['sensors']['max_skew_ns'] for f in images.values()):
            raise ValueError('image_state_skew')
        try:
            ranging = self.range()
        except ValueError as exc:
            ranging = dict(trusted=False, reason=str(exc))
        return dict(images=images, tcp=state['tcp'].tolist(), range=ranging,
                    state_source_ns=state['source_ns'])

    def stop(self, reason, fault):
        if fault:
            self.executor.fault(reason)
        else:
            self.executor.stop(reason)

    def close(self):
        self.sensors.monitor.on_fault = None
        self.executor.close()
