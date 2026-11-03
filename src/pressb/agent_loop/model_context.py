"""Lossy, explicitly labelled model views. Source artifacts remain lossless."""
import json
import math

FORMAT = 'pressb-conversation-v1'


def text(value):
    return dict(type='text', text=json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')))


def rounded(value, places=1, scale=1):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [rounded(v, places, scale) for v in value]
    return round(value * scale, places) if isinstance(value, (int, float)) and math.isfinite(value) else None


def pose(value):
    if not value or len(value) != 7:
        return None
    return dict(xyz_mm=rounded(value[:3], scale=1000), quaternion_xyzw=rounded(value[3:], 4))


def state(value):
    value = value or {}
    gripper = value.get('joints', {}).get('astribot_gripper_right', {})
    return dict(measured_pose=pose(value.get('pose_m_xyzw')),
                command_pose=pose(value.get('command_pose_m_xyzw')),
                head_deg=rounded(value.get('head_deg')), gripper_command=value.get('gripper_command'),
                measured_gripper_sdk_native=rounded(gripper.get('position'), 4),
                rgb_state_skew_ms=rounded(value.get('rgb_state_skew_ms')))


def depth(value, selection=None):
    value, selection = value or {}, selection or {}
    return dict(median_mm=rounded(value.get('median_m'), scale=1000),
                status=value.get('status', 'unknown'), valid_pixels=value.get('valid_pixels'),
                window_pixels=value.get('window_pixels'), center_px=value.get('center_px', selection.get('center_px')),
                point_normalized=rounded(selection.get('point_normalized'), 6),
                revision=selection.get('revision'), rgb_depth_skew_ms=rounded(value.get('skew_ms')))


def observation(obs):
    return dict(context_type='LIVE_OBSERVATION', observation_id=obs['observation_id'],
                state=state(obs.get('state')), depth=depth(obs.get('depth'), obs.get('selection', obs.get('fixed_point'))))


def session(value):
    return {k: value.get(k) for k in ('session_id', 'frame_id', 'reference_point', 'speed_mm_s',
            'head_speed_deg_s', 'completion_basis')}


def review(value):
    value = value or {}
    keys = ('case_id', 'analysis_version', 'outcome', 'outcome_source', 'outcome_scope', 'user_feedback',
            'button_activation_verified', 'withdrawal_verified', 'image_review', 'limitations', 'lessons',
            'findings', 'behavior_preserved', 'success_reference')
    result = {k: value[k] for k in keys if k in value}
    # Round evidence in findings is useful; transport acknowledgements are not.
    if 'result_assessment' in value:
        result['result_assessment'] = {k: v for k, v in value['result_assessment'].items() if k != 'robot_stop_ack'}
    return result


def demonstration(summary):
    rows = []
    for source in summary['operations']:
        op = source['operation']
        row = dict(step=op['operation_id'], type=op['kind'], command=op['command'], status=op.get('status', 'unknown'),
                   start_command_pose=pose(op.get('start_command_pose_m_xyzw')),
                   target_command_pose=pose(op.get('target_command_pose_m_xyzw')))
        for phase in ('before', 'after'):
            boundary = source['observations'][phase]
            metadata = boundary.get('metadata') or {}
            row[phase] = dict(status=op.get(phase, {}).get('status', 'missing'),
                              depth=depth(metadata.get('depth'), metadata.get('selection')),
                              missing_images=boundary['missing_images'])
        rows.append(row)
    original_session = summary['session']
    result = dict(context_type=summary['context_type'], case_review=review(summary.get('case_review')),
                  session={k: original_session.get(k) for k in ('recording_complete', 'final')}, operations=rows,
                  state_semantics='All demonstration poses are COMMAND targets, not measured poses; xyz mm, quaternion xyzw.')
    # Derive soft reference only from reviewed activation image boundaries, not a hardcoded step/value.
    references = []
    for path in summary.get('case_review', {}).get('evidence', []):
        parts = path.split('/') if isinstance(path, str) else []
        if len(parts) == 4 and parts[0] == 'operations' and parts[2] in ('before', 'after'):
            item = next((r for r in rows if r['step'] == parts[1]), None)
            if item:
                candidate = dict(step=parts[1], boundary=parts[2], **item[parts[2]]['depth'])
                if candidate not in references:
                    references.append(candidate)
    result['reviewed_activation_depth_references'] = references
    result['depth_reference_semantics'] = ('Approximate historical camera-depth only, useful when point/configuration/alignment match. '
        'Not fingertip clearance, gripper length, depth-minus offset, activation threshold or a motion limit.')
    return result


def execution(result):
    motion = result.get('motion') or {}
    command = result.get('request') or {}
    elapsed = motion.get('elapsed_s')
    if elapsed is None and motion.get('started_monotonic_ns') is not None and motion.get('completed_monotonic_ns') is not None:
        elapsed = (motion['completed_monotonic_ns']-motion['started_monotonic_ns'])/1e9
    compact = dict(command_id=command.get('command_id'),
                command={k: command[k] for k in ('type', 'xyz_mm', 'head_deg', 'key') if k in command},
                status=result.get('status'), sends_completed=motion.get('action_completed'),
                completion_basis=motion.get('completion_basis'),
                elapsed_s=rounded(elapsed, 2),
                observation_id=(result.get('capture') or {}).get('observation_id'), error=result.get('error'))
    if 'replay' in result:
        compact['replay'] = result['replay']
    return compact


def failure(summary):
    observations, rows = {}, []
    for source in summary['rounds']:
        row = {k: source.get(k) for k in ('round', 'proposal', 'executed', 'status', 'error', 'decision_stage', 'perception',
                                        'alignment_progress', 'activation', 'withdrawal', 'decision_basis',
                                        'control_plan', 'progress_summary', 'view_summary', 'localization_summary') if k in source}
        for phase in ('before', 'after'):
            obs = source.get(phase)
            row[phase + '_observation_id'] = obs['observation_id'] if obs else None
            if obs:
                compact = observation(obs)
                compact['context_type'] = 'HISTORICAL_FAILURE_NOT_LIVE'
                observations[obs['observation_id']] = compact
        if source.get('result'):
            row['result'] = execution(dict(source['result'], status=source['status'], request=source.get('command', {})))
        rows.append(row)
    # Short references avoid repeatedly tokenizing UUIDs; original IDs remain mapped once.
    aliases = {oid: f'F{n:02d}' for n, oid in enumerate(observations)}
    def refs(value):
        if isinstance(value, dict):
            return {(key.removesuffix('_id')+'_ref' if key.endswith('observation_id') and isinstance(v, str) and v in aliases else key): refs(v)
                    for key, v in value.items()}
        if isinstance(value, list):
            return [refs(v) for v in value]
        return aliases.get(value, value) if isinstance(value, str) else value
    return dict(context_type=summary['context_type'], review=refs(review(summary['review'])),
                completed_commands=summary['completed_commands'], rounds=refs(rows),
                observations={aliases[oid]: obs for oid, obs in observations.items()},
                reference_semantics='Fxx refers to a historical observation below, which records its original observation_id; never a live ID.',
                semantics=summary['semantics'])
