"""Depth geometry, timestamped ROS observations and read-only capability probing."""
import base64
import io
import threading
import time
import numpy as np
from .geometry import transform, vector


def depth_distance(depth, intrinsics, camera_in_chassis, tcp, normal, config, exclude=None):
    """Fit a robust local surface around the TCP ray, not the image centre.

    Input must be rectified, metric depth with the corresponding depth intrinsics.
    Exclusion mask is True for tool/occluded pixels. No RGB-depth alignment assumed.
    """
    z = np.asarray(depth, dtype=float)
    k = np.asarray(intrinsics, dtype=float)
    t = transform(camera_in_chassis)
    tcp, normal = vector(tcp, 3), vector(normal, 3)
    if (z.ndim != 2 or k.shape != (3, 3) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.isclose(np.linalg.norm(normal), 1)):
        raise ValueError('invalid_depth_geometry')
    yy, xx = np.indices(z.shape)
    valid = np.isfinite(z) & (z >= config['depth_min_m']) & (z <= config['depth_max_m'])
    if exclude is not None:
        mask = np.asarray(exclude, bool)
        if mask.shape != z.shape:
            raise ValueError('invalid_depth_mask')
        valid &= ~mask
    camera_points = np.column_stack(((xx[valid]-k[0, 2])*z[valid]/k[0, 0],
                                     (yy[valid]-k[1, 2])*z[valid]/k[1, 1], z[valid]))
    points = camera_points @ t[:3, :3].T + t[:3, 3]
    relative = points - tcp
    along = relative @ normal
    lateral = np.linalg.norm(relative - along[:, None]*normal, axis=1)
    selected = (along > config['minimum_gap_m']) & (along < config['maximum_gap_m']) & (lateral < config['ray_radius_m'])
    points, along = points[selected], along[selected]
    if len(points) < config['min_points']:
        raise ValueError('insufficient_depth_support')
    keep = np.abs(along - np.median(along)) <= config['surface_band_m']
    inliers = points[keep]
    if len(inliers) < config['min_points'] or len(inliers)/len(points) < config['min_inlier_ratio']:
        raise ValueError('ambiguous_depth_surface')
    centre = np.median(inliers, axis=0)
    _, singular, vt = np.linalg.svd(inliers-centre, full_matrices=False)
    if singular[1]/np.sqrt(len(inliers)) < config['minimum_surface_span_m']:
        raise ValueError('degenerate_depth_surface')
    surface_normal = vt[-1]
    cosine = abs(float(surface_normal @ normal))
    tilt = float(np.arccos(np.clip(cosine, 0, 1)))
    residual = float(np.sqrt(np.mean(((inliers-centre) @ surface_normal)**2)))
    if tilt > config['max_panel_tilt_rad'] or residual > config['max_surface_residual_m']:
        raise ValueError('surface_tilt_or_noise')
    gap = float((centre-tcp) @ surface_normal / (normal @ surface_normal))
    if not config['minimum_gap_m'] < gap < config['maximum_gap_m']:
        raise ValueError('invalid_surface_gap')
    return dict(source='depth', gap_m=gap, tilt_rad=tilt, residual_m=residual,
                support=len(inliers), inlier_ratio=len(inliers)/len(points), trusted=True)


class Sensors:
    """ROS subscriptions only: constructing this object never acquires SDK rights."""
    def __init__(self, config, emit):
        from pressb.s1b_deployment.robot_state import RobotState
        from sensor_msgs.msg import CompressedImage, Image, CameraInfo
        from rclpy.qos import qos_profile_sensor_data
        self.c, self.emit = config, emit
        self.monitor = RobotState(emit, shared_context=True, control_only=True)
        self.lock = threading.Lock()
        self.frames = {}
        self.last_recorded = {}
        self.depth = None
        self.info = None
        self.error = None
        for key, topic in config['rgb_topics'].items():
            self.monitor.node.create_subscription(CompressedImage, topic,
                lambda m, key=key: self.rgb(key, m), qos_profile_sensor_data)
        # Topics must come from live discovery; do not guess vendor depth names.
        if config.get('depth_topic'):
            self.monitor.node.create_subscription(Image, config['depth_topic'], self.on_depth, qos_profile_sensor_data)
        if config.get('depth_info_topic'):
            self.monitor.node.create_subscription(CameraInfo, config['depth_info_topic'], self.on_info, qos_profile_sensor_data)

    def stamp(self, message):
        source = message.header.stamp.sec*10**9 + message.header.stamp.nanosec
        if source <= 0 or abs(time.time_ns()-source) > self.c['max_source_age_ns']:
            raise ValueError('stale_sensor_source_timestamp')
        return source

    def rgb(self, key, message):
        try:
            source = self.stamp(message)
            if 'jpeg' not in message.format.lower():
                raise ValueError('unsupported_rgb_encoding')
            from pressb.s1b_deployment.observation import JpegFrame
            data = bytes(message.data)
            JpegFrame(data).rgb(key)
            with self.lock:
                if key in self.frames and source <= self.frames[key]['source_ns']:
                    raise ValueError('nonmonotonic_rgb')
                self.frames[key] = dict(source_ns=source, receipt_ns=time.monotonic_ns(), jpeg_base64=base64.b64encode(data).decode())
                record = source-self.last_recorded.get(key, 0) >= self.c.get('record_interval_ns', 100_000_000)
                frame = dict(self.frames[key])
                if record:
                    self.last_recorded[key] = source
            if record:
                self.emit(dict(event='camera_frame', camera=key, **frame))
        except Exception as exc:
            self.error = str(exc)

    def on_info(self, message):
        with self.lock:
            self.info = dict(k=np.asarray(message.k).reshape(3, 3), frame=message.header.frame_id,
                             width=message.width, height=message.height, distortion=list(message.d))

    def on_depth(self, message):
        try:
            source = self.stamp(message)
            if message.encoding not in ('16UC1', '32FC1'):
                raise ValueError('unsupported_depth_encoding')
            dtype = np.dtype(('>' if message.is_bigendian else '<') + ('u2' if message.encoding == '16UC1' else 'f4'))
            array = np.ndarray((message.height, message.width), dtype=dtype,
                               buffer=bytes(message.data), strides=(message.step, dtype.itemsize)).astype(float)
            scale = self.c['depth_scale_m'][message.encoding]
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('unverified_depth_scale')
            with self.lock:
                if self.depth and source <= self.depth['source_ns']:
                    raise ValueError('nonmonotonic_depth')
                self.depth = dict(array=array*scale, source_ns=source, receipt_ns=time.monotonic_ns(), frame=message.header.frame_id)
        except Exception as exc:
            self.error = str(exc)

    def images(self):
        with self.lock:
            now = time.monotonic_ns()
            if set(self.frames) != {'head', 'right'} or any(
                    not 0 <= now-f['receipt_ns'] <= self.c['max_age_ns'] for f in self.frames.values()):
                raise ValueError('fresh_dual_images_required')
            frames = {k:dict(v) for k,v in self.frames.items()}
        if abs(frames['head']['source_ns']-frames['right']['source_ns']) > self.c['max_skew_ns']:
            raise ValueError('camera_skew')
        return frames

    def ranging(self, wrist, tcp, normal, config, state_source_ns):
        with self.lock:
            d, info = self.depth, self.info
        if d is None or info is None:
            raise ValueError('depth_unavailable')
        if not config['verified']:
            raise ValueError('depth_calibration_unverified')
        if not 0 <= time.monotonic_ns()-d['receipt_ns'] <= self.c['max_age_ns']:
            raise ValueError('stale_depth')
        if abs(d['source_ns']-state_source_ns) > self.c['max_skew_ns']:
            raise ValueError('depth_state_skew')
        if (d['frame'] != info['frame'] or d['frame'] != config['frame_id']
                or d['array'].shape != (info['height'], info['width'])
                or any(abs(x) > 1e-9 for x in info['distortion'])):
            raise ValueError('depth_requires_verified_rectified_intrinsics')
        mask = np.zeros(d['array'].shape, bool)
        for x0, y0, x1, y1 in config['exclude_rectangles']:
            mask[y0:y1, x0:x1] = True
        result = depth_distance(d['array'], info['k'], wrist @ transform(config['camera_in_wrist']),
                                tcp, normal, config, mask)
        result.update(source_ns=d['source_ns'], receipt_ns=d['receipt_ns'])
        archive = io.BytesIO()
        np.savez_compressed(archive, depth_m=d['array'], intrinsics=info['k'],
                            camera_in_chassis=wrist @ transform(config['camera_in_wrist']),
                            exclude_mask=mask)
        self.emit(dict(event='depth_estimate', **result,
                       depth_npz_base64=base64.b64encode(archive.getvalue()).decode()))
        return result

    def probe(self):
        rows = self.monitor.snapshot()
        with self.lock:
            depth = None if self.depth is None else dict(shape=list(self.depth['array'].shape),
                valid_fraction=float(np.mean(np.isfinite(self.depth['array']) & (self.depth['array'] > 0))),
                source_ns=self.depth['source_ns'], frame=self.depth['frame'])
        return dict(topics=self.monitor.node.get_topic_names_and_types(), depth=depth,
                    torque=rows['astribot_arm_right']['torque'], sensor_error=self.error)

    def close(self):
        self.monitor.close()
