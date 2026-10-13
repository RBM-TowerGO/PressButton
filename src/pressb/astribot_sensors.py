"""Robot-mounted simulated sensors. Import only after SimulationApp startup.

Camera depth is optical Z (metres); lidar points are in the lidar frame (metres).
IMU acceleration includes gravity; orientations use wxyz. Each stream retains
its own simulation timestamp. This is an ideal simulation rig, not S1 calibration.
"""
import copy
import json
from pathlib import Path

import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics
import omni.kit.commands
from isaacsim.core.utils.extensions import enable_extension

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/astribot_sensors.json"


class AstribotSensors:
    def __init__(self, world, robot, adapter, config=DEFAULT_CONFIG):
        enable_extension("isaacsim.sensors.physx")
        enable_extension("isaacsim.sensors.physics")
        from isaacsim.sensors.camera import Camera
        from isaacsim.sensors.physics import IMUSensor, ContactSensor
        from isaacsim.sensors.physx import _range_sensor

        self.world, self.robot = world, robot
        self.config = json.loads(Path(config).read_text())
        cfg = self.config
        self.epoch = 0
        self.cameras, self.imus, self.contacts = {}, {}, {}
        self.paths = {}
        robot_path = adapter["robot_path"]
        for name, mount in cfg["cameras"].items():
            path = robot_path + "/" + mount["link"] + "/RGBD"
            prim = UsdGeom.Camera.Define(world.stage, path)
            transform = Gf.Matrix4d().SetLookAt(*[Gf.Vec3d(*mount[k]) for k in ("eye", "target", "up")]).GetInverse()
            prim.AddTransformOp().Set(transform)
            width, height = cfg["camera_resolution"]
            prim.CreateFocalLengthAttr(cfg["camera_focal_length_mm"])
            prim.CreateHorizontalApertureAttr(cfg["camera_horizontal_aperture_mm"])
            prim.CreateVerticalApertureAttr(cfg["camera_horizontal_aperture_mm"] * height / width)
            prim.CreateClippingRangeAttr(Gf.Vec2f(*cfg["camera_clip_m"]))
            self.cameras[name] = Camera(prim_path=path, name=name + "_rgbd", resolution=(width, height), frequency=-1)
            self.paths[name + "_camera"] = path
        mount = cfg["lidar"]
        ok, lidar = omni.kit.commands.execute("RangeSensorCreateLidar", path="/Lidar",
            parent=robot_path + "/" + mount["link"], draw_points=False, draw_lines=False,
            high_lod=True, yaw_offset=0., **{k: v for k, v in mount.items() if k not in ("link", "translation")})
        if not ok:
            raise RuntimeError("Failed to create PhysX lidar")
        lidar.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(*mount["translation"]))
        self.lidar_path = str(lidar.GetPath())
        self.paths["lidar"] = self.lidar_path
        self.lidar = _range_sensor.acquire_lidar_sensor_interface()
        for name, link in cfg["imus"].items():
            path = robot_path + "/" + link + "/IMU"
            self.imus[name] = IMUSensor(prim_path=path, name=name + "_imu", dt=1 / cfg["physics_hz"],
                translation=np.zeros(3), orientation=np.array([1., 0., 0., 0.]))
            self.paths[name + "_imu"] = path
        # Attach one sensor per rigid body that actually has collision geometry.
        bodies = set()
        for path in adapter["collision_shapes"]:
            body = world.stage.GetPrimAtPath(path)
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if body:
                bodies.add(str(body.GetPath()))
        for body in sorted(bodies):
            name = body.rsplit("/", 1)[-1]
            self.contacts[name] = ContactSensor(prim_path=body + "/Contact", name=name + "_contact",
                dt=1 / cfg["physics_hz"], min_threshold=0., max_threshold=1e6, radius=-1.)
            self.contacts[name].add_raw_contact_data_to_frame()
            self.paths[name + "_contact"] = body + "/Contact"
        self.coverage = {"contact_bodies": sorted(bodies), "bodies_without_collision": adapter["bodies_without_collision"]}

    def initialize(self):
        """Call after world.reset(), including every subsequent reset."""
        self.epoch += 1
        for camera in self.cameras.values():
            camera.initialize()
            camera.add_distance_to_image_plane_to_frame()
        for sensor in [*self.imus.values(), *self.contacts.values()]:
            sensor.initialize()
        # Isaac Sim 5.0 exposes this map through articulation metadata; DOF
        # indices are not a safe substitute for incoming-joint wrench rows.
        metadata = self.robot._articulation_view._metadata
        self.wrench_rows = {name: int(index) + 1 for name, index in metadata.joint_indices.items()}

    def physics_frame(self):
        return dict(epoch=self.epoch, time=float(self.world.current_time),
            joints=dict(names=list(self.robot.dof_names), position=self.robot.get_joint_positions().copy(),
                velocity=self.robot.get_joint_velocities().copy(), effort=self.robot.get_measured_joint_efforts().copy(),
                reaction_wrench=self.robot.get_measured_joint_forces().copy()),
            imus={n: copy.deepcopy(s.get_current_frame(read_gravity=True)) for n, s in self.imus.items()},
            contacts={n: copy.deepcopy(s.get_current_frame()) for n, s in self.contacts.items()})

    def capture(self):
        """Read after a rendered physics step; no hidden stepping or time advance."""
        frame = self.physics_frame()
        frame["cameras"] = {}
        for name, camera in self.cameras.items():
            raw = camera.get_current_frame()
            rgba = camera.get_rgba()
            if rgba.ndim != 3 or raw.get("distance_to_image_plane") is None:
                raise RuntimeError(f"Camera {name} is not ready; render warm-up frames after initialize()")
            frame["cameras"][name] = dict(rgb=rgba[:, :, :3].copy(),
                depth_m=np.array(raw.get("distance_to_image_plane"), copy=True),
                time=float(raw.get("rendering_time", -1)), reference_time=copy.deepcopy(raw.get("rendering_frame")),
                intrinsics=np.asarray(camera.get_intrinsics_matrix()).copy())
        depth = np.array(self.lidar.get_linear_depth_data(self.lidar_path), copy=True)
        points = np.array(self.lidar.get_point_cloud_data(self.lidar_path), copy=True)
        valid = np.isfinite(depth) & (depth >= self.config["lidar"]["min_range"]) & (depth < self.config["lidar"]["max_range"] - .001)
        frame["lidar"] = dict(time=frame["time"], range_m=depth, points_local_m=points, hit=valid,
            azimuth_rad=np.array(self.lidar.get_azimuth_data(self.lidar_path), copy=True),
            zenith_rad=np.array(self.lidar.get_zenith_data(self.lidar_path), copy=True))
        cache = UsdGeom.XformCache()
        # USD matrices are transposed here for column-vector p_world = T @ p_local.
        frame["world_from_sensor"] = {n: np.array(cache.GetLocalToWorldTransform(self.world.stage.GetPrimAtPath(p))).T
                                       for n, p in self.paths.items()}
        return frame

    def calibration(self):
        return dict(config=self.config, paths=self.paths, coverage=self.coverage,
            camera_axes="USD: +X right, +Y up, -Z forward; depth is positive optical Z",
            lidar_axes="local XYZ; metres; no-hit points must be removed using hit mask",
            imu_axes="sensor-local acceleration m/s^2 including gravity, angular velocity rad/s; world orientation wxyz",
            wrench_rows=getattr(self, "wrench_rows", {}),
            wrench_axes="link incoming joint frame, [Fx,Fy,Fz,Tx,Ty,Tz], N and Nm; row 0 base; includes gravity and dynamics, not a calibrated wrist F/T sensor",
            transforms="world_from_sensor: column-vector 4x4; metres",
            timing="physics 120 Hz; camera and recorded full lidar sweep 10 Hz; per-stream simulation seconds; epoch changes on reset",
            model="ideal RGB-D, PhysX full-sweep raycast lidar, physics IMU/contact/joint state; no hardware noise or ROS publication")

    def save_snapshot(self, output):
        """Save actual observations and calibration, ready for offline consumers."""
        from PIL import Image
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        frame = self.capture()
        arrays = {}
        camera_metadata = {}
        for name, cam in frame["cameras"].items():
            Image.fromarray(cam["rgb"].astype(np.uint8)).save(output / (name + "_rgb.png"))
            np.save(output / (name + "_depth_m.npy"), cam["depth_m"], allow_pickle=False)
            camera_metadata[name] = {k: v for k, v in cam.items() if k not in ("rgb", "depth_m")}
        arrays.update({"lidar_" + k: v for k, v in frame["lidar"].items()})
        arrays.update({"world_from_" + k: v for k, v in frame["world_from_sensor"].items()})
        arrays.update({"joint_" + k: v for k, v in frame["joints"].items()})
        np.savez_compressed(output / "sensors.npz", **arrays)

        def serialize(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)

        metadata = dict(epoch=frame["epoch"], time=frame["time"], cameras=camera_metadata,
                        imus=frame["imus"], contacts=frame["contacts"], calibration=self.calibration())
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=serialize) + "\n")
        return output
