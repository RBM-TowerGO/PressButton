"""Astribot S1 asset adapter for Isaac Sim 5; import after SimulationApp starts.

This supplies a fixed-base articulation and joint-space control metadata.
It does not replace PressB's PiPER IK or its button-task configuration.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "assets/generated/astribot/astribot_s1.usdc"
URDF = ROOT / "vendor/astribot_descriptions/urdf/astribot_s1_urdf/astribot_whole_body_maniskill.urdf"
ARM_JOINTS = {side: [f"astribot_arm_{side}_joint_{i}" for i in range(1, 8)]
              for side in ("left", "right")}


def add_astribot(stage, path="/World/Astribot", *, floor_clearance=0.002):
    if not ASSET.is_file():
        raise FileNotFoundError(f"Run scripts/prepare_astribot_asset.py first: {ASSET}")
    if stage.GetPrimAtPath(path):
        raise ValueError(f"Robot prim already exists: {path}")
    container = UsdGeom.Xform.Define(stage, path)
    container.GetPrim().GetReferences().AddReference(str(ASSET))
    robot_path = path + "/astribot_whole_body"
    robot = stage.GetPrimAtPath(robot_path)
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide", "proxy"]).ComputeWorldBound(robot).ComputeAlignedBox()
    # Include guide-purpose collision geometry below the visible base mesh.
    lift = float(-bbox.GetMin()[2] + floor_clearance)
    container.AddTranslateOp().Set(Gf.Vec3d(0, 0, lift))
    limits = {j.get("name"): j.find("limit") for j in ET.parse(URDF).findall("joint")
              if j.get("type") == "revolute"}
    joint_paths = {}
    collision_shapes = []
    # The upstream file applies CollisionAPI to grouping Xforms. PhysX needs
    # the API on the actual collision shapes; use the supplied geometry only.
    for prim in Usd.PrimRange(robot):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsA(UsdGeom.Gprim):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
        if prim.IsA(UsdGeom.Gprim) and "/collisions/" in str(prim.GetPath()):
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            collision_shapes.append(str(prim.GetPath()))
    for prim in Usd.PrimRange(robot):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            phys = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            phys.CreateSolverPositionIterationCountAttr(32)
            phys.CreateSolverVelocityIterationCountAttr(4)
            phys.CreateEnabledSelfCollisionsAttr(True)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            name = prim.GetName()
            limit = limits[name]
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            # Explicit simulation gains; retain URDF effort limits and geometry.
            if "torso" in name:
                stiffness, damping = 12000., 600.
            elif "gripper" in name:
                stiffness, damping = 80., 4.
            else:
                stiffness, damping = 2000., 100.
            drive.CreateStiffnessAttr(stiffness)
            drive.CreateDampingAttr(damping)
            drive.CreateMaxForceAttr(float(limit.get("effort")))
            drive.CreateTargetPositionAttr(0.)
            drive.CreateTargetVelocityAttr(0.)
            PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(
                float(limit.get("velocity")) * 180. / 3.141592653589793)
            joint_paths[name] = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(0.001)
            collision.CreateRestOffsetAttr(0.)
    root_joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(robot_path + "/root_joint"))
    if not root_joint:
        raise ValueError("Missing fixed-base root joint")
    # The world side of a joint has no parent transform: move its anchor too.
    root_joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, lift))
    return dict(robot_path=robot_path, articulation_path=robot_path + "/root_joint",
                base_link=robot_path + "/astribot_torso_base", base_lift_m=lift,
                arm_joints=ARM_JOINTS, joint_paths=joint_paths,
                wrist_links={s: robot_path + f"/astribot_arm_{s}_link_7" for s in ARM_JOINTS},
                collision_shapes=collision_shapes,
                bodies_without_collision=[str(p.GetPath()) for p in Usd.PrimRange(robot)
                    if p.HasAPI(UsdPhysics.RigidBodyAPI) and not any(
                        c.startswith(str(p.GetPath()) + "/") for c in collision_shapes)],
                self_collisions=True, fixed_base=True)
