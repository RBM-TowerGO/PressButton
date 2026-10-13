"""Geometry preparation for pressing with the original, closed S1 right hand."""
import numpy as np
from scipy.optimize import brentq

from .astribot_kinematics import AstribotKinematics, gripper_targets


def prepare_hand(stage, metadata):
    from pxr import Usd, UsdGeom
    kin = AstribotKinematics()
    cache = UsdGeom.XformCache()
    points = {}
    for suffix in ("L11", "R11"):
        name = "astribot_gripper_right_Link_" + suffix
        body = stage.GetPrimAtPath(metadata["robot_path"] + "/" + name)
        mesh = next(p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Mesh) and "/collisions/" in str(p.GetPath()))
        transform = np.array(cache.GetLocalToWorldTransform(mesh) * cache.GetLocalToWorldTransform(body).GetInverse()).T
        raw = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
        points[name] = raw @ transform[:3, :3].T + transform[:3, 3]

    def finger_points(angle):
        result = []
        for name, vertices in points.items():
            transform = kin.fk(name, gripper_targets(angle), root="astribot_gripper_right_base")
            result.append(vertices @ transform[:3, :3].T + transform[:3, 3])
        return result

    def gap(angle):
        left, right = finger_points(angle)
        return left[:, 0].min() - right[:, 0].max()

    # Keep 0.3 mm of numerical clearance between opposing collision meshes.
    # The upstream example's 0.5-rad clamp leaves a 55-mm opening in this asset.
    closed = brentq(lambda angle: gap(angle) - .0003, .5, 1.3)
    vertices = np.concatenate(finger_points(closed))
    tcp = np.eye(4)
    tcp[:3, 3] = [0, 0, vertices[:, 2].max()]
    tcp = kin.fk("astribot_gripper_right_base", root="astribot_arm_right_link_7") @ tcp
    front = vertices[vertices[:, 2] > vertices[:, 2].max() - .001]
    return dict(closed_angle_rad=closed, closed_gap_m=gap(closed),
                tcp_in_wrist=tcp.tolist(), front_bounds_m=[front.min(0).tolist(), front.max(0).tolist()],
                finger_vertices={name: value.tolist() for name, value in points.items()})


def complete_collision_geometry(stage, metadata):
    """Add conservative visual-mesh hulls for previously uncovered rigid links.

    Only the compound gripper linkage receives explicit internal filtering;
    opposite fingertips and all robot/environment pairs remain collision enabled.
    """
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    added = []
    for path in metadata["bodies_without_collision"]:
        body = stage.GetPrimAtPath(path)
        meshes = [p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Mesh) and "/visuals/" in str(p.GetPath())]
        for index, prim in enumerate(meshes):
            source = UsdGeom.Mesh(prim)
            transform = np.array(cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse()).T
            points = np.asarray(source.GetPointsAttr().Get())
            points = points @ transform[:3, :3].T + transform[:3, 3]
            target = UsdGeom.Mesh.Define(stage, path + f"/TaskCollision{index}")
            target.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
            target.CreateFaceVertexCountsAttr(source.GetFaceVertexCountsAttr().Get())
            target.CreateFaceVertexIndicesAttr(source.GetFaceVertexIndicesAttr().Get())
            target.CreatePurposeAttr("guide")
            UsdPhysics.CollisionAPI.Apply(target.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(target.GetPrim()).CreateApproximationAttr("convexHull")
            added.append(str(target.GetPath()))
    filtered = []
    # Housings around consecutive axes overlap by construction (e.g. the
    # shoulder shell and the rigid torso mounting). Keep their external hulls
    # while filtering only nearby links along the original kinematic graph.
    kin = AstribotKinematics()
    graph = {}
    for child, (parent, *_rest) in kin.joints.items():
        child_prim = stage.GetPrimAtPath(metadata["robot_path"] + "/" + child)
        if not child_prim or not child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        while parent in kin.joints:
            parent_prim = stage.GetPrimAtPath(metadata["robot_path"] + "/" + parent)
            if parent_prim and parent_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                break
            parent = kin.joints[parent][0]
        graph.setdefault(child, set()).add(parent)
        graph.setdefault(parent, set()).add(child)
    for path in metadata["bodies_without_collision"]:
        name = path.rsplit("/", 1)[-1]
        near = set(graph.get(name, ()))
        near.update(n for parent in list(near) for n in graph.get(parent, ()))
        for other_name in sorted(near - {name}):
            other = metadata["robot_path"] + "/" + other_name
            prim = stage.GetPrimAtPath(other)
            if prim and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path)).CreateFilteredPairsRel().AddTarget(other)
                filtered.append([path, other])
    for side in ("left", "right"):
        paths = [metadata["robot_path"] + f"/astribot_gripper_{side}_Link_{suffix}"
                 for suffix in ("L1", "L2", "L11", "R1", "R2", "R11")]
        for i, path in enumerate(paths):
            for other in paths[i+1:]:
                if path.endswith("_L11") and other.endswith("_R11"):
                    continue
                UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path)).CreateFilteredPairsRel().AddTarget(other)
                filtered.append([path, other])
    for prim in Usd.PrimRange(stage.GetPrimAtPath(metadata["robot_path"])):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateContactOffsetAttr(.0001 if "gripper" in str(prim.GetPath()) else .001)
            api.CreateRestOffsetAttr(0.)
    return dict(added_colliders=added, filtered_mechanism_pairs=filtered,
                note="Visual convex hulls; new hulls filter two-hop kinematic neighbours and gripper linkage internals; opposing tips retained")


def smooth_fraction(t):
    t = np.clip(t, 0., 1.)
    return t*t*t*(10. - 15.*t + 6.*t*t)
