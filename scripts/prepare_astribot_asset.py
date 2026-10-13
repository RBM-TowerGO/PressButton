#!/usr/bin/env python3
"""Prepare the official flattened S1 USDA for referencing in a metre-based scene.

Run with the isolated OpenUSD environment, or after Isaac Sim has started.
The upstream binary USD is not used: it contains an unavailable absolute reference.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

    manifest = json.loads((ROOT / "assets/astribot_sources.lock.json").read_text())
    entry = next(x for x in manifest["files"] if x["path"].endswith("astribot_s1.usda"))
    source = ROOT / manifest["destination"] / entry["path"]
    with source.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != entry["sha256"]:
        raise ValueError("Upstream USDA checksum mismatch")
    stage = Usd.Stage.Open(str(source))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("Expected metres and Z up")
    deinstanced = 0
    # Expand prototype geometry before referencing /World in another scene.
    # Otherwise the flattened prototypes live outside the referenced subtree.
    for prim in list(stage.Traverse()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
            deinstanced += 1
    output = ROOT / "assets/generated/astribot"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "astribot_s1.usdc"
    stage.Flatten().Export(str(target))
    check = Usd.Stage.Open(str(target))
    _, dependencies, unresolved = UsdUtils.ComputeAllDependencies(str(target))
    # OmniPBR is supplied by Isaac Sim, not a missing mesh or downloaded texture.
    missing = [x for x in unresolved if x != "OmniPBR.mdl"]
    if missing or dependencies:
        raise ValueError(f"Unexpected external assets: {dependencies}, {missing}")
    robot = check.GetPrimAtPath("/World/astribot_whole_body")
    prims = list(Usd.PrimRange(robot))
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(robot).ComputeAlignedBox()
    joints = [p.GetName() for p in prims if p.IsA(UsdPhysics.RevoluteJoint)]
    if len(joints) != 32:
        raise ValueError(f"Expected 32 revolute joints, got {len(joints)}")
    report = dict(source=str(source.relative_to(ROOT)), source_sha256=checksum,
                  revision=manifest["revision"], output=str(target.relative_to(ROOT)),
                  deinstanced_prims=deinstanced, joint_names=joints,
                  meshes=sum(p.IsA(UsdGeom.Mesh) for p in prims),
                  rigid_bodies=sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims),
                  colliders=sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims),
                  bbox_min_m=list(bbox.GetMin()), bbox_max_m=list(bbox.GetMax()),
                  unresolved_assets=missing, runtime_materials=unresolved,
                  upstream_binary_usd_usable=False,
                  upstream_binary_usd_issue="Unavailable absolute author-machine USD reference")
    with target.open("rb") as stream:
        report["output_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
