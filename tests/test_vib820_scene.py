"""Measured geometry, a real button aperture, and independent request feedback."""
import json
from pathlib import Path

import numpy as np
import pytest

Usd=pytest.importorskip("pxr.Usd")
from pxr import UsdGeom, UsdPhysics
from pressb.vib820_scene import build_scene, dimensions, set_button_light

ROOT=Path(__file__).parents[1]


@pytest.fixture
def model():
    cfg=json.loads((ROOT/"configs/astribot_vib820.json").read_text())
    stage=Usd.Stage.CreateInMemory()
    return stage,cfg,build_scene(stage,cfg)


def test_measured_contact_geometry_and_joint(model):
    stage,cfg,scene=model
    assert list(scene["buttons"]) == [0]
    b=scene["buttons"][0]
    assert b["center"] == [.46,0.,1.2]
    cap=UsdGeom.Cylinder.Get(stage,b["collision_path"])
    assert cap.GetRadiusAttr().Get() == .015
    assert b["rest_x"]-cap.GetHeightAttr().Get()/2 == pytest.approx(.46)
    joint=UsdPhysics.PrismaticJoint.Get(stage,b["joint_path"])
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(.002)
    assert joint.GetAxisAttr().Get() == "X"
    assert scene["dimensions"]["display_top"] == pytest.approx(1.469)
    assert scene["dimensions"]["panel_top"]-scene["dimensions"]["panel_bottom"] == pytest.approx(.4)


def test_static_meshes_preserve_open_button_bore(model):
    stage,cfg,scene=model
    for path,min_radius in [("/World/Panel/Faceplate",.0186),("/World/Panel/FixedButtonRing",.0153)]:
        mesh=UsdGeom.Mesh.Get(stage,path)
        assert mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "none"
        points=np.array(mesh.GetPointsAttr().Get())
        radial=np.linalg.norm(points[:,1:]-[0.,1.2],axis=1)
        assert radial.min() == pytest.approx(min_radius,abs=1e-7)
    assert not stage.GetPrimAtPath("/World/Panel/LightRing").HasAPI(UsdPhysics.CollisionAPI)


def test_feedback_changes_without_moving_cap(model):
    stage,cfg,scene=model
    b=scene["buttons"][0]; body=stage.GetPrimAtPath(b["body_path"])
    before=body.GetAttribute("xformOp:translate").Get()
    for on in [True,False]:
        set_button_light(stage,b,on)
        assert body.GetAttribute("pressb:illuminated").Get() is on
        assert body.GetAttribute("xformOp:translate").Get() == before


def test_bad_dimensions_reject_overlap(model):
    _,cfg,_=model
    cfg["button_outer_diameter"] = .020
    with pytest.raises(ValueError,match="nested"):
        dimensions(cfg)


def test_measured_recess_and_clear_opening(model):
    stage,cfg,scene=model
    cache=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_])
    def bounds(name):
        return cache.ComputeWorldBound(stage.GetPrimAtPath('/World/Environment/'+name)).ComputeAlignedRange()
    near,far=bounds('JambReturn0'),bounds('JambReturn1')
    assert far.GetMin()[1]-near.GetMax()[1] == pytest.approx(1.30)
    wall_x=bounds('PanelWall').GetMin()[0]
    for i in range(2):
        door=bounds(f'DoorLeaf{i}')
        ret=bounds(f'JambReturn{i}')
        assert door.GetMin()[0]-wall_x == pytest.approx(.35)
        assert ret.GetMin()[0] == pytest.approx(wall_x)
        assert ret.GetMax()[0] == pytest.approx(door.GetMin()[0])
    assert bounds('HeaderReturn').GetMin()[2] == pytest.approx(cfg['door_height'])
    assert scene['buttons'][0]['center'] == [.46,0.,1.2]
