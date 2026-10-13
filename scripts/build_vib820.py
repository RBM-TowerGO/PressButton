#!/usr/bin/env python3
"""Build and render the measured VIB-820 panel, with portable local USD assets."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/astribot_vib820.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--renderer",choices=["RayTracedLighting","PathTracing"],default="RayTracedLighting")
    parser.add_argument("--gpu",type=int,default=0,help="RTX/physics device index; make it visible through CUDA_VISIBLE_DEVICES")
    args=parser.parse_args()
    args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True,active_gpu=args.gpu,physics_gpu=args.gpu,multi_gpu=False,
        renderer=args.renderer,extra_args=["--/app/asyncRendering=false",
        "--/rtx/ecoMode/enabled=false","--/validate/p2p/enabled=false","--/validate/iommu/enabled=false"]))
    passed=False
    try:
        from pxr import Sdf,Usd
        import numpy as np
        from PIL import Image
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera
        from pressb.vib820_scene import build_scene,set_button_light
        cfg=json.loads(args.config.read_text())
        world=World(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/30,backend="numpy",device="cpu")
        scene=build_scene(world.stage,cfg)
        print("Scene authored; initializing cameras and material shaders",flush=True)
        cameras={name:Camera(prim_path=scene[key],name=name,resolution=res,frequency=-1)
                 for name,key,res in [("front","front_camera_path",(900,1200)),
                     ("panel","panel_camera_path",(1000,1000)),("lobby","lobby_camera_path",(1600,1000))]}
        world.reset()
        print("World reset complete",flush=True)
        for camera in cameras.values(): camera.initialize()
        for i in range(60): world.step(render=i%4==0)
        for state in [False,True]:
            set_button_light(world.stage,scene["buttons"][0],state)
            for _ in range(48): world.render()
            for name,camera in cameras.items():
                rgb=camera.get_rgba()[:,:,:3].astype(np.uint8)
                if np.std(rgb)<5: raise RuntimeError("Blank render: "+name)
                Image.fromarray(rgb).save(args.output/f"{name}_{'on' if state else 'off'}.png")
            print("Rendered", "light on" if state else "light off",flush=True)
        set_button_light(world.stage,scene["buttons"][0],False)
        # Bundle image assets; built-in MDL modules resolve through Isaac's library.
        import shutil
        texture_dir=args.output/"textures"; texture_dir.mkdir()
        output=args.output/"vib820_lobby.usda"
        world.stage.GetRootLayer().Export(str(output))
        stage=Usd.Stage.Open(str(output))
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                if attr.GetTypeName()==Sdf.ValueTypeNames.Asset:
                    value=attr.Get()
                    if value and value.path:
                        src=Path(value.path)
                        if src.name in {"OmniSurface.mdl","OmniPBR.mdl"} and not src.is_absolute():
                            continue
                        if not src.is_file():
                            raise FileNotFoundError(f"Unresolved asset {attr.GetPath()}: {src}")
                        dst=texture_dir/src.name; shutil.copy2(src,dst)
                        attr.Set(Sdf.AssetPath(os.path.relpath(dst,args.output)))
        stage.GetRootLayer().Save()
        report=dict(status="built",scope="Geometry/render preview; light-on images illustrate appearance, not contact validation",
                    config=cfg,dimensions_m=scene["dimensions"],button=scene["buttons"][0],
                    usd=str(output),reference="User measurements and reference video",renderer=args.renderer,
                    material_manifest="material_manifest.json",mdl_runtime="Isaac Sim 5.0 builtin OmniSurface/OmniPBR; USD Preview fallback included")
        shutil.copy2(ROOT/"assets/materials/vib820/manifest.json",args.output/"material_manifest.json")
        (args.output/"model_report.json").write_text(json.dumps(report,indent=2)+"\n")
        (args.output/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
        print("BUILT",output,flush=True); passed=True
    finally:
        app.app.post_quit(0 if passed else 1); app.close()


if __name__=="__main__": main()
