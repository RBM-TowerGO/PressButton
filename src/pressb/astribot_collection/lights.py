"""Immutable on/off materials with the verified VIB-820 optical parameters.

Binding state-specific materials invalidates tiled-render material history more
reliably than mutating a shared shader's inputs across cloned environments.
No collider, light intensity, button geometry or physical threshold changes.
"""
def prepare_feedback_materials(stage,info,prefix):
    from pxr import Gf,Sdf,UsdShade
    paths={}
    for on in (False,True):
        path=prefix+'/Looks/CollectionButton'+('On' if on else 'Off')
        material=UsdShade.Material.Define(stage,path)
        shader=UsdShade.Shader.Define(stage,path+'/Shader');shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*((.85,.91,1.) if on else (.055,.065,.075))))
        shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(.05)
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.35)
        shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*((1.5,1.7,2.) if on else (0,0,0))))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        paths[str(on)]=path
    info['feedback_materials']=paths
    info['feedback_geometry']=[prefix+'/Panel/LightRing']+[info['body_path']+'/'+n for n in ('ArrowStem','ArrowLeft','ArrowRight')]
    set_button_light(stage,info,False)

def set_button_light(stage,info,on=True):
    from pxr import UsdShade
    material=UsdShade.Material.Get(stage,info['feedback_materials'][str(bool(on))])
    for path in info['feedback_geometry']:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(material)
    stage.GetPrimAtPath(info['body_path']).GetAttribute('pressb:illuminated').Set(bool(on))
