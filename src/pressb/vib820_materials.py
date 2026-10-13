"""VIB-820 appearance: CC0 maps, layered dielectrics and MDL/PreviewSurface.

All optical parameters are reference-matched estimates, not measured BRDFs.
Material normal/rounded-edge shading never changes the collision geometry.
"""
from pxr import Gf, Sdf, UsdShade, UsdGeom
from .scene import ROOT, _material

ASSETS = ROOT / "assets/materials/vib820"


def mdl_surface(stage, material, module="OmniSurface", **inputs):
    shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/MDL")
    shader.CreateImplementationSourceAttr("sourceAsset")
    shader.SetSourceAsset(Sdf.AssetPath(module + ".mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier(module, "mdl")
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    for name, value in inputs.items():
        if isinstance(value, bool): typ = Sdf.ValueTypeNames.Bool
        elif isinstance(value, int): typ = Sdf.ValueTypeNames.Int
        elif isinstance(value, (str, Sdf.AssetPath)):
            typ = Sdf.ValueTypeNames.Asset
            value = Sdf.AssetPath(str(value)) if isinstance(value, str) else value
        elif isinstance(value, tuple):
            typ = Sdf.ValueTypeNames.Color3f if len(value) == 3 else Sdf.ValueTypeNames.Float2
            value = Gf.Vec3f(*value) if len(value) == 3 else Gf.Vec2f(*value)
        else: typ = Sdf.ValueTypeNames.Float
        attr = shader.CreateInput(name, typ)
        attr.Set(value)
        if typ == Sdf.ValueTypeNames.Asset and ("normal" in name or "roughness" in name):
            attr.GetAttr().SetColorSpace("raw")
    return shader


def _preview_texture(stage, material, channel, file, output="rgb", raw=False):
    base = str(material.GetPath())
    reader = UsdShade.Shader.Define(stage, base + "/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, base + "/" + channel + "Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(file)))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    shader = UsdShade.Shader.Get(stage, base + "/Shader")
    typ = Sdf.ValueTypeNames.Float if output == "r" else Sdf.ValueTypeNames.Normal3f if channel == "normal" else Sdf.ValueTypeNames.Color3f
    shader.CreateInput(channel, typ).ConnectToSource(texture.ConnectableAPI(), output)
    if channel == "normal":
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2,2,2,1))
        texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1,-1,-1,0))


def steel(stage, name, projected=False):
    color = ASSETS / "steel_clean_color.png"
    rough = ASSETS / "steel_clean_roughness.png"
    normal = ASSETS / "steel_clean_normal_gl.png"
    for p in [color, rough, normal]:
        if not p.is_file(): raise FileNotFoundError(f"Missing VIB-820 material asset: {p}")
    mat = _material(stage, name, (.54,.55,.56), 1., .25)
    for channel, file, output, raw in [("diffuseColor",color,"rgb",False),
            ("roughness",rough,"r",True), ("normal",normal,"rgb",True)]:
        _preview_texture(stage, mat, channel, file, output, raw)
    if projected:
        mdl_surface(stage, mat, "OmniPBR", diffuse_texture=str(color),
            metallic_constant=1., reflectionroughness_texture=str(rough),
            reflection_roughness_texture_influence=1., normalmap_texture=str(normal),
            flip_tangent_v=False, bump_factor=1., project_uvw=True, world_or_object=True,
            texture_scale=(5.,5.), texture_rotate=90., round_edges_radius=.0004)
    else:
        mdl_surface(stage, mat, diffuse_reflection_weight=1., diffuse_reflection_color_image=str(color),
            metalness=1., specular_reflection_roughness_image=str(rough),
            specular_reflection_anisotropy=.55, specular_reflection_anisotropy_rotation=.25,
            geometry_normal_image=str(normal), round_edges_radius=.0003)
    return mat


def planar_steel_uv(mesh):
    # Source hairlines run along image U. Map U vertically, one tile per 160 mm.
    points = mesh.GetPointsAttr().Get()
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
        [(float(p[2])/.16, -float(p[1])/.16) for p in points])


def coated(stage, name, color, roughness=.035, ior=1.52):
    mat = _material(stage, name, color, 0., .28)
    shader = UsdShade.Shader.Get(stage, str(mat.GetPath()) + "/Shader")
    shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(1.)
    shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mdl_surface(stage, mat, diffuse_reflection_weight=1., diffuse_reflection_color=color,
        specular_reflection_weight=0., coat_weight=1., coat_ior=ior,
        coat_roughness=roughness, round_edges_radius=.0003)
    return mat


def coat_print(stage, material, file, display=False):
    # Opaque rear print + a dielectric front interface; faster and more stable than
    # nested coplanar refractive plates. Actual refraction through edges is omitted.
    inputs = dict(diffuse_reflection_weight=1., diffuse_reflection_color_image=str(file),
        specular_reflection_weight=0., coat_weight=1., coat_ior=1.52 if display else 1.49,
        coat_roughness=.025 if display else .075)
    if display:
        inputs.update(emission_weight=1., emission_intensity=160., emission_color_image=str(file), emission_mode=1)
    mdl_surface(stage, material, **inputs)
    shader = UsdShade.Shader.Get(stage, str(material.GetPath()) + "/Shader")
    shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(1.)
    shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(inputs["coat_roughness"])
