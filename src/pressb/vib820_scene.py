"""Measured single down-call panel and approximate local elevator lobby.

Metres, +Z up, +X into the wall, +Y left when facing the panel.
The 0 button ID denotes down-call, never a floor number.
"""
import hashlib
import math
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade
from .scene import ROOT, PhysxSchema, _bind, _camera, _cube, _material, _texture_material
from .vib820_materials import ASSETS, steel, coated, mdl_surface, coat_print, planar_steel_uv


def dimensions(cfg):
    """Resolve measured elevations and explicitly provisional mounting margins."""
    keys = ("panel_width", "panel_height", "panel_depth", "display_width", "display_height",
            "button_cap_diameter", "button_outer_diameter", "button_travel", "display_top_margin")
    if any(not math.isfinite(cfg[k]) or cfg[k] <= 0 for k in keys):
        raise ValueError("VIB-820 dimensions must be finite and positive")
    top = cfg["display_bottom_z"] + cfg["display_height"] + cfg["display_top_margin"]
    bottom = top - cfg["panel_height"]
    z = cfg["display_bottom_z"] - cfg["button_below_display"]
    if not (cfg["button_cap_diameter"] < cfg["button_outer_diameter"] < cfg["panel_width"]):
        raise ValueError("Cap, outer ring and panel widths must be strictly nested")
    if cfg["display_width"] >= cfg["panel_width"] or z-cfg["button_outer_diameter"]/2 <= bottom:
        raise ValueError("Display or button does not fit measured panel")
    if z+cfg["button_outer_diameter"]/2 >= cfg["display_bottom_z"]:
        raise ValueError("Button overlaps display")
    recess=cfg.get("jamb_recess_depth",0.)
    if not math.isfinite(recess) or recess < 0:
        raise ValueError("Jamb recess depth must be finite and nonnegative")
    return dict(panel_top=top, panel_bottom=bottom, button_z=z,
                display_top=cfg["display_bottom_z"]+cfg["display_height"],
                wall_x=cfg["button_face_x"]+cfg["panel_depth"],
                door_face_x=cfg["button_face_x"]+cfg["panel_depth"]+recess,
                door_clear_width=cfg["door_width"],jamb_recess_depth=recess)


def _mesh(stage, path, points, faces, material, collision=False):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    _bind(mesh.GetPrim(), material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        # Static triangle mesh preserves the aperture; a convex hull fills it.
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        if PhysxSchema:
            api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            api.CreateContactOffsetAttr(.0001)
            api.CreateRestOffsetAttr(0.)
    return mesh


def _annulus(stage, path, x, y, z, inner, outer, depth, material, collision=False):
    n = 96
    points = [(xx, y+r*math.cos(2*math.pi*i/n), z+r*math.sin(2*math.pi*i/n))
              for xx,r in [(x,outer),(x,inner),(x+depth,outer),(x+depth,inner)] for i in range(n)]
    faces = []
    for i in range(n):
        j = (i+1)%n
        faces.extend([(i,j,n+j,n+i),(2*n+i,3*n+i,3*n+j,2*n+j),
                      (i,2*n+i,2*n+j,j),(n+i,n+j,3*n+j,3*n+i)])
    return _mesh(stage,path,points,[tuple(reversed(f)) for f in faces],material,collision)


def _decal(stage, path, center, width, height, kind):
    """Deterministic vector-drawn reference signage, not extracted reflections."""
    from PIL import Image, ImageDraw, ImageFont
    import io
    directory = ROOT/"assets/generated/vib820"
    directory.mkdir(parents=True, exist_ok=True)
    w,h = 600, max(80, round(600*height/width))
    colors = {"display":(3,6,10),"fire":(150,24,35),"access":(18,47,104),
              "stairs":(242,240,223),"warning":(238,238,229)}
    im = Image.new("RGB",(w,h),colors[kind]); d=ImageDraw.Draw(im)
    fontpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    def font(s): return ImageFont.truetype(fontpath,s)
    if kind=="display":
        # Static state from the video, with a true physical down button below.
        d.line([(220,int(h*.38)),(300,int(h*.30)),(380,int(h*.38))],fill=(43,151,246),width=18)
        d.line([(300,int(h*.31)),(300,int(h*.46))],fill=(43,151,246),width=18)
        d.text((300,int(h*.60)),"1",font=font(245),anchor="mm",fill=(43,151,246))
    elif kind=="access":
        d.rounded_rectangle((14,14,w-14,h-14),radius=18,outline="white",width=6)
        d.ellipse((240,85,305,150),fill="white")
        d.arc((120,240,390,510),35,300,fill="white",width=28)
        d.line([(271,150),(278,312),(383,312),(440,423),(493,397)],fill="white",width=29)
        d.line([(275,235),(369,235)],fill="white",width=24)
    elif kind=="stairs":
        yellow=(235,198,47)
        d.rectangle((0,0,w,int(h*.13)),fill=yellow)
        d.rectangle((0,int(h*.90),w,h),fill=yellow)
        cx,cy,r=125,int(h*.48),49
        d.ellipse((cx-r,cy-r,cx+r,cy+r),fill=yellow)
        for i in range(16):
            a=2*math.pi*i/16
            d.line([(cx+65*math.cos(a),cy+65*math.sin(a)),
                    (cx+82*math.cos(a),cy+82*math.sin(a))],fill=yellow,width=9)
        d.multiline_text((390,h*.50),"Going up one?\nGoing down two?\nUse the stairs",font=font(28),
                         anchor="mm",align="center",spacing=15,fill=(53,51,43))
    elif kind=="warning":
        d.rectangle((3,3,w-3,h-3),outline=(86,86,77),width=3)
        d.rectangle((4,int(h*.85),w-4,h-4),fill=(228,192,47))
        d.text((26,17),"SAFETY NOTICE",font=font(24),fill=(152,32,30))
        d.multiline_text((26,58),"Passenger lift\nPlease keep the entrance clear\nUse the stairs in an emergency\nFollow the building safety instructions",font=font(19),
                         spacing=9,fill=(36,36,33))
        d.text((w/2,h*.925),"W A R N I N G",font=font(23),anchor="mm",fill=(103,35,22))
    else:
        d.rectangle((12,12,w-12,h-12),outline="white",width=5)
        d.multiline_text((w/2,h/2),"WHEN THERE\nIS A FIRE\nDO NOT USE\nTHE LIFT",font=font(55),
                         anchor="mm",align="center",spacing=18,fill="white")
    stream=io.BytesIO(); im.save(stream,format="PNG"); data=stream.getvalue()
    file=directory/(kind+"_"+hashlib.sha256(data).hexdigest()[:12]+".png")
    if not file.exists(): file.write_bytes(data)
    material=_texture_material(stage,"VIB_"+kind,file,.2 if kind=="display" else .45)
    coat_print(stage,material,file,display=kind=="display")
    if kind=="display":
        shader=UsdShade.Shader.Get(stage,str(material.GetPath())+"/Shader")
        tex=UsdShade.Shader.Get(stage,str(material.GetPath())+"/Texture")
        shader.GetInput("emissiveColor").ConnectToSource(tex.ConnectableAPI(),"rgb")
    x,y,z=center
    mesh=_mesh(stage,path,[(x,y+width/2,z-height/2),(x,y-width/2,z-height/2),
                           (x,y-width/2,z+height/2),(x,y+width/2,z+height/2)],[(0,1,2,3)],material)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,"vertex").Set(
        [(0,0),(1,0),(1,1),(0,1)])


def set_button_light(stage, info, on=True):
    shader=UsdShade.Shader.Get(stage,info["shader_path"])
    shader.GetInput("diffuseColor").Set(Gf.Vec3f(*((.85,.91,1.) if on else (.055,.065,.075))))
    shader.GetInput("emissiveColor").Set(Gf.Vec3f(*((1.5,1.7,2.) if on else (0,0,0))))
    stage.GetPrimAtPath(info["body_path"]).GetAttribute("pressb:illuminated").Set(bool(on))


def build_scene(stage,cfg):
    d=dimensions(cfg)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    world=UsdGeom.Xform.Define(stage,"/World"); stage.SetDefaultPrim(world.GetPrim())
    for p in ["/World/Environment","/World/Panel","/World/Lights"]: UsdGeom.Xform.Define(stage,p)
    UsdGeom.Scope.Define(stage,"/World/Looks")
    steelmat=steel(stage,"VIB_SatinSteel")
    doorsteel=steel(stage,"VIB_DoorSteel",projected=True)
    capmat=_material(stage,"VIB_ButtonSteel",(.52,.54,.56),.82,.34)
    mdl_surface(stage,capmat,diffuse_reflection_weight=1.,diffuse_reflection_color=(.55,.56,.57),
                metalness=1.,specular_reflection_roughness=.21,round_edges_radius=.0003)
    wallmat=coated(stage,"VIB_GlossWall",(.79,.80,.77),.028,1.52)
    floor=_material(stage,"VIB_Stone",(.54,.51,.45),0,.5)
    dark=_material(stage,"VIB_Gap",(.012,.016,.019),.0,.6)
    light=_material(stage,"VIB_ButtonLight",(.055,.065,.075),.05,.35)
    x=cfg["button_face_x"]; y=cfg.get("panel_center_y",0.)
    width=cfg["panel_width"]; bottom=d["panel_bottom"]; top=d["panel_top"]; z=d["button_z"]
    wallx=d["wall_x"]
    # Door is to the viewer's left: positive world Y.
    jamb_y=y+width/2+cfg["panel_to_jamb_gap"]
    door_right=jamb_y+cfg["jamb_strip_width"]
    dw,dh=cfg["door_width"],cfg["door_height"]
    _cube(stage,"/World/Environment/Floor",(-1,0,-.04),(4,5,.08),floor,True)
    _cube(stage,"/World/Environment/PanelWall",(wallx+.05,(jamb_y-1.5)/2,1.4),(.1,jamb_y+1.5,2.8),wallmat,True)
    _cube(stage,"/World/Environment/LeftWall",(wallx+.05,door_right+dw+.30,1.4),(.1,.60,2.8),wallmat,True)
    _cube(stage,"/World/Environment/OverDoor",(wallx+.05,door_right+dw/2,(dh+2.8)/2),(.1,dw,2.8-dh),wallmat,True)
    for i in range(2):
        _cube(stage,f"/World/Environment/DoorLeaf{i}",(d["door_face_x"]+.0125,door_right+(i+.5)*dw/2,dh/2),(.025,dw/2-.003,dh),doorsteel,True)
    for i,jy in enumerate([jamb_y+cfg["jamb_strip_width"]/2,door_right+dw+cfg["jamb_strip_width"]/2]):
        _cube(stage,f"/World/Environment/Jamb{i}",(wallx-.002,jy,dh/2),(.018,cfg["jamb_strip_width"],dh),doorsteel,True)
        if d["jamb_recess_depth"] > 0:
            # Return faces run into the wall; their inner edges define clear width.
            _cube(stage,f"/World/Environment/JambReturn{i}",
                  ((wallx+d["door_face_x"])/2,jy,dh/2),
                  (d["jamb_recess_depth"],cfg["jamb_strip_width"],dh),doorsteel,True)
    _cube(stage,"/World/Environment/DoorHeader",(wallx-.002,door_right+dw/2,dh+.025),(.018,dw+.03,.05),doorsteel,True)
    if d["jamb_recess_depth"] > 0:
        _cube(stage,"/World/Environment/HeaderReturn",
              ((wallx+d["door_face_x"])/2,door_right+dw/2,dh+.025),
              (d["jamb_recess_depth"],dw+.03,.05),doorsteel,True)
    for i in range(7):
        _cube(stage,f"/World/Environment/TileSeam{i}",(-2.5+i*.6,0,.0002),(.002,5,.0004),dark)
    for i in range(8):
        _cube(stage,f"/World/Environment/TileCrossSeam{i}",(-1,-2.1+i*.6,.0002),(4,.002,.0004),dark)
    # Box shell has a shallow rear plate; faceplate uses an actual aperture.
    dep=cfg["panel_depth"]
    _cube(stage,"/World/Panel/Back",(wallx-.001,y,(top+bottom)/2),(.002,width,cfg["panel_height"]),doorsteel,True)
    for name,cy,cz,sy,sz in [("Left",y+width/2-.001,(top+bottom)/2,.002,top-bottom),
        ("Right",y-width/2+.001,(top+bottom)/2,.002,top-bottom),
        ("Top",y,top-.001,width,.002),("Bottom",y,bottom+.001,width,.002)]:
        _cube(stage,"/World/Panel/"+name,(x+dep/2,cy,cz),(dep,sy,sz),doorsteel,True)
    # Watertight rectangular plate with circular bore; radial intersections at outer rectangle.
    angles=sorted({2*math.pi*i/96 for i in range(96)} |
                  {math.atan2(dz,dy)%(2*math.pi) for dy in [-width/2,width/2] for dz in [bottom-z,top-z]})
    n=len(angles); bore=cfg["button_outer_diameter"]/2+.0001
    loops=[]
    for xx,inner in [(x,False),(x,True),(x+.002,False),(x+.002,True)]:
        for a in angles:
            c,s=math.cos(a),math.sin(a)
            r=bore if inner else min(width/2/max(abs(c),1e-12),
                    (top-z if s>=0 else z-bottom)/max(abs(s),1e-12))
            loops.append((xx,y+r*c,z+r*s))
    faces=[]
    for i in range(n):
        j=(i+1)%n
        faces.extend([(i,j,n+j,n+i),(2*n+i,3*n+i,3*n+j,2*n+j),
                      (i,2*n+i,2*n+j,j),(n+i,n+j,3*n+j,3*n+i)])
    faceplate=_mesh(stage,"/World/Panel/Faceplate",loops,[tuple(reversed(f)) for f in faces],steelmat,True)
    planar_steel_uv(faceplate)
    _decal(stage,"/World/Panel/Display",(x-.00015,y,cfg["display_bottom_z"]+cfg["display_height"]/2),
           cfg["display_width"],cfg["display_height"],"display")
    r=cfg["button_cap_diameter"]/2; outer=cfg["button_outer_diameter"]/2
    ring=_annulus(stage,"/World/Panel/FixedButtonRing",x,y,z,r+.0003,outer,.003,steelmat,True)
    planar_steel_uv(ring)
    _annulus(stage,"/World/Panel/LightRing",x-.0001,y,z,outer-.0012,outer-.00035,.0001,light)
    bodypath="/World/Panel/CallDown/Cap"
    body=UsdGeom.Xform.Define(stage,bodypath); rest=x+.003
    body.AddTranslateOp().Set(Gf.Vec3d(rest,y,z))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.015)
    body.GetPrim().CreateAttribute("pressb:illuminated",Sdf.ValueTypeNames.Bool).Set(False)
    body.GetPrim().CreateAttribute("pressb:task",Sdf.ValueTypeNames.String).Set("call_down")
    if PhysxSchema:
        api=PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim()); api.CreateDisableGravityAttr(True)
        api.CreateEnableCCDAttr(True); api.CreateSolverPositionIterationCountAttr(16)
        api.CreateSolverVelocityIterationCountAttr(4)
        PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.)
    cap=UsdGeom.Cylinder.Define(stage,bodypath+"/Collision")
    cap.CreateAxisAttr("X"); cap.CreateRadiusAttr(r); cap.CreateHeightAttr(.006)
    _bind(cap.GetPrim(),capmat); UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    if PhysxSchema:
        api=PhysxSchema.PhysxCollisionAPI.Apply(cap.GetPrim()); api.CreateContactOffsetAttr(.0001); api.CreateRestOffsetAttr(0.)
    # Down-arrow inset visual follows the moving cap. It is not a collider.
    _cube(stage,bodypath+"/ArrowStem",(-.0031,0,.0015),(.0001,.002,.010),light)
    for name,cy,angle in [("Left",.0025,-45.),("Right",-.0025,45.)]:
        arm=_cube(stage,bodypath+"/Arrow"+name,(-.0031,cy,-.0025),(.0001,.002,.0071),light)
        UsdGeom.Xformable(arm).AddRotateXOp().Set(angle)
        # Rotation must precede dimension scaling in the transform order.
        xf=UsdGeom.Xformable(arm); ops=xf.GetOrderedXformOps()
        xf.SetXformOpOrder([ops[0],ops[2],ops[1]])
    joint=UsdPhysics.PrismaticJoint.Define(stage,"/World/Panel/CallDown/SpringJoint")
    joint.CreateBody1Rel().SetTargets([body.GetPath()]); joint.CreateAxisAttr("X")
    joint.CreateLocalPos0Attr(Gf.Vec3f(rest,y,z)); joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLowerLimitAttr(0.); joint.CreateUpperLimitAttr(cfg["button_travel"])
    joint.CreateCollisionEnabledAttr(False)
    drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),"linear")
    drive.CreateTypeAttr("force"); drive.CreateTargetPositionAttr(0.); drive.CreateTargetVelocityAttr(0.)
    drive.CreateStiffnessAttr(cfg["button_stiffness"]); drive.CreateDampingAttr(cfg["button_damping"])
    drive.CreateMaxForceAttr(5.)
    info=dict(body_path=bodypath,collision_path=str(cap.GetPath()),joint_path=str(joint.GetPath()),
              shader_path=str(light.GetPath())+"/Shader",rest_x=rest,center=[x,y,z],
              body_center=[rest,y,z],travel=cfg["button_travel"],label="call_down")
    # Reference signage: approximate placement and simplified text, with no collision role.
    signage=cfg.get("signage",{
        "warning":dict(right_offset=.0375,center_z=top+.0545,width=.170,height=.085),
        "fire":dict(right_offset=.1375,center_z=1.4165,width=.100,height=.105),
        "stairs":dict(right_offset=.130,center_z=1.255,width=.095,height=.095),
        "access":dict(right_offset=.120,center_z=1.129,width=.090,height=.090)})
    for name,sign in signage.items():
        # Visual-only 1.6 mm print board. The artwork retains the previous layout.
        edge_colors={"warning":(.55,.56,.53),"fire":(.18,.018,.024),
                     "stairs":(.70,.61,.24),"access":(.013,.025,.08)}
        edgemat=coated(stage,"VIB_SignEdge_"+name,edge_colors[name],.065,1.49)
        _cube(stage,"/World/Environment/SignEdge_"+name,
              (wallx-.0010,y-sign["right_offset"],sign["center_z"]),
              (.0016,sign["width"],sign["height"]),edgemat)
        _decal(stage,"/World/Environment/Sign_"+name,
               (wallx-.00182,y-sign["right_offset"],sign["center_z"]),sign["width"],sign["height"],name)
    dome=UsdLux.DomeLight.Define(stage,"/World/Lights/Ambient"); dome.CreateIntensityAttr(300)
    dome.CreateTextureFileAttr(Sdf.AssetPath(str(ASSETS/"large_corridor_2k.hdr")))
    dome.CreateTextureFormatAttr("latlong")
    dome.AddRotateZOp().Set(95.)
    for name,eye,target,size,intensity in [
        ("Ceiling",(-.4,.4,2.7),(wallx,0,1.0),(1.5,.6),1500),
        ("Window",(-2.5,-1.5,1.9),(wallx,.5,1.2),(1.6,1.8),850),
        ("WindowLow",(-1.5,.25,1.0),(wallx,0,1.2),(1.2,.22),600),
        ("StripA",(-1.8,-.70,2.15),(wallx,-.2,1.35),(.85,.065),3200),
        ("StripB",(-1.8,.35,2.15),(wallx,.2,1.35),(.85,.065),3200)]:
        lamp=UsdLux.RectLight.Define(stage,"/World/Lights/"+name)
        lamp.CreateWidthAttr(size[0]); lamp.CreateHeightAttr(size[1]); lamp.CreateIntensityAttr(intensity)
        lamp.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
    overview=_camera(stage,"/World/Camera",cfg["camera_eye"],cfg["camera_target"],30)
    mid=(top+bottom)/2
    panel=_camera(stage,"/World/PanelCamera",[x-.40,y-.40,mid+.10],[x,y,mid],30)
    front=_camera(stage,"/World/FrontCamera",[x-.85,y,mid],[x,y,mid],45)
    lobby=_camera(stage,"/World/LobbyCamera",[-2.9,-.6,1.6],[wallx,.40,1.15],28)
    return dict(buttons={0:info},camera_path=overview,panel_camera_path=panel,
                front_camera_path=front,lobby_camera_path=lobby,dimensions=d,assets=cfg["provenance"])
