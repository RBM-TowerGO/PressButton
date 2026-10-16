"""Offline geometry check of the recorded head view; changes no camera settings."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom
from pressb.astribot_kinematics import AstribotKinematics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
run = ROOT/'outputs/vib820_spec_v1_1_all_cameras_20261014/run'
r = json.loads((run/'report.json').read_text())
t = np.load(run/'trajectory.npz')
stage = Usd.Stage.Open(str(run/'scene.usda'))
prim = stage.GetPrimAtPath(r['camera_paths']['head'])
local = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).T
camera = UsdGeom.Camera(prim)
focal = camera.GetFocalLengthAttr().Get()
vfov = 2*np.arctan(camera.GetVerticalApertureAttr().Get()/(2*focal))
hfov = 2*np.arctan(camera.GetHorizontalApertureAttr().Get()/(2*focal))
fy = 720/(2*np.tan(vfov/2))
kin = AstribotKinematics()
button = np.array([r['config']['button_face_x'],r['config']['panel_center_y'],r['scene_dimensions']['button_z']])
records = []
for phase in ['settle','close_gripper','move_to_work_pose','press','return_to_ready','ready_hold']:
    indices = np.flatnonzero(t['phase']==phase)
    i = int(indices[len(indices)//2])
    q = dict(zip(r['joint_names'],t['q'][i]))
    base = np.eye(4)
    base[:3,:3] = Rotation.from_quat(np.roll(t['base_quat'][i],-1)).as_matrix()
    base[:3,3] = t['base'][i]
    world = base @ kin.fk('astribot_head_link_2',q) @ local
    points = {'button':button}
    for side in ['left','right']:
        # Midpoint of the two terminal finger link origins, not full hand bounds.
        points[side+'_finger_link_midpoint'] = np.mean([
            (base @ kin.fk(f'astribot_gripper_{side}_Link_{suffix}',q))[:3,3]
            for suffix in ['L11','R11']],axis=0)
    projections = {}
    for name,p in points.items():
        c = np.linalg.inv(world) @ np.r_[p,1]
        uv = [640+fy*c[0]/(-c[2]),360-fy*c[1]/(-c[2])]
        projections[name] = dict(world_xyz_m=p.tolist(), pixel_uv=uv,
            angle_below_axis_deg=float(np.degrees(np.arctan2(-c[1],-c[2]))),
            inside_image=bool(c[2]<0 and 0<=uv[0]<1280 and 0<=uv[1]<720))
    lower_ray = -world[:3,2]-np.tan(vfov/2)*world[:3,1]
    lower_at_panel = world[:3,3]+lower_ray*((button[0]-world[0,3])/lower_ray[0])
    records.append(dict(phase=phase,time_s=float(t['time'][i]),camera_xyz_m=world[:3,3].tolist(),
        camera_forward=(-world[:3,2]).tolist(),lower_view_height_at_panel_m=float(lower_at_panel[2]),points=projections))
head_indices=[r['joint_names'].index(f'astribot_head_joint_{j}') for j in [1,2]]
result=dict(recording=str(run.relative_to(ROOT)),resolution=[1280,720],fov_deg=[float(np.degrees(hfov)),float(np.degrees(vfov))],
    focal_length_mm=focal,head_target_max_abs_rad=float(np.max(np.abs(t['target'][:,head_indices]))),
    head_actual_max_abs_rad=float(np.max(np.abs(t['q'][:,head_indices]))),samples=records,
    scope='Six representative recorded poses. Finger link midpoints are reference points, not full mesh visibility or occlusion tests. No simulation rerun or camera changes.')
(OUT/'geometry.json').write_text(json.dumps(result,indent=2)+'\n')

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':12})
a=records[0];cx,cy,cz=a['camera_xyz_m']; pitch=np.arctan2(a['camera_forward'][2],a['camera_forward'][0])
fig,ax=plt.subplots(figsize=(10,6.3));fig.subplots_adjust(bottom=.18,top=.86,left=.1,right=.96)
xs=np.linspace(cx,.62,100)
bottom=cz+np.tan(pitch-vfov/2)*(xs-cx);top=cz+np.tan(pitch+vfov/2)*(xs-cx)
ax.fill_between(xs,bottom,top,color='#dceaf7',label='头部相机垂直视野（侧面示意）')
ax.plot(xs,bottom,color='#5186b8');ax.plot(xs,top,color='#5186b8')
ax.plot(xs,cz+np.tan(pitch)*(xs-cx),'--',color='#5186b8')
ax.plot(cx,cz,'o',color='#2463b4',ms=9)
ax.annotate(f'头部相机\n高 {cz:.3f} m，基本水平朝前',xy=(cx,cz),xytext=(-.23,1.82),fontsize=11,
            arrowprops={'arrowstyle':'->','color':'#2463b4'})
ax.axvline(button[0],color='#a89980',alpha=.6)
ax.plot(button[0],button[2],'o',color='#d69416',ms=10,label='按钮中心')
ax.annotate('按钮高 1.20 m\n在画面下方之外',xy=(button[0],button[2]),xytext=(.43,1.04),fontsize=11,ha='center',
            arrowprops={'arrowstyle':'->','color':'#ad7614'})
p=a['points']['left_finger_link_midpoint']['world_xyz_m']
ax.plot(p[0],p[2],'s',color='#15966a',ms=9,label='预备姿态双手指节参考点（侧投影重合）')
ax.annotate('双手约在光轴下方 47°\n相机向下仅覆盖约 25.7°',xy=(p[0],p[2]),xytext=(-.18,1.07),fontsize=11,
            arrowprops={'arrowstyle':'->','color':'#15966a'},color='#117452')
ax.plot([cx,button[0]],[cz,button[2]],':',color='#d69416')
ax.text(.29,1.43,f'面板处视野下沿 ≈ {a["lower_view_height_at_panel_m"]:.2f} m',fontsize=11,color='#3c6f9e')
ax.set(xlim=(-.25,.64),ylim=(.99,2.08),xlabel='世界 X（m）→ 面板',ylabel='世界高度 Z（m）',title='为什么头部单目看不到双手？当前相机朝向与任务高度不匹配')
ax.grid(alpha=.15);ax.legend(loc='upper right',fontsize=9)
fig.text(.1,.04,'依据本次录像的关节轨迹与相机参数离线计算；图为 X–Z 侧面投影，未改变相机或机器人姿态。',fontsize=10,color='#536479')
fig.savefig(OUT/'head_view_geometry.png',dpi=160)
print(json.dumps({k:v for k,v in result.items() if k!='samples'},ensure_ascii=False))
print(json.dumps(records[0],ensure_ascii=False))
