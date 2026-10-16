"""Draw scene geometry and observed base-center outcomes; no new simulation."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BATCH = ROOT / 'outputs/vib820_ready_parallel/20261014_seed20261014_n50_ready_dx30_5_dy30_v1'
cfg = json.loads((ROOT / 'configs/astribot_vib820_ready.json').read_text())
rows = list(csv.DictReader((BATCH / 'summary.csv').open()))
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 11,
                     'axes.unicode_minus': False, 'svg.fonttype': 'path'})
green, blue, dark = '#15966a', '#2463b4', '#20354b'
inside = [r for r in rows if -.30 <= float(r['x_m']) <= -.14 and .00 <= float(r['y_m']) <= .40]
assert len(inside) == 20 and all(r['primary_outcome'] == 'success' for r in inside)
fig = plt.figure(figsize=(16, 9), facecolor='#f7f9fc')
gs = fig.add_gridspec(1, 2, left=.06, right=.96, bottom=.24, top=.83, width_ratios=[1.6, 1], wspace=.20)
ax, detail = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
fig.text(.06, .94, '双臂预备姿态 · 底盘候选站位俯视图', fontsize=24, weight='bold', color=dark)
fig.text(.06, .893, '从上往下看  |  +X 朝向面板，+Y 朝电梯门一侧  |  坐标单位：米', fontsize=13, color='#536479')

for a in [ax, detail]:
    a.set_facecolor('white')
    a.set_aspect('equal')
    a.grid(color='#e5eaf0', linewidth=.7, zorder=0)
    a.set_axisbelow(True)
    a.tick_params(labelsize=10)
    for s in a.spines.values():
        s.set_color('#c9d2de')
    a.add_patch(Rectangle((-.42, -.02), .35, .60, fill=False, linestyle=(0,(5,4)), edgecolor='#8d99a8', lw=1.5))
    a.add_patch(Rectangle((-.30, .00), .16, .40, facecolor=green, alpha=.13, edgecolor='none'))
    a.add_patch(Rectangle((-.30, .00), .16, .40, fill=False, edgecolor=green, lw=2))
    a.set_xlabel('世界 X → 朝向面板')
    a.set_ylabel('世界 Y → 电梯门一侧')

# Scene geometry follows vib820_scene.py: wall .48, front plate .46,
# panel spans Y +/- .0475; door is recessed .35 m and starts at Y .2625.
face = cfg['button_face_x']; wall = face + cfg['panel_depth']
cy = cfg['panel_center_y']; half = cfg['panel_width']/2
jamb = cy + half + cfg['panel_to_jamb_gap']
door_y = jamb + cfg['jamb_strip_width']; door_x = wall + cfg['jamb_recess_depth']
ax.add_patch(Rectangle((wall, -.19), .10, jamb+.19, facecolor='#c8ced6', edgecolor='#8c98a6'))
ax.add_patch(Rectangle((wall, jamb), door_x-wall, cfg['jamb_strip_width'], facecolor='#adb8c7'))
ax.add_patch(Rectangle((door_x, door_y), .025, 1.30, facecolor='#a6b8cb', edgecolor='#667c93'))
ax.add_patch(Rectangle((face, cy-half), cfg['panel_depth'], cfg['panel_width'], facecolor='#f0b33a', edgecolor='#9c6600', lw=2, zorder=5))
ax.scatter([face], [cy], s=35, color='#a55d00', zorder=6)
ax.annotate('电梯按钮面板\n正面 X=0.46，中心 Y=0\n宽 9.5 cm', xy=(face, cy), xytext=(.10, -.14), fontsize=11,
            arrowprops={'arrowstyle':'->','color':'#9c6600'}, color='#825600', ha='center')
ax.text(.535, .065, '墙体', color='#657183', rotation=90, ha='center')
ax.text(.86, .48, '电梯门（局部）', rotation=90, color='#4e6680', ha='center', va='center')
ax.text(.66, .30, '门洞向内凹入', fontsize=10, ha='center', color='#667c93')
ax.text(-.22, .49, '底盘中心候选区域', color=green, ha='center', weight='bold')
ax.text(-.22, .415, 'X ∈ [−0.30, −0.14]\nY ∈ [0.00, 0.40]', ha='center', va='bottom', fontsize=10, color=green)

# Icon only, deliberately not an unsupported footprint estimate.
rx, ry = -.22, .28
ax.add_patch(Circle((rx,ry), .034, facecolor='white', edgecolor=blue, lw=2, zorder=7))
ax.plot(rx,ry,'o',color=blue,ms=3,zorder=9)
ax.annotate('', xy=(rx+.105,ry), xytext=(rx+.035,ry), arrowprops={'arrowstyle':'-|>','color':blue,'lw':2}, zorder=8)
ax.annotate('推荐机器人位置\n(−0.22, 0.28)\n箭头为朝向，yaw=0', xy=(rx,ry), xytext=(-.02,.52),
            arrowprops={'arrowstyle':'-','color':blue}, color=blue, fontsize=11)
ax.plot([rx,face],[ry,ry], linestyle=':', color='#92a6bf', zorder=1)
ax.annotate('', xy=(face,.19), xytext=(rx,.19), arrowprops={'arrowstyle':'<->','color':'#6b7f95'})
ax.text(.12,.205,'至面板正面的 X 向距离：68 cm',ha='center',fontsize=10,color='#526b86')
ax.scatter([-.12],[.28], marker='+', s=90, color='#737d8a', zorder=10)
ax.annotate('原采样中心\n(−0.12, 0.28)', xy=(-.12,.28), xytext=(.06,.35), fontsize=9, color='#737d8a',
            arrowprops={'arrowstyle':'-','color':'#939ca6'})
ax.set(xlim=(-.49,.94), ylim=(-.19,.65), title='场景关系（等比例；机器人符号不代表实际外形）')

styles = [('success', '完整成功 34', 'o',green),
          ('target_ik_failed','目标位姿不可达 8','^','#d49023'),
          ('initial_collision','初始重叠 6','X','#c84c50'),
          ('unexpected_contact','合爪碰撞 2','x','#8f376f')]
for key,label,marker,color in styles:
    group = [r for r in rows if r['primary_outcome']==key]
    detail.scatter([float(r['x_m']) for r in group], [float(r['y_m']) for r in group],
                   marker=marker,c=color,s=45,label=label,zorder=5,linewidths=1.4)
detail.scatter([rx],[ry], marker='*',s=220,c=blue,edgecolor='white',linewidths=.7,zorder=8,label='推荐起点（另有成功演示）')
detail.set(xlim=(-.45,-.035), ylim=(-.065,.625), title='50 个实测站位 · 局部放大')
detail.set_xticks([-.4,-.3,-.2,-.1])
detail.annotate('16 cm', xy=(-.30,-.04),xytext=(-.14,-.04), ha='center',va='center',color=green,
                arrowprops={'arrowstyle':'<->','color':green})
detail.text(-.308,.20,'40 cm',rotation=90,ha='right',va='center',color=green,fontsize=10)
detail.legend(loc='upper center', bbox_to_anchor=(.5,-.14), frameon=False, fontsize=10, ncol=2)
fig.text(.06,.17,'绿色矩形：候选底盘中心区域，宽 16 cm × 横向 40 cm；内部已测 20 个站位全部完成任务。',fontsize=13,color=green,weight='bold')
fig.text(.06,.12,'这是依据本轮结果提出的补测候选区域，不代表矩形内任意位置都已验证安全。灰色虚线为本轮完整采样范围。',fontsize=11,color='#536479')
fig.text(.06,.075,'适用条件：双臂已处于预备姿态，底盘高度与配置一致，朝向固定 yaw=0；区域约束的是底盘坐标原点，不是整机占地。',fontsize=11,color='#536479')
fig.text(.06,.035,'数据：20261014 / 50 个随机站位；几何：astribot_vib820_ready.json 与 vib820_scene.py。',fontsize=9,color='#8793a3')
for suffix in ['png','svg','pdf']:
    fig.savefig(OUT / f'ready_base_top_view.{suffix}',dpi=180,facecolor=fig.get_facecolor())
print(json.dumps({'candidate_samples':len(inside),'candidate_successes':sum(r['primary_outcome'] == 'success' for r in inside),'outputs':str(OUT)},ensure_ascii=False))
