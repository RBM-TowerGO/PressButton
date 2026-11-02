"""Render saved token accounting; does not contact model or robot services."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter, FuncFormatter

root = Path(__file__).resolve().parent
d = json.loads((root/'analysis.json').read_text())
font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                     'font.size': 10, 'svg.fonttype': 'path'})
from matplotlib import font_manager
font_manager.fontManager.addfont(font.get_file())
rows = d['rows']
x = [r['round'] for r in rows]
keys = ['system_tools', 'task_cases', 'earlier_history', 'current_observation']
labels = ['系统、工具与基础模板', '固定任务与案例', '本次运行历史', '当前双视角与状态']
colors = ['#8495a7', '#5285bf', '#e4a454', '#259b83']
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                         gridspec_kw={'height_ratios': [1.45, 1]}, layout='constrained')
for ax, pct in zip(axes, [False, True]):
    bottom = [0]*len(rows)
    for key, label, color in zip(keys, labels, colors):
        values = [r[key+('_pct' if pct else '_tokens')] for r in rows]
        bars = ax.bar(x, values, bottom=bottom, label=label, color=color, width=.72)
        bars[-1].set_hatch('//')
        bottom = [a+b for a,b in zip(bottom, values)]
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    ax.set_ylabel('当轮输入构成' if pct else '当轮输入 token')
axes[0].axhline(d['application_budget'], color='#b64848', ls='--', lw=1.4)
axes[0].text(1, d['application_budget']+1100, '应用压缩预算：65,536 token', color='#b64848')
axes[0].set_ylim(0, 71000)
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{int(v):,}'))
axes[0].set_title('原 direct 运行：固定材料不变，历史逐轮累积', loc='left', fontsize=16, pad=14)
axes[0].legend(ncol=2, loc='upper left', bbox_to_anchor=(0, .90), framealpha=.95)
axes[1].set_ylim(0,100)
axes[1].yaxis.set_major_formatter(PercentFormatter(100))
axes[1].set_xticks(x)
axes[1].set_xlabel('决策轮次（第15轮为已准备请求，未取得模型响应）')
axes[1].text(.99,.04,'token 占比不等于模型注意力占比', transform=axes[1].transAxes,
             ha='right', color='#333333', bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
fig.savefig(root/'context_growth.png', dpi=170)
fig.savefig(root/'context_growth.svg')
