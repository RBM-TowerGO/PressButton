#!/usr/bin/env python3
"""Presentation-only station plot from existing CSV; preserves experiment records."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'success':'#16803c','unexpected_contact':'#d52b2b','initial_collision':'#b11c1c',
        'target_ik_failed':'#ed7d19','path_ik_failed':'#9e4a12','runtime_error':'#7b4ba3',
        'pending':'#9b9b9b','running':'#555555','audit_failed':'#a67b14',
        'press_not_triggered':'#c68517','release_or_return_failed':'#ad6216','reset_failed':'#995c18'}

def plot(csv_path,output,title,manifest=None):
    with Path(csv_path).open() as f:rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(12,8))
    for outcome in sorted(set(r['primary_outcome'] for r in rows)):
        group=[r for r in rows if r['primary_outcome']==outcome]
        collision=outcome in ('unexpected_contact','initial_collision')
        ax.scatter([float(r['x_m']) for r in group],[float(r['y_m']) for r in group],
            c=COLORS.get(outcome,'#333333'),marker='X' if collision else 'o',s=90,
            linewidths=.7,edgecolors='white',label=f'{outcome} ({len(group)})',zorder=3)
    for r in rows:
        number=int(r['sample_id'].rsplit('_',1)[-1])
        ax.annotate(f'{number:02d}',(float(r['x_m']),float(r['y_m'])),fontsize=8,
            xytext=(5,5 if number%2 else -11),textcoords='offset points',zorder=4)
    if manifest is None:
        manifest=json.loads((Path(csv_path).parent/'manifest.json').read_text())
    if 'yaw_bounds_deg' in manifest:
        angles=np.array([float(r['yaw_rad']) for r in rows])
        ax.quiver([float(r['x_m']) for r in rows],[float(r['y_m']) for r in rows],
                  .018*np.cos(angles),.018*np.sin(angles),angles='xy',scale_units='xy',scale=1,
                  width=.003,color='#333333',zorder=2)
        ax.text(.02,.98,'Arrows: actual root yaw (0–10 deg)',transform=ax.transAxes,va='top',fontsize=9)
    nominal=manifest['nominal_xy_m']
    bounds=manifest.get('delta_bounds_xy_m',[manifest.get('delta_bounds_m',[-.2,.2])]*2)
    lo=[nominal[i]+bounds[i][0] for i in range(2)]
    hi=[nominal[i]+bounds[i][1] for i in range(2)]
    ax.scatter(*nominal,marker='*',s=200,c='black',label='sampling reference',zorder=5)
    ax.add_patch(plt.Rectangle(lo,hi[0]-lo[0],hi[1]-lo[1],fill=False,linestyle='--',linewidth=1.2,color='black'))
    ax.set(xlabel='World X (m; +X toward panel)',ylabel='World Y (m)',title=title,
           xlim=(lo[0]-.03,hi[0]+.03),ylim=(lo[1]-.03,hi[1]+.03))
    ax.set_aspect('equal');ax.grid(alpha=.2)
    ax.legend(loc='center left',bbox_to_anchor=(1.03,.5),frameon=True)
    fig.subplots_adjust(right=.67,left=.1,bottom=.12,top=.9)
    fig.savefig(output,dpi=180,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--csv',default='summary.csv');p.add_argument('--output',default='station_results_display.png');a=p.parse_args()
    plot(a.directory/a.csv,a.directory/a.output,a.directory.name+'\n'+a.csv)
