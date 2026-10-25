#!/usr/bin/env python3
"""Local desktop review UI. Only review JSON and explicit export copies are written."""
from __future__ import annotations

import copy
import io
import os
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import h5py
import numpy as np
from PIL import Image, ImageTk

from astribot_data_review import (
    CAMERAS, checked_source, export, get_entry, image_offsets,
    load_review, mark_entry, sha256, write_json,
)

STATUS = {'pending': '待审核', 'keep': '保留', 'reject': '已删除', 'hold': '待排查'}


class ReviewStore:
    """Persist edits with undo and detection of external CLI edits."""
    def __init__(self, path):
        self.path = Path(path)
        self.data = load_review(path)
        self.fingerprint = sha256(path)
        self.history = []

    def assert_current(self):
        if sha256(self.path) != self.fingerprint:
            raise ValueError('审核文件已被其他程序修改。请关闭并重新打开窗口，以免覆盖。')

    def commit(self, mutation):
        self.assert_current()
        candidate = copy.deepcopy(self.data)
        mutation(candidate)
        write_json(self.path, candidate)
        self.history.append(self.data)
        self.history = self.history[-30:]
        self.data = candidate
        self.fingerprint = sha256(self.path)

    def undo(self):
        if not self.history:
            return
        self.assert_current()
        previous = self.history[-1]
        write_json(self.path, previous)
        self.history.pop()
        self.data = previous
        self.fingerprint = sha256(self.path)

    def set_status(self, episode, status, start, end, note):
        def apply(data):
            e = get_entry(data, episode)
            if status == 'reject' and e['status'] != 'reject':
                e['before_delete_status'] = e['status']
            if status == 'restore':
                e['status'] = e.pop('before_delete_status', 'pending')
            elif status in ('reject', 'hold', 'pending') and e['audit']['frames'] < 2:
                e.update(status=status, note=note)
            else:
                if status == 'keep' and end - start < 2:
                    raise ValueError('至少保留 2 帧。结束帧包含在保留范围内。')
                mark_entry(data, episode, status, start, end, note)
        self.commit(apply)

    def move(self, episode, delta):
        def apply(data):
            entries = data['episodes']
            i = next(i for i, e in enumerate(entries) if e['original_id'] == episode)
            j = max(0, min(len(entries) - 1, i + delta))
            entries.insert(j, entries.pop(i))
        self.commit(apply)

    def keep_all_pending(self):
        self.assert_current()
        eligible, skipped = [], []
        for e in self.data['episodes']:
            if e['status'] != 'pending':
                continue
            valid = (not e['audit']['errors'] and
                     0 <= e['start_frame'] < e['end_frame'] <= e['audit']['frames'] and
                     e['end_frame'] - e['start_frame'] >= 2)
            (eligible if valid else skipped).append(e['original_id'])
        if eligible:
            ids = set(eligible)
            def apply(data):
                for e in data['episodes']:
                    if e['original_id'] in ids:
                        e['status'] = 'keep'
            self.commit(apply)
        return eligible, skipped


def read_episode(path):
    with h5py.File(path, 'r') as f:
        t = f['time'][:]
        result = {'time': t, 'images': {},
                  'state': f['joints_dict/joints_position_state'][:],
                  'command': f['joints_dict/joints_position_command'][:],
                  'offset': f['joints_dict/command_timestamp'][:] - t}
        for cam in CAMERAS:
            g = f['images_dict/' + cam]
            result['images'][cam] = (g['rgb'][:], image_offsets(g, len(t)))
    return result


class ReviewWindow:
    def __init__(self, root, path):
        self.root, self.store = root, ReviewStore(path)
        self.current_id = None
        self.data = None
        self.frame = 0
        self.playing = False
        self.busy = False
        self.loading = False
        self.generation = 0
        self.next_play_target = None
        self.autoplay_generation = None
        self.play_after = None
        self.events = queue.Queue()
        self.photos = []
        self.root.title('Astribot · 数据审核')
        self.root.geometry('1500x980')
        self.root.minsize(1100, 780)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.build()
        self.refresh_list()
        if self.store.data['episodes']:
            self.tree.selection_set(str(self.store.data['episodes'][0]['original_id']))
        self.root.after(40, self.poll)

    def button(self, parent, text, command):
        b = ttk.Button(parent, text=text, command=command)
        b.pack(side='left', padx=3, pady=4)
        return b

    def build(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('Treeview', rowheight=29)
        style.configure('TButton', padding=(9, 7))
        outer = ttk.Frame(self.root, padding=12); outer.pack(fill='both', expand=True)
        header = ttk.Frame(outer); header.pack(fill='x')
        ttk.Label(header, text='Astribot 轨迹审核', font=('', 19, 'bold')).pack(side='left')
        self.summary = ttk.Label(header); self.summary.pack(side='left', padx=25)
        self.export_button = self.button(header, '导出保留项并重新编号…', self.export_clicked)
        self.button(header, '将所有待审核项设为保留', self.keep_all_pending_clicked)
        ttk.Label(outer, text='删除仅从导出集合中剔除，可恢复；原始 HDF5 保留。所有标记和顺序自动保存。').pack(anchor='w', pady=(8, 10))
        body = ttk.Panedwindow(outer, orient='horizontal'); body.pack(fill='both', expand=True)
        left = ttk.Frame(body, width=300); body.add(left, weight=1)
        controls = ttk.Frame(left); controls.pack(fill='x')
        self.filter = tk.StringVar(value='全部')
        selector = ttk.Combobox(controls, textvariable=self.filter, values=['全部'] + list(STATUS.values()), state='readonly', width=9)
        selector.pack(side='left'); selector.bind('<<ComboboxSelected>>', lambda _: self.refresh_list())
        self.search = tk.StringVar()
        ttk.Label(controls, text='原编号').pack(side='left', padx=5)
        ttk.Entry(controls, textvariable=self.search, width=8).pack(side='left')
        self.search.trace_add('write', lambda *_: self.refresh_list())
        listbox = ttk.Frame(left); listbox.pack(fill='both', expand=True, pady=8)
        self.tree = ttk.Treeview(listbox, columns=('order', 'id', 'state', 'frames'), show='headings', selectmode='browse')
        for name, label, width in [('order', '顺序', 45), ('id', '原编号', 65), ('state', '审核状态', 90), ('frames', '帧数', 60)]:
            self.tree.heading(name, text=label); self.tree.column(name, width=width, stretch=False)
        sb = ttk.Scrollbar(listbox, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y'); self.tree.pack(fill='both', expand=True)
        self.tree.tag_configure('reject', foreground='#a33b3b')
        self.tree.tag_configure('keep', foreground='#18744b')
        self.tree.tag_configure('warning', foreground='#9b6100')
        self.tree.bind('<<TreeviewSelect>>', self.selected)
        order = ttk.Frame(left); order.pack(fill='x')
        self.button(order, '↑ 上移', lambda: self.move(-1))
        self.button(order, '↓ 下移', lambda: self.move(1))
        self.button(order, '撤销', self.undo)
        ttk.Label(left, text='顺序列决定导出顺序；只对保留项连续编号。', wraplength=280).pack(anchor='w', pady=5)

        right = ttk.Frame(body, padding=(12, 0, 0, 0)); body.add(right, weight=4)
        self.title = ttk.Label(right, font=('', 14, 'bold')); self.title.pack(anchor='w')
        self.warning = ttk.Label(right, foreground='#a35900', wraplength=1000)
        self.warning.pack(anchor='w', pady=5)
        image_row = ttk.Frame(right); image_row.pack(fill='both', expand=True)
        self.image_labels = {}
        for col, (cam, label) in enumerate(zip(CAMERAS, ['头部', '左腕', '右腕'])):
            image_row.columnconfigure(col, weight=1, uniform='camera')
            panel = ttk.Frame(image_row); panel.grid(row=0, column=col, sticky='nsew', padx=3)
            ttk.Label(panel, text=label).pack(anchor='w')
            image = tk.Label(panel, bg='#141c29', fg='white', text='选择轨迹', width=1, height=1)
            image.pack(fill='both', expand=True); self.image_labels[cam] = image
        image_row.rowconfigure(0, weight=1)
        bar = ttk.Frame(right); bar.pack(fill='x', pady=8)
        self.button(bar, '上一条', lambda: self.adjacent(-1))
        self.button(bar, '◀ 帧', lambda: self.show_frame(self.frame - 1))
        self.play_button = self.button(bar, '▶ 播放', self.toggle_play)
        self.button(bar, '帧 ▶', lambda: self.show_frame(self.frame + 1))
        self.button(bar, '下一条', lambda: self.adjacent(1))
        self.speed = tk.StringVar(value='1.0')
        ttk.Label(bar, text='倍速').pack(side='left', padx=5)
        ttk.Combobox(bar, textvariable=self.speed, values=['0.25', '0.5', '1.0', '2.0'], width=5, state='readonly').pack(side='left')
        self.autoplay_next = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text='下一条自动播放', variable=self.autoplay_next).pack(side='left', padx=10)
        self.position = ttk.Label(right); self.position.pack(anchor='w')
        self.slider = ttk.Scale(right, from_=0, to=1, command=self.seek)
        self.slider.pack(fill='x', pady=4)
        crop = ttk.LabelFrame(right, text='保留范围 · 零起始帧号，起始帧与结束帧均包含', padding=6)
        crop.pack(fill='x', pady=6)
        self.start = tk.IntVar(value=0); self.end = tk.IntVar(value=0)
        ttk.Label(crop, text='起始帧').pack(side='left')
        self.start_spin = ttk.Spinbox(crop, from_=0, to=100000, textvariable=self.start, width=7); self.start_spin.pack(side='left', padx=5)
        self.button(crop, '当前帧设为起点', lambda: self.start.set(self.frame))
        ttk.Label(crop, text='结束帧').pack(side='left', padx=(15, 0))
        self.end_spin = ttk.Spinbox(crop, from_=0, to=100000, textvariable=self.end, width=7); self.end_spin.pack(side='left', padx=5)
        self.button(crop, '当前帧设为终点', lambda: self.end.set(self.frame))
        self.button(crop, '保存范围 / 备注', self.save_range)
        jointrow = ttk.Frame(right); jointrow.pack(fill='x')
        ttk.Label(jointrow, text='关节列（厂商名称尚未映射）').pack(side='left')
        self.joint = tk.StringVar(value='0')
        jointselect = ttk.Combobox(jointrow, textvariable=self.joint, values=[str(i) for i in range(25)], state='readonly', width=5)
        jointselect.pack(side='left', padx=5); jointselect.bind('<<ComboboxSelected>>', lambda _: self.draw_plot())
        ttk.Label(jointrow, text='蓝：实测    橙：指令').pack(side='left', padx=10)
        self.plot = tk.Canvas(right, height=130, bg='#f3f6fb', highlightthickness=0)
        self.plot.pack(fill='x', pady=5)
        self.plot.bind('<Configure>', lambda _: self.draw_plot())
        self.plot.bind('<Button-1>', self.plot_seek)
        notes = ttk.Frame(right); notes.pack(fill='x')
        ttk.Label(notes, text='备注').pack(side='left')
        self.note = tk.StringVar(); ttk.Entry(notes, textvariable=self.note).pack(side='left', fill='x', expand=True, padx=8)
        actions = ttk.Frame(right); actions.pack(fill='x', pady=7)
        self.button(actions, '✓ 保留', lambda: self.status_clicked('keep'))
        self.button(actions, '删除', lambda: self.status_clicked('reject'))
        self.button(actions, '恢复', lambda: self.status_clicked('restore'))
        self.button(actions, '待排查', lambda: self.status_clicked('hold'))
        self.button(actions, '重置为待审核', lambda: self.status_clicked('pending'))
        self.notice = ttk.Label(outer, text='准备就绪'); self.notice.pack(anchor='w', pady=(8, 0))

    def report_error(self, e):
        messagebox.showerror('操作未完成', str(e), parent=self.root)

    def refresh_list(self):
        selected = str(self.current_id) if self.current_id is not None else None
        self.tree.delete(*self.tree.get_children())
        counts = {k: 0 for k in STATUS}
        for i, e in enumerate(self.store.data['episodes']):
            counts[e['status']] += 1
            if self.filter.get() not in ('全部', STATUS[e['status']]): continue
            if self.search.get() and self.search.get() not in str(e['original_id']): continue
            tags = [e['status']]
            if e['audit']['warnings'] and e['status'] not in ('keep', 'reject'): tags.append('warning')
            self.tree.insert('', 'end', iid=str(e['original_id']), values=(i + 1, e['original_id'], STATUS[e['status']], e['audit']['frames']), tags=tags)
        if selected and self.tree.exists(selected):
            self.tree.selection_set(selected); self.tree.see(selected)
        self.summary.config(text='  ·  '.join(f'{STATUS[k]} {v}' for k, v in counts.items()))

    def selected(self, _=None):
        selection = self.tree.selection()
        if not selection: return
        episode = int(selection[0])
        if episode == self.current_id: return
        try:
            self.persist_fields()
        except Exception as e:
            self.next_play_target = None
            if self.current_id is not None and self.tree.exists(str(self.current_id)):
                self.tree.selection_set(str(self.current_id))
            self.report_error(e)
            return
        self.pause()
        self.current_id = episode
        self.data = None; self.loading = True; self.generation += 1
        generation = self.generation
        self.autoplay_generation = generation if self.next_play_target == episode else None
        self.next_play_target = None
        self.plot.delete('all')
        for label in self.image_labels.values(): label.config(image='', text='载入中…')
        self.photos = []
        self.sync_fields()
        self.notice.config(text=f'正在核验并载入原编号 {episode}…')
        snapshot = copy.deepcopy(self.store.data); entry = get_entry(snapshot, episode)
        def work():
            try:
                data = read_episode(checked_source(snapshot, entry))
                self.events.put(('load', generation, data))
            except Exception as e: self.events.put(('load_error', generation, str(e)))
        threading.Thread(target=work, daemon=True).start()

    def sync_fields(self):
        e = get_entry(self.store.data, self.current_id)
        self.title.config(text=f'原编号 {self.current_id}   ·   {STATUS[e["status"]]}   ·   {e["file"]}')
        self.start.set(e['start_frame']); self.end.set(max(0, e['end_frame'] - 1)); self.note.set(e['note'])
        self.warning.config(text='；'.join(e['audit']['errors'] + e['audit']['warnings']) or '结构检查通过；是否成功按下按钮仍需回放确认。')

    def poll(self):
        try:
            while True:
                kind, token, payload = self.events.get_nowait()
                if kind == 'load' and token == self.generation:
                    self.loading = False; self.data = payload
                    self.slider.configure(to=len(payload['time']) - 1)
                    self.show_frame(0)
                    self.notice.config(text='已载入。点击保留、删除或保存范围；修改立即保存。')
                    if token == self.autoplay_generation and self.autoplay_next.get():
                        self.toggle_play()
                elif kind == 'load_error' and token == self.generation:
                    self.loading = False
                    for label in self.image_labels.values(): label.config(text='无法载入')
                    self.report_error(payload)
                elif kind == 'progress': self.notice.config(text=f'正在导出并校验图像 {token}/{payload}…')
                elif kind in ('export_done', 'export_error'):
                    self.busy = False; self.export_button.config(state='normal')
                    self.notice.config(text=f'导出完成：{payload}' if kind == 'export_done' else '导出失败，原始数据未改动')
                    if kind == 'export_error': self.report_error(payload)
                    else: messagebox.showinfo('导出完成', f'已保存到：\n{payload}\n包含连续编号 HDF5、新旧编号映射和审核快照。', parent=self.root)
        except queue.Empty: pass
        self.root.after(40, self.poll)

    def seek(self, value):
        if self.data is not None and int(float(value)) != self.frame:
            self.pause(); self.show_frame(int(float(value)))

    def show_frame(self, frame):
        if self.data is None: return
        self.frame = max(0, min(len(self.data['time']) - 1, int(frame)))
        self.slider.set(self.frame)
        photos = []
        try:
            for cam in CAMERAS:
                blob, offsets = self.data['images'][cam]
                raw = blob[offsets[self.frame]:offsets[self.frame + 1]].tobytes()
                im = Image.open(io.BytesIO(raw)); im.load()
                label = self.image_labels[cam]
                im.thumbnail((max(240, label.winfo_width() - 8), max(160, label.winfo_height() - 8)))
                photo = ImageTk.PhotoImage(im); photos.append(photo)
                label.config(image=photo, text='')
        except Exception as e:
            self.pause(); self.report_error(f'帧 {self.frame} 解码失败：{e}')
        self.photos = photos
        t = self.data['time']; offset = self.data['offset'][self.frame]
        self.position.config(text=f'帧 {self.frame} / {len(t)-1}     时间 {t[self.frame]-t[0]:.3f} / {t[-1]-t[0]:.3f} 秒     指令时间偏差 {offset*1000:+.1f} ms')
        self.draw_plot()

    def draw_plot(self):
        self.plot.delete('all')
        if self.data is None: return
        j = int(self.joint.get()); width = max(200, self.plot.winfo_width()); height = self.plot.winfo_height()
        arrays = [self.data[k][:, j] for k in ('state', 'command')]
        lo, hi = min(float(np.min(a)) for a in arrays), max(float(np.max(a)) for a in arrays)
        span = max(hi - lo, 1e-6); n = len(arrays[0])
        for a, color in zip(arrays, ['#2878be', '#d87522']):
            coords = [v for i, y in enumerate(a) for v in (40 + i / max(1, n-1) * (width-50), 12 + (hi-y) / span * (height-30))]
            if len(coords) >= 4: self.plot.create_line(*coords, fill=color, width=2)
        x = 40 + self.frame / max(1, n-1) * (width-50)
        self.plot.create_line(x, 0, x, height, fill='#768296', dash=(4, 3))
        self.plot.create_text(3, 10, text=f'{hi:.3f}', anchor='w')
        self.plot.create_text(3, height-15, text=f'{lo:.3f}', anchor='w')

    def plot_seek(self, event):
        if self.data is not None:
            self.pause(); self.show_frame(round((event.x-40)/max(1, self.plot.winfo_width()-50)*(len(self.data['time'])-1)))

    def pause(self):
        self.playing = False
        if self.play_after is not None:
            self.root.after_cancel(self.play_after)
            self.play_after = None
        self.play_button.config(text='▶ 播放')

    def toggle_play(self):
        if self.playing: self.pause(); return
        if self.data is None: return
        if self.frame == len(self.data['time']) - 1: self.show_frame(0)
        self.playing = True; self.play_button.config(text='暂停')
        self.play_wall = time.monotonic(); self.play_time = self.data['time'][self.frame]
        self.play_speed = float(self.speed.get()); self.tick()

    def tick(self):
        self.play_after = None
        if not self.playing or self.data is None: return
        target = self.play_time + (time.monotonic()-self.play_wall)*self.play_speed
        index = int(np.searchsorted(self.data['time'], target, side='right')-1)
        if index != self.frame: self.show_frame(index)
        if self.frame >= len(self.data['time'])-1: self.pause()
        else: self.play_after = self.root.after(15, self.tick)

    def editable(self):
        if self.busy: raise ValueError('导出进行中，请等待完成后再修改。')
        if self.current_id is None: raise ValueError('请先选择轨迹。')

    def persist_fields(self):
        if self.current_id is None or self.busy: return
        e = get_entry(self.store.data, self.current_id)
        start, end, note = self.start.get(), self.end.get()+1, self.note.get()
        if e['audit']['frames'] < 2:
            return
        if (start, end, note) != (e['start_frame'], e['end_frame'], e['note']):
            self.store.set_status(self.current_id, e['status'], start, end, note)

    def status_clicked(self, status):
        try:
            self.editable()
            self.store.set_status(self.current_id, status, self.start.get(), self.end.get()+1, self.note.get())
            self.sync_fields(); self.refresh_list()
            self.notice.config(text='已保存审核决定。原始文件未删除，可点击恢复或撤销。')
        except Exception as e: self.report_error(e)

    def save_range(self):
        if self.current_id is not None:
            self.status_clicked(get_entry(self.store.data, self.current_id)['status'])

    def keep_all_pending_clicked(self):
        try:
            if self.busy:
                raise ValueError('导出进行中，请等待完成后再修改。')
            self.persist_fields()
            kept, skipped = self.store.keep_all_pending()
            if self.current_id is not None:
                self.sync_fields()
            self.refresh_list()
            text = f'已将 {len(kept)} 条待审核轨迹设为保留（整个数据集，不受列表筛选影响）。'
            if kept:
                text += '可点击撤销恢复本次批量修改。'
            if skipped:
                text += f' 跳过 {len(skipped)} 条结构或裁剪范围异常的轨迹，原编号：' + ', '.join(map(str, skipped))
            self.notice.config(text=text)
            if skipped:
                messagebox.showwarning('部分轨迹未保留', text, parent=self.root)
        except Exception as e:
            self.report_error(e)

    def move(self, delta):
        try:
            self.editable(); self.store.move(self.current_id, delta); self.refresh_list()
            self.notice.config(text='已保存顺序。导出时按此顺序对保留项从 0 连续编号。')
        except Exception as e: self.report_error(e)

    def undo(self):
        try:
            self.editable(); self.store.undo(); self.refresh_list(); self.sync_fields()
            self.notice.config(text='已撤销最近一次修改。')
        except Exception as e: self.report_error(e)

    def adjacent(self, delta):
        ids = list(self.tree.get_children())
        if not ids: return
        current = str(self.current_id)
        i = ids.index(current) if current in ids else 0
        target = ids[max(0, min(len(ids)-1, i+delta))]
        self.next_play_target = int(target) if delta > 0 and target != current else None
        self.tree.selection_set(target)

    def export_clicked(self):
        if self.busy: return
        try:
            self.persist_fields()
            self.store.assert_current()
            count = sum(e['status']=='keep' for e in self.store.data['episodes'])
            if not count: raise ValueError('还没有保留项。请先回放并点击“保留”。')
            parent = filedialog.askdirectory(title=f'导出 {count} 条保留轨迹：选择父目录', initialdir=self.store.path.parent.parent, parent=self.root)
            if not parent: return
            name = simpledialog.askstring('新建导出目录', f'将按当前顺序导出 {count} 条保留轨迹；其他状态不导出。\n输入新目录名称：', initialvalue='astribot_reviewed_v1', parent=self.root)
            if not name: return
            if Path(name).name != name or name in ('.', '..'): raise ValueError('请输入单个目录名，不要包含路径。')
            output = Path(parent)/name
            if output.exists(): raise ValueError('此目录已存在，请换一个名称。')
            snapshot = copy.deepcopy(self.store.data)
            self.pause(); self.busy = True; self.export_button.config(state='disabled')
            self.notice.config(text=f'正在导出 {count} 条轨迹…')
            def work():
                try:
                    export(snapshot, output, progress=lambda a,b:self.events.put(('progress', a, b)))
                    self.events.put(('export_done', None, str(output)))
                except Exception as e: self.events.put(('export_error', None, str(e)))
            threading.Thread(target=work, daemon=True).start()
        except Exception as e: self.report_error(e)

    def close(self):
        if self.busy:
            messagebox.showinfo('正在导出', '请等待导出和校验完成后关闭窗口。', parent=self.root); return
        try:
            self.persist_fields()
        except Exception as e:
            self.report_error(e)
            # External changes cannot be overwritten, but must not trap the user in the UI.
            if not messagebox.askyesno('关闭窗口', '不保存当前输入并关闭？已经保存的审核结果不受影响。', parent=self.root): return
        self.pause(); self.root.destroy()


def launch(path):
    if not os.environ.get('DISPLAY'):
        auth = Path(f'/run/user/{os.getuid()}/gdm/Xauthority')
        if auth.exists() and Path('/tmp/.X11-unix/X1').exists():
            os.environ['DISPLAY'] = ':1'; os.environ.setdefault('XAUTHORITY', str(auth))
    root = tk.Tk()
    ReviewWindow(root, path)
    root.mainloop()
