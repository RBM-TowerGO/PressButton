#!/usr/bin/env python3
"""Replay one recorded localization request to DeepSeek at several thinking levels.

Source: run 20261105_141334_26369ffd, round 0027, observation ac60a72b... The
recorded head button_box landed on the white gripper-base block instead of the
down-call button. Only the thinking-related fields of the original body are
changed; prompt, images and tool stay byte-identical.

Usage: DEEPSEEK_API_KEY=... python script_used.py [--trials 3] [--levels off low medium high]
"""
import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from pressb.agent_loop import model_provider  # noqa: E402
from pressb.agent_loop.common import request, atomic_json  # noqa: E402

RUN = ROOT/'outputs/agent_web/20261105_141334_26369ffd/agent'
SOURCE = RUN/'rounds/0027/localization/request.json'
HEAD_IMAGE = RUN/'observations/ac60a72be04e47108a91eaa5028d54f9/head.jpg'
# Manual ground truth from a zoomed crop of head.jpg (1280x720): button ring
# about px [618,382,660,415] -> normalized_1000.
HEAD_GT = [483, 531, 516, 576]
# The recorded (wrong) answer, for reference.
RECORDED_HEAD = [485, 711, 512, 754]
OUT = Path(__file__).resolve().parent/'results'


def variant(body, level):
    """Same body; only thinking switch, effort, tool_choice and max_tokens differ."""
    body = copy.deepcopy(body)
    for key in ('thinking', 'reasoning_effort'):
        body.pop(key, None)
    if level == 'off':
        # Production values before the first gripper close.
        body.update(thinking=dict(type='disabled'), tool_choice='required', max_tokens=512)
    else:
        # Production uses low with 4096; larger efforts get more room so truncation
        # does not masquerade as a localization error.
        body.update(thinking=dict(type='enabled'), reasoning_effort=level, tool_choice='auto',
                    max_tokens=4096 if level == 'low' else 8192)
    return body


def iou(a, b):
    w = min(a[2], b[2])-max(a[0], b[0]); h = min(a[3], b[3])-max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w*h
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def score(box):
    if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
        return dict(center_in_gt=None, iou=None)
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    inside = HEAD_GT[0] <= cx <= HEAD_GT[2] and HEAD_GT[1] <= cy <= HEAD_GT[3]
    on_block = RECORDED_HEAD[1]-20 <= cy <= RECORDED_HEAD[3]+20 and abs(cx-500) < 40
    return dict(center=[round(cx, 1), round(cy, 1)], center_in_gt=inside,
                iou=round(iou(box, HEAD_GT), 3), near_recorded_wrong_box=on_block)


def arguments(message):
    calls = message.get('tool_calls') or []
    if calls:
        return json.loads(calls[0]['function']['arguments'])
    content = (message.get('content') or '').strip()
    start, end = content.find('{'), content.rfind('}')
    return json.loads(content[start:end+1]) if start >= 0 else None


def run(level, trial, body):
    started = time.monotonic()
    record = dict(level=level, trial=trial)
    try:
        response = request(model_provider.base(), '/chat/completions', variant(body, level), timeout=300)
    except Exception as exc:
        record.update(error=f'{type(exc).__name__}: {exc}'[:500], elapsed_s=round(time.monotonic()-started, 1))
        return record
    choice = response['choices'][0]; message = choice['message']
    usage = response.get('usage') or {}
    record.update(elapsed_s=round(time.monotonic()-started, 1), finish_reason=choice.get('finish_reason'),
                  completion_tokens=usage.get('completion_tokens'),
                  reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'))
    atomic_json(OUT/f'{level}-{trial}.json', response)
    try:
        args = arguments(message)
    except (ValueError, KeyError, TypeError) as exc:
        record.update(parse_error=str(exc)); return record
    head = (args or {}).get('head') or {}
    wrist = (args or {}).get('wrist') or {}
    record.update(head_visibility=head.get('button_visibility'), head_box=head.get('button_box'),
                  wrist_box=wrist.get('button_box'), **score(head.get('button_box')))
    return record


def overlay(records):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    colors = dict(off='red', low='orange', medium='cyan', high='magenta')
    im = Image.open(HEAD_IMAGE).convert('RGB'); d = ImageDraw.Draw(im); W, H = im.size
    px = lambda b: [b[0]*W/1000, b[1]*H/1000, b[2]*W/1000, b[3]*H/1000]
    d.rectangle(px(HEAD_GT), outline='lime', width=3)
    for r in records:
        if isinstance(r.get('head_box'), list) and len(r['head_box']) == 4:
            d.rectangle(px(r['head_box']), outline=colors.get(r['level'], 'white'), width=2)
    im.save(OUT/'head_boxes.png')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--levels', nargs='+', default=['off', 'low', 'medium', 'high'])
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    model_provider.configure('deepseek')
    body = json.loads(SOURCE.read_text())
    jobs = [(lvl, t) for lvl in a.levels for t in range(a.trials)]
    with ThreadPoolExecutor(a.workers) as pool:
        records = list(pool.map(lambda j: run(j[0], j[1], body), jobs))
    atomic_json(OUT/'summary.json', dict(source=str(SOURCE), head_gt_normalized_1000=HEAD_GT,
                                         recorded_head_box=RECORDED_HEAD, records=records))
    overlay(records)
    for lvl in a.levels:
        rows = [r for r in records if r['level'] == lvl]
        hit = sum(1 for r in rows if r.get('center_in_gt'))
        print(f'== {lvl}: {hit}/{len(rows)} head center in GT')
        for r in rows:
            print('  ', json.dumps({k: r.get(k) for k in ('trial', 'head_box', 'center_in_gt', 'iou',
                  'near_recorded_wrong_box', 'elapsed_s', 'completion_tokens', 'reasoning_tokens',
                  'finish_reason', 'error', 'parse_error')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
