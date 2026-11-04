#!/usr/bin/env python3
"""Display fixed crops of completed closed-gripper-end predictions.

This renderer never requests model inference. Original images, archived model
coordinates, and annotations are read without modification. The crop sheet is
only a viewing aid and was not supplied to the model.
"""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
CROP = (240, 220, 440, 330)
SCALE = 3
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def font(size):
    return ImageFont.truetype(FONT, size)


def main():
    plan = json.loads((ROOT / 'plan.json').read_text())
    summary = json.loads((ROOT / 'summary.json').read_text())
    samples, results = plan['samples'], summary.get('results', [])
    if (summary.get('status') != 'complete' or len(samples) != 8 or len(results) != 8
            or [row['id'] for row in results] != [row['id'] for row in samples]):
        raise RuntimeError('All eight first-response evaluations must be complete before rendering.')

    cell_w, cell_h = 200 * SCALE, 110 * SCALE
    left, gap, top, footer = 76, 16, 151, 80
    width = left + 2 * cell_w + gap + 20
    height = top + len(samples) * cell_h + (len(samples) - 1) * gap + footer
    sheet = Image.new('RGB', (width, height), '#17202b')
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 14), 'Qwen3.8 合拢夹爪物理末端：原图与模型预测', font=font(27), fill='white')
    draw.text((20, 57), '固定裁剪仅用于展示，未作为模型输入；直接放大既有标注，不修改预测框或点。',
              font=font(18), fill='#d5dee8')
    draw.text((20, 87), '原图区域 x=240..440, y=220..330；3× 最近邻放大。每行对应同一张原图。',
              font=font(18), fill='#aebdce')
    for column, label in enumerate(('原图局部', '模型末端预测')):
        draw.text((left + column * (cell_w + gap), 118), label, font=font(22), fill='white')

    for index, (sample, result) in enumerate(zip(samples, results)):
        folder = ROOT / 'cases' / sample['id']
        original_path = folder / 'original.jpg'
        if hashlib.sha256(original_path.read_bytes()).hexdigest() != sample['sha256']:
            raise RuntimeError('Original image hash changed: ' + sample['id'])
        original = Image.open(original_path).convert('RGB')
        annotated = Image.open(folder / 'annotated.png').convert('RGB')
        if original.size != (640, 360) or annotated.size != original.size:
            raise RuntimeError('Fixed crops require matching 640x360 images: ' + sample['id'])
        y = top + index * (cell_h + gap)
        draw.text((20, y + 8), f'{index + 1:02}', font=font(28), fill='white')
        for column, picture in enumerate((original, annotated)):
            x = left + column * (cell_w + gap)
            detail = picture.crop(CROP).resize((cell_w, cell_h), Image.Resampling.NEAREST)
            sheet.paste(detail, (x, y))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline='#506276', width=1)
        if not result['format_valid'] or result.get('status') != 'located':
            x = left + cell_w + gap
            label = ('INVALID RESPONSE' if not result['format_valid']
                     else 'End: ' + result['status'])
            draw.rectangle((x, y, x + cell_w - 1, y + 34),
                           fill='#b00020' if not result['format_valid'] else '#17202b')
            draw.text((x + 8, y + 4), label, font=font(21), fill='white')

    y = height - footer + 13
    draw.text((20, y), '青框：模型预测的物理末端区域；粉色十字：模型预测的末端表面代表点。',
              font=font(19), fill='#d5dee8')
    draw.text((20, y + 30), '固定范围可能截去框点；完整预测请查看 annotated_contact_sheet.jpg。坐标未作人工修正。',
              font=font(17), fill='#aebdce')
    destination = ROOT / 'end_details.jpg'
    sheet.save(destination, quality=95, subsampling=0)
    print(destination)


if __name__ == '__main__':
    main()
