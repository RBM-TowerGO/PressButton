#!/usr/bin/env python3
"""Display fixed fingertip crops after all eight model responses are complete.

This renderer makes no model requests. It reads the archived original JPEGs
and existing annotated PNGs without modifying model boxes, points or JSON.
The resulting crop sheet was never used as model input.
"""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
CROPS = ((130, 245, 240, 335), (440, 245, 550, 335))
SCALE = 3
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def font(size):
    return ImageFont.truetype(FONT, size)


def main():
    plan = json.loads((ROOT / 'plan.json').read_text())
    summary = json.loads((ROOT / 'summary.json').read_text())
    samples = plan['samples']
    if (summary.get('status') != 'complete' or len(samples) != 8
            or len(summary.get('results', [])) != 8
            or [row['id'] for row in summary['results']] != [s['id'] for s in samples]):
        raise RuntimeError('All eight first-response evaluations must be complete before rendering.')

    cell_w, cell_h = 110 * SCALE, 90 * SCALE
    left, gap, top, footer = 76, 12, 150, 85
    width = left + 4 * cell_w + 3 * gap + 20
    height = top + 8 * cell_h + 7 * gap + footer
    sheet = Image.new('RGB', (width, height), '#17202b')
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 15), 'Qwen3.8 物理指尖：原图与预测局部对照', font=font(28), fill='white')
    draw.text((20, 57), '仅用于展示：固定区域裁剪并 3× 放大；未作为模型输入，不修改预测点或框。',
              font=font(19), fill='#d5dee8')
    draw.text((20, 87), '左区域 x=130..240, y=245..335；右区域 x=440..550, y=245..335（原图像素）。',
              font=font(17), fill='#aebdce')
    for column, title in enumerate(('左指原图', '左指预测', '右指原图', '右指预测')):
        draw.text((left + column * (cell_w + gap), 117), title, font=font(21), fill='white')

    for row, sample in enumerate(samples):
        folder = ROOT / 'cases' / sample['id']
        original_path = folder / 'original.jpg'
        if hashlib.sha256(original_path.read_bytes()).hexdigest() != sample['sha256']:
            raise RuntimeError('Original image hash changed: ' + sample['id'])
        original = Image.open(original_path).convert('RGB')
        annotated = Image.open(folder / 'annotated.png').convert('RGB')
        if original.size != (640, 360) or annotated.size != original.size:
            raise RuntimeError('Fixed crops require matching 640x360 images: ' + sample['id'])
        y = top + row * (cell_h + gap)
        draw.text((20, y + 8), f'{row + 1:02}', font=font(27), fill='white')
        for column, (picture, crop) in enumerate(((original, CROPS[0]), (annotated, CROPS[0]),
                                                 (original, CROPS[1]), (annotated, CROPS[1]))):
            # Nearest-neighbor enlargement preserves the original raster pixels.
            detail = picture.crop(crop).resize((cell_w, cell_h), Image.Resampling.NEAREST)
            x = left + column * (cell_w + gap)
            sheet.paste(detail, (x, y))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline='#506276', width=1)
        if not summary['results'][row]['format_valid']:
            for column in (1, 3):
                x = left + column * (cell_w + gap)
                draw.rectangle((x, y, x + cell_w, y + 32), fill='#b00020')
                draw.text((x + 5, y + 4), 'INVALID RESPONSE', font=font(18), fill='white')

    y = height - footer + 16
    draw.text((20, y), '蓝色：左指预测；粉色：右指预测。十字圆点表示模型输出的物理指尖。',
              font=font(19), fill='#d5dee8')
    draw.text((20, y + 29), '固定裁剪范围可能截去错误或偏远的预测；完整框与点请查看 annotated_contact_sheet.jpg。',
              font=font(17), fill='#aebdce')
    destination = ROOT / 'fingertip_details.jpg'
    sheet.save(destination, quality=95, subsampling=0)
    print(destination)


if __name__ == '__main__':
    main()
