#!/usr/bin/env python3
"""Combine two immutable reviewed releases, retaining the first release's split."""
import argparse
import copy
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np

from export_s1b_real_lerobot import CONTRACT, build_arrays, sha256, write_json

TASK = 'Use the right gripper to press the circular button on the elevator panel.'


def mixed_splits(first_count, second_records, first_split):
    if (set(first_split['train']) & set(first_split['validation'])
            or set(first_split['train']) | set(first_split['validation']) != set(range(first_count))):
        raise ValueError('Invalid first-release split')
    order = sorted(range(len(second_records)), key=lambda i: second_records[i]['original_id'])
    blocks = [order[i:i+8] for i in range(0, len(order), 8)]
    if len(blocks) < 2:
        raise ValueError('Need at least two acquisition blocks in the second release')
    shuffled = np.random.default_rng(42).permutation(len(blocks))
    val_blocks = set(shuffled[:min(len(blocks)-1, math.ceil(len(blocks)/6))].tolist())
    second_val = sorted(first_count+i for b, block in enumerate(blocks) if b in val_blocks for i in block)
    validation = sorted(first_split['validation'] + second_val)
    return dict(seed=42, method='preserve release1 split; release2 original-ID ordered blocks of eight',
                train=[i for i in range(first_count+len(second_records)) if i not in validation],
                validation=validation, second_validation_blocks=sorted(val_blocks),
                second_acquisition_blocks=[[first_count+i for i in b] for b in blocks],
                limitation='Acquisition blocks, not independent scenes; no frame-level random split.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first-source', type=Path, default=Path('data/astribot_reviewed_v1'))
    p.add_argument('--second-source', type=Path, default=Path('data/astribot_reviewed_2'))
    p.add_argument('--first-split', type=Path, default=Path('data/astribot_s1b_real_joint10_v1/meta/splits.json'))
    p.add_argument('--output', type=Path, default=Path('data/astribot_reviewed_mixed_v2'))
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    sources = [a.first_source.resolve(), a.second_source.resolve()]
    mappings = [json.loads((s/'episode_mapping.json').read_text()) for s in sources]
    split = mixed_splits(len(mappings[0]), mappings[1], json.loads(a.first_split.read_text()))
    records, arrays, hashes, batch_reports = [], [], set(), []
    # Validate every input before creating the merged release.
    for batch, (source, mapping) in enumerate(zip(sources, mappings)):
        batch_arrays = []
        for row in mapping:
            path = source/'episodes'/row['output_file']
            digest = sha256(path)
            if digest != row['output_sha256'] or digest in hashes:
                raise ValueError(f'Changed or duplicated episode: {path}')
            if row['audit']['errors']:
                raise ValueError(f'Invalid reviewed episode: {path}')
            hashes.add(digest)
            with h5py.File(path, 'r') as f:
                values = build_arrays(f)
            if values['valid_anchor'].sum() == 0:
                raise ValueError(f'No valid action windows: {path}')
            index = len(records)
            record = copy.deepcopy(row)
            record.update(new_id=index, output_file=f'episode_{index:06d}.hdf5',
                          source_batch=f'elevator{batch+1}', source_reviewed_root=str(source),
                          source_reviewed_file=row['output_file'], source_reviewed_episode_id=row['new_id'])
            records.append(record)
            arrays.append(values)
            batch_arrays.append(values)
        head = np.concatenate([v['action'][:, 8:] for v in batch_arrays]).astype(np.float64)
        batch_reports.append(dict(batch=f'elevator{batch+1}', episodes=len(mapping),
            frames=sum(len(v['action']) for v in batch_arrays),
            anchors=sum(int(v['valid_anchor'].sum()) for v in batch_arrays),
            head_command_min=head.min(0).tolist(), head_command_max=head.max(0).tolist(),
            head_command_std=head.std(0).tolist()))
    (a.output/'episodes').mkdir(parents=True)
    for r in records:
        # No data edits: read-only use of links to the immutable reviewed exports.
        os.link(Path(r['source_reviewed_root'])/'episodes'/r['source_reviewed_file'],
                a.output/'episodes'/r['output_file'])
    write_json(a.output/'episode_mapping.json', records)
    write_json(a.output/'splits.json', split)
    contract = json.loads(CONTRACT.read_text())
    contract.update(version='s1b-real-joint10-record-time-v2', dataset=str(a.output), task=TASK,
                    source_batches=[str(s) for s in sources],
                    head_supervision='Recorded absolute joint targets: fixed in elevator1, following-hand in elevator2',
                    action_normalization='Train split mean/std for arm/head; binary gripper identity; minimum scale 1e-6')
    write_json(a.output/'data_contract.json', contract)
    report = dict(task=TASK, batches=batch_reports, episodes=len(records),
        splits={k:dict(episodes=len(split[k]), anchors=sum(int(arrays[i]['valid_anchor'].sum()) for i in split[k]))
                for k in ('train', 'validation')},
        source_mapping_sha256={str(s):sha256(s/'episode_mapping.json') for s in sources})
    write_json(a.output/'merge_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
