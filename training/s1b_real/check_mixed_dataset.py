"""Verify the real mixed dataset through the training reader before GPU training."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent/'runtime'))
from starVLA.dataloader.s1b_real import S1BRealDataset
from starVLA.training.train_s1b_real import validation_anchors, epoch_geometry


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    task = 'Use the right gripper to press the circular button on the elevator panel.'
    mapping = json.loads((a.dataset/'meta/source_mapping.json').read_text())
    splits = json.loads((a.dataset/'meta/splits.json').read_text())
    assert len(mapping) == 212
    assert len({r['output_sha256'] for r in mapping}) == len(mapping)
    assert [r['new_id'] for r in mapping] == list(range(len(mapping)))
    tasks = pq.read_table(a.dataset/'meta/tasks.parquet').to_pylist()
    assert tasks == [dict(task_index=0, task=task)]
    table = pq.read_table(sorted((a.dataset/'data').rglob('*.parquet'))).sort_by([('index','ascending')])
    mask = np.isin(table['episode_index'].to_numpy(), splits['train'])
    raw = {key:np.asarray(table[field].to_pylist(), dtype=np.float32)
           for key,field in [('state','observation.state'),('action','action')]}
    c = json.loads(a.contract.read_text())
    for key, values in raw.items():
        x = values[mask].astype(np.float64)
        mean, scale = x.mean(0), np.maximum(x.std(0),1e-6)
        if key == 'action': mean[7], scale[7] = 0, 1
        np.testing.assert_allclose(c['normalization'][key]['mean'],mean,rtol=0,atol=1e-10)
        np.testing.assert_allclose(c['normalization'][key]['scale'],scale,rtol=0,atol=1e-10)
    assert not any(c['normalization']['action']['constant'][8:])
    counts, sample_count = {}, 0
    for split in ('train','val'):
        ds = S1BRealDataset(a.dataset,a.contract,split=split)
        try:
            for sid in ds.source_episode_ids:
                valid = ds.valid_frames[sid]
                for idx in sorted({0,len(valid)//2,len(valid)-1}):
                    item = ds.get_frame(sid,int(valid[idx]))
                    assert item['lang'] == task
                    assert item['state'].shape == (1,10) and item['action'].shape == (7,10)
                    assert item['video'].shape == (2,8,224,224,3)
                    assert not item['action_is_pad'].any() and np.isfinite(item['action']).all()
                    assert np.isin(item['action'][:,7],[0,1]).all()
                    start = item['dataset_index']
                    np.testing.assert_allclose(ds.action_raw(item['action']),raw['action'][start:start+7],rtol=0,atol=2e-7)
                    sample_count += 1
            counts[split] = dict(episodes=len(ds.source_episode_ids),anchors=len(ds))
            if split == 'val': counts['validation_samples'] = len(validation_anchors(ds,12))
            else: counts['epoch_geometry'] = epoch_geometry(len(ds),8,2,4)
        finally: ds.close()
    report = dict(status='passed',task=task,decoded_samples=sample_count,**counts)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
