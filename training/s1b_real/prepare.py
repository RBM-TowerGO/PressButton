"""Hash published dataset and bind training schema, frozen splits and normalization."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from starVLA.dataloader.s1b_real import SEMANTICS,_sha256,_canonical_sha
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=a.dataset.resolve();stats=json.loads((root/'meta/training_normalization.json').read_text())
hashes={str(f.relative_to(root)):_sha256(f) for f in sorted(root.rglob('*')) if f.is_file()}
contract=dict(schema=SEMANTICS,dataset_sha256=_canonical_sha(hashes),file_sha256=hashes,
 file_sizes={f:(root/f).stat().st_size for f in hashes},normalization={
 k:dict(mean=stats[field]['mean'],scale=stats[field]['std'],constant=stats[field]['constant'])
 for k,field in [('state','observation.state'),('action','action')]})
a.output.parent.mkdir(parents=True,exist_ok=True)
text=json.dumps(contract,indent=2)+'\n'
if a.output.exists() and a.output.read_text()!=text:raise ValueError('Existing contract changed')
a.output.write_text(text)
print(contract['dataset_sha256'])
