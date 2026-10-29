#!/usr/bin/env python3
"""Offline S1B/OpenPI tooling. Training/conversion require explicit subcommands."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pressb.openpi_s1b.dataset import Dataset, read_json
from pressb.openpi_s1b.artifacts import prepare, validate_bundle, export, seal_training, write_json
from pressb.openpi_s1b.transforms import require
from pressb.s1b_deployment.contract import sha


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('audit', 'prepare'):
        c = sub.add_parser(name)
        c.add_argument('--dataset', required=True)
        c.add_argument('--splits')
        c.add_argument('--anchors')
        c.add_argument('--contract')
        if name == 'prepare': c.add_argument('--output', required=True)
    sub.add_parser('check-deps')
    c = sub.add_parser('convert')
    c.add_argument('--checkpoint', required=True, help='Local official pi05_base JAX checkpoint')
    c.add_argument('--output', required=True)
    c = sub.add_parser('train')
    c.add_argument('--dataset', required=True)
    c.add_argument('--bundle', required=True)
    c.add_argument('--base-weights', required=True)
    c.add_argument('--exp-name', required=True)
    c.add_argument('--checkpoint-base', default='runs/openpi')
    c.add_argument('--batch-size', type=int, default=8, help='Global batch across all ranks')
    c.add_argument('--steps', type=int, default=30000)
    c.add_argument('--save-interval', type=int, default=1000)
    c.add_argument('--num-workers', type=int, default=2)
    c.add_argument('--resume', action='store_true')
    c = sub.add_parser('export')
    c.add_argument('--checkpoint', required=True)
    c.add_argument('--output', required=True)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.command in ('audit', 'prepare'):
        ds = Dataset(a.dataset, splits=a.splits, anchors=a.anchors, contract=a.contract)
        # Decode distinct first/middle/last samples without modifying source data.
        for index in sorted({0, len(ds)//2, len(ds)-1}):
            row = ds[index]
            require(row['head'].shape == (720,1280,3) and row['right'].shape == (360,640,3), 'Camera shape mismatch')
        if a.command == 'prepare': prepare(ds, a.output)
        print(json.dumps(dict(train_windows=len(ds), validation_windows=len(ds.manifests['anchors']['validation']),
                              decoded_samples=min(len(ds),3), model_inference_verified=False), indent=2))
        return
    if a.command == 'export':
        print(json.dumps(export(a.checkpoint, a.output), indent=2))
        return

    from pressb.openpi_s1b.runtime import upstream, config, load_script
    path, lock = upstream()
    if a.command == 'check-deps':
        import torch, transformers, av, pyarrow
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.training import data_loader
        from openpi.policies.policy_config import create_trained_policy
        from transformers.models.gemma import modeling_gemma
        import inspect
        require('cond' in inspect.signature(modeling_gemma.GemmaRMSNorm.forward).parameters,
                'Official transformers patch is missing')
        model = Pi0Config(pi05=True, action_dim=32, action_horizon=7, pytorch_compile_mode=None)
        require(model.discrete_state_input and model.action_horizon == 7, 'Wrong pi05 config')
        # Exercise our real config and upstream transforms without downloading
        # the tokenizer. This stand-in is local to this import-only check.
        import tempfile
        import dataclasses
        import io
        from unittest.mock import patch
        import numpy as np
        from pressb.openpi_s1b.transforms import fit_stats
        with tempfile.TemporaryDirectory(prefix='s1b-openpi-check-') as folder:
            folder = Path(folder)
            write_json(folder/'config.json', dict(name='pi05_s1b_joint10', pi05=True, action_dim=32,
                action_horizon=7, discrete_state_input=True, fps=30,
                cameras={'head':'base_0_rgb','right':'right_wrist_0_rgb'}))
            write_json(folder/'normalization.json', fit_stats(np.zeros((8,10)), np.zeros((8,10))))
            cfg = config(folder)
            torch.save({'config': dataclasses.asdict(cfg)}, io.BytesIO())
            class TestTokenizer:
                def __init__(self, max_len): self.max_len = max_len
                def tokenize(self, *args):
                    return np.zeros(self.max_len,np.int32), np.ones(self.max_len,np.bool_)
            with patch('openpi.training.config._tokenizer.PaligemmaTokenizer', TestTokenizer):
                data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
                raw = dict(head=np.zeros((720,1280,3),np.uint8), right=np.zeros((360,640,3),np.uint8),
                    state=np.zeros(10,np.float32), actions=np.zeros((7,10),np.float32), prompt='offline check')
                transformed = data_loader.transform_dataset([raw], data_config)[0]
                require(transformed['actions'].shape == (7,32) and transformed['state'].shape == (32,),
                        'Upstream pad transform mismatch')
                require(all(v.shape == (224,224,3) for v in transformed['image'].values()),
                        'Upstream image transform mismatch')
                require(not transformed['image_mask']['left_wrist_0_rgb'], 'Missing camera mask mismatch')
        print(json.dumps(dict(openpi_commit=lock['runtime_commit'], torch=torch.__version__,
            transformers=transformers.__version__, python=sys.version.split()[0],
            cuda_runtime=torch.version.cuda, cuda_model_execution_verified=False,
            model_config='pi05_s1b_joint10', model_inference_verified=False,
            note='Imports/config only; no weights loaded, no tokenizer downloaded'), indent=2))
        return
    if a.command == 'convert':
        require(Path(a.checkpoint).is_dir(), 'A local pi05_base checkpoint is required; no automatic download')
        require((Path(a.checkpoint)/'params').is_dir(), 'Expected checkpoint root containing params/')
        # This pinned official converter selects adaptive norms using the path.
        require('pi05' in str(Path(a.checkpoint).resolve()), 'Official converter requires pi05 in checkpoint path')
        require(not Path(a.output).exists(), 'Conversion output already exists')
        from openpi.models.pi0_config import Pi0Config
        converter = load_script(path/'examples/convert_jax_model_to_pytorch.py', '_s1b_official_converter')
        converter.convert_pi0_checkpoint(str(Path(a.checkpoint).resolve()), 'bfloat16', a.output,
                                         Pi0Config(pi05=True, action_dim=32, action_horizon=7, pytorch_compile_mode=None))
        write_json(Path(a.output)/'s1b_base.json', dict(source='pi05_base', upstream=lock['runtime_commit'],
                    model_sha256=sha(Path(a.output)/'model.safetensors')))
        return

    bundle = Path(a.bundle).resolve()
    validate_bundle(bundle)
    require(Path(a.base_weights, 'model.safetensors').is_file(), 'Converted PyTorch weights required')
    base_identity = read_json(Path(a.base_weights)/'s1b_base.json')
    require(base_identity == dict(source='pi05_base', upstream=lock['runtime_commit'],
                                 model_sha256=sha(Path(a.base_weights)/'model.safetensors')),
            'Use the S1B convert entrypoint for pi05_base weights')
    require(a.steps > 0 and a.batch_size > 0 and a.save_interval > 0 and a.num_workers >= 0, 'Invalid training options')
    ds = Dataset(a.dataset, splits=bundle/'split.json', anchors=bundle/'anchors.json', contract=bundle/'source_contract.json')
    # Bundled JSON may have different formatting than source manifests; compare
    # content and numeric hashes rather than reserialized source bytes.
    provenance = read_json(bundle/'provenance.json')
    require(ds.numeric_hashes == provenance['numeric_files'], 'Prepared numeric dataset changed')
    require(ds.provenance()['source_files'] == provenance['source_files'], 'Prepared images/metadata changed')
    require(ds.statistics() == read_json(bundle/'normalization.json'), 'Training statistics changed')
    cfg = config(bundle, base_weights=a.base_weights, exp_name=a.exp_name, checkpoint_base=a.checkpoint_base,
                 batch_size=a.batch_size, steps=a.steps, save_interval=a.save_interval,
                 num_workers=a.num_workers, resume=a.resume)
    import torch
    from pressb.openpi_s1b.runtime import check_cuda_build
    check_cuda_build(torch, f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    from openpi.training import data_loader
    original_factory = data_loader.create_torch_dataset
    def factory(data_config, action_horizon, model_config):
        require(data_config.repo_id == cfg.name and action_horizon == 7, 'Unexpected training dataset')
        return ds
    trainer = load_script(path/'scripts/train_pytorch.py', '_s1b_official_trainer')
    original_save = trainer.save_checkpoint
    training = {k:v for k,v in vars(a).items() if k not in ('command', 'resume')}
    training['base_identity'] = base_identity
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    require(a.batch_size % world_size == 0 and a.batch_size >= world_size, 'Global batch must divide world size')
    # Upstream resume chooses latest checkpoint, so ensure its S1B contract is identical.
    if a.resume:
        existing = sorted((p for p in cfg.checkpoint_dir.iterdir() if p.name.isdigit()), key=lambda p:int(p.name))
        require(bool(existing), 'No checkpoint to resume')
        latest = existing[-1]
        validate_bundle(latest/'s1b')
        require(read_json(latest/'s1b/bundle.json') == read_json(bundle/'bundle.json'), 'Resume bundle differs')
    else:
        require(not cfg.checkpoint_dir.exists(), 'Run exists; use --resume or a new exp-name')
    def save(model, optimizer, global_step, config, is_main, data_config):
        original_save(model, optimizer, global_step, config, is_main, data_config)
        checkpoint = config.checkpoint_dir/str(global_step)
        if is_main and checkpoint.is_dir():
            seal_training(checkpoint, bundle, global_step, training)
    trainer.save_checkpoint = save
    data_loader.create_torch_dataset = factory
    try:
        trainer.init_logging()
        trainer.train_loop(cfg)
    finally:
        data_loader.create_torch_dataset = original_factory
        trainer.save_checkpoint = original_save


if __name__ == '__main__':
    main()
