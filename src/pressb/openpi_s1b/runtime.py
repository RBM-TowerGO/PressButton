"""Pinned upstream integration. Heavy imports stay behind explicit entrypoints."""
import dataclasses
import importlib.util
from pathlib import Path
import subprocess
import sys
from .dataset import read_json
from .transforms import require

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pi05_s1b_joint10'


def check_cuda_build(torch, device):
    """Fail before allocating model weights on unsupported Blackwell builds."""
    require(torch.cuda.is_available(), 'CUDA is unavailable in the policy environment')
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 10:
        arches = torch.cuda.get_arch_list()
        require(any(a.startswith(f'sm_{major}') or a.startswith(f'compute_{major}') for a in arches),
                'This PyTorch wheel lacks support for the selected Blackwell GPU. '
                'Install a compatible CUDA 12.8+ PyTorch build in .venv-openpi and revalidate before deployment.')


def upstream(root=None):
    require(sys.version_info[:2] == (3, 11), 'Use the isolated Python 3.11 OpenPI environment')
    lock = read_json(ROOT/'repos.lock.json')['repositories']['openpi']
    path = Path(root or ROOT/lock['path']).resolve()
    require(subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
            == lock['runtime_commit'], 'OpenPI revision mismatch')
    require(not subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'],
                                       text=True).strip(), 'OpenPI tracked source is modified')
    # No network or model loading: only make the pinned packages available.
    sys.path.insert(0, str(path/'src'))
    sys.path.insert(0, str(path/'packages/openpi-client/src'))
    import openpi
    require(Path(openpi.__path__[0]).resolve() == path/'src/openpi', 'Wrong imported OpenPI source')
    return path, lock


def config(bundle, *, base_weights=None, exp_name='s1b', checkpoint_base='runs/openpi',
           batch_size=8, steps=30000, save_interval=1000, num_workers=2, resume=False):
    from openpi.training import config as cfg
    from openpi.models.pi0_config import Pi0Config
    from openpi import transforms
    from .transforms import Inputs, Outputs
    bundle = Path(bundle).resolve()
    settings = read_json(bundle/'config.json')
    expected = dict(name=NAME, pi05=True, action_dim=32, action_horizon=7, discrete_state_input=True,
                    fps=30, cameras={'head': 'base_0_rgb', 'right': 'right_wrist_0_rgb'})
    require(settings == expected, 'Unsupported S1B model config')
    stats = read_json(bundle/'normalization.json')

    @dataclasses.dataclass(frozen=True)
    class S1BDataConfig(cfg.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            # Our reversible transform excludes binary gripper actions. Upstream
            # normalization remains an explicit no-op; never normalize twice.
            return cfg.DataConfig(repo_id=NAME, asset_id=NAME, norm_stats={},
                data_transforms=transforms.Group(inputs=[Inputs(stats)], outputs=[Outputs(stats)]),
                model_transforms=cfg.ModelTransformFactory()(model_config), use_quantile_norm=False)

    return cfg.TrainConfig(name=NAME, exp_name=exp_name,
        model=Pi0Config(pi05=True, action_dim=32, action_horizon=7, discrete_state_input=True,
                        pytorch_compile_mode=None),
        data=S1BDataConfig(), pytorch_weight_path=str(base_weights) if base_weights else None,
        checkpoint_base_dir=str(checkpoint_base), batch_size=batch_size, num_train_steps=steps,
        save_interval=save_interval, num_workers=num_workers, wandb_enabled=False, resume=resume,
        ema_decay=None, policy_metadata={'backend': 'openpi_pi05'})


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
