"""Offline initialization/LR ablations on trusted local smoke artifacts; no rollout."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pressb.online_rl.learner import LearnerConfig, SACLearner, _EnsembleLinear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--variants', nargs='+', choices=('baseline', 'lr_3e-5', 'lr_1e-5', 'fanin_first', 'fanin_last', 'fanin_all', 'fixed'),
                        default=['baseline', 'lr_3e-5', 'lr_1e-5', 'fanin_first', 'fanin_last', 'fanin_all'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = torch.load(args.source / 'learner.pt', map_location='cpu', weights_only=False)['config']
    with np.load(args.source / 'replay.npz') as archive:
        data = {k: archive[k].copy() for k in archive.files if k != 'meta'}
    z = data['obs'][:, 64:]
    results = {'source': str(args.source), 'steps': args.steps,
               'latent_norm': np.linalg.norm(z, axis=1).tolist(),
               'latent_cosine': ((z @ z.T) / np.outer(np.linalg.norm(z, axis=1), np.linalg.norm(z, axis=1))).tolist(),
               'runs': []}
    for variant in args.variants:
        for seed in (20261024, 42, 43, 44, 45):
            config = dict(cfg, seed=seed)
            if variant == 'fixed':
                config.update(critic_init='fan_in', q_lr=3e-5)
            if variant.startswith('lr_'):
                config['q_lr'] = float(variant[3:])
            learner = SACLearner(LearnerConfig(**config))
            layers = [m for m in learner.qs.modules() if isinstance(m, _EnsembleLinear)]
            selected = {'fanin_first': layers[:1], 'fanin_last': layers[-1:], 'fanin_all': layers}.get(variant, [])
            with torch.no_grad():
                for layer in selected:
                    # Same random draw, corrected fan_in for the (in,out) layout.
                    factor = math.sqrt(layer.weight.shape[2] / layer.weight.shape[1])
                    layer.weight.mul_(factor)
                    layer.bias.mul_(factor)
            learner.q_targets = copy.deepcopy(learner.qs).requires_grad_(False)
            rng = np.random.default_rng(seed)
            rows = []
            for step in range(args.steps):
                indices = rng.integers(0, len(data['obs']), size=config['batch_size'])
                batch = {k: v[indices] for k, v in data.items()}
                values = learner._prepare_batch(batch)
                def logits():
                    action = learner._critic_action(values['action'], values['base_action'])
                    x = torch.cat((values['obs'], action), -1).unsqueeze(0).expand(learner.qs.count, -1, -1)
                    with torch.no_grad():
                        return learner.qs.net(x).flatten().tolist()
                before = logits()
                metrics = learner.update(batch)
                rows.append(dict(metrics, logits_before=before, logits_after=logits()))
            results['runs'].append(dict(variant=variant, seed=seed, metrics=rows))
            print(variant, seed, 'q1', round(rows[0]['q_predicted'], 5), 'logit_after1', [round(x, 3) for x in rows[0]['logits_after']], 'zero_grad_steps', sum(r['critic_grad_norm'] == 0 for r in rows), 'final_q', round(rows[-1]['q_predicted'], 5), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
