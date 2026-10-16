#!/usr/bin/env python3
"""Matched reset/ready-pose image preflight, before expensive video benchmarks."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from pressb.astribot_collection.contract import atomic_json
from pressb.astribot_collection.dispatch import source_files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--envs-per-gpu', type=int, choices=[1, 2, 4, 8], required=True)
    p.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu')
    p.add_argument('--render-layout', choices=['individual', 'homogeneous', 'isolated'], default='homogeneous')
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    started = time.perf_counter()
    atomic_json(out / 'configuration.json', dict(source_sha256=source_files(ROOT),
        config=cfg, gpu=args.gpu, envs_per_gpu=args.envs_per_gpu,
        backend=args.backend, render_layout=args.render_layout, reset_rounds=20, ready_ticks=180))
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, active_gpu=args.gpu, physics_gpu=args.gpu,
        multi_gpu=False, limit_cpu_threads=8, renderer='RayTracedLighting', anti_aliasing=3,
        width=960, height=600, disable_viewport_updates=True,
        extra_args=['--/app/asyncRendering=false', '--/rtx/ecoMode/enabled=false',
                    '--/app/extensions/fsWatcherEnabled=false']))
    runtime = None
    try:
        from pressb.astribot_collection.runtime import Runtime
        from PIL import Image
        runtime = Runtime(app, cfg, out / 'shared', args.envs_per_gpu, args.gpu,
                          args.backend, 1, args.render_layout)
        runtime.probe_resets(cfg, 20, out / 'reset_probe')
        directories = [out / f'env{i}' for i in range(args.envs_per_gpu)]
        runtime.prepare_wave([cfg] * args.envs_per_gpu, directories)
        for tick in range(180):
            runtime.world.step(render=False)
            runtime.refresh()
            if tick % 4 == 0:
                runtime.render()
        if args.render_layout == 'isolated':
            # Compare while each product is active, before another room renders.
            for state in runtime.isolated_render_state.values():
                state['recorded_buffer_diagnostic'] = False
        images = runtime.render()
        import numpy as np
        equality = []
        if args.render_layout == 'isolated':
            from pressb.astribot_collection.contract import CAMERAS
            for env in runtime.envs:
                diagnostic = json.loads((out / 'shared' / f'buffer_diagnostic_env{env.index}.json').read_text())
                for name, row in zip(CAMERAS, diagnostic):
                    equality.append(dict(environment=env.index, camera=name,
                        byte_identical=row['max_cpu_cuda_difference'] == 0))
        else:
            for _, annotator, split, entries in runtime.products:
                cpu = annotator.get_data(device='cpu', do_array_copy=True)
                if isinstance(cpu, dict):
                    cpu = cpu['data']
                for (env, name), reference in zip(entries, split(cpu)):
                    identical = bool(np.array_equal(images[env.index][name], reference))
                    equality.append(dict(environment=env.index, camera=name, byte_identical=identical))
        atomic_json(out / 'cpu_cuda_equality.json', equality)
        if not all(row['byte_identical'] for row in equality):
            raise RuntimeError('Packed CUDA frames differ from native CPU camera products')
        for env in runtime.envs:
            for name, rgb in images[env.index].items():
                Image.fromarray(rgb).save(env.output / f'initial_ready_{name}.png')
            atomic_json(env.output / 'report.json', dict(config=cfg, environment_offset_m=env.offset.tolist()))
        atomic_json(out / 'result.json', dict(status='finished', elapsed_s=time.perf_counter()-started,
            timings=runtime.timings, directories=[str(d) for d in directories]))
    finally:
        if runtime:
            runtime.close()
        app.app.post_quit()
        app.close()


if __name__ == '__main__':
    main()
