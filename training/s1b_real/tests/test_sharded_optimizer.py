"""Two CPU ranks exercise consolidation and AdamW resume without an HPC job."""
from copy import deepcopy

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from starVLA.training.train_s1b_real import build_optimizer


def worker(rank, rendezvous, checkpoint):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        reference = deepcopy(model)
        cfg = OmegaConf.create(dict(optimizer_sharding=True, optimizer=dict(betas=[.9, .95], eps=1e-8), weight_decay=1e-8))
        def groups(m):
            return [dict(params=list(layer.parameters()), lr=1e-3, name=str(i)) for i, layer in enumerate(m)]
        optimizer = build_optimizer(groups(model), cfg)
        reference_optimizer = torch.optim.AdamW(groups(reference), betas=(.9, .95), eps=1e-8, weight_decay=1e-8, foreach=False)
        def step(m, opt):
            opt.zero_grad(set_to_none=True)
            m(torch.ones(2, 4)).square().mean().backward()
            opt.step()
        step(model, optimizer)
        step(reference, reference_optimizer)
        from starVLA.training import optimizer_checkpoint
        import torch.distributed.optim.zero_redundancy_optimizer as zero
        original = zero._broadcast_object
        optimizer_checkpoint.CHUNK_BYTES = 64  # Force many chunks for tiny states.
        optimizer_checkpoint.consolidate_optimizer_state(optimizer, to=0)
        assert zero._broadcast_object is original
        if rank == 0:
            torch.save(optimizer.state_dict(), checkpoint)
        dist.barrier()
        restored = build_optimizer(groups(model), cfg)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        assert restored.optim.state
        assert all(int(state["step"]) == 1 for state in restored.optim.state.values())
        step(model, restored)
        step(reference, reference_optimizer)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    finally:
        dist.destroy_process_group()


def test_sharded_adamw_save_resume(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / "rendezvous"), str(tmp_path / "optimizer.pt")), nprocs=2, join=True)
