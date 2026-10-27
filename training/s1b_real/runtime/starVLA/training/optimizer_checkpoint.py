"""Bounded staging for ZeRO-1 checkpoint consolidation (PyTorch 2.7).

Keep the upstream state-dict format and parameter-index mapping. Replace only
its private object transport during consolidation, then restore that transport.
Each DDP process calls this synchronously on its training thread.
"""
import io

import torch
import torch.distributed as dist

CHUNK_BYTES = 64 * 1024 * 1024


def _broadcast_state(obj, src_rank, group=dist.group.WORLD, device=torch.device('cpu')):
    source = dist.get_rank() == src_rank
    if source:
        stream = io.BytesIO()
        torch.save(obj, stream)
        # ByteTensor(bytearray) expands gigabytes into Python sequence elements.
        # frombuffer instead views the serialized bytes without that conversion.
        payload = torch.frombuffer(stream.getbuffer(), dtype=torch.uint8)
        size = payload.numel()
    else:
        size = 0
    length = torch.tensor([size], dtype=torch.int64, device=device)
    dist.broadcast(length, src=src_rank, group=group)
    size = int(length.item())
    if not source:
        storage = bytearray(size)
        payload = torch.frombuffer(storage, dtype=torch.uint8)
    for offset in range(0, size, CHUNK_BYTES):
        count = min(CHUNK_BYTES, size - offset)
        if source:
            wire = payload[offset:offset + count].to(device)
        else:
            wire = torch.empty(count, dtype=torch.uint8, device=device)
        dist.broadcast(wire, src=src_rank, group=group)
        if not source:
            payload[offset:offset + count].copy_(wire.cpu())
        del wire
    if source:
        return obj
    # Consolidation stores states on CPU; do not reconstruct Adam moments on GPU.
    return torch.load(io.BytesIO(storage), map_location='cpu', weights_only=False)


def consolidate_optimizer_state(optimizer, to=0):
    if not hasattr(optimizer, 'consolidate_state_dict'):
        return
    import torch.distributed.optim.zero_redundancy_optimizer as zero
    if not isinstance(optimizer, zero.ZeroRedundancyOptimizer):
        optimizer.consolidate_state_dict(to=to)
        return
    original = zero._broadcast_object
    try:
        zero._broadcast_object = _broadcast_state
        optimizer.consolidate_state_dict(to=to)
    finally:
        zero._broadcast_object = original
