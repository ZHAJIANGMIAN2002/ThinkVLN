import os
from typing import Dict

import torch
import torch.distributed as dist


def all_reduce_scalar_dict(stats: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return dict(stats)
    keys = sorted(stats.keys())
    values = torch.tensor([float(stats[k]) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {k: float(values[i].item()) for i, k in enumerate(keys)}


def init_dist_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
    else:
        rank = 0
        world_size = 1
        gpu = 0
    return rank, world_size, gpu


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


__all__ = [
    "all_reduce_scalar_dict",
    "init_dist_mode",
    "get_rank",
    "get_world_size",
]
