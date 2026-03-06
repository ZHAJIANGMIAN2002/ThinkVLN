import os
from datetime import timedelta
from typing import Dict, Optional

import torch
import torch.distributed as dist

_SCALAR_GROUP = None


def all_reduce_scalar_dict(stats: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return dict(stats)
    keys = sorted(stats.keys())
    if not keys:
        return {}
    group = _SCALAR_GROUP
    tensor_device = torch.device("cpu") if group is not None else device
    values = torch.tensor([float(stats[k]) for k in keys], dtype=torch.float64, device=tensor_device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM, group=group)
    return {k: float(values[i].item()) for i, k in enumerate(keys)}


def init_dist_mode(timeout_minutes: int = 120, scalar_timeout_minutes: Optional[int] = None):
    global _SCALAR_GROUP
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=int(timeout_minutes)),
            )
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        if world_size > 1 and _SCALAR_GROUP is None:
            scalar_minutes = (
                int(scalar_timeout_minutes) if scalar_timeout_minutes is not None else int(timeout_minutes)
            )
            _SCALAR_GROUP = dist.new_group(
                backend="gloo",
                timeout=timedelta(minutes=scalar_minutes),
            )
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
