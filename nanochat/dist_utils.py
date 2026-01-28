"""
Dist utilities for local Windows use (no DDP).
"""

def dist_enabled():
    import torch.distributed as dist
    return dist.is_available() and dist.is_initialized()

def get_rank_safe():
    import torch.distributed as dist
    if dist_enabled():
        return dist.get_rank()
    return 0

def get_world_size_safe():
    import torch.distributed as dist
    if dist_enabled():
        return dist.get_world_size()
    return 1
