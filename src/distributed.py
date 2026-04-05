import os

import torch
from loguru import logger

_RANK = 0
_WORLD_SIZE = 1
_LOCAL_RANK = 0


def is_distributed():
    return "SLURM_PROCID" in os.environ


def setup():
    global _RANK, _WORLD_SIZE, _LOCAL_RANK

    _RANK = int(os.environ["SLURM_PROCID"])
    _WORLD_SIZE = int(os.environ["SLURM_NTASKS"])
    _LOCAL_RANK = int(os.environ["SLURM_LOCALID"])

    if "MASTER_ADDR" not in os.environ:
        raise RuntimeError("MASTER_ADDR environment variable not set.")

    torch.distributed.init_process_group(
        backend="nccl", world_size=_WORLD_SIZE, rank=_RANK
    )
    torch.cuda.set_device(_LOCAL_RANK)

    print(
        f"Initialized distributed training: RANK {_RANK}, WORLD_SIZE {_WORLD_SIZE}, LOCAL_RANK {_LOCAL_RANK}"
    )

    if _RANK != 0:
        logger.remove()


def get_rank():
    return _RANK


def get_world_size():
    return _WORLD_SIZE


@staticmethod
def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
