import os

import torch
from loguru import logger

RANK = 0
WORLD_SIZE = 1
LOCAL_RANK = 0


def is_distributed():
    return "SLURM_PROCID" in os.environ


def setup():
    global RANK, WORLD_SIZE, LOCAL_RANK

    RANK = int(os.environ["SLURM_PROCID"])
    WORLD_SIZE = int(os.environ["SLURM_NTASKS"])
    LOCAL_RANK = int(os.environ["SLURM_LOCALID"])

    if "MASTER_ADDR" not in os.environ:
        raise RuntimeError("MASTER_ADDR environment variable not set.")

    torch.distributed.init_process_group(
        backend="nccl", world_size=WORLD_SIZE, rank=RANK
    )
    torch.cuda.set_device(LOCAL_RANK)

    if RANK != 0:
        logger.remove()


@staticmethod
def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
