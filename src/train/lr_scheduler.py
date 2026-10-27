from enum import StrEnum

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class LRSchedule(StrEnum):
    WARMUP_COSINE = "warmup_cosine"
    EDM2 = "edm2"


def edm2_learning_rate(
    *,
    cur_nimg: int,
    ref_lr: float,
    ref_batches: int = 35000,
    ref_batch_size: int = 2048,
    rampup_mimg: float = 10.0,
) -> float:
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * ref_batch_size), 1))
    if rampup_mimg > 0:
        lr *= min(cur_nimg / (rampup_mimg * 1e6), 1)
    return float(lr)


class WarmupCosineLR(SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps_percentage: float = 0.05,
        warmup_start_factor: float = 1e-5,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        warmup_steps = int(total_steps * warmup_steps_percentage)
        assert warmup_steps < total_steps, "Warmup steps must be less than total steps"

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=eta_min,
        )

        super().__init__(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
            last_epoch=-1,
        )

        # Ugly but works
        if last_epoch > 0:
            for _ in range(last_epoch):
                self.step()
