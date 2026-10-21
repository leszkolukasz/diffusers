import copy

import torch
from torch import nn


class ExpMovingAverageWrapper:
    ema_model: nn.Module
    decay: float

    def __init__(
        self,
        model: nn.Module,
        *,
        total_steps: int,
        halflife_ratio: float = 0.05,
        warmup_ratio: float = 0.05,
        update_every: int = 10,
    ):
        self.total_steps = max(1, total_steps)
        self.halflife_steps = max(1, int(halflife_ratio * self.total_steps))
        self.warmup_ratio = warmup_ratio
        self.update_every = max(1, update_every)
        self.decay = 0.0

        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def _decay(self, step: int) -> float:
        if self.warmup_ratio > 0:
            halflife = min(
                self.halflife_steps,
                max(1, int(step * self.warmup_ratio)),
            )
        else:
            halflife = self.halflife_steps

        return 0.5 ** (self.update_every / halflife)

    @torch.no_grad()
    def update(self, model: nn.Module, *, step: int):
        self.decay = self._decay(step)
        for ema_param, new_param in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(new_param.data, alpha=1.0 - self.decay)
