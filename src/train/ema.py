import copy

import torch
from torch import nn


class ExpMovingAverageWrapper:
    ema_model: nn.Module
    decay: float

    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_param, new_param in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(new_param.data, alpha=1.0 - self.decay)
