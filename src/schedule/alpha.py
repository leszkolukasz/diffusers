from abc import ABC, abstractmethod

import torch

from src.schedule.hf import HuggingFaceDDPMBaseSchedule
from src.timestep import Timestep, TimestepConfig


class AlphaSchedule(ABC):
    timestep_config: TimestepConfig = TimestepConfig(kind="continuous", T=1.0)

    @abstractmethod
    def __call__(self, t: Timestep) -> torch.Tensor:
        pass

    def derivative(self, t: Timestep) -> torch.Tensor:
        raise NotImplementedError("Discrete only schedule does not support derivative.")


class ConstantAlphaSchedule(AlphaSchedule):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, t: Timestep):
        return torch.tensor(self.alpha, device=t.steps.device)

    def derivative(self, t: Timestep):
        return torch.tensor(0.0, device=t.steps.device)


class LinearAlphaSchedule(AlphaSchedule):
    def __call__(self, t: Timestep):
        return 1.0 - t.adapt(self.timestep_config).steps

    def derivative(self, t: Timestep):
        return torch.tensor(-1.0, device=t.steps.device)


class CosineAlphaSchedule(AlphaSchedule):
    def __call__(self, t: Timestep):
        adapted_t = t.adapt(self.timestep_config)
        return torch.cos(adapted_t.steps * (torch.pi / 2))

    def derivative(self, t: Timestep):
        adapted_t = t.adapt(self.timestep_config)
        return -torch.pi / 2 * torch.sin(adapted_t.steps * (torch.pi / 2))


class HuggingFaceDDPMAlphaSchedule(AlphaSchedule, HuggingFaceDDPMBaseSchedule):
    def __call__(self, t: Timestep) -> torch.Tensor:
        t_adapted = t.adapt(self.timestep_config)
        return self.get_alpha_at_t(t_adapted)
