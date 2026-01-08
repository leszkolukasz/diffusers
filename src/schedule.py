from abc import ABC, abstractmethod

import torch

from src.timestep import Timestep, TimestepConfig


class AlphaSchedule(ABC):
    timestep_config: TimestepConfig = TimestepConfig(kind="continuous", max_t=1.0)

    @abstractmethod
    def __call__(self, t: Timestep) -> torch.Tensor:
        pass

    @abstractmethod
    def derivative(self, t: Timestep) -> torch.Tensor:
        pass


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


class SigmaSchedule(ABC):
    timestep_config: TimestepConfig = TimestepConfig(kind="continuous", max_t=1.0)

    @abstractmethod
    def __call__(self, t: Timestep) -> torch.Tensor:
        pass

    @abstractmethod
    def derivative(self, t: Timestep) -> torch.Tensor:
        pass


class LinearSigmaSchedule(SigmaSchedule):
    def __call__(self, t: Timestep):
        return t.adapt(self.timestep_config).steps

    def derivative(self, t: Timestep):
        return torch.tensor(1.0, device=t.steps.device)


class CosineSigmaSchedule(SigmaSchedule):
    def __call__(self, t: Timestep):
        adapted_t = t.adapt(self.timestep_config)
        return torch.sin(adapted_t.steps * (torch.pi / 2))

    def derivative(self, t: Timestep):
        adapted_t = t.adapt(self.timestep_config)
        return torch.pi / 2 * torch.cos(adapted_t.steps * (torch.pi / 2))


class LambdaSchedule:
    alpha: AlphaSchedule
    sigma: SigmaSchedule

    def __init__(self, alpha_schedule: AlphaSchedule, sigma_schedule: SigmaSchedule):
        self.alpha = alpha_schedule
        self.sigma = sigma_schedule

    def __call__(self, t: Timestep) -> torch.Tensor:
        alpha = self.alpha(t)
        sigma = self.sigma(t)
        return torch.log((alpha / sigma) ** 2)

    def derivative(self, t: Timestep) -> torch.Tensor:
        alpha = self.alpha(t)
        sigma = self.sigma(t)
        d_alpha = self.alpha.derivative(t)
        d_sigma = self.sigma.derivative(t)
        return 2 * (d_alpha / alpha - d_sigma / sigma)


class ScheduleGroup:
    alpha: AlphaSchedule
    sigma: SigmaSchedule
    lambda_: LambdaSchedule

    def __init__(self, alpha_schedule: AlphaSchedule, sigma_schedule: SigmaSchedule):
        self.alpha = alpha_schedule
        self.sigma = sigma_schedule
        self.lambda_ = LambdaSchedule(alpha_schedule, sigma_schedule)
