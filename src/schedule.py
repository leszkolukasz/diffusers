from abc import ABC, abstractmethod
from typing import cast

import torch
from diffusers import DDPMScheduler

from src.common import assert_type
from src.timestep import Timestep, TimestepConfig


class HuggingFaceDDPMBaseSchedule(ABC):
    def __init__(self, model_id: str):
        config = cast(dict, DDPMScheduler.load_config(model_id))  # type: ignore[possibly-missing-attribute]

        self.max_t = assert_type(config.get("num_train_timesteps"), int)
        beta_start = assert_type(config.get("beta_start"), float)
        beta_end = assert_type(config.get("beta_end"), float)
        schedule_type = assert_type(config.get("beta_schedule"), str)

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, self.max_t)
        elif schedule_type == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, self.max_t) ** 2
        else:
            raise NotImplementedError(f"Unknown schedule: {schedule_type}")

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def get_alpha_at_t(self, timestep: Timestep) -> torch.Tensor:
        t_adapted = timestep.adapt(
            TimestepConfig(kind="discrete", max_t=self.max_t - 1)
        )
        t_indices = t_adapted.steps.long()
        self.alphas_cumprod = self.alphas_cumprod.to(t_indices.device)
        return torch.sqrt(self.alphas_cumprod[t_indices])

    def get_sigma_at_t(self, timestep: Timestep) -> torch.Tensor:
        t_adapted = timestep.adapt(
            TimestepConfig(kind="discrete", max_t=self.max_t - 1)
        )
        t_indices = t_adapted.steps.long()
        self.alphas_cumprod = self.alphas_cumprod.to(t_indices.device)
        return torch.sqrt(1.0 - self.alphas_cumprod[t_indices])


class AlphaSchedule(ABC):
    timestep_config: TimestepConfig = TimestepConfig(kind="continuous", max_t=1.0)

    @abstractmethod
    def __call__(self, t: Timestep) -> torch.Tensor:
        pass

    def derivative(self, t: Timestep) -> torch.Tensor:
        raise NotImplementedError("Discrete only schedule does not support derivative.")


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


class SigmaSchedule(ABC):
    timestep_config: TimestepConfig = TimestepConfig(kind="continuous", max_t=1.0)

    @abstractmethod
    def __call__(self, t: Timestep) -> torch.Tensor:
        pass

    def derivative(self, t: Timestep) -> torch.Tensor:
        raise NotImplementedError("Discrete only schedule does not support derivative.")


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


class HuggingFaceDDPMSigmaSchedule(SigmaSchedule, HuggingFaceDDPMBaseSchedule):
    def __call__(self, t: Timestep) -> torch.Tensor:
        t_adapted = t.adapt(self.timestep_config)
        return self.get_sigma_at_t(t_adapted)


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


class AYSSigmaSchedule:
    alpha: AlphaSchedule
    sigma: SigmaSchedule

    def __init__(self, alpha_schedule: AlphaSchedule, sigma_schedule: SigmaSchedule):
        self.alpha = alpha_schedule
        self.sigma = sigma_schedule

    def __call__(self, t: Timestep) -> torch.Tensor:
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        return sigma_t / alpha_t

    def derivative(self, t: Timestep) -> torch.Tensor:
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        d_alpha = self.alpha.derivative(t)
        d_sigma = self.sigma.derivative(t)
        return (d_sigma * alpha_t - sigma_t * d_alpha) / (alpha_t**2)


class ScheduleGroup:
    alpha: AlphaSchedule
    sigma: SigmaSchedule
    lambda_: LambdaSchedule
    ays_sigma: AYSSigmaSchedule

    def __init__(self, alpha_schedule: AlphaSchedule, sigma_schedule: SigmaSchedule):
        self.alpha = alpha_schedule
        self.sigma = sigma_schedule
        self.lambda_ = LambdaSchedule(alpha_schedule, sigma_schedule)
        self.ays_sigma = AYSSigmaSchedule(alpha_schedule, sigma_schedule)
