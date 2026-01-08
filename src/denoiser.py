import math
from abc import ABC, abstractmethod

import torch
from loguru import logger

from src.model import ErrorPredictor, ModuleMetadata
from src.schedule import ScheduleGroup
from src.timestep import Timestep


class Denoiser(ABC):
    model: ErrorPredictor
    schedules: ScheduleGroup

    def __init__(
        self,
        *,
        model: ErrorPredictor,
        schedules: ScheduleGroup,
    ):
        self.model = model
        self.schedules = schedules

        alpha_schedule_meta = model.metadata.get(ModuleMetadata.AlphaSchedule, None)
        sigma_schedule_meta = model.metadata.get(ModuleMetadata.SigmaSchedule, None)

        if (
            alpha_schedule_meta is not None
            and schedules.alpha.__class__.__name__ != alpha_schedule_meta
        ):
            logger.warning(
                f"Denoiser model was trained with alpha schedule '{alpha_schedule_meta}', but current schedule is '{schedules.alpha.__class__.__name__}'"
            )

        if (
            sigma_schedule_meta is not None
            and schedules.sigma.__class__.__name__ != sigma_schedule_meta
        ):
            logger.warning(
                f"Denoiser model was trained with sigma schedule '{sigma_schedule_meta}', but current schedule is '{schedules.sigma.__class__.__name__}'"
            )

    # Assumes t_prev < t
    @torch.no_grad()
    def denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep) -> torch.Tensor:
        return self._denoise(x_t, t, t_prev)

    @abstractmethod
    def _denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep):
        pass


class DiscreteDenoiser(Denoiser):
    def _denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep):
        max_t = int(self.model.timestep_config.max_t)
        t = t.as_discrete(max_t)
        t_prev = t_prev.as_discrete(max_t)

        alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
        sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
        alpha_t_prev = self.schedules.alpha(t_prev).view(-1, 1, 1, 1)
        sigma_t_prev = self.schedules.sigma(t_prev).view(-1, 1, 1, 1)

        # Assumes eta = 1

        err_pred = self.model(x_t, timestep=t)
        mean = (
            alpha_t_prev / alpha_t
        ) * x_t - sigma_t * alpha_t_prev / alpha_t * err_pred

        noise = torch.randn_like(x_t) * sigma_t_prev

        return mean + noise


class ContinuousDenoiser(Denoiser):
    @torch.no_grad()
    def denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep) -> torch.Tensor:
        return self._denoise(x_t, t.as_continuous(), t_prev.as_continuous())


class EulerODEDenoiser(ContinuousDenoiser):
    def _denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep):
        alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
        d_alpha_t = self.schedules.alpha.derivative(t).view(-1, 1, 1, 1)
        sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
        d_lambda_t = self.schedules.lambda_.derivative(t).view(-1, 1, 1, 1)

        h = t_prev.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)

        err_pred = self.model(x_t, timestep=t)
        f = d_alpha_t / alpha_t * x_t - d_lambda_t * sigma_t / 2 * err_pred
        x_t_prev = x_t + h * f

        return x_t_prev


class HeunODEDenoiser(ContinuousDenoiser):
    def _denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep):
        h = t_prev.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)

        def f(x_t: torch.Tensor, t: Timestep):
            alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
            d_alpha_t = self.schedules.alpha.derivative(t).view(-1, 1, 1, 1)
            sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
            d_lambda_t = self.schedules.lambda_.derivative(t).view(-1, 1, 1, 1)

            err_pred = self.model(x_t, timestep=t)
            return d_alpha_t / alpha_t * x_t - d_lambda_t * sigma_t / 2 * err_pred

        k1 = f(x_t, t)
        k2 = f(x_t + h * k1, t_prev)
        x_t_prev = x_t + h / 2 * (k1 + k2)

        return x_t_prev


class EulerMaruyamaSDEDenoiser(ContinuousDenoiser):
    def _denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep):
        alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
        d_alpha_t = self.schedules.alpha.derivative(t).view(-1, 1, 1, 1)
        sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
        d_lambda_t = self.schedules.lambda_.derivative(t).view(-1, 1, 1, 1)

        h = t_prev.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)
        eta_t_inf = 1.0

        err_pred = self.model(x_t, timestep=t)
        drift = (
            d_alpha_t / alpha_t * x_t
            + sigma_t * (eta_t_inf**2 - d_lambda_t) / 2 * err_pred
        )
        diffusion = sigma_t * eta_t_inf * math.sqrt(-h) * torch.randn_like(x_t)

        x_t_prev = x_t + h * drift + diffusion

        return x_t_prev


class HeunSDEDenoiser(ContinuousDenoiser):
    def _denoise(self, x_t: torch.Tensor, t: Timestep, t_prev: Timestep):
        h = t_prev.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)
        eta_t_inf = 1.0

        def drift(x_t: torch.Tensor, t: Timestep):
            alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
            d_alpha_t = self.schedules.alpha.derivative(t).view(-1, 1, 1, 1)
            sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
            d_lambda_t = self.schedules.lambda_.derivative(t).view(-1, 1, 1, 1)

            err_pred = self.model(x_t, timestep=t)
            return (
                d_alpha_t / alpha_t * x_t
                + sigma_t * (eta_t_inf**2 - d_lambda_t) / 2 * err_pred
            )

        def diffusion(t: Timestep):
            sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
            return sigma_t * eta_t_inf * math.sqrt(-h)

        k1 = drift(x_t, t)
        k2 = drift(x_t + h * k1, t_prev)
        dr = (k1 + k2) / 2

        dif = (diffusion(t) + diffusion(t_prev)) / 2 * torch.randn_like(x_t)
        # dif = diffusion(t) * torch.randn_like(x_t)

        x_t_prev = x_t + h * dr + dif

        return x_t_prev
