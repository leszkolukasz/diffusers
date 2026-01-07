from abc import ABC, abstractmethod
import math
import torch

from src.schedule import SchedulePack


class Denoiser(ABC):
    model: torch.nn.Module
    schedule_pack: SchedulePack
    max_t: int

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        max_t: int,
        schedules: SchedulePack,
    ):
        self.model = model
        self.schedules = schedules
        self.max_t = max_t

    @torch.no_grad()
    def denoise(self, x_t: torch.Tensor, t: torch.Tensor):
        return self._denoise(x_t, t)

    @abstractmethod
    def _denoise(self, x_t: torch.Tensor, t: torch.Tensor):
        pass

class DiscreteDenoiser(Denoiser):
    def _denoise(self, x_t: torch.Tensor, t: torch.Tensor):
        assert torch.all((t > 0) & (t < self.max_t))

        alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
        sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
        alpha_t_minus_one = self.schedules.alpha(t-1).view(-1, 1, 1, 1)
        sigma_t_minus_one = self.schedules.sigma(t-1).view(-1, 1, 1, 1)

        # Assumes eta = 1

        err_pred = self.model(x_t, timestep=t)
        mean = (alpha_t_minus_one / alpha_t) * x_t - sigma_t * alpha_t_minus_one / alpha_t * err_pred

        noise = torch.randn_like(x_t) * sigma_t_minus_one

        return mean + noise

class EulerODEDenoiser(Denoiser):
    def _denoise(self, x_t: torch.Tensor, t: torch.Tensor):
        assert torch.all((t > 0) & (t < self.max_t))

        alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
        d_alpha_t = self.schedules.alpha.derivative(t).view(-1, 1, 1, 1)
        sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
        d_lambda_t = self.schedules.lambda_.derivative(t).view(-1, 1, 1, 1)

        h = 1 / self.max_t

        err_pred = self.model(x_t, timestep=t)
        f = d_alpha_t / alpha_t * x_t - d_lambda_t * sigma_t / 2 * err_pred
        x_t_minus_one = x_t - h * f

        return x_t_minus_one
    
class HeunODEDenoiser(Denoiser):
    def _denoise(self, x_t: torch.Tensor, t: torch.Tensor):
        assert torch.all((t > 0) & (t < self.max_t))

        h = 1 / self.max_t

        def f(x_t: torch.Tensor, t: torch.Tensor):
            alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
            d_alpha_t = self.schedules.alpha.derivative(t).view(-1, 1, 1, 1)
            sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
            d_lambda_t = self.schedules.lambda_.derivative(t).view(-1, 1, 1, 1)

            err_pred = self.model(x_t, timestep=t)
            return d_alpha_t / alpha_t * x_t - d_lambda_t * sigma_t / 2 * err_pred
        
        k1 = f(x_t, t)
        k2 = f(x_t - h * k1, t - 1)
        x_t_minus_one = x_t - h / 2 * (k1 + k2)

        return x_t_minus_one
    

class EulerMaruyamaSDEDenoiser(Denoiser):
    def _denoise(self, x_t: torch.Tensor, t: torch.Tensor):
        assert torch.all((t > 0) & (t < self.max_t))

        alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
        d_alpha_t = self.schedules.alpha.derivative(t).view(-1, 1, 1, 1)
        sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
        d_lambda_t = self.schedules.lambda_.derivative(t).view(-1, 1, 1, 1)

        h = 1 / self.max_t
        eta_t_inf = 1.0

        err_pred = self.model(x_t, timestep=t)
        drift = d_alpha_t / alpha_t * x_t + sigma_t * (eta_t_inf**2 - d_lambda_t) / 2 * err_pred
        diffusion = sigma_t * eta_t_inf * math.sqrt(h) * torch.randn_like(x_t)

        x_t_minus_one = x_t - h * drift + diffusion

        return x_t_minus_one
    

class HeunSDEDenoiser(Denoiser):
    def _denoise(self, x_t: torch.Tensor, t: torch.Tensor):
        assert torch.all((t > 0) & (t < self.max_t))

        h = 1 / self.max_t
        eta_t_inf = 1.0

        def drift(x_t: torch.Tensor, t: torch.Tensor):
            alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
            d_alpha_t = self.schedules.alpha.derivative(t).view(-1, 1, 1, 1)
            sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
            d_lambda_t = self.schedules.lambda_.derivative(t).view(-1, 1, 1, 1)

            err_pred = self.model(x_t, timestep=t)
            return d_alpha_t / alpha_t * x_t + sigma_t * (eta_t_inf**2 - d_lambda_t) / 2 * err_pred
        
        def diffusion(t: torch.Tensor):
            sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
            return sigma_t * eta_t_inf * math.sqrt(h)

        k1 = drift(x_t, t)
        k2 = drift(x_t - h * k1, t - 1)
        dr = (k1 + k2) / 2

        dif = (diffusion(t) + diffusion(t - 1)) / 2 * torch.randn_like(x_t)
        # dif = diffusion(t) * torch.randn_like(x_t)

        x_t_minus_one = x_t - h * dr + dif

        return x_t_minus_one