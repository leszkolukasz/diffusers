from abc import abstractmethod

import torch

from src.equation import Equation
from src.model import PredictionTarget
from src.timestep import Timestep


class DifferentialEquation(Equation):
    @abstractmethod
    def drift(self, x_t: torch.Tensor, t: Timestep) -> torch.Tensor:
        pass

    @abstractmethod
    def diffusion_coeff(self, x_t: torch.Tensor, t: Timestep) -> torch.Tensor:
        pass


# From: Sampler Performance in Guided Diffusion Models
class GeneralizedDifferential(DifferentialEquation):
    def drift(self, x_t: torch.Tensor, t: Timestep) -> torch.Tensor:
        alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
        d_alpha_t = self.schedules.alpha.derivative(t).view(-1, 1, 1, 1)
        sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
        d_lambda_t = self.schedules.lambda_.derivative(t).view(-1, 1, 1, 1)

        eta_t_inf = self.schedules.eta(t).view(-1, 1, 1, 1)

        assert self.model.target == PredictionTarget.Noise
        noise_pred = self.model(x_t, timestep=t)
        drift = (
            d_alpha_t / alpha_t * x_t
            + sigma_t * (eta_t_inf**2 - d_lambda_t) / 2 * noise_pred
        )

        return drift

    def diffusion_coeff(self, x_t: torch.Tensor, t: Timestep) -> torch.Tensor:
        sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
        eta_t_inf = self.schedules.eta(t).view(-1, 1, 1, 1)

        return sigma_t * eta_t_inf
