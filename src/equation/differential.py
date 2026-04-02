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
        noise_pred = self.model(x_t, timestep=t, schedules=self.schedules)
        drift = (
            d_alpha_t / alpha_t * x_t
            + sigma_t * (eta_t_inf**2 - d_lambda_t) / 2 * noise_pred
        )

        return drift

    def diffusion_coeff(self, x_t: torch.Tensor, t: Timestep) -> torch.Tensor:
        sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
        eta_t_inf = self.schedules.eta(t).view(-1, 1, 1, 1)

        return sigma_t * eta_t_inf


class ProbabilityFlow(GeneralizedDifferential):
    def drift(self, x_t: torch.Tensor, t: Timestep) -> torch.Tensor:
        edm_sigma_t = self.schedules.edm_sigma(t).view(-1, 1, 1, 1)
        d_edm_sigma_t = self.schedules.edm_sigma.derivative(t).view(-1, 1, 1, 1)

        s_t = self.schedules.s(t).view(-1, 1, 1, 1)
        d_s_t = self.schedules.s.derivative(t).view(-1, 1, 1, 1)

        assert self.model.target == PredictionTarget.x0
        pred = self.model(x_t, timestep=t, schedules=self.schedules)

        return (
            d_edm_sigma_t / edm_sigma_t + d_s_t / s_t
        ) * x_t - d_edm_sigma_t * s_t / edm_sigma_t * pred

    def diffusion_coeff(self, x_t: torch.Tensor, t: Timestep) -> torch.Tensor:
        raise NotImplementedError("Probability flow is not stochastic")
