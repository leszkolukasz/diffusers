from abc import abstractmethod

import torch

from src.equation import Equation
from src.model import PredictionTarget, reparametrize
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

        pred = self.model(x_t, timestep=t, schedules=self.schedules)
        noise_pred = reparametrize(
            pred=pred,
            x_t=x_t,
            t=t,
            schedules=self.schedules,
            src=self.model.target,
            dest=PredictionTarget.Noise,
        )

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

        pred = self.model(x_t, timestep=t, schedules=self.schedules)
        x0_pred = reparametrize(
            pred=pred,
            x_t=x_t,
            t=t,
            schedules=self.schedules,
            src=self.model.target,
            dest=PredictionTarget.x0,
        )

        # Note: Align Your Steps has wrong formula here, with additional multiplication by s_t.
        # This stems from incorrect calculation of score(x_t / s(t)).
        # Though, it does not matter as we use s(t) = 1.
        return (
            d_edm_sigma_t / edm_sigma_t + d_s_t / s_t
        ) * x_t - d_edm_sigma_t * s_t / edm_sigma_t * x0_pred

    def diffusion_coeff(self, x_t: torch.Tensor, t: Timestep) -> torch.Tensor:
        raise NotImplementedError("Probability flow is not stochastic")


class ScoreSDE(DifferentialEquation):
    def drift(self, x_t: torch.Tensor, t: Timestep) -> torch.Tensor:
        edm_sigma_t = self.schedules.edm_sigma(t).view(-1, 1, 1, 1)
        d_edm_sigma_t = self.schedules.edm_sigma.derivative(t).view(-1, 1, 1, 1)

        s_t = self.schedules.s(t).view(-1, 1, 1, 1)
        d_s_t = self.schedules.s.derivative(t).view(-1, 1, 1, 1)

        pred = self.model(x_t, timestep=t, schedules=self.schedules)
        x0_pred = reparametrize(
            pred=pred,
            x_t=x_t,
            t=t,
            schedules=self.schedules,
            src=self.model.target,
            dest=PredictionTarget.x0,
        )

        return (
            2 * d_edm_sigma_t / edm_sigma_t + d_s_t / s_t
        ) * x_t - 2 * d_edm_sigma_t * s_t / edm_sigma_t * x0_pred

    def diffusion_coeff(self, x_t: torch.Tensor, t: Timestep) -> torch.Tensor:
        edm_sigma_t = self.schedules.edm_sigma(t).view(-1, 1, 1, 1)
        d_edm_sigma_t = self.schedules.edm_sigma.derivative(t).view(-1, 1, 1, 1)

        s_t = self.schedules.s(t).view(-1, 1, 1, 1)

        return s_t * torch.sqrt(2 * d_edm_sigma_t * edm_sigma_t)
