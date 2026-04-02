from abc import abstractmethod

import torch

from src.equation import Equation
from src.model import PredictionTarget
from src.timestep import Timestep


class DiscreteEquation(Equation):
    @abstractmethod
    def mean(self, x_t: torch.Tensor, t: Timestep, s: Timestep) -> torch.Tensor:
        pass

    @abstractmethod
    def std(self, x_t: torch.Tensor, t: Timestep, s: Timestep) -> torch.Tensor:
        pass


# From: Sampler Performance in Guided Diffusion Models
class GeneralizedDiscrete(DiscreteEquation):
    def mean(self, x_t: torch.Tensor, t: Timestep, s: Timestep) -> torch.Tensor:
        alpha_t = self.schedules.alpha(t).view(-1, 1, 1, 1)
        sigma_t = self.schedules.sigma(t).view(-1, 1, 1, 1)
        alpha_s = self.schedules.alpha(s).view(-1, 1, 1, 1)
        sigma_s = self.schedules.sigma(s).view(-1, 1, 1, 1)
        eta_s = self.schedules.eta(t, s).view(-1, 1, 1, 1)

        assert self.model.target == PredictionTarget.Noise
        noise_pred = self.model(x_t, timestep=t, schedules=self.schedules)

        mean = (alpha_s / alpha_t) * x_t + (
            sigma_s * torch.sqrt(1 - eta_s**2) - sigma_t * alpha_s / alpha_t
        ) * noise_pred

        return mean

    def std(self, x_t: torch.Tensor, t: Timestep, s: Timestep) -> torch.Tensor:
        sigma_s = self.schedules.sigma(s).view(-1, 1, 1, 1)
        eta_s = self.schedules.eta(t, s).view(-1, 1, 1, 1)

        return sigma_s * eta_s
