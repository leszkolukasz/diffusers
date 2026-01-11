import torch

from src.schedule.alpha import AlphaSchedule
from src.schedule.sigma import SigmaSchedule
from src.timestep import Timestep


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
