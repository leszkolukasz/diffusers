import torch

from src.common import assert_type
from src.equation import DifferentialEquation
from src.model import PredictionTarget, reparametrize
from src.solver import Solver
from src.timestep import Timestep

EPSILON = 1e-9


class ContinuousSolver(Solver):
    equation: DifferentialEquation

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: Timestep, s: Timestep) -> torch.Tensor:
        T = assert_type(self.T, float)
        return self._step(x_t, t.as_continuous(T), s.as_continuous(T))


class EulerODESolver(ContinuousSolver):
    def _step(self, x_t: torch.Tensor, t: Timestep, s: Timestep):
        h = s.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)

        drift = self.equation.drift(x_t, t)
        x_s = x_t + h * drift

        return x_s


class HeunODESolver(ContinuousSolver):
    def _step(self, x_t: torch.Tensor, t: Timestep, s: Timestep):
        h = s.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)

        k1 = self.equation.drift(x_t, t)
        x_s_euler = x_t + h * k1

        if s.steps.max() > 0.0:
            k2 = self.equation.drift(x_s_euler, s)
            x_s = x_t + h / 2 * (k1 + k2)
        else:
            x_s = x_s_euler

        return x_s


class EulerMaruyamaSDESolver(ContinuousSolver):
    def _step(self, x_t: torch.Tensor, t: Timestep, s: Timestep):
        h = s.steps.view(-1, 1, 1, 1) - t.steps.view(-1, 1, 1, 1)

        drift = self.equation.drift(x_t, t)

        diff_coeff = self.equation.diffusion_coeff(x_t, t)
        diffusion = diff_coeff * torch.sqrt(-h) * torch.randn_like(x_t)

        x_s = x_t + h * drift + diffusion

        return x_s


# I still have no idea why it is called Stochastic-DDIM solver
# Note: This solves SDE for s(t) = 1, sigma_EDM(t) = t analitically,
# so it does not really need Equation class.
class StochasticDDIMSolver(ContinuousSolver):
    def _step(self, x_t: torch.Tensor, t: Timestep, s: Timestep):
        schedules = self.equation.schedules
        sigma_t = schedules.edm_sigma(t).view(-1, 1, 1, 1)
        sigma_s = schedules.edm_sigma(s).view(-1, 1, 1, 1)
        ratio = (sigma_s / sigma_t.clamp_min(EPSILON)) ** 2

        pred = self.equation.model(x_t, timestep=t, schedules=schedules)
        x0 = reparametrize(
            pred=pred,
            x_t=x_t,
            t=t,
            schedules=schedules,
            src=self.equation.model.target,
            dest=PredictionTarget.x0,
        )

        diff_coeff = sigma_s * torch.sqrt(1 - ratio)
        diffusion = diff_coeff * torch.randn_like(x_t)

        return ratio * x_t + (1 - ratio) * x0 + diffusion
