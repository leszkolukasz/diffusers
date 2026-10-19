import math

import torch

from src.schedule.sampling.base import SamplingSchedule
from src.timestep import Timestep, TimestepConfig


# KL-optimal schedule from Theorem 3.1 of Align Your Steps
class GaussKLOptimalSamplingSchedule(SamplingSchedule):
    def __init__(
        self,
        *,
        c: float = 0.5,
        T: float,
        max_t: float = 80.0,
        min_t: float = 0.0,
    ):
        c = float(c)
        max_t = float(max_t)
        min_t = float(min_t)

        super().__init__(T=T, max_t=max_t)
        self.c = c
        self.min_t = min_t

    def get_timesteps(self, n_steps: int, **kwargs) -> Timestep:
        alpha_min = math.atan(self.min_t / self.c)
        alpha_max = math.atan(self.max_t / self.c)

        alphas = torch.linspace(alpha_min, alpha_max, n_steps + 1)
        steps = self.c * torch.tan(alphas)
        # tan(atan(x)) is not exactly x in floating point.
        steps[0] = self.min_t
        steps[-1] = self.max_t
        steps = torch.flip(steps, dims=[0])

        return Timestep(TimestepConfig(kind="continuous", T=self.T), steps)


# KLUB-optimal schedule from Appendix A.3 of Align Your Steps
class GaussKLUBOptimalSamplingSchedule(SamplingSchedule):
    def __init__(
        self,
        *,
        c: float = 0.5,
        T: float,
        max_t: float = 80.0,
        min_t: float = 1e-3,
    ):
        c = float(c)
        max_t = float(max_t)
        min_t = float(min_t)

        super().__init__(T=T, max_t=max_t)
        self.c = c
        self.min_t = min_t

    def get_timesteps(self, n_steps: int, **kwargs) -> Timestep:
        v_min = math.log1p(self.c**2 / self.min_t**2)
        v_max = math.log1p(self.c**2 / self.max_t**2)

        fractions = torch.linspace(0.0, 1.0, n_steps)
        values = v_min + fractions * (v_max - v_min)
        steps = self.c / torch.sqrt(torch.expm1(values))
        steps[0] = self.min_t
        steps[-1] = self.max_t
        steps = torch.flip(steps, dims=[0])
        steps = torch.cat([steps, steps.new_zeros(1)])

        return Timestep(TimestepConfig(kind="continuous", T=self.T), steps)
