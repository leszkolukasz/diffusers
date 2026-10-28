from abc import ABC, abstractmethod

import torch

from src.timestep import Timestep, TimestepConfig

MAX_T_COEFF = 0.95


class SamplingSchedule(ABC):
    max_t: float
    T: float

    def __init__(self, *, T: float, max_t: float | None = None):
        self.max_t = max_t if max_t is not None else MAX_T_COEFF * T
        self.T = T

    @abstractmethod
    def get_timesteps(self, n_steps: int, **kwargs) -> Timestep:
        pass


class LinearSamplingSchedule(SamplingSchedule):
    def get_timesteps(self, n_steps: int, **kwargs) -> Timestep:
        steps = torch.linspace(self.max_t, 0.0, n_steps + 1)
        return Timestep(TimestepConfig(kind="continuous", T=self.T), steps)


class QuadraticSamplingSchedule(SamplingSchedule):
    def get_timesteps(self, n_steps: int, **kwargs) -> Timestep:
        steps = self.max_t * torch.linspace(1.0, 0.0, n_steps + 1).pow(2)
        return Timestep(TimestepConfig(kind="continuous", T=self.T), steps)
