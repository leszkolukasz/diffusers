from abc import ABC, abstractmethod

import torch

from src.equation import Equation
from src.timestep import Timestep


class Solver(ABC):
    equation: Equation
    T: int | float

    def __init__(
        self,
        equation: Equation,
        T: int | float,
    ):
        self.equation = equation
        self.T = T

    # Assumes s < t
    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: Timestep, s: Timestep) -> torch.Tensor:
        return self._step(x_t, t, s)

    @abstractmethod
    def _step(self, x_t: torch.Tensor, t: Timestep, s: Timestep):
        pass
