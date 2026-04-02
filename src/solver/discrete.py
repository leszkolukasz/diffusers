import torch

from src.common import assert_type
from src.equation import DiscreteEquation
from src.solver import Solver
from src.timestep import Timestep


class DiscreteSolver(Solver):
    equation: DiscreteEquation

    def _step(self, x_t: torch.Tensor, t: Timestep, s: Timestep):
        T = assert_type(self.T, int)
        t = t.as_discrete(T)
        s = s.as_discrete(T)

        mean = self.equation.mean(x_t, t, s)
        std = self.equation.std(x_t, t, s)
        noise = torch.randn_like(x_t) * std

        return mean + noise
