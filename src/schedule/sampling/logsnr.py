import torch

from src.schedule.misc import ScheduleGroup
from src.schedule.sampling.base import SamplingSchedule
from src.timestep import Timestep, TimestepConfig

SEARCH_ITERATIONS = 64


class LinearLogSNRSamplingSchedule(SamplingSchedule):
    def __init__(
        self,
        *,
        schedules: ScheduleGroup,
        min_t: float = 1e-3,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.schedules = schedules
        self.min_t = min_t
        self.timestep_config = TimestepConfig(kind="continuous", T=self.T)

    def get_timesteps(self, n_steps: int, **kwargs) -> Timestep:
        lambda_noisy = self._logsnr(self.max_t)
        lambda_clean = self._logsnr(self.min_t)

        lambdas = torch.linspace(lambda_noisy, lambda_clean, n_steps)
        steps = self._invert_logsnr(lambdas)
        steps = torch.cat([steps, torch.tensor([0.0])])

        return Timestep(self.timestep_config, steps)

    def _logsnr(self, t: float) -> float:
        timestep = Timestep(self.timestep_config, torch.tensor([t]))
        return self.schedules.lambda_(timestep).item()

    def _invert_logsnr(self, lambdas: torch.Tensor) -> torch.Tensor:
        lower = torch.full_like(lambdas, self.min_t)
        upper = torch.full_like(lambdas, self.max_t)

        for _ in range(SEARCH_ITERATIONS):
            mid = (lower + upper) / 2
            mid_lambda = self.schedules.lambda_(Timestep(self.timestep_config, mid))

            too_clean = mid_lambda > lambdas
            lower = torch.where(too_clean, mid, lower)
            upper = torch.where(too_clean, upper, mid)

        return (lower + upper) / 2
