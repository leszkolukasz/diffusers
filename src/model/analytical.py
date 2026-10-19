import torch

from src.model.base import PredictionTarget, Predictor
from src.schedule import ScheduleGroup
from src.timestep import Timestep, TimestepConfig


# Ideal x0-denoiser for p_data = N(0, c^2 I)
class AnalyticalGaussianPredictor(Predictor):
    def __init__(
        self,
        *,
        c: float = 0.5,
        n_channels: int = 1,
        img_width: int = 32,
        img_height: int = 32,
        T: float = 1000.0,
    ):
        super().__init__()

        c = float(c)

        self.c = c
        self.n_channels = n_channels
        self.img_width = img_width
        self.img_height = img_height
        self.target = PredictionTarget.x0
        self.normalize_x = True
        self.timestep_config = TimestepConfig(kind="continuous", T=T)
        self.file_name = f"analytical_gaussian_c{c:g}.pth"

    def _forward(
        self,
        x: torch.Tensor,
        timestep: Timestep,
        schedules: ScheduleGroup | None = None,
    ) -> torch.Tensor:
        t = timestep.adapt(self.timestep_config).steps
        t = t.view(-1, *([1] * (x.dim() - 1)))
        c_sq = self.c**2
        return x * (c_sq / (c_sq + t**2))
