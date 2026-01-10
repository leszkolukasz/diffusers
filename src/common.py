import torch
from torch import Tensor

from src.schedule import ScheduleGroup
from src.timestep import Timestep


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffusionMixin:
    def diffuse(
        self, x0: Tensor, t: Timestep, schedules: ScheduleGroup
    ) -> tuple[Tensor, Tensor]:
        noise = torch.randn_like(x0)
        alpha = schedules.alpha(t).view(-1, 1, 1, 1)
        sigma = schedules.sigma(t).view(-1, 1, 1, 1)

        return alpha * x0 + sigma * noise, noise

    # Samples from p(x_t | x__{t_prev}) where t > t_prev
    def diffuse_from(
        self,
        x_0: Tensor,
        x_prev: Tensor,
        t_prev: Timestep,
        t: Timestep,
        schedules: ScheduleGroup,
    ) -> Tensor:
        noise = torch.rand_like(x_0)
        alpha_t = schedules.alpha(t).view(-1, 1, 1, 1)
        sigma_t = schedules.sigma(t).view(-1, 1, 1, 1)
        alpha_t_prev = schedules.alpha(t_prev).view(-1, 1, 1, 1)
        sigma_t_prev = schedules.sigma(t_prev).view(-1, 1, 1, 1)

        return (
            x_prev
            + (alpha_t - alpha_t_prev) * x_0
            + torch.sqrt(sigma_t**2 - sigma_t_prev**2) * noise
        )
