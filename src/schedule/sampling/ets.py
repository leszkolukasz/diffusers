import os
from dataclasses import dataclass
from typing import Generator

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import get_device
from src.diffusion import diffuse
from src.model import VAE, Predictor
from src.schedule import ScheduleGroup
from src.schedule.sampling import SamplingSchedule
from src.timestep import Timestep, TimestepConfig

@dataclass
class ETSConfig:
    n_grid_steps: int = 200
    n_monte_carlo_iter: int = 256
    device: torch.device = get_device()
    save_file: str = "generated/ets_timesteps.pth"


class ETSSamplingSchedule(SamplingSchedule):
    model: Predictor
    schedules: ScheduleGroup
    dataloader: DataLoader
    timestep_config: TimestepConfig
    vae: VAE | None

    def __init__(
        self,
        *,
        max_t: float = 0.95,
        model: Predictor,
        schedules: ScheduleGroup,
        dataloader: DataLoader,
        solver_T: float,
        vae: VAE | None = None,
        config: ETSConfig = ETSConfig(),
    ):
        super().__init__(max_t=max_t, T=solver_T)
        self.model = model
        self.schedules = schedules
        self.dataloader = dataloader
        self.vae = vae
        self.config = config
        self.timestep_config = TimestepConfig(kind="continuous", T=solver_T)

    def get_timesteps(self, n_steps: int, **kwargs) -> Timestep:
        if os.path.exists(self.config.save_file):
            state = torch.load(self.config.save_file, weights_only=False)
            steps = state["steps"].to(self.config.device)
            return Timestep(self.timestep_config, steps).reverse()

        t_grid, R = self._estimate_rescaled_entropy()
        steps = self._invert_entropy(t_grid, R, n_steps)

        os.makedirs(os.path.dirname(self.config.save_file), exist_ok=True)
        torch.save({"steps": steps.cpu()}, self.config.save_file)

        return Timestep(self.timestep_config, steps).reverse()

    def _estimate_rescaled_entropy(self) -> tuple[torch.Tensor, torch.Tensor]:
        t_grid = torch.linspace(
            0.0, self.max_t, self.config.n_grid_steps + 1, device=self.config.device
        )
        R = torch.zeros(self.config.n_grid_steps + 1, device=self.config.device)

        pbar = tqdm(range(self.config.n_grid_steps), desc="ETS: Estimating rescaled entropy, timestep:")
        for i in pbar:
            t_i = t_grid[i].item()
            dt = t_grid[i + 1].item() - t_i

            eps_sq = self._estimate_denoising_error(t_i)

            t_tensor = torch.tensor([t_i], device=self.config.device)
            timestep = Timestep(self.timestep_config, t_tensor)
            sigma_dot = self.schedules.edm_sigma.derivative(timestep).item()
            sigma = self.schedules.edm_sigma(timestep).item()

            R[i + 1] = R[i] + (sigma_dot / sigma**2) * dt * eps_sq

        return t_grid, R

    def _estimate_denoising_error(self, t: float) -> float:
        eps_sq_sum = 0.0
        count = 0

        for X in self._get_data_samples():
            timestep = Timestep(
                self.timestep_config,
                torch.full((X.size(0),), t, device=X.device),
            )
            X_t, _ = diffuse(X, timestep, self.schedules)
            x_hat_0 = self.model(X_t, timestep=timestep, schedules=self.schedules)

            eps_sq_sum += (x_hat_0 - X).view(X.size(0), -1).norm(dim=1).pow(2).sum().item()
            count += X.size(0)

        return eps_sq_sum / count

    def _invert_entropy(self, t_grid: torch.Tensor, R: torch.Tensor, n_steps: int) -> torch.Tensor:
        target_levels = np.linspace(0.0, R[-1].item(), n_steps + 1)
        steps_np = np.interp(target_levels, R.cpu().numpy(), t_grid.cpu().numpy())
        return torch.tensor(steps_np, dtype=torch.float32, device=self.config.device)

    def _get_data_samples(self) -> Generator[torch.Tensor, None, None]:
        count = 0
        while count < self.config.n_monte_carlo_iter:
            for X, _ in self.dataloader:
                X = X.to(self.config.device)
                yield self.vae.encode(X.half()).float() if self.vae is not None else X
                count += X.size(0)
                if count >= self.config.n_monte_carlo_iter:
                    break
