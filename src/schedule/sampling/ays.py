import math
from dataclasses import dataclass
from typing import Generator

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import get_device
from src.config import EquationType
from src.diffusion import diffuse, diffuse_from
from src.model import VAE, PredictionTarget, Predictor
from src.schedule import ScheduleGroup
from src.schedule.sampling import EPSILON, SamplingSchedule
from src.timestep import Timestep, TimestepConfig


@dataclass
class AYSConfig:
    max_iter: int = 300
    max_finetune_iter: int = 10
    device: torch.device = get_device()
    n_candidates: int = 11
    n_monte_carlo_iter: int = 1000
    save_interval_iter: int = 10
    importance_sampling: bool = True
    inverse_transform_sampling_grid_size: int = int(1e5)
    save_file: str = "generated/ays_timesteps.pt"


class AYSSamplingSchedule(SamplingSchedule):
    model: Predictor
    schedules: ScheduleGroup
    dataloader: DataLoader
    timestep_config: TimestepConfig
    equation_type: EquationType
    vae: VAE | None = None

    def __init__(
        self,
        *,
        max_t: float = 0.95,
        model: Predictor,
        schedules: ScheduleGroup,
        dataloader: DataLoader,
        solver_T: float,
        equation_type: EquationType,
        vae: VAE | None = None,
        config: AYSConfig = AYSConfig(),
    ):
        super().__init__(max_t=max_t, T=solver_T)
        self.model = model
        self.schedules = schedules
        self.equation_type = equation_type
        self.dataloader = dataloader
        self.vae = vae
        self.config = config

        self.timestep_config = TimestepConfig(kind="continuous", T=solver_T)

    def get_timesteps(
        self, n_steps: int = 10, *, initial_t: Timestep | None = None, **kwargs
    ) -> Timestep:
        if initial_t is not None:
            t = initial_t.adapt(self.timestep_config).steps
        else:
            logger.warning(
                f"No initial_t provided. Remember to set correct max_t: {self.max_t}"
            )
            t = torch.linspace(0.0, self.max_t, 11)

        t = self._get_10_timesteps(t)

        if n_steps <= 10:
            return self._interpolate_timesteps(t, n_steps)

        t = self.get_20_timesteps(t)

        if n_steps <= 20:
            return self._interpolate_timesteps(t, n_steps)

        t = self.get_40_timesteps(t)

        return self._interpolate_timesteps(t, n_steps)

    def _get_10_timesteps(self, initial_steps: torch.Tensor) -> Timestep:
        assert len(initial_steps) == 11

        steps = self._optimize(
            initial_steps,
            max_iter=self.config.max_iter,
            desc="AYS 10-step",
            suffix="_10",
        )

        return Timestep(self.timestep_config, steps)

    def get_20_timesteps(self, steps_10: Timestep) -> Timestep:
        assert len(steps_10) == 11
        steps = steps_10.adapt(self.timestep_config).steps
        steps = self._subdivide(steps)

        steps = self._optimize(
            steps,
            max_iter=self.config.max_finetune_iter,
            desc="AYS 20-step",
            skip_even=True,
            suffix="_20",
        )

        return Timestep(self.timestep_config, steps)

    def get_40_timesteps(self, steps_20: Timestep) -> Timestep:
        assert len(steps_20) == 21
        steps = steps_20.adapt(self.timestep_config).steps
        steps = self._subdivide(steps)

        steps = self._optimize(
            steps,
            max_iter=self.config.max_finetune_iter,
            desc="AYS 40-step",
            skip_even=True,
            suffix="_40",
        )

        return Timestep(self.timestep_config, steps)

    def _optimize(
        self,
        steps: torch.Tensor,
        max_iter: int,
        desc: str,
        *,
        skip_even: bool = False,
        suffix: str = "",
    ) -> torch.Tensor:
        pbar_outer = tqdm(range(max_iter), desc=desc)

        no_change = False
        current_iter = 0

        while not no_change and current_iter < max_iter:
            no_change = True
            current_iter += 1

            t_indices = range(1, len(steps) - 1)
            pbar_steps = tqdm(
                t_indices, desc=f"Iter {current_iter}: Steps", leave=False
            )

            for i in pbar_steps:
                if skip_even and i % 2 == 0:
                    continue

                s = steps[i - 1].item()
                t = steps[i].item()
                t_next = steps[i + 1].item()

                candidates = self._get_candidates(s, t, t_next)
                klub_per_candidate = []

                pbar_steps.set_description(f"Iter {current_iter}: Optimizing t_{i}")

                pbar_candidates = tqdm(candidates, desc="Candidates", leave=False)
                for cand in pbar_candidates:
                    pbar_candidates.set_description(
                        f"t_{i} candidate: {cand.item():.4f}"
                    )

                    klub = self._estimate_klub(s, cand.item())
                    klub += self._estimate_klub(cand.item(), t_next)
                    klub_per_candidate.append(klub)

                argmin = int(torch.argmin(torch.tensor(klub_per_candidate)).item())

                if candidates[argmin].item() != t:
                    steps[i] = candidates[argmin]
                    no_change = False

            pbar_outer.update(1)

            if current_iter % self.config.save_interval_iter == 0:
                torch.save(
                    Timestep(self.timestep_config, steps),
                    f"{self.config.save_file}{suffix}",
                )

        pbar_outer.close()

        return steps

    def _interpolate_timesteps(self, timesteps: Timestep, n_steps: int) -> Timestep:
        steps = timesteps.adapt(self.timestep_config).steps

        if len(steps) == n_steps + 1:
            return Timestep(self.timestep_config, steps)

        xs = torch.linspace(0, self.T, len(steps)).cpu().detach().numpy()
        ys = torch.log(steps).cpu().detach().numpy()

        new_xs = torch.linspace(0, self.T, n_steps + 1).cpu().detach().numpy()
        new_ys = np.interp(new_xs, xs, ys)
        new_steps = torch.exp(torch.tensor(new_ys, device=steps.device))

        return Timestep(self.timestep_config, new_steps)

    def _subdivide(self, steps: torch.Tensor) -> torch.Tensor:
        new_steps = []

        # log t_{2n+1} = 0.5 * (log t_{n} + log t_{n+1})

        for i in range(len(steps) - 1):
            t_start = steps[i].item()
            t_end = steps[i + 1].item()

            new_steps.append(t_start)

            if t_start < EPSILON:
                t_mid = (t_start + t_end) / 2.0
            else:
                # t_mid = math.exp(0.5 * (math.log(t_start) + math.log(t_end)))
                t_mid = math.sqrt(t_start * t_end)

            new_steps.append(t_mid)

        new_steps.append(steps[-1].item())

        return torch.tensor(new_steps, device=steps.device)

    def _get_candidates(self, s: float, t: float, t_next: float) -> torch.Tensor:
        candidates = torch.linspace(
            s + EPSILON, t_next - EPSILON, self.config.n_candidates - 1
        )
        candidates = torch.cat([torch.tensor([t]), candidates])

        return candidates.sort().values

    def _estimate_klub(self, t_start: float, t_end: float) -> float:
        klub_sum = 0.0
        sample_count = 0

        for X in self._get_data_samples():
            t_samples = (
                self._importance_sample(X.size(0), t_start, t_end)
                if self.config.importance_sampling
                else (
                    torch.rand(X.size(0), device=X.device) * (t_end - t_start) + t_start
                )
            )
            timestep_samples = Timestep(self.timestep_config, t_samples)
            timestep_end = Timestep(
                self.timestep_config, torch.tensor([t_end], device=X.device)
            )

            # (batch, channels, height, width)
            X_t, _ = diffuse(
                X,
                timestep_samples,
                self.schedules,
            )

            X_t_end = diffuse_from(
                X,
                X_t,
                timestep_samples,
                timestep_end,
                self.schedules,
            )

            pred_t = self.model(
                X_t, timestep=timestep_samples, schedules=self.schedules
            )
            pred_t_end = self.model(
                X_t_end,
                timestep=timestep_end,
                schedules=self.schedules,
            )

            pred_t_timesteps = Timestep(self.timestep_config, pred_t)

            pred_diff_norm = (pred_t_end - pred_t).view(X.size(0), -1).norm(dim=1) ** 2

            factor = (
                (t_end - t_start)
                if not self.config.importance_sampling
                else self.schedules.edm_sigma(pred_t_timesteps).view(-1, 1, 1, 1) ** 3
                / (1 / (t_start**2 + 0.5**2) - 1 / (t_end**2 + 0.5**2))
            )

            if self.equation_type in [
                EquationType.song_sde,
                EquationType.probability_flow,
            ]:
                assert self.model.target == PredictionTarget.x0

                integral = (
                    self.schedules.s(pred_t_timesteps).view(-1, 1, 1, 1)
                    * self.schedules.edm_sigma.derivative(pred_t_timesteps).view(
                        -1, 1, 1, 1
                    )
                    * (
                        1
                        / self.schedules.edm_sigma(pred_t_timesteps).view(-1, 1, 1, 1)
                        ** 3
                    )
                    * pred_diff_norm
                    * factor
                )
            elif self.equation_type == EquationType.generalized_differential:
                assert self.model.target == PredictionTarget.Noise

                integral = (
                    -0.5
                    * (
                        self.schedules.lambda_.derivative(pred_t_timesteps)
                        * pred_diff_norm
                    )
                    * factor
                )
            else:
                raise ValueError(f"Unsupported equation type: {self.equation_type}")

            klub_sum += integral.sum().item()
            sample_count += X.size(0)

        return klub_sum / sample_count

    # Inverse Transform Sampling
    def _importance_sample(self, n: int, t_start: float, t_end: float) -> torch.Tensor:
        c = 0.5

        t_grid = torch.linspace(
            t_start,
            t_end,
            self.config.inverse_transform_sampling_grid_size,
            device=self.config.device,
        )

        pi_t = (1.0 / t_grid**3) * (1.0 / (t_grid**2 + c**2) - 1.0 / (t_end**2 + c**2))
        pi_t = torch.clamp(pi_t, min=1e-10)

        cdf = torch.cumsum(pi_t, dim=0)
        cdf = cdf / cdf[-1]

        u = torch.rand(n, device=self.config.device)

        indices = torch.searchsorted(cdf, u)
        indices = torch.clamp(indices, 0, len(t_grid) - 1)

        return t_grid[indices]

    def _get_data_samples(self) -> Generator[torch.Tensor, None, None]:
        count = 0
        while count < self.config.n_monte_carlo_iter:
            for batch_idx, (X, _) in enumerate(self.dataloader):
                X = X.to(self.config.device)
                yield self.vae.encode(X.half()).float() if self.vae is not None else X
                count += X.size(0)
                if count >= self.config.n_monte_carlo_iter:
                    break
