import math
import os
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum

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
from src.schedule.sampling import SamplingSchedule
from src.timestep import Timestep, TimestepConfig

STAGE_10 = 10
STAGE_20 = 20
STAGE_40 = 40

MIN_T = 0.5 * 1e-3  # No point going lower than EDM
MIN_GAP = 1e-5

REL_TOL = 1e-7
ABS_TOL = 1e-9


class CandidateSpacing(str, Enum):
    linear = "linear"
    log = "log"


@dataclass
class AYSConfig:
    max_iter: int = 128
    max_finetune_iter: int = 16
    device: torch.device = field(default_factory=get_device)
    n_candidates: int = 11
    n_monte_carlo_iter: int = 1024
    save_interval_iter: int = 10
    importance_sampling: bool = True
    inverse_transform_sampling_grid_size: int = 4096
    candidate_spacing: CandidateSpacing = CandidateSpacing.linear
    save_file: str = "generated/ays_timesteps.pth"


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

    def _save_state(self, steps: torch.Tensor, stage: int, current_iter: int):
        saved_steps = torch.flip(steps.cpu().to(torch.float32), dims=[0])
        if getattr(self, "append_zero", False):
            saved_steps = torch.cat([saved_steps, saved_steps.new_zeros(1)])

        state = {
            "steps": saved_steps,
            "stage": stage,
            "current_iter": current_iter,
            "candidate_spacing": self.config.candidate_spacing.value,
            "append_zero": getattr(self, "append_zero", False),
        }
        os.makedirs(os.path.dirname(self.config.save_file), exist_ok=True)
        torch.save(state, self.config.save_file)

    def get_timesteps(
        self, n_steps: int = 40, *, initial_t: Timestep | None = None, **kwargs
    ) -> Timestep:
        assert n_steps >= 10
        append_zero = False

        stage = STAGE_10
        start_iter = 0
        steps = None

        if os.path.exists(self.config.save_file):
            logger.info(
                f"Resuming AYS optimization from checkpoint: {self.config.save_file}"
            )
            state = torch.load(
                self.config.save_file,
                weights_only=False,
            )
            steps = torch.flip(
                state["steps"].to(device=self.config.device, dtype=torch.float64),
                dims=[0],
            )
            append_zero = bool(state.get("append_zero", False))
            if append_zero:
                if math.isclose(
                    float(steps[0].item()), 0.0, rel_tol=REL_TOL, abs_tol=ABS_TOL
                ):
                    steps = steps[1:]
                else:
                    logger.warning(
                        "Checkpoint has append_zero=True but does not end with 0; "
                        "continuing without removing a terminal zero."
                    )
            stage = state["stage"]
            start_iter = state["current_iter"]
            logger.info(f"Resuming Stage: {stage}-step, Iteration: {start_iter}")
        else:
            logger.info("No checkpoint found. Starting fresh AYS optimization.")
            if initial_t is not None:
                steps = (
                    initial_t.adapt(self.timestep_config)
                    .reverse()
                    .steps.to(device=self.config.device, dtype=torch.float64)
                )
                assert len(steps) == 11
            else:
                logger.warning(
                    f"No initial_t provided. Falling back to linear schedule. Remember to set max_t: {self.max_t}"
                )
                steps = torch.linspace(
                    MIN_T,
                    self.max_t,
                    11,
                    device=self.config.device,
                    dtype=torch.float64,
                )

        if math.isclose(float(steps[0].item()), 0.0, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            logger.warning(
                f"Clamping steps[0]={steps[0].item():.6f} to MIN_T={MIN_T} for numerical stability."
            )
            steps[0] = MIN_T
            append_zero = True

        self.append_zero = append_zero

        if steps[1].item() < MIN_T:
            raise ValueError(
                f"Initial steps[1]={steps[1].item():.6f} is less than MIN_T={MIN_T}. Please provide a valid initial_t."
            )

        assert all(steps[:-1] < steps[1:])

        logger.info(f"Starting with steps: {steps}")

        while True:
            is_first_stage = stage == STAGE_10
            max_iterations = (
                self.config.max_iter
                if is_first_stage
                else self.config.max_finetune_iter
            )

            steps = self._optimize(
                steps,
                max_iter=max_iterations,
                desc=f"AYS {stage}-step",
                stage=stage,
                start_iter=start_iter,
                skip_even=not is_first_stage,
            )

            if (stage >= n_steps and not is_first_stage) or stage == STAGE_40:
                break

            stage *= 2
            start_iter = 0
            steps = self._subdivide(steps)

            logger.info(f"Subdivided steps for next stage ({stage}-step): {steps}")

        result = self._interpolate_timesteps(
            Timestep(self.timestep_config, steps.to(torch.float32)), n_steps
        ).reverse()

        if append_zero:
            result = Timestep(
                self.timestep_config,
                torch.cat(
                    [result.steps, torch.tensor([0.0], device=result.steps.device)]
                ),
            )

        self._save_state(steps, stage, self.config.max_finetune_iter)

        return result

    def _optimize(
        self,
        steps: torch.Tensor,
        *,
        max_iter: int,
        desc: str,
        stage: int,
        start_iter: int = 0,
        skip_even: bool = False,
    ) -> torch.Tensor:
        if start_iter >= max_iter:
            return steps

        pbar_outer = tqdm(total=max_iter, initial=start_iter, desc=desc)

        no_change = False
        current_iter = start_iter

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

                s = float(steps[i - 1].item())
                t = float(steps[i].item())
                t_next = float(steps[i + 1].item())

                candidates = self._get_candidates(s, t, t_next)
                klub_per_candidate = []

                if len(candidates) == 1:
                    logger.warning(
                        f"Skipping optimization for t_{i} due to reaching min gap."
                    )
                else:
                    pbar_steps.set_description(f"Iter {current_iter}: Optimizing t_{i}")

                    pbar_candidates = tqdm(candidates, desc="Candidates", leave=False)
                    for cand in pbar_candidates:
                        cand_val = float(cand.item())
                        pbar_candidates.set_description(
                            f"t_{i} candidate: {cand_val:.4f}"
                        )

                        klub = self._estimate_klub(s, cand_val)
                        klub += self._estimate_klub(cand_val, t_next)
                        klub_per_candidate.append(klub)

                    klub_tensor = torch.tensor(klub_per_candidate, dtype=torch.float64)
                    argmin = int(torch.argmin(klub_tensor).item())

                    if not math.isclose(
                        float(candidates[argmin].item()),
                        t,
                        rel_tol=REL_TOL,
                        abs_tol=ABS_TOL,
                    ):
                        steps[i] = candidates[argmin]
                        no_change = False

            pbar_outer.update(1)

            assert torch.all(steps[1:] >= steps[:-1])

            if (
                current_iter % self.config.save_interval_iter == 0
                or current_iter == max_iter
                or no_change
            ):
                self._save_state(
                    steps, stage, current_iter if not no_change else max_iter
                )

        pbar_outer.close()
        return steps

    def _interpolate_timesteps(self, timesteps: Timestep, n_steps: int) -> Timestep:
        steps = timesteps.adapt(self.timestep_config).steps

        if len(steps) == n_steps + 1:
            return Timestep(self.timestep_config, steps)

        xs = torch.linspace(0, self.T, len(steps)).cpu().detach().numpy()
        ys = torch.log(torch.clamp(steps, min=MIN_T)).cpu().detach().numpy()

        new_xs = torch.linspace(0, self.T, n_steps + 1).cpu().detach().numpy()
        new_ys = np.interp(new_xs, xs, ys)
        new_steps = torch.exp(
            torch.tensor(new_ys, device=steps.device, dtype=steps.dtype)
        )

        return Timestep(self.timestep_config, new_steps)

    def _subdivide(self, steps: torch.Tensor) -> torch.Tensor:
        new_steps = []

        for i in range(len(steps) - 1):
            t_start = float(steps[i].item())
            t_end = float(steps[i + 1].item())
            new_steps.append(t_start)

            if t_start < MIN_T:
                t_mid = (t_start + t_end) / 2.0
            else:
                t_mid = math.sqrt(t_start * t_end)

            new_steps.append(t_mid)

        new_steps.append(float(steps[-1].item()))
        return torch.tensor(new_steps, device=steps.device, dtype=torch.float64)

    def _get_candidates(self, s: float, t: float, t_next: float) -> torch.Tensor:
        if t_next - s <= 2 * MIN_GAP:
            return torch.tensor([t], device=self.config.device, dtype=torch.float64)

        if self.config.candidate_spacing == CandidateSpacing.log:
            log_lower = math.log(s)
            log_upper = math.log(t_next)
            interior_candidates = torch.exp(
                torch.linspace(
                    log_lower,
                    log_upper,
                    self.config.n_candidates + 1,
                    device=self.config.device,
                    dtype=torch.float64,
                )[1:-1]
            )
        else:
            interior_candidates = torch.linspace(
                s,
                t_next,
                self.config.n_candidates + 1,
                device=self.config.device,
                dtype=torch.float64,
            )[1:-1]

        t_tensor = torch.tensor([t], device=self.config.device, dtype=torch.float64)
        candidates = torch.cat([t_tensor, interior_candidates])

        lower_bound = s + MIN_GAP
        upper_bound = t_next - MIN_GAP
        candidates = torch.clamp(candidates, min=lower_bound, max=upper_bound)

        return candidates.sort().values

    @torch.no_grad()
    def _estimate_klub(self, t_start: float, t_end: float) -> float:
        klub_sum = 0.0
        sample_count = 0

        t_start = max(t_start, MIN_T)
        assert t_start < t_end

        if t_end - t_start < MIN_GAP * (1.0 - ABS_TOL):
            logger.warning(
                f"Gap between t_start={t_start:.6f} and t_end={t_end:.6f} is too small and may lead to numerical instability."
            )

        for X in self._get_data_samples():
            batch_size = X.size(0)
            if self.config.importance_sampling:
                t_samples, norm_const = self._importance_sample(
                    batch_size, t_start, t_end
                )
                pi_t = self._importance_sampling_dist(t_samples, t_end)
                weight = norm_const / pi_t
            else:
                u = torch.rand(batch_size, device=X.device, dtype=torch.float64)
                u = torch.clamp(u, min=MIN_GAP, max=1.0 - MIN_GAP)
                t_samples = u * (t_end - t_start) + t_start
                weight = torch.full_like(t_samples, t_end - t_start)

            timestep_samples = Timestep(self.timestep_config, t_samples.to(X.dtype))
            timestep_end = Timestep(
                self.timestep_config,
                torch.full((batch_size,), t_end, device=X.device, dtype=X.dtype),
            )

            X_t, _ = diffuse(
                x0=X,
                t=timestep_samples,
                schedules=self.schedules,
            )

            X_t_end = diffuse_from(
                x_0=X,
                x_s=X_t,
                t_s=timestep_samples,
                t=timestep_end,
                schedules=self.schedules,
            )

            pred_t = self.model(
                X_t, timestep=timestep_samples, schedules=self.schedules
            )
            pred_t_end = self.model(
                X_t_end, timestep=timestep_end, schedules=self.schedules
            )

            pred_diff_norm = (
                pred_t_end.to(torch.float64) - pred_t.to(torch.float64)
            ).view(batch_size, -1).norm(dim=1) ** 2

            ts = Timestep(self.timestep_config, t_samples)

            if self.equation_type in [
                EquationType.score_sde,
                EquationType.probability_flow,
            ]:
                assert self.model.target == PredictionTarget.x0
                sigma_edm_t = self.schedules.edm_sigma(ts).to(torch.float64)
                d_sigma_edm_t = self.schedules.edm_sigma.derivative(ts).to(
                    torch.float64
                )

                integrand = (d_sigma_edm_t / (sigma_edm_t**3)) * pred_diff_norm
                integral = integrand * weight

            elif self.equation_type == EquationType.generalized_differential:
                assert self.model.target == PredictionTarget.Noise
                d_lambda_t = self.schedules.lambda_.derivative(ts).to(torch.float64)

                integrand = -0.5 * d_lambda_t * pred_diff_norm
                integral = integrand * weight
            else:
                raise ValueError(f"Unsupported equation type: {self.equation_type}")

            klub_sum += float(integral.sum().item())
            sample_count += batch_size

        return klub_sum / sample_count

    # Inverse Transform Sampling
    def _importance_sample(
        self, n: int, t_start: float, t_end: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_size = self.config.inverse_transform_sampling_grid_size
        edges = torch.linspace(
            t_start,
            t_end,
            grid_size + 1,
            device=self.config.device,
            dtype=torch.float64,
        )

        # We use midpoints to avoid case where linspace edges are outside of [t_start, t_end] range
        # due to numerical precision issues
        t_grid = 0.5 * (edges[:-1] + edges[1:])
        dt = (t_end - t_start) / grid_size

        pi_t = self._importance_sampling_dist(t_grid, t_end)
        assert torch.all(pi_t > 0)

        norm_const = pi_t.sum() * dt

        cdf = torch.cumsum(pi_t, dim=0)
        cdf = cdf / cdf[-1]

        u = torch.rand(n, device=self.config.device, dtype=torch.float64)
        indices = torch.searchsorted(cdf, u)

        # indices = torch.clamp(indices, min=0, max=grid_size - 1)
        assert torch.all(indices >= 0) and torch.all(indices < grid_size)

        return t_grid[indices], norm_const

    # Note: This returns unnormalized distribution
    def _importance_sampling_dist(
        self, t_tensor: torch.Tensor, t_end: float, c: float = 0.5
    ) -> torch.Tensor:
        ts = Timestep(self.timestep_config, t_tensor)
        ts_end = Timestep(
            self.timestep_config,
            torch.tensor([t_end], device=t_tensor.device, dtype=torch.float64),
        )

        sigma_edm_t = self.schedules.edm_sigma(ts).to(torch.float64)
        sigma_edm_end = self.schedules.edm_sigma(ts_end).to(torch.float64)
        d_sigma_edm_t = self.schedules.edm_sigma.derivative(ts).to(torch.float64)

        # Stable computation of 1/(sigma_t^2 + c^2) - 1/(sigma_end^2 + c^2)
        num = (sigma_edm_end - sigma_edm_t) * (sigma_edm_end + sigma_edm_t)
        den = (sigma_edm_t**2 + c**2) * (sigma_edm_end**2 + c**2)
        diff_inv = torch.clamp(num / den, min=1e-18)

        return (d_sigma_edm_t / (sigma_edm_t**3)) * diff_inv

    def _get_data_samples(self) -> Generator[torch.Tensor, None, None]:
        count = 0
        while count < self.config.n_monte_carlo_iter:
            for _, (X, _) in enumerate(self.dataloader):
                X = X.to(self.config.device)
                yield self.vae.encode(X.half()).float() if self.vae is not None else X
                count += X.size(0)
                if count >= self.config.n_monte_carlo_iter:
                    break
