from dataclasses import dataclass

import torch
from loguru import logger
from PIL.GifImagePlugin import TYPE_CHECKING

from src.denoiser import Denoiser
from src.timestep import Timestep, TimestepConfig


@dataclass
class Generator:
    denoiser: Denoiser

    n_channels: int
    img_width: int
    img_height: int

    # Assumes timesteps are provided in decreasing order.
    # Should not contain values close to max_t.
    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        *,
        max_t: float | int | None = None,
        n_steps: int | None = None,
        timesteps: Timestep | None = None,
    ) -> torch.Tensor:
        if n_steps is None and timesteps is None:
            n_steps = int(self.denoiser.model.timestep_config.max_t) - 1
            logger.info(
                f"Neither n_steps nor timesteps were provided. Using max number of steps according to model config: {n_steps}."
            )

        device = next(self.denoiser.model.parameters()).device
        x_t = torch.randn(
            n_samples, self.n_channels, self.img_width, self.img_height, device=device
        )

        if TYPE_CHECKING:
            assert n_steps is not None

        max_t = max_t or self.denoiser.model.timestep_config.max_t

        if timesteps is None:
            timesteps_tensor = torch.linspace(
                max_t,
                0.0,
                n_steps + 1,
                device=device,
            )
            timesteps = Timestep(
                TimestepConfig(
                    kind="continuous", max_t=self.denoiser.model.timestep_config.max_t
                ),
                timesteps_tensor,
            )

        for i in range(len(timesteps) - 1):
            x_t = self.denoiser.denoise(x_t, timesteps[i], timesteps[i + 1])

        return x_t
