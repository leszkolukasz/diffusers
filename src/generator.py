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

    # Assumes timesteps are provided in decreasing order. Should not contain 0 or max_t.
    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        *,
        n_steps: int | None = None,
        timesteps: Timestep | None = None,
    ) -> torch.Tensor:
        if n_steps is None and timesteps is None:
            n_steps = int(self.denoiser.model.timestep_config.max_t)
            logger.info(
                f"Neither n_steps nor timesteps were provided. Using model's max_t: {n_steps}"
            )

        device = next(self.denoiser.model.parameters()).device
        x_t = torch.randn(
            n_samples, self.n_channels, self.img_width, self.img_height, device=device
        )

        if TYPE_CHECKING:
            assert n_steps is not None

        if timesteps is None:
            # Skip last step as it causes issues (TODO: check this)
            timesteps_tensor = torch.arange(
                n_steps - 1, 0, -1, device=x_t.device, dtype=torch.long
            )
            timesteps = Timestep(
                TimestepConfig(kind="discrete", max_t=n_steps), timesteps_tensor
            )

        for i in range(len(timesteps) - 1):
            x_t = self.denoiser.denoise(x_t, timesteps[i], timesteps[i + 1])

        return x_t
