import torch

from dataclasses import dataclass
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
    def generate(self, n_samples: int, *, n_steps: int | None = None, timesteps: Timestep | None = None) -> torch.Tensor:
        assert n_steps is not None or timesteps is not None, "Either n_steps or timesteps must be provided"

        device = next(self.denoiser.model.parameters()).device
        x_t = torch.randn(n_samples, self.n_channels, self.img_width, self.img_height, device=device)

        if timesteps is None:
            # Skip first and last step as it causes issues
            timesteps = torch.arange(n_steps-1, 1, -1, device=x_t.device, dtype=torch.long)
            timesteps = Timestep(TimestepConfig(kind="discrete", max_t=n_steps), timesteps)

        for i in range(len(timesteps)-1):
            x_t = self.denoiser.denoise(x_t, timesteps[i], timesteps[i+1])

        return x_t