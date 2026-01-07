from abc import ABC, abstractmethod
import os

import torch
from torch import nn
from diffusers.models import UNet2DModel

class PersistableModule(nn.Module):
    file_name: str

    def save(self):
        torch.save(self.state_dict(), f"models/{self.file_name}")

    def load(self):
        self.load_state_dict(torch.load(f"models/{self.file_name}"))

    def try_load(self):
        if os.path.exists(f"models/{self.file_name}"):
            self.load()

class ErrorPredictor(PersistableModule, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, timestep: torch.LongTensor) -> torch.Tensor:
        pass

class SimpleErrorPredictor(ErrorPredictor):
    file_name = "simple_error_predictor.pth"

    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, _timestep: torch.LongTensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.upsample(x)
        return x
    
# Model is conditioned on timestep from range [0, max_steps] inclusive.
class ErrorPredictorUNet(ErrorPredictor):
    max_steps: int

    def __init__(self, max_steps: int, suffix: str = ""):
        super().__init__()
        self.max_steps = max_steps
        self.file_name = f"error_predictor_unet_{max_steps}{suffix}.pth"
        self.unet = UNet2DModel(
            sample_size=28,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 128),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    # Input: (batch_size, C, H, W)
    def forward(self, x: torch.Tensor, timestep: torch.LongTensor) -> torch.Tensor:
        if torch.any(timestep > self.max_steps) or torch.any(timestep < 0):
            raise ValueError(f"Timestep must be in range [0, {self.max_steps}]")
        
        if torch.dtype(timestep) != torch.long:
            raise ValueError("Timestep must be of type torch.LongTensor")

        return self.unet(x, timestep=timestep).sample