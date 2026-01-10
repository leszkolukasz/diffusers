import os
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Self

import torch
from diffusers.models import UNet2DModel
from loguru import logger
from torch import nn

from src.timestep import Timestep, TimestepConfig


class ModuleMetadata(StrEnum):
    AlphaSchedule = "alpha_schedule"
    SigmaSchedule = "sigma_schedule"


class PersistableModule(nn.Module):
    file_name: str
    metadata: dict = {}

    def save(self, **extra_metadata):
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {"type": self.__class__.__name__, **extra_metadata},
        }
        torch.save(checkpoint, f"models/{self.file_name}")

    def load(self):
        checkpoint = torch.load(f"models/{self.file_name}", weights_only=False)
        self.load_state_dict(checkpoint["state_dict"])
        self.metadata = checkpoint.get("config", {})

        type_name = self.metadata.get("type", None)
        if type_name is not None and type_name != self.__class__.__name__:
            logger.warning(
                f"Loaded model type '{type_name}' does not match current class '{self.__class__.__name__}'"
            )

    def try_load(self):
        if os.path.exists(f"models/{self.file_name}"):
            self.load()

    @classmethod
    def load_from_file(cls, file_path: str) -> Self:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model not found: {file_path}")

        checkpoint = torch.load(file_path, weights_only=False)
        config = checkpoint.get("config", {})
        type_name = config.get("type")

        if not type_name:
            raise ValueError(f"Model {file_path} does not contain metadata 'type'.")

        target_class = cls._get_subclass_by_name(type_name)
        model = target_class(**config)

        model.load_state_dict(checkpoint["state_dict"])
        model.metadata = config
        model.file_name = os.path.basename(file_path)

        return model

    @classmethod
    def _get_subclass_by_name(cls, name: str):
        if cls.__name__ == name:
            return cls

        for subclass in cls.__subclasses__():
            found = subclass._get_subclass_by_name(name)
            if found:
                return found

        raise ValueError(
            f"Class '{name}' not found in subclass hierarchy of {cls.__name__}"
        )


class NoisePredictor(PersistableModule, ABC):
    timestep_config: TimestepConfig

    @abstractmethod
    def forward(self, x: torch.Tensor, timestep: Timestep) -> torch.Tensor:
        pass

    def save(self, **extra_metadata):
        super().save(max_t=self.timestep_config.max_t, **extra_metadata)

    def load(self):
        super().load()
        meta_max_t = self.metadata.get("max_t", None)
        if meta_max_t is not None and meta_max_t != self.timestep_config.max_t:
            logger.warning(
                f"Loaded model max_t '{meta_max_t}' does not match current timestep_config.max_t '{self.timestep_config.max_t}'"
            )

    @classmethod
    def load_from_file(cls, file_path: str) -> "NoisePredictor":
        instance = super().load_from_file(file_path)

        if not issubclass(instance.__class__, NoisePredictor):
            raise ValueError(f"Loaded model from {file_path} is not an NoisePredictor")

        return instance


# Model is conditioned on timestep from range [0, max_t] inclusive.
class NoisePredictorUNet(NoisePredictor):
    timestep_config: TimestepConfig

    def __init__(self, *, max_t: int, suffix: str | None = None, **_kwargs):
        super().__init__()
        self.timestep_config = TimestepConfig(kind="discrete", max_t=max_t)
        self.file_name = (
            f"noise_predictor_unet{suffix if suffix is not None else ''}.pth"
        )
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
    def forward(self, x: torch.Tensor, timestep: Timestep) -> torch.Tensor:
        timestep = timestep.adapt(self.timestep_config)
        return self.unet(x, timestep=timestep.steps).sample
