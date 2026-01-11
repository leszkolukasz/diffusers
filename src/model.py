import os
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Self, Type, cast

import torch
from diffusers import DDPMScheduler, SchedulerMixin
from diffusers.models import UNet2DModel
from loguru import logger
from torch import nn

from src.common import assert_type
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
    n_channels: int
    img_width: int
    img_height: int

    @abstractmethod
    def forward(self, x: torch.Tensor, timestep: Timestep) -> torch.Tensor:
        pass

    def save(self, **extra_metadata):
        super().save(
            max_t=self.timestep_config.max_t,
            n_channels=self.n_channels,
            img_width=self.img_width,
            img_height=self.img_height,
            **extra_metadata,
        )

    def load(self):
        super().load()
        meta_max_t = self.metadata.get("max_t", None)
        meta_n_channels = self.metadata.get("n_channels", None)
        meta_img_width = self.metadata.get("img_width", None)
        meta_img_height = self.metadata.get("img_height", None)

        if meta_max_t is not None and meta_max_t != self.timestep_config.max_t:
            logger.warning(
                f"Loaded model max_t '{meta_max_t}' does not match current timestep_config.max_t '{self.timestep_config.max_t}'"
            )

        for attr_name, meta_value, current_value in [
            ("n_channels", meta_n_channels, self.n_channels),
            ("img_width", meta_img_width, self.img_width),
            ("img_height", meta_img_height, self.img_height),
        ]:
            if meta_value is not None and meta_value != current_value:
                logger.warning(
                    f"Loaded model {attr_name} '{meta_value}' does not match current value '{current_value}'"
                )

    @classmethod
    def load_from_file(cls, file_path: str) -> "NoisePredictor":
        instance = super().load_from_file(file_path)

        if not issubclass(instance.__class__, NoisePredictor):
            raise ValueError(f"Loaded model from {file_path} is not an NoisePredictor")

        return instance


# Model is conditioned on timestep from range [0, max_t] inclusive.
class NoisePredictorUNet(NoisePredictor):
    def __init__(
        self,
        *,
        n_channels: int,
        img_width: int,
        img_height: int,
        max_t: int,
        suffix: str | None = None,
        **_kwargs,
    ):
        super().__init__()
        assert img_width == img_height, "Only square images are supported"

        self.n_channels = n_channels
        self.img_width = img_width
        self.img_height = img_height

        self.timestep_config = TimestepConfig(kind="discrete", max_t=max_t)
        self.file_name = (
            f"noise_predictor_unet{suffix if suffix is not None else ''}.pth"
        )
        self.unet = UNet2DModel(
            sample_size=img_width,
            in_channels=n_channels,
            out_channels=n_channels,
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


class NoisePredictorHuggingface(NoisePredictorUNet):
    def __init__(
        self,
        *,
        model_id: str,
        scheduler_class: Type[SchedulerMixin] = DDPMScheduler,  # type: ignore
        suffix: str | None = None,
        **_kwargs,
    ):
        suffix_str = "" if suffix is None else suffix

        model_config = cast(dict, UNet2DModel.load_config(model_id))
        n_channels = assert_type(model_config.get("in_channels"), int)
        img_width = assert_type(model_config.get("sample_size"), int)
        img_height = img_width

        scheduler_config = cast(dict, scheduler_class.load_config(model_id))  # ty: ignore
        max_t = assert_type(scheduler_config.get("num_train_timesteps"), int)

        super().__init__(
            n_channels=n_channels,
            img_width=img_width,
            img_height=img_height,
            max_t=max_t,
            suffix=f"_{model_id.replace('/', '_')}{suffix_str}",
        )
        self.unet = UNet2DModel.from_pretrained(model_id)
