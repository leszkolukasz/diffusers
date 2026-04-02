from typing import Type

import torch
from diffusers import DDPMScheduler
from diffusers.models import UNet2DModel
from loguru import logger

from src.common import (
    assert_type,
    load_scheduler_config,
    load_unet_config,
    load_unet_pretrained,
)
from src.model import PredictionTarget, Predictor
from src.schedule import ScheduleGroup
from src.timestep import Timestep, TimestepConfig


# Model is conditioned on timestep from range [0, T] inclusive.
class PredictorUNet(Predictor):
    def __init__(
        self,
        *,
        n_channels: int,
        img_width: int,
        img_height: int,
        T: int,
        suffix: str | None = None,
        target: PredictionTarget | str = PredictionTarget.Noise,
        **_kwargs,
    ):
        super().__init__()
        assert img_width == img_height, "Only square images are supported"

        self.n_channels = n_channels
        self.img_width = img_width
        self.img_height = img_height
        self.target = (
            PredictionTarget.from_value(target) if isinstance(target, str) else target
        )

        self.timestep_config = TimestepConfig(kind="continuous", T=T)
        self.file_name = f"{self.target.value}_predictor_unet{suffix if suffix is not None else ''}.pth"
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
    def forward(
        self,
        x: torch.Tensor,
        timestep: Timestep,
        schedules: ScheduleGroup | None = None,
    ) -> torch.Tensor:
        timestep = timestep.adapt(self.timestep_config)
        return self.unet(x, timestep=timestep.steps).sample


class PredictorHuggingface(PredictorUNet):
    def __init__(
        self,
        *,
        model_id: str,
        dtype: torch.dtype = torch.float32,
        unet_class: Type = UNet2DModel,
        scheduler_class: Type = DDPMScheduler,
        suffix: str | None = None,
        **_kwargs,
    ):
        suffix_str = "" if suffix is None else suffix

        model_config = load_unet_config(model_id, unet_class)
        n_channels = assert_type(model_config.get("in_channels"), int)
        img_width = assert_type(model_config.get("sample_size"), int)
        img_height = img_width

        scheduler_config = load_scheduler_config(model_id, scheduler_class)
        T = assert_type(scheduler_config.get("num_train_timesteps"), int)
        prediction_type = scheduler_config.get("prediction_type")

        if prediction_type != PredictionTarget.Noise.to_hf():
            logger.warning(
                f"Loaded scheduler prediction_type '{prediction_type}' is not '{PredictionTarget.Noise.to_hf()}'."
            )

        super().__init__(
            n_channels=n_channels,
            img_width=img_width,
            img_height=img_height,
            T=T,
            target=PredictionTarget.Noise,
            suffix=f"_{model_id.replace('/', '_')}{suffix_str}",
        )
        self.unet = load_unet_pretrained(
            model_id, unet_class=unet_class, torch_dtype=dtype
        )

    def load(self):
        raise NotImplementedError("Not supported")

    def save(self, **extra_metadata):
        raise NotImplementedError("Not supported")
