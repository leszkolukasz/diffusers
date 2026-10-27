from typing import override

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
from src.model.presets import UNET_PRESETS, ModelSize
from src.schedule import ScheduleGroup
from src.timestep import Timestep, TimestepConfig


# Model is conditioned on timestep from range [0, T] inclusive.
class PredictorUNet(Predictor):
    model_size: ModelSize

    def __init__(
        self,
        *,
        n_channels: int,
        img_width: int,
        img_height: int,
        T: int,
        normalize_x: bool,
        model_size: ModelSize | str = ModelSize.SMALL,
        prefix: str | None = None,
        suffix: str | None = None,
        target: PredictionTarget | str = PredictionTarget.Noise,
        **_kwargs,
    ):
        super().__init__()
        assert img_width == img_height, "Only square images are supported"

        self.n_channels = n_channels
        self.img_width = img_width
        self.img_height = img_height
        self.prefix = prefix.strip("_") if prefix else ""
        self.normalize_x = normalize_x
        self.target = (
            PredictionTarget.from_value(target) if isinstance(target, str) else target
        )
        self.model_size = (
            ModelSize.from_value(model_size)
            if isinstance(model_size, str)
            else model_size
        )

        self.timestep_config = TimestepConfig(kind="continuous", T=T)
        prefix_str = f"{self.prefix}_" if self.prefix else ""
        self.file_name = f"{prefix_str}{self.target.value}{'_norm_x' if self.normalize_x else ''}_predictor_unet_{self.model_size.value}{suffix if suffix is not None else ''}.pth"
        self.unet = self._build_unet()

    def _build_unet(self):
        return UNet2DModel(
            sample_size=self.img_width,
            in_channels=self.n_channels,
            out_channels=self.n_channels,
            **UNET_PRESETS[self.model_size],
        )

    # Input: (batch_size, C, H, W)
    @override
    def _forward(
        self,
        x: torch.Tensor,
        timestep: Timestep,
        schedules: ScheduleGroup | None = None,
        return_logvar: bool = False,
    ) -> torch.Tensor:
        if return_logvar:
            raise NotImplementedError("This predictor has no logvar head.")
        timestep = timestep.adapt(self.timestep_config)
        return self.unet(x, timestep=timestep.steps).sample

    def loss_weight(
        self, t: Timestep, schedules: ScheduleGroup | None = None
    ) -> torch.Tensor:
        if self.target == PredictionTarget.Score:
            if schedules is None:
                raise ValueError("schedules must be provided for score loss weight")
            # Without clamping loss will explode for some schedules.
            return torch.clamp(schedules.edm_sigma(t) ** 2, max=1.0)
        return super().loss_weight(t, schedules)

    def save(self, path=None, **extra_metadata):
        super().save(
            path=path, model_size=self.model_size.value, **extra_metadata
        )

    def load(self):
        super().load()
        meta_model_size = self.metadata.get("model_size", None)
        if meta_model_size is not None and meta_model_size != self.model_size.value:
            logger.warning(
                f"Loaded model model_size '{meta_model_size}' does not match current model_size '{self.model_size.value}'"
            )


class PredictorHuggingface(PredictorUNet):
    def __init__(
        self,
        *,
        model_id: str,
        dtype: torch.dtype = torch.float32,
        unet_class: type = UNet2DModel,
        scheduler_class: type = DDPMScheduler,
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

        logger.warning("Normalization of x is disabled for Huggingface models.")

        super().__init__(
            n_channels=n_channels,
            img_width=img_width,
            img_height=img_height,
            T=T,
            normalize_x=False,
            model_size=ModelSize.MICRO,  # Created UNet will be discarded
            target=PredictionTarget.Noise,
            suffix=f"_{model_id.replace('/', '_')}{suffix_str}",
        )
        self.unet = load_unet_pretrained(
            model_id, unet_class=unet_class, torch_dtype=dtype
        )

    def load(self):
        raise NotImplementedError("Not supported")

    def save(self, path=None, **extra_metadata):
        raise NotImplementedError("Not supported")
