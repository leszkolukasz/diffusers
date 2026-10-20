from typing import override

import torch

from src.model import PredictionTarget, PredictorUNet, reparametrize
from src.model.nvidia.edm2 import UNet as UNetEDM2
from src.model.presets import EDM2_PRESETS, ModelSize
from src.schedule import ScheduleGroup
from src.timestep import Timestep


class PredictorEDM(PredictorUNet):
    def __init__(
        self,
        *,
        n_channels: int,
        img_width: int,
        img_height: int,
        T: int,
        normalize_x: bool,
        model_size: ModelSize = ModelSize.SMALL,
        target: PredictionTarget | str = PredictionTarget.x0,
        suffix: str | None = None,
        **kwargs,
    ):
        model_size = (
            ModelSize.from_value(model_size)
            if isinstance(model_size, str)
            else model_size
        )

        super().__init__(
            n_channels=n_channels,
            img_width=img_width,
            img_height=img_height,
            T=T,
            normalize_x=normalize_x,
            model_size=model_size,
            suffix=suffix,
            target=target,
            **kwargs,
        )

        self.file_name = f"{self.target.value}{'_norm_x' if self.normalize_x else ''}_predictor_edm_{model_size.value}{suffix if suffix is not None else ''}.pth"

    # Input: (batch_size, C, H, W)
    # Assumes x is normalized by s(t) already, i.e. x = x_t / s(t)
    @override
    def _forward(
        self, x: torch.Tensor, timestep: Timestep, schedules: ScheduleGroup
    ) -> torch.Tensor:
        if not self.normalize_x:
            x = x / schedules.s(timestep).view(-1, 1, 1, 1)

        timestep = timestep.adapt(self.timestep_config)
        sigma_t = torch.clamp(schedules.edm_sigma(timestep).view(-1, 1, 1, 1), min=1e-9)

        sigma_data = 0.5
        c_noise = (1.0 / 4 * torch.log(sigma_t)).view(-1)
        c_in = 1.0 / torch.sqrt(sigma_t**2 + sigma_data**2)
        c_out = sigma_t * sigma_data / torch.sqrt(sigma_t**2 + sigma_data**2)
        c_skip = sigma_data**2 / (sigma_t**2 + sigma_data**2)
        pred = self._predict(c_in * x, c_noise)
        x0 = c_skip * x + c_out * pred

        if self.target == PredictionTarget.x0:
            return x0

        x_t = x * schedules.s(timestep).view(-1, 1, 1, 1)
        return reparametrize(
            pred=x0,
            x_t=x_t,
            t=timestep,
            schedules=schedules,
            src=PredictionTarget.x0,
            dest=self.target,
        )

    def _predict(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.unet(x, timestep=timestep).sample

    def loss_weight(self, t: Timestep, schedules: ScheduleGroup) -> torch.Tensor:
        if self.target == PredictionTarget.Score:
            # From NCSN
            return schedules.edm_sigma(t) ** 2
        if self.target == PredictionTarget.Noise:
            return super().loss_weight(t, schedules)

        sigma_t = schedules.edm_sigma(t)
        sigma_data = 0.5

        return (sigma_t**2 + sigma_data**2) / (sigma_t * sigma_data) ** 2


class PredictorEDM2(PredictorEDM):
    def __init__(
        self,
        n_channels: int,
        img_width: int,
        img_height: int,
        T: int,
        model_size: ModelSize = ModelSize.SMALL,
        suffix: str | None = None,
        **kwargs,
    ):
        model_size = (
            ModelSize.from_value(model_size)
            if isinstance(model_size, str)
            else model_size
        )

        super().__init__(
            n_channels=n_channels,
            img_width=img_width,
            img_height=img_height,
            T=T,
            model_size=model_size,
            suffix=suffix,
            **kwargs,
        )

        self.file_name = f"{self.target.value}{'_norm_x' if self.normalize_x else ''}_predictor_edm2_{model_size.value}{suffix if suffix is not None else ''}.pth"

        self.unet = UNetEDM2(
            img_resolution=img_width,
            img_channels=n_channels,
            label_dim=0,
            **EDM2_PRESETS[model_size],
        )

    def _predict(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.unet(x, timestep, None)
