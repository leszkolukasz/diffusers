import torch

from src.model import PredictionTarget, PredictorUNet
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
        model_size: ModelSize = ModelSize.SMALL,
        suffix: str | None = None,
        **kwargs,
    ):
        kwargs.pop("target", None)  # Needed for load_from_file to work

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
            target=PredictionTarget.x0,
            **kwargs,
        )

        self.file_name = f"{self.target.value}_predictor_edm_{model_size.value}{suffix if suffix is not None else ''}.pth"

    # Input: (batch_size, C, H, W)
    def forward(
        self, x: torch.Tensor, timestep: Timestep, schedules: ScheduleGroup
    ) -> torch.Tensor:
        timestep = timestep.adapt(self.timestep_config)
        s_t = schedules.s(timestep).view(-1, 1, 1, 1)
        sigma_t = schedules.edm_sigma(timestep).view(-1, 1, 1, 1)

        sigma_data = 0.5

        normalized_x = x / s_t
        c_noise = (1.0 / 4 * torch.log(sigma_t)).view(-1)
        c_in = 1.0 / torch.sqrt(sigma_t**2 + sigma_data**2)
        c_out = sigma_t * sigma_data / torch.sqrt(sigma_t**2 + sigma_data**2)
        c_skip = sigma_data**2 / (sigma_t**2 + sigma_data**2)
        pred = self._predict(c_in * normalized_x, c_noise)

        return c_skip * normalized_x + c_out * pred

    def _predict(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.unet(x, timestep=timestep).sample

    def loss_weight(self, t: Timestep, schedules: ScheduleGroup) -> torch.Tensor:
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

        self.file_name = f"{self.target.value}_predictor_edm2_{model_size.value}{suffix if suffix is not None else ''}.pth"

        self.unet = UNetEDM2(
            img_resolution=img_width,
            img_channels=n_channels,
            label_dim=0,
            **EDM2_PRESETS[model_size],
        )

    def _predict(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.unet(x, timestep, None)
