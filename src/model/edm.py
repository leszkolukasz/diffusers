import torch
from diffusers.models import UNet2DModel

from src.model import PredictionTarget, Predictor
from src.schedule import ScheduleGroup
from src.timestep import Timestep, TimestepConfig
from src.model.nvidia.edm2 import UNet as UNetEDM2


class PredictorEDM(Predictor):
    def __init__(
        self,
        *,
        n_channels: int,
        img_width: int,
        img_height: int,
        T: int,
        suffix: str | None = None,
        **_kwargs,
    ):
        super().__init__()
        assert img_width == img_height, "Only square images are supported"

        self.n_channels = n_channels
        self.img_width = img_width
        self.img_height = img_height
        self.target = PredictionTarget.x0

        self.timestep_config = TimestepConfig(kind="continuous", T=T)
        self.file_name = f"{self.target.value}_predictor_edm{suffix if suffix is not None else ''}.pth"
        self.unet = UNet2DModel(  # TODO: Add EDM2 network
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
        suffix: str | None = None,
        **_kwargs,
    ):
        super().__init__(
            n_channels=n_channels,
            img_width=img_width,
            img_height=img_height,
            T=T,
            suffix=suffix,
            **_kwargs,
        )

        self.file_name = f"{self.target.value}_predictor_edm2{suffix if suffix is not None else ''}.pth"

        self.unet = UNetEDM2(
            img_resolution=img_width,
            img_channels=n_channels,
            label_dim=0,
            model_channels=32,
            channel_mult=[1, 2, 4],
            num_blocks=2,
            attn_resolutions=[16, 8],
        )

    def _predict(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.unet(x, timestep, None)
