import os
import pickle
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, override
from urllib.parse import urlparse

import torch
from loguru import logger

from src.model import PredictionTarget, PredictorUNet, reparametrize
from src.model.nvidia.edm2 import MPConv, MPFourier
from src.model.nvidia.edm2 import UNet as UNetEDM2
from src.model.presets import EDM2_PRESETS, EDM2_PRETRAINED_PRESETS, ModelSize
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
        prefix: str | None = None,
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
            prefix=prefix,
            suffix=suffix,
            target=target,
            **kwargs,
        )

        self.file_name = f"{self.prefix + '_' if self.prefix else ''}{self.target.value}{'_norm_x' if self.normalize_x else ''}_predictor_edm_{model_size.value}{suffix if suffix is not None else ''}.pth"

    # Input: (batch_size, C, H, W)
    # Assumes x is normalized by s(t) already, i.e. x = x_t / s(t)
    @override
    def _forward(
        self,
        x: torch.Tensor,
        timestep: Timestep,
        schedules: ScheduleGroup,
        return_logvar: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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

        if return_logvar:
            return x0, self._predict_logvar(c_noise)

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
        prefix: str | None = None,
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
            prefix=prefix,
            suffix=suffix,
            **kwargs,
        )

        self.file_name = f"{self.prefix + '_' if self.prefix else ''}{self.target.value}{'_norm_x' if self.normalize_x else ''}_predictor_edm2_{model_size.value}{suffix if suffix is not None else ''}.pth"

        self.logvar_fourier = MPFourier(128)
        self.logvar_linear = MPConv(128, 1, kernel=[])

    def _build_unet(self):
        return UNetEDM2(
            img_resolution=self.img_width,
            img_channels=self.n_channels,
            label_dim=0,
            **EDM2_PRESETS[self.model_size],
        )

    def _predict(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.unet(x, timestep, None)

    def _predict_logvar(self, c_noise: torch.Tensor) -> torch.Tensor:
        return self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)


@contextmanager
def _edm2_pickle_modules():
    """Fixes pickling issues as EDM2 uses different import paths."""

    from src.model.nvidia import dnnlib, torch_utils
    from src.model.nvidia.torch_utils import misc, persistence

    stubs = {
        name: types.ModuleType(name)
        for name in ("torch_utils.distributed", "torch_utils.training_stats")
    }
    aliases = {
        "dnnlib": dnnlib,
        "torch_utils": torch_utils,
        "torch_utils.persistence": persistence,
        "torch_utils.misc": misc,
        **stubs,
    }

    previous_modules = {name: sys.modules.get(name) for name in aliases}
    previous_attributes = {}
    for name, module in (
        ("persistence", persistence),
        ("misc", misc),
        ("distributed", stubs["torch_utils.distributed"]),
        ("training_stats", stubs["torch_utils.training_stats"]),
    ):
        previous_attributes[name] = getattr(torch_utils, name, None)
        setattr(torch_utils, name, module)

    sys.modules.update(aliases)
    try:
        yield
    finally:
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for name, module in previous_attributes.items():
            if module is None:
                delattr(torch_utils, name)
            else:
                setattr(torch_utils, name, module)


def _edm2_cache_dir() -> Path:
    cache_dir = os.environ.get("EDM2_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir).expanduser()
    return Path.home() / ".cache" / "diffusion-model" / "edm2"


def _download_edm2_checkpoint(url: str, expected_size: int) -> Path:
    cache_path = _edm2_cache_dir() / Path(urlparse(url).path).name
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and cache_path.stat().st_size == expected_size:
        return cache_path

    if cache_path.exists():
        logger.warning(f"Removing incomplete EDM2 checkpoint: {cache_path}")
        cache_path.unlink()

    logger.info(f"Downloading EDM2 checkpoint: {url}")
    torch.hub.download_url_to_file(url, str(cache_path), progress=True)

    if cache_path.stat().st_size != expected_size:
        raise RuntimeError(
            f"EDM2 checkpoint has wrong size: {cache_path} "
            f"({cache_path.stat().st_size} bytes, expected {expected_size})"
        )
    return cache_path


def _extract_edm2_unet_state(checkpoint_path: Path) -> dict[str, Any]:
    state_path = checkpoint_path.with_suffix(".unet.pth")
    if state_path.exists():
        return torch.load(state_path, map_location="cpu", weights_only=True)[
            "state_dict"
        ]

    logger.info(f"Extracting UNet weights from EDM2 checkpoint: {checkpoint_path}")
    with _edm2_pickle_modules(), checkpoint_path.open("rb") as file:
        data = pickle.load(file)

    ema = data["ema"]
    payload = {
        "unet": ema.unet.state_dict(),
        "logvar_fourier": ema.logvar_fourier.state_dict(),
        "logvar_linear": ema.logvar_linear.state_dict(),
    }
    torch.save(payload, state_path)
    return payload


class PredictorEDM2Pretrained(PredictorEDM2):
    def __init__(
        self,
        n_channels: int,
        img_width: int,
        img_height: int,
        T: int,
        model_size: ModelSize | str = ModelSize.PRETRAINED_XS_IMG64,
        class_idx: int | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        **kwargs,
    ):
        model_size = (
            ModelSize.from_value(model_size)
            if isinstance(model_size, str)
            else model_size
        )
        if model_size not in EDM2_PRETRAINED_PRESETS:
            logger.warning(
                f"Model size '{model_size.value}' is not a pretrained EDM2 preset; "
                f"using {ModelSize.PRETRAINED_XS_IMG64.value}."
            )
            model_size = ModelSize.PRETRAINED_XS_IMG64

        self.pretrained_preset = EDM2_PRETRAINED_PRESETS[model_size]
        self.label_dim = int(self.pretrained_preset["label_dim"])
        if class_idx == "random":
            self.class_idx: int | str | None = "random"
        elif class_idx is not None:
            class_idx = int(class_idx)
            if not 0 <= class_idx < self.label_dim:
                raise ValueError(
                    f"class_idx must be in [0, {self.label_dim}), got {class_idx}."
                )
            self.class_idx = class_idx
        else:
            self.class_idx = None
        self._random_class_labels: torch.Tensor | None = None

        super().__init__(
            n_channels=n_channels,
            img_width=img_width,
            img_height=img_height,
            T=T,
            model_size=model_size,
            prefix=prefix,
            suffix=suffix,
            **kwargs,
        )

        prefix_str = f"{self.prefix}_" if self.prefix else ""
        suffix_str = suffix if suffix is not None else ""
        self.file_name = (
            f"{prefix_str}{self.target.value}"
            f"{'_norm_x' if self.normalize_x else ''}"
            f"_predictor_edm2_{model_size.value.lower()}{suffix_str}.pth"
        )

    def _build_unet(self):
        preset = self.pretrained_preset
        expected_width = int(preset["img_resolution"])
        expected_channels = int(preset["img_channels"])
        if self.img_width != expected_width or self.img_height != expected_width:
            raise ValueError(
                f"{self.__class__.__name__} requires {expected_width}x{expected_width} "
                f"images, but got {self.img_height}x{self.img_width}."
            )
        if self.n_channels != expected_channels:
            raise ValueError(
                f"{self.__class__.__name__} requires {expected_channels} input "
                f"channels, but got {self.n_channels}."
            )

        self.label_dim = int(preset["label_dim"])
        return UNetEDM2(
            img_resolution=expected_width,
            img_channels=expected_channels,
            label_dim=self.label_dim,
            **preset["unet"],
        )

    def set_random_class_labels(self, batch_size: int, device=None) -> None:
        if self.label_dim == 0:
            return
        if device is None:
            device = next(self.parameters()).device
        labels = torch.zeros(
            (batch_size, self.label_dim), device=device, dtype=torch.float32
        )
        indices = torch.randint(
            self.label_dim, (batch_size,), device=device, dtype=torch.long
        )
        labels[torch.arange(batch_size, device=device), indices] = 1
        self._random_class_labels = labels

    def _predict(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if self.label_dim == 0:
            labels = None
        elif self.class_idx == "random":
            if (
                self._random_class_labels is None
                or self._random_class_labels.shape[0] < x.shape[0]
            ):
                self.set_random_class_labels(x.shape[0], device=x.device)
            labels = self._random_class_labels[: x.shape[0]].to(
                device=x.device, dtype=x.dtype
            )
        else:
            labels = torch.zeros(
                (x.shape[0], self.label_dim), device=x.device, dtype=x.dtype
            )
            if self.class_idx is not None:
                labels[:, self.class_idx] = 1
        return self.unet(x, timestep, labels)

    def load_pretrained(self):
        preset = self.pretrained_preset
        checkpoint_path = _download_edm2_checkpoint(
            str(preset["checkpoint_url"]), int(preset["checkpoint_size"])
        )
        state_dict = _extract_edm2_unet_state(checkpoint_path)
        self.unet.load_state_dict(state_dict["unet"], strict=True)
        if "logvar_fourier" in state_dict and "logvar_linear" in state_dict:
            self.logvar_fourier.load_state_dict(state_dict["logvar_fourier"])
            self.logvar_linear.load_state_dict(state_dict["logvar_linear"])
        else:
            logger.warning(
                "EDM2 checkpoint cache has no logvar head; using a fresh one."
            )

        self.metadata = {
            "type": self.__class__.__name__,
            "T": self.timestep_config.T,
            "n_channels": self.n_channels,
            "img_width": self.img_width,
            "img_height": self.img_height,
            "target": self.target.value,
            "normalize_x": self.normalize_x,
            "model_size": self.model_size.value,
            "pretrained_preset": self.model_size.value,
            "class_idx": self.class_idx,
            "alpha_schedule": "ConstantAlphaSchedule",
            "sigma_schedule": "LinearSigmaSchedule",
            "eta_schedule": "ConstantEtaSchedule",
        }
        logger.info(
            f"Loaded pretrained EDM2 UNet from {self.model_size.value} "
            f"({sum(p.numel() for p in self.unet.parameters()):,} parameters)"
        )

    def load(self):
        local_path = f"models/{self.file_name}"
        if os.path.exists(local_path):
            super().load()
        else:
            self.load_pretrained()

    def try_load(self) -> bool:
        if super().try_load():
            return True
        self.load_pretrained()
        return True
