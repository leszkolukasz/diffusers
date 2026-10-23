import os
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Self

import torch
from loguru import logger
from torch import nn

from src.schedule import ScheduleGroup
from src.timestep import Timestep, TimestepConfig

EPSILON = 1e-9


class PredictorMetadata(StrEnum):
    AlphaSchedule = "alpha_schedule"
    SigmaSchedule = "sigma_schedule"
    EtaSchedule = "eta_schedule"
    VAE = "vae"
    Dataset = "dataset"


class PersistableModule(nn.Module):
    file_name: str
    metadata: dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = {}

    @property
    def raw_file_name(self) -> str:
        return f"raw_{self.file_name}"

    def save(
        self, path: str | os.PathLike | None = None, **extra_metadata
    ):
        self.save_to(path or f"models/{self.file_name}", **extra_metadata)

    def save_to(self, path: str | os.PathLike, **extra_metadata):
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {"type": self.__class__.__name__, **extra_metadata},
        }
        parent = os.path.dirname(os.fspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(checkpoint, path)

    def load(self):
        self.load_from_path(f"models/{self.file_name}")

    def load_from_path(self, path: str | os.PathLike):
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint["state_dict"])
        self.metadata = checkpoint.get("config", {})

        type_name = self.metadata.get("type", None)
        if type_name is not None and type_name != self.__class__.__name__:
            logger.warning(
                f"Loaded model type '{type_name}' does not match current class '{self.__class__.__name__}'"
            )

    def try_load(self) -> bool:
        if os.path.exists(f"models/{self.file_name}"):
            self.load()
            return True
        return False

    def try_load_raw(self) -> bool:
        path = f"models/{self.raw_file_name}"
        if os.path.exists(path):
            self.load_from_path(path)
            return True
        return False

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
        if not target_class:
            raise ValueError(f"Subclass '{type_name}' not found.")

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

        return None


class PredictionTarget(StrEnum):
    Noise = "noise"
    x0 = "x0"
    Score = "score"
    Vecolcity = "velocity"

    @staticmethod
    def from_value(value: str) -> "PredictionTarget":
        for target in PredictionTarget:
            if target.value == value:
                return target
        raise ValueError(f"Unsupported PredictionTarget value: {value}")

    def to_hf(self) -> str:
        match self:
            case PredictionTarget.Noise:
                return "epsilon"
            case PredictionTarget.x0:
                return "v_prediction"
            case PredictionTarget.Vecolcity:
                return "v_prediction"
            case PredictionTarget.Score:
                raise NotImplementedError("Not supported")
            case _:
                raise ValueError(f"Unsupported PredictionTarget: {self}")


class Predictor(PersistableModule, ABC):
    timestep_config: TimestepConfig
    n_channels: int
    img_width: int
    img_height: int
    target: PredictionTarget
    normalize_x: (
        bool  # Whether to normalize the input x by s(t) before passing it to the model
    )

    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        x: torch.Tensor,
        timestep: Timestep,
        schedules: ScheduleGroup | None = None,
    ) -> torch.Tensor:
        if self.normalize_x:
            if schedules is None:
                raise ValueError("schedules must be provided when normalize_x is True")
            x = x / schedules.s(timestep).view(-1, 1, 1, 1)

        return self._forward(x=x, timestep=timestep, schedules=schedules)

    @abstractmethod
    def _forward(
        self,
        x: torch.Tensor,
        timestep: Timestep,
        schedules: ScheduleGroup | None = None,
    ) -> torch.Tensor:
        pass

    def loss_weight(
        self, t: Timestep, schedules: ScheduleGroup | None = None
    ) -> torch.Tensor:
        return torch.tensor(1.0, device=t.device)

    def save(
        self, path: str | os.PathLike | None = None, **extra_metadata
    ):
        super().save(
            path=path,
            T=self.timestep_config.T,
            n_channels=self.n_channels,
            img_width=self.img_width,
            img_height=self.img_height,
            target=self.target.value,
            normalize_x=self.normalize_x,
            **extra_metadata,
        )

    def load(self):
        super().load()
        meta_T = self.metadata.get("T", None)
        meta_n_channels = self.metadata.get("n_channels", None)
        meta_img_width = self.metadata.get("img_width", None)
        meta_img_height = self.metadata.get("img_height", None)
        meta_normalize_x = self.metadata.get("normalize_x", None)
        meta_target = self.metadata.get("target", None)

        if meta_T is not None and meta_T != self.timestep_config.T:
            logger.warning(
                f"Loaded model T '{meta_T}' does not match current timestep_config.T '{self.timestep_config.T}'"
            )

        for attr_name, meta_value, current_value in [
            ("n_channels", meta_n_channels, self.n_channels),
            ("img_width", meta_img_width, self.img_width),
            ("img_height", meta_img_height, self.img_height),
            ("target", meta_target, self.target.value if self.target else None),
            ("normalize_x", meta_normalize_x, self.normalize_x),
        ]:
            if meta_value is not None and meta_value != current_value:
                logger.warning(
                    f"Loaded model {attr_name} '{meta_value}' does not match current value '{current_value}'"
                )

    @classmethod
    def load_from_file(cls, file_path: str) -> "Predictor":
        instance = super().load_from_file(file_path)

        if not issubclass(instance.__class__, Predictor):
            raise TypeError(f"Loaded model from {file_path} is not a Predictor")

        return instance


def reparametrize(
    *,
    pred: torch.Tensor,
    x_t: torch.Tensor,
    t: Timestep,
    schedules: ScheduleGroup,
    src: PredictionTarget,
    dest: PredictionTarget,
) -> torch.Tensor:
    if src == dest:
        return pred

    alpha_t = schedules.alpha(t).view(-1, 1, 1, 1)
    sigma_t = schedules.sigma(t).view(-1, 1, 1, 1)
    edm_sigma_t = schedules.edm_sigma(t).view(-1, 1, 1, 1)

    if dest == PredictionTarget.Noise:
        if src == PredictionTarget.x0:
            return (x_t - alpha_t * pred) / torch.clamp(sigma_t, min=EPSILON)
        elif src == PredictionTarget.Score:
            return -edm_sigma_t * pred

    elif dest == PredictionTarget.x0:
        if src == PredictionTarget.Noise:
            return (x_t - sigma_t * pred) / torch.clamp(alpha_t, min=EPSILON)
        elif src == PredictionTarget.Score:
            return x_t / torch.clamp(alpha_t, min=EPSILON) + (edm_sigma_t**2) * pred

    elif dest == PredictionTarget.Score:
        if src == PredictionTarget.Noise:
            return -pred / torch.clamp(edm_sigma_t, min=EPSILON)
        elif src == PredictionTarget.x0:
            return (pred - x_t / torch.clamp(alpha_t, min=EPSILON)) / torch.clamp(
                edm_sigma_t**2, min=EPSILON
            )

    raise NotImplementedError(
        f"Reparametrization from {src} to {dest} not implemented."
    )
