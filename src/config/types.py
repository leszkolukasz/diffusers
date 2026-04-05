from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Type

from torchvision.datasets import VisionDataset

if TYPE_CHECKING:
    from src.schedule import AlphaSchedule, SigmaSchedule


class SolverType(str, Enum):
    discrete = "discrete"
    euler = "euler"
    heun = "heun"
    euler_maruyama = "euler_maruyama"


class ScheduleType(str, Enum):
    linear = "linear"
    cosine = "cosine"
    ddpm = "ddpm"
    edm = "edm"


class SamplingScheduleType(str, Enum):
    linear = "linear"
    edm = "edm"
    discrete = "discrete"


class EtaType(str, Enum):
    deterministic = "deterministic"
    stochastic = "stochastic"
    ddpm = "ddpm"


class EquationType(str, Enum):
    generalized_discrete = "generalized_discrete"
    generalized_differential = "generalized_differential"
    probability_flow = "probability_flow"


class DatasetType(str, Enum):
    mnist = "mnist"
    fashion = "fashion"
    cifar10 = "cifar10"
    celeb = "celeb"
    flowers = "flowers"
    stl10 = "stl10"
    food101 = "food101"


class ModelType(str, Enum):
    edm = "edm"
    edm2 = "edm2"
    unet = "unet"
    huggingface = "huggingface"


@dataclass
class DatasetConfig:
    dataset_class: Type[VisionDataset]
    channels: int
    img_size: int
    split: str | None = None
    path: str | None = None


@dataclass
class ScheduleConfig:
    alpha_schedule_factory: Callable[..., "AlphaSchedule"]
    sigma_schedule_factory: Callable[..., "SigmaSchedule"]
