from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from torch.utils.data import Dataset
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
    logsnr_linear = "logsnr_linear"
    gauss_kl_optimal = "gauss_kl_optimal"
    gauss_klub_optimal = "gauss_klub_optimal"
    discrete = "discrete"
    custom = "custom"


class EtaType(str, Enum):
    deterministic = "deterministic"
    stochastic = "stochastic"
    ddpm = "ddpm"


class EquationType(str, Enum):
    generalized_discrete = "generalized_discrete"
    generalized_differential = "generalized_differential"
    probability_flow = "probability_flow"
    score_sde = "score_sde"


class DatasetType(str, Enum):
    mnist = "mnist"
    fashion = "fashion"
    cifar10 = "cifar10"
    cifar10_test = "cifar10_test"
    celeb = "celeb"
    flowers = "flowers"
    stl10 = "stl10"
    food101 = "food101"
    ays_toy = "ays_toy"
    pattern_spiral = "pattern_spiral"
    pattern_x = "pattern_x"


class ModelType(str, Enum):
    edm = "edm"
    edm2 = "edm2"
    unet = "unet"
    huggingface = "huggingface"


@dataclass
class DatasetConfig:
    dataset_class: type[Dataset | VisionDataset]
    channels: int
    img_size: int
    split: str | None = None
    path: str | None = None
    kwargs: dict | None = None


@dataclass
class ScheduleConfig:
    alpha_schedule_factory: Callable[..., "AlphaSchedule"]
    sigma_schedule_factory: Callable[..., "SigmaSchedule"]
