from dataclasses import dataclass
from enum import Enum
from typing import Callable, Type

from torchvision import datasets
from torchvision.datasets import VisionDataset

from src.equation import (
    Equation,
    GeneralizedDifferential,
    GeneralizedDiscrete,
    ProbabilityFlow,
)
from src.model import Predictor, PredictorEDM, PredictorEDM2, PredictorUNet
from src.schedule import (
    AlphaSchedule,
    ConstantAlphaSchedule,
    ConstantEtaSchedule,
    CosineAlphaSchedule,
    CosineSigmaSchedule,
    DDPMEtaSchedule,
    EtaSchedule,
    HuggingFaceDDPMAlphaSchedule,
    HuggingFaceDDPMSigmaSchedule,
    LinearAlphaSchedule,
    LinearSigmaSchedule,
    SigmaSchedule,
)
from src.solver import (
    DiscreteSolver,
    EulerMaruyamaSDESolver,
    EulerODESolver,
    HeunODESolver,
    Solver,
)
from src.trainer import TimeSampler


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


class ModelType(str, Enum):
    edm = "edm"
    edm2 = "edm2"
    unet = "unet"
    huggingface = "huggingface"


@dataclass
class DatasetConfig:
    dataset_class: Type[VisionDataset]
    channels: int
    img_width: int
    img_height: int


@dataclass
class ScheduleConfig:
    alpha_schedule_factory: Callable[..., AlphaSchedule]
    sigma_schedule_factory: Callable[..., SigmaSchedule]


SOLVER_CONFIGS: dict[SolverType, Type[Solver]] = {
    SolverType.discrete: DiscreteSolver,
    SolverType.euler: EulerODESolver,
    SolverType.heun: HeunODESolver,
    SolverType.euler_maruyama: EulerMaruyamaSDESolver,
}

SCHEDULE_CONFIGS: dict[ScheduleType, ScheduleConfig] = {
    ScheduleType.linear: ScheduleConfig(
        alpha_schedule_factory=LinearAlphaSchedule,
        sigma_schedule_factory=LinearSigmaSchedule,
    ),
    ScheduleType.cosine: ScheduleConfig(
        alpha_schedule_factory=CosineAlphaSchedule,
        sigma_schedule_factory=CosineSigmaSchedule,
    ),
    ScheduleType.ddpm: ScheduleConfig(
        alpha_schedule_factory=HuggingFaceDDPMAlphaSchedule,
        sigma_schedule_factory=HuggingFaceDDPMSigmaSchedule,
    ),
    ScheduleType.edm: ScheduleConfig(
        alpha_schedule_factory=lambda **kwargs: ConstantAlphaSchedule(1.0),
        sigma_schedule_factory=lambda **kwargs: LinearSigmaSchedule(exploding=True),
    ),
}

ETA_CONFIGS: dict[EtaType, Callable[[AlphaSchedule, SigmaSchedule], EtaSchedule]] = {
    EtaType.deterministic: lambda alpha, sigma: ConstantEtaSchedule(0.0),
    EtaType.stochastic: lambda alpha, sigma: ConstantEtaSchedule(1.0),
    EtaType.ddpm: lambda alpha, sigma: DDPMEtaSchedule(alpha, sigma),
}

EQUATION_CONFIGS: dict[EquationType, Type[Equation]] = {
    EquationType.generalized_discrete: GeneralizedDiscrete,
    EquationType.generalized_differential: GeneralizedDifferential,
    EquationType.probability_flow: ProbabilityFlow,
}

DATASET_CONFIGS: dict[DatasetType, DatasetConfig] = {
    DatasetType.mnist: DatasetConfig(datasets.MNIST, 1, 28, 28),
    DatasetType.fashion: DatasetConfig(datasets.FashionMNIST, 1, 28, 28),
    DatasetType.cifar10: DatasetConfig(datasets.CIFAR10, 3, 32, 32),
    DatasetType.celeb: DatasetConfig(datasets.CelebA, 3, 256, 256),
    DatasetType.flowers: DatasetConfig(datasets.Flowers102, 3, 128, 128),
}

MODEL_CONFIGS: dict[ModelType, Type[Predictor]] = {
    ModelType.edm: PredictorEDM,
    ModelType.edm2: PredictorEDM2,
    ModelType.unet: PredictorUNet,
}


def get_timesampler(schedule_name: ScheduleType) -> TimeSampler:
    if schedule_name == ScheduleType.edm:
        return TimeSampler.EDM
    if schedule_name == ScheduleType.ddpm:
        return TimeSampler.UNIFORM_DISCRETE
    return TimeSampler.UNIFORM_CONTINUOUS


def get_solver_T(
    equation_name: EquationType, schedule_name: ScheduleType, predictor_t: int
) -> float:
    """Returns maximum time step T solver can be run with."""

    if equation_name == EquationType.generalized_discrete:
        return predictor_t
    if equation_name == EquationType.generalized_differential:
        return 1.0
    if equation_name == EquationType.probability_flow:
        return float(predictor_t) if schedule_name == ScheduleType.edm else 1.0
    raise ValueError(f"Unknown equation config name: {equation_name}")
