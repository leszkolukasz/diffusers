from collections.abc import Callable

from torchvision import datasets

from src.config import (
    DatasetConfig,
    DatasetType,
    EquationType,
    EtaType,
    ModelType,
    ScheduleConfig,
    ScheduleType,
    SolverType,
)
from src.dataset import AYSToyDataset, PatternDataset
from src.equation import (
    Equation,
    GeneralizedDifferential,
    GeneralizedDiscrete,
    ProbabilityFlow,
    ScoreSDE,
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
    StochasticDDIMSolver,
)
from src.train import TimeSampler

SOLVER_CONFIGS: dict[SolverType, type[Solver]] = {
    SolverType.discrete: DiscreteSolver,
    SolverType.euler: EulerODESolver,
    SolverType.heun: HeunODESolver,
    SolverType.euler_maruyama: EulerMaruyamaSDESolver,
    SolverType.stochastic_ddim: StochasticDDIMSolver,
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

EQUATION_CONFIGS: dict[EquationType, type[Equation]] = {
    EquationType.generalized_discrete: GeneralizedDiscrete,
    EquationType.generalized_differential: GeneralizedDifferential,
    EquationType.probability_flow: ProbabilityFlow,
    EquationType.score_sde: ScoreSDE,
}

N_SAMPLES = 10000

# Image sizes should be divisible by 8.
DATASET_CONFIGS: dict[DatasetType, DatasetConfig] = {
    DatasetType.mnist: DatasetConfig(datasets.MNIST, 1, 32),
    DatasetType.fashion: DatasetConfig(datasets.FashionMNIST, 1, 32),
    DatasetType.cifar10: DatasetConfig(datasets.CIFAR10, 3, 32),
    DatasetType.cifar10_test: DatasetConfig(
        datasets.CIFAR10, 3, 32, kwargs={"train": False}
    ),
    DatasetType.celeb: DatasetConfig(datasets.CelebA, 3, 256, path="data/celebA"),
    DatasetType.flowers: DatasetConfig(datasets.Flowers102, 3, 64),
    DatasetType.stl10: DatasetConfig(datasets.STL10, 3, 128, split="unlabeled"),
    DatasetType.food101: DatasetConfig(datasets.Food101, 3, 128, split="train"),
    DatasetType.ays_toy: DatasetConfig(
        AYSToyDataset,
        1,
        32,
        kwargs={"num_samples": N_SAMPLES},
    ),
    DatasetType.pattern_spiral: DatasetConfig(
        PatternDataset,
        2,
        1,
        kwargs={"num_samples": N_SAMPLES, "pattern_path": "assets/spiral.png"},
    ),
    DatasetType.pattern_x: DatasetConfig(
        PatternDataset,
        2,
        1,
        kwargs={"num_samples": N_SAMPLES, "pattern_path": "assets/x.png"},
    ),
}

MODEL_CONFIGS: dict[ModelType, type[Predictor]] = {
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
    equation_name: EquationType,
    schedule_name: ScheduleType,
    predictor_t: float,
) -> float:
    """Returns maximum time step T solver can be run with."""
    is_discrete = equation_name == EquationType.generalized_discrete

    if schedule_name == ScheduleType.edm or is_discrete:
        val = predictor_t
    elif equation_name in (
        EquationType.probability_flow,
        EquationType.score_sde,
        EquationType.generalized_differential,
    ):
        val = 1.0
    else:
        raise ValueError(f"Unknown equation config name: {equation_name}")

    return int(val) if is_discrete else float(val)
