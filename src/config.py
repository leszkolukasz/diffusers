from torchvision import datasets

from src.equation import GeneralizedDifferential, GeneralizedDiscrete
from src.schedule import (
    ConstantEtaSchedule,
    CosineAlphaSchedule,
    CosineSigmaSchedule,
    DDPMEtaSchedule,
    HuggingFaceDDPMAlphaSchedule,
    HuggingFaceDDPMSigmaSchedule,
    LinearAlphaSchedule,
    LinearSigmaSchedule,
)
from src.solver import (
    DiscreteSolver,
    EulerMaruyamaSDESolver,
    EulerODESolver,
    HeunODESolver,
)

BATCH_SIZE = 512
PREDICTOR_T = 1000
SOLVER_T = 1000

SOLVER_CONFIGS = {
    "discrete": DiscreteSolver,
    "euler": EulerODESolver,
    "heun": HeunODESolver,
    "euler_maruyama": EulerMaruyamaSDESolver,
}

SCHEDULE_CONFIGS = {
    "linear": {
        "alpha_schedule": LinearAlphaSchedule,
        "sigma_schedule": LinearSigmaSchedule,
    },
    "cosine": {
        "alpha_schedule": CosineAlphaSchedule,
        "sigma_schedule": CosineSigmaSchedule,
    },
    "ddpm": {
        "alpha_schedule": HuggingFaceDDPMAlphaSchedule,
        "sigma_schedule": HuggingFaceDDPMSigmaSchedule,
    },
}

ETA_CONFIGS = {
    "deterministic": lambda alpha, sigma: ConstantEtaSchedule(0.0),
    "stochastic": lambda alpha, sigma: ConstantEtaSchedule(1.0),
    "ddpm": lambda alpha, sigma: DDPMEtaSchedule(alpha, sigma),
}

EQUATION_CONFIGS = {
    "generalized_discrete": GeneralizedDiscrete,
    "generalized_differential": GeneralizedDifferential,
}

DATASET_CONFIGS = {
    "mnist": {
        "class": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
        "channels": 1,
        "img_width": 28,
        "img_height": 28,
    },
    "fashion": {
        "class": datasets.FashionMNIST,
        "mean": (0.5,),
        "std": (0.5,),
        "channels": 1,
        "img_width": 28,
        "img_height": 28,
    },
    "cifar10": {
        "class": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "channels": 3,
        "img_width": 32,
        "img_height": 32,
    },
    "celeb": {
        "class": datasets.CelebA,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "channels": 3,
        "img_width": 256,
        "img_height": 256,
    },
}

SOLVER_CONFIG_NAME = "discrete"
solver_config = SOLVER_CONFIGS[SOLVER_CONFIG_NAME]

SCHEDULE_CONFIG_NAME = "linear"
schedule_config = SCHEDULE_CONFIGS[SCHEDULE_CONFIG_NAME]

ETA_CONFIG_NAME = "stochastic"
eta_config = ETA_CONFIGS[ETA_CONFIG_NAME]

EQUATION_CONFIG_NAME = "generalized_discrete"
equation_config = EQUATION_CONFIGS[EQUATION_CONFIG_NAME]

DATASET_CONFIG_NAME = "mnist"
dataset_config = DATASET_CONFIGS[DATASET_CONFIG_NAME]
