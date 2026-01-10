import os
import sys

import torchvision
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.denoiser import (
    Denoiser,
    DiscreteDenoiser,
    EulerMaruyamaSDEDenoiser,
    EulerODEDenoiser,
    HeunODEDenoiser,
    HeunSDEDenoiser,
)
from src.generator import Generator
from src.model import NoisePredictor, NoisePredictorUNet
from src.sampling import AYSConfig, AYSSamplingSchedule
from src.schedule import (
    CosineAlphaSchedule,
    CosineSigmaSchedule,
    LinearAlphaSchedule,
    LinearSigmaSchedule,
    ScheduleGroup,
)
from src.trainer import Trainer

BATCH_SIZE = 512
MAX_T = 1000
NORM_MEAN = 0.5
NORM_STD = 0.5

DENOISER_CONFIGS = {
    "discrete": {
        "denoiser": DiscreteDenoiser,
    },
    "euler": {
        "denoiser": EulerODEDenoiser,
    },
    "heun": {
        "denoiser": HeunODEDenoiser,
    },
    "euler_maruyama": {
        "denoiser": EulerMaruyamaSDEDenoiser,
    },
    "heun_sde": {
        "denoiser": HeunSDEDenoiser,
    },
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
}

DENOISER_CONFIG_NAME = "euler"
denoiser_config = DENOISER_CONFIGS[DENOISER_CONFIG_NAME]

SCHEDULE_CONFIG_NAME = "cosine"
schedule_config = SCHEDULE_CONFIGS[SCHEDULE_CONFIG_NAME]

unnormalize = transforms.Normalize((-NORM_MEAN / NORM_STD,), (1.0 / NORM_STD,))


def get_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=2):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((NORM_MEAN,), (NORM_STD,))]
    )

    data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


def train():
    logger.info("Starting training")

    logger.info(f"Using schedule config: {SCHEDULE_CONFIG_NAME}")
    schedules = ScheduleGroup(
        alpha_schedule=schedule_config["alpha_schedule"](),  # ty: ignore
        sigma_schedule=schedule_config["sigma_schedule"](),  # ty: ignore
    )

    logger.info(f"Using MAX_T: {MAX_T}")
    model = NoisePredictorUNet(max_t=MAX_T, suffix="test").cuda()
    model.try_load()

    trainer = Trainer(
        model=model,
        schedules=schedules,
    )
    trainer.load_checkpoint()

    dataloader = get_dataloader()
    trainer.train(dataloader)

    logger.success("Training complete")


def test():
    if not os.path.exists(f"generated/{DENOISER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}"):
        os.makedirs(f"generated/{DENOISER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}")

    # model: PersistableModule = NoisePredictorUNet(max_steps=MAX_T).cuda()
    # model.load()
    model = NoisePredictor.load_from_file(
        "./models/noise_predictor_unettest.pth"
    ).cuda()
    model.eval()

    logger.info(f"Using schedule config: {SCHEDULE_CONFIG_NAME}")
    schedules = ScheduleGroup(
        alpha_schedule=schedule_config["alpha_schedule"](),  # ty: ignore
        sigma_schedule=schedule_config["sigma_schedule"](),  # ty: ignore
    )

    logger.info(f"Using denoiser config: {DENOISER_CONFIG_NAME}")
    denoiser: Denoiser = denoiser_config["denoiser"](
        model=model,
        schedules=schedules,
    )

    generator = Generator(
        denoiser=denoiser,
        n_channels=1,
        img_width=28,
        img_height=28,
    )

    n_samples = 16
    generated = generator.generate(n_samples=n_samples)

    for i in range(n_samples):
        img = generated[i]
        torchvision.utils.save_image(
            img, f"generated/{DENOISER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}/{i + 1}.png"
        )


def ays():
    model = NoisePredictorUNet(max_t=MAX_T).cuda()
    model.eval()

    logger.info(f"Using schedule config: {SCHEDULE_CONFIG_NAME}")
    schedules = ScheduleGroup(
        alpha_schedule=schedule_config["alpha_schedule"](),  # ty: ignore
        sigma_schedule=schedule_config["sigma_schedule"](),  # ty: ignore
    )

    logger.info(f"Using denoiser config: {DENOISER_CONFIG_NAME}")
    denoiser: Denoiser = denoiser_config["denoiser"](
        model=model,
        schedules=schedules,
    )

    ays_schedule = AYSSamplingSchedule(
        denoiser=denoiser,
        dataloader=get_dataloader(batch_size=64, shuffle=False),
        config=AYSConfig(max_iter=1),
    )

    timesteps = ays_schedule.get_timesteps()
    print(timesteps)


if __name__ == "__main__":
    mode = "train"

    if len(sys.argv) == 0:
        raise ValueError("Please provide a mode: train, test, ays")

    mode = sys.argv[1]

    match mode:
        case "train":
            train()
        case "test":
            test()
        case "ays":
            ays()
        case _:
            logger.error(f"Unknown mode: {mode}")
