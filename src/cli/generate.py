import os
from pathlib import Path
from typing import Any

import torch
import torchvision
import typer
from loguru import logger

from src.config import (
    DatasetType,
    EquationType,
    EtaType,
    ModelType,
    SamplingScheduleType,
    ScheduleType,
    SolverType,
)
from src.config.presets import (
    DATASET_CONFIGS,
    EQUATION_CONFIGS,
    ETA_CONFIGS,
    SCHEDULE_CONFIGS,
    SOLVER_CONFIGS,
    get_solver_T,
)
from src.dataset import AYSToyDataset, PatternDataset, unnormalize
from src.generator import Generator
from src.model import VAE, Predictor, PredictorHuggingface, PredictorMetadata
from src.schedule import ScheduleGroup
from src.schedule.sampling import (
    EDMSamplingSchedule,
    LinearLogSNRSamplingSchedule,
    LinearSamplingSchedule,
    OptimalGaussianSamplingSchedule,
    Timestep,
    TimestepConfig,
)

app = typer.Typer(help="Generate images")

TOY_DATASETS = (AYSToyDataset, PatternDataset)


def is_toy_dataset(dataset: DatasetType | None) -> bool:
    return (
        dataset is not None and DATASET_CONFIGS[dataset].dataset_class in TOY_DATASETS
    )


def dataset_from_metadata(metadata: dict) -> DatasetType | None:
    value = metadata.get(PredictorMetadata.Dataset)
    if not value:
        return None
    try:
        return DatasetType(value)
    except ValueError:
        logger.warning(f"Unknown dataset in model metadata: {value}")
        return None


def save_generation_output(
    generated: torch.Tensor,
    output_path: str | Path,
    dataset: DatasetType | None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    is_pattern_shape = generated.ndim == 4 and generated.shape[1:] == (2, 1, 1)

    if is_toy_dataset(dataset) or is_pattern_shape:
        name = f"samples_{dataset.value}.pth" if dataset else "samples.pth"
        save_path = out / name
        torch.save(
            {
                "samples": generated.detach().cpu(),
                "dataset": dataset.value if dataset is not None else None,
                **(metadata or {}),
            },
            save_path,
        )
        return save_path

    for index in range(generated.shape[0]):
        torchvision.utils.save_image(
            unnormalize(generated[index]),
            out / f"{index + 1}.png",
        )

    return out


@app.callback(invoke_without_command=True)
def generate(
    model_name: ModelType = typer.Option(
        ModelType.edm,
        "--model",
        help="Network architecture",
    ),
    model_path: str | None = typer.Option(
        None, "--model-path", help="Path to the local pre-trained weights"
    ),
    model_id: str | None = typer.Option(
        None,
        "--model-id",
        help="HuggingFace repository ID (required if model=huggingface)",
    ),
    schedule: ScheduleType = typer.Option(
        ScheduleType.edm, "--schedule", help="Signal/Noise schedule"
    ),
    eta: EtaType = typer.Option(EtaType.deterministic, "--eta", help="Stochasticity"),
    solver_name: SolverType = typer.Option(
        SolverType.heun, "--solver", help="Discrete/ODE/SDE Solver"
    ),
    equation_name: EquationType = typer.Option(
        EquationType.probability_flow,
        "--equation",
        help="Reverse diffusion equation",
    ),
    sampling_schedule: SamplingScheduleType = typer.Option(
        SamplingScheduleType.edm, "--sampling-schedule", help="Time sampling"
    ),
    sampling_timesteps_path: str | None = typer.Option(
        None,
        "--sampling-timesteps-path",
        help="Path to custom sampling timesteps (AYS, ETS)",
    ),
    optimal_c: float = typer.Option(
        0.5,
        "--optimal-c",
        help="Standard deviation c for the optimal Gaussian sampling schedule.",
    ),
    vae_low_memory: bool = typer.Option(
        False, "--vae-low-memory", help="Enable low memory mode for VAE"
    ),
    n_steps: int = typer.Option(100, "--n-steps", help="Number of solver steps"),
    n_samples: int = typer.Option(
        30, "--n-samples", help="Number of images to generate"
    ),
):
    output_path = f"generated/{solver_name.value}_{schedule.value}"
    os.makedirs(output_path, exist_ok=True)

    logger.info("Starting the generation")
    logger.info(f"Model Architecture: {model_name.value}")
    logger.info(f"Signal/Noise Schedule: {schedule.value}")
    logger.info(f"Stochasticity (eta): {eta.value}")
    logger.info(f"Equation: {equation_name.value}")
    logger.info(f"Solver: {solver_name.value}")

    if model_name == ModelType.huggingface:
        if not model_id:
            logger.error(
                "You must provide '--model-id' when using the huggingface model."
            )
            raise typer.Exit(code=1)

        logger.info(f"Loading HuggingFace model: {model_id}")
        model = PredictorHuggingface(model_id=model_id).cuda()
    else:
        if not model_path:
            logger.error(
                "You must provide '--model-path' when using a local model type."
            )
            raise typer.Exit(code=1)

        logger.info(f"Loading local model from: {model_path}")
        model = Predictor.load_from_file(model_path).cuda()
        model.load()

    dataset = dataset_from_metadata(model.metadata)
    if dataset is not None:
        logger.info(f"Model dataset: {dataset.value}")

    model.eval()
    logger.info(
        f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    if (
        equation_name in [EquationType.score_sde, EquationType.probability_flow]
        and not model.normalize_x
    ):
        logger.error(
            f"Model '{model_name.value}' is not trained to operate on normalized x. This is required for the '{equation_name.value}' equation."
        )
        raise typer.Exit(code=1)

    schedule_kwargs = {}
    if model_id:
        schedule_kwargs["model_id"] = model_id

    alpha_schedule = SCHEDULE_CONFIGS[schedule].alpha_schedule_factory(
        **schedule_kwargs
    )
    sigma_schedule = SCHEDULE_CONFIGS[schedule].sigma_schedule_factory(
        **schedule_kwargs
    )
    eta_schedule = ETA_CONFIGS[eta](alpha_schedule, sigma_schedule)

    schedules = ScheduleGroup(
        alpha_schedule=alpha_schedule,
        sigma_schedule=sigma_schedule,
        eta_schedule=eta_schedule,
    )

    equation = EQUATION_CONFIGS[equation_name](
        model=model,
        schedules=schedules,
    )

    predictor_T = model.timestep_config.T
    solver_T = get_solver_T(equation_name, schedule, predictor_T)
    solver = SOLVER_CONFIGS[solver_name](
        equation=equation,
        T=solver_T,
    )

    vae_model_id = model.metadata.get(PredictorMetadata.VAE, None)
    vae = None
    if vae_model_id:
        logger.info(f"Using VAE: {vae_model_id}")
        vae = VAE(model_id=vae_model_id, low_memory=vae_low_memory).cuda()

    generator = Generator(solver=solver, vae=vae)

    logger.info(f"Generating {n_samples} samples...")

    variance_exploding = (
        schedules.sigma.exploding if hasattr(schedules.sigma, "exploding") else False
    )

    generator_kwargs: dict[str, Any] = {}
    generator_kwargs["variance_exploding"] = variance_exploding

    match sampling_schedule:
        case SamplingScheduleType.edm:
            generator_kwargs["timesteps"] = (
                EDMSamplingSchedule(T=solver_T).get_timesteps(n_steps=n_steps).cuda()
            )
        case SamplingScheduleType.linear:
            scheduler = LinearSamplingSchedule(T=solver_T)
            generator_kwargs["timesteps"] = scheduler.get_timesteps(  # , max_t = ...
                n_steps=n_steps
            ).cuda()
            logger.warning(f"Remember to set correct max_t: {scheduler.max_t}")
        case SamplingScheduleType.logsnr_linear:
            generator_kwargs["timesteps"] = (
                LinearLogSNRSamplingSchedule(schedules=schedules, T=solver_T)
                .get_timesteps(n_steps=n_steps)
                .cuda()
            )
        case SamplingScheduleType.optimal_gaussian:
            if schedule != ScheduleType.edm:
                logger.warning(
                    "The optimal Gaussian schedule assumes s(t)=1 and sigma(t)=t "
                    "(EDM schedule)."
                )
            generator_kwargs["timesteps"] = (
                OptimalGaussianSamplingSchedule(T=solver_T, c=optimal_c)
                .get_timesteps(n_steps=n_steps)
                .cuda()
            )
        case SamplingScheduleType.discrete:
            logger.info("n_steps is ignored. Using all steps supported by model.")
            generator_kwargs["skip_last_step"] = True
        case SamplingScheduleType.custom:
            if not sampling_timesteps_path:
                logger.error(
                    "You must provide '--sampling-timesteps-path' when using a custom sampling schedule."
                )
                raise typer.Exit(code=1)
            custom_schedule = torch.load(sampling_timesteps_path)["steps"]
            generator_kwargs["timesteps"] = Timestep(
                TimestepConfig("continuous", T=solver_T), custom_schedule
            ).cuda()
        case _:
            logger.error(f"Unsupported sampling schedule: {sampling_schedule.value}")
            raise typer.Exit(code=1)

    with torch.no_grad():
        generated = generator.generate(
            n_samples=n_samples,
            **generator_kwargs,
        )

    saved = save_generation_output(
        generated,
        output_path,
        dataset,
        metadata={
            "n_samples": n_samples,
            "model_path": model_path,
            "solver": solver_name.value,
            "schedule": schedule.value,
            "sampling_schedule": sampling_schedule.value,
        },
    )
    logger.success(f"Saved generation output to {saved}")
