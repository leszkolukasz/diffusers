import os
from pathlib import Path
from typing import Any

import torch
import torchvision
import typer
from loguru import logger
from tqdm import tqdm

from src.config import (
    DatasetType,
    EquationType,
    EtaType,
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
from src.model import (
    VAE,
    Predictor,
    PredictorEDM2Pretrained,
    PredictorHuggingface,
    PredictorMetadata,
    dataset_from_metadata,
    schedule_from_metadata,
)
from src.model.presets import EDM2_PRETRAINED_PRESETS, ModelSize
from src.schedule import ScheduleGroup
from src.schedule.sampling import (
    EDMSamplingSchedule,
    GaussKLOptimalSamplingSchedule,
    GaussKLUBOptimalSamplingSchedule,
    LinearLogSNRSamplingSchedule,
    LinearSamplingSchedule,
    Timestep,
    TimestepConfig,
)

app = typer.Typer(help="Generate images")

TOY_DATASETS = (AYSToyDataset, PatternDataset)


def is_toy_dataset(dataset: DatasetType | None) -> bool:
    return (
        dataset is not None and DATASET_CONFIGS[dataset].dataset_class in TOY_DATASETS
    )


def parse_class_idx(value: str | None) -> int | str | None:
    if value is None:
        return None
    if value.lower() == "random":
        return "random"
    try:
        return int(value)
    except ValueError as error:
        raise typer.BadParameter(
            f"--class-idx must be an integer or 'random', got '{value}'."
        ) from error


def load_pretrained_edm2(
    preset_name: str, class_idx: int | str | None = None
) -> PredictorEDM2Pretrained:
    try:
        model_size = ModelSize.from_value(preset_name)
    except ValueError as error:
        available = ", ".join(size.value for size in EDM2_PRETRAINED_PRESETS)
        raise typer.BadParameter(
            f"Unknown pretrained EDM2 preset '{preset_name}'. Available: {available}"
        ) from error

    if model_size not in EDM2_PRETRAINED_PRESETS:
        available = ", ".join(size.value for size in EDM2_PRETRAINED_PRESETS)
        raise typer.BadParameter(
            f"'{preset_name}' is not a pretrained EDM2 preset. Available: {available}"
        )

    preset = EDM2_PRETRAINED_PRESETS[model_size]
    model = PredictorEDM2Pretrained(
        n_channels=int(preset["img_channels"]),
        img_width=int(preset["img_resolution"]),
        img_height=int(preset["img_resolution"]),
        T=1000,
        normalize_x=True,
        model_size=model_size,
        class_idx=class_idx,
    )
    model.load_pretrained()
    return model


def save_generation_output(
    generated: torch.Tensor,
    output_path: str | Path,
    dataset: DatasetType | None,
    metadata: dict[str, Any] | None = None,
    start_index: int = 0,
) -> Path:
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    if is_toy_dataset(dataset):
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
            out / f"{start_index + index + 1}.png",
        )

    return out


@app.callback(invoke_without_command=True)
def generate(
    model_path: str | None = typer.Option(
        None,
        "--model-path",
        help="Path to the local pre-trained weights (used when --model-id is not given).",
    ),
    model_id: str | None = typer.Option(
        None,
        "--model-id",
        help="HuggingFace repository ID. If given, a HuggingFace model is loaded.",
    ),
    pretrained: bool = typer.Option(
        False,
        "--pretrained",
        help="Load an official NVIDIA EDM2 checkpoint directly.",
    ),
    pretrained_preset: str = typer.Option(
        ModelSize.PRETRAINED_XS_IMG64.value,
        "--pretrained-preset",
        help="Official EDM2 preset to use with --pretrained.",
    ),
    class_idx: str | None = typer.Option(
        None,
        "--class-idx",
        help=(
            "ImageNet class id for a class-conditional pretrained EDM2 model, "
            "or 'random' to sample one class per image."
        ),
    ),
    schedule: ScheduleType | None = typer.Option(
        None,
        "--schedule",
        help="Signal/Noise schedule. Defaults to the one stored in the model metadata.",
    ),
    eta: EtaType = typer.Option(EtaType.deterministic, "--eta", help="Stochasticity"),
    solver_name: SolverType = typer.Option(
        SolverType.heun, "--solver", help="Discrete/ODE/SDE Solver"
    ),
    solver_T: float | None = typer.Option(
        None,
        "--solver-T",
        help="Maximum time step T solver can be run with. Defaults to the model's predictor T.",
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
    c: float = typer.Option(
        0.5,
        "--c",
        help="Standard deviation c for the Gaussian optimal schedules.",
    ),
    vae_low_memory: bool = typer.Option(
        False, "--vae-low-memory", help="Enable low memory mode for VAE"
    ),
    n_steps: int = typer.Option(100, "--n-steps", help="Number of solver steps"),
    n_samples: int = typer.Option(
        30, "--n-samples", help="Number of images to generate"
    ),
    batch_size: int = typer.Option(
        256, "--batch-size", help="Samples per generation batch (lower if OOM)."
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: generated/<solver>_<schedule>).",
    ),
):
    logger.info("Starting the generation")
    logger.info(f"Stochasticity (eta): {eta.value}")
    logger.info(f"Equation: {equation_name.value}")
    logger.info(f"Solver: {solver_name.value}")
    logger.info(f"Solver T: {solver_T if solver_T is not None else 'default'}")
    parsed_class_idx = parse_class_idx(class_idx)

    sources = int(bool(model_path)) + int(bool(model_id)) + int(pretrained)
    if sources > 1:
        logger.error(
            "Use only one model source: '--model-path', '--model-id', or '--pretrained'."
        )
        raise typer.Exit(code=1)
    if sources == 0:
        logger.error(
            "You must provide '--model-path', '--model-id', or '--pretrained'."
        )
        raise typer.Exit(code=1)

    if pretrained:
        logger.info(f"Loading pretrained EDM2 preset: {pretrained_preset}")
        model = load_pretrained_edm2(
            pretrained_preset, class_idx=parsed_class_idx
        ).cuda()
        if parsed_class_idx is None:
            logger.warning(
                "No --class-idx supplied for a class-conditional EDM2 checkpoint; "
                "using the all-zero label vector."
            )
        if schedule is None:
            schedule = ScheduleType.edm
        elif schedule != ScheduleType.edm:
            logger.warning(
                "Official EDM2 checkpoints were trained with the EDM schedule; "
                f"using --schedule {schedule.value} may give unexpected results."
            )
    elif model_id:
        logger.info(f"Loading HuggingFace model: {model_id}")
        model = PredictorHuggingface(model_id=model_id).cuda()
    else:
        if model_path is None:
            raise typer.Exit(code=1)
        logger.info(f"Loading local model from: {model_path}")
        model = Predictor.load_from_file(model_path).cuda()

    dataset = dataset_from_metadata(model.metadata)
    if dataset is not None:
        logger.info(f"Model dataset: {dataset.value}")

    metadata_schedule = schedule_from_metadata(model.metadata)
    if schedule is None:
        schedule = metadata_schedule or ScheduleType.edm
        if metadata_schedule is not None:
            logger.info(f"Using schedule from model metadata: {schedule.value}")

    logger.info(f"Signal/Noise Schedule: {schedule.value}")

    if solver_name == SolverType.stochastic_ddim and schedule != ScheduleType.edm:
        logger.error("Stochastic DDIM requires the EDM schedule.")
        raise typer.Exit(code=1)
    else:
        logger.info("Stochastic-DDIM solver ignores the passed equation")

    output_path = output_dir or f"generated/{solver_name.value}_{schedule.value}"
    os.makedirs(output_path, exist_ok=True)

    model.eval()
    logger.info(
        f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    if (
        equation_name in [EquationType.score_sde, EquationType.probability_flow]
        and not model.normalize_x
    ):
        logger.error(
            f"Model '{model.__class__.__name__}' is not trained to operate on "
            f"normalized x. This is required for the '{equation_name.value}' equation."
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
    solver_T = (
        get_solver_T(equation_name, schedule, predictor_T)
        if solver_T is None
        else solver_T
    )
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
        case (
            SamplingScheduleType.gauss_kl_optimal
            | SamplingScheduleType.gauss_klub_optimal
        ) as optimal_name:
            if schedule != ScheduleType.edm:
                logger.warning(
                    "The Gaussian optimal schedules assume s(t)=1 and sigma(t)=t "
                    "(EDM schedule)."
                )
            optimal_cls = (
                GaussKLOptimalSamplingSchedule
                if optimal_name == SamplingScheduleType.gauss_kl_optimal
                else GaussKLUBOptimalSamplingSchedule
            )
            generator_kwargs["timesteps"] = (
                optimal_cls(T=solver_T, c=c).get_timesteps(n_steps=n_steps).cuda()
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

    metadata = {
        "n_samples": n_samples,
        "model_path": f"pretrained:{pretrained_preset}"
        if pretrained
        else model_path,
        "class_idx": parsed_class_idx if pretrained else None,
        "solver": solver_name.value,
        "schedule": schedule.value,
        "sampling_schedule": sampling_schedule.value,
    }
    save_as_tensor = is_toy_dataset(dataset)

    logger.info(f"Generating {n_samples} samples in batches of {batch_size}...")
    stored_chunks: list[torch.Tensor] = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    progress = tqdm(total=n_batches, desc="Generating", unit="batch")
    try:
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                current = min(batch_size, n_samples - start)
                if (
                    isinstance(model, PredictorEDM2Pretrained)
                    and model.class_idx == "random"
                ):
                    model.set_random_class_labels(
                        current, device=next(model.parameters()).device
                    )
                chunk = generator.generate(
                    n_samples=current,
                    verbose=start == 0,
                    **generator_kwargs,
                )
                if save_as_tensor:
                    stored_chunks.append(chunk.cpu())
                else:
                    save_generation_output(
                        chunk, output_path, dataset, metadata, start_index=start
                    )

                progress.update(1)
                progress.set_postfix_str(
                    f"{min(start + current, n_samples)}/{n_samples} samples"
                )
    finally:
        progress.close()

    if save_as_tensor:
        generated = torch.cat(stored_chunks, dim=0)
        saved = save_generation_output(generated, output_path, dataset, metadata)
        logger.success(f"Saved generation output to {saved}")
    else:
        logger.success(f"Saved {n_samples} images to {output_path}")
