import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from loguru import logger

from src.common import safe_filename_part
from src.config import (
    DatasetType,
    EquationType,
    EtaType,
    SamplingScheduleType,
    ScheduleType,
)
from src.config.presets import (
    DATASET_CONFIGS,
    ETA_CONFIGS,
    SCHEDULE_CONFIGS,
    get_solver_T,
)
from src.dataset import get_dataloader
from src.model import (
    VAE,
    AnalyticalGaussianPredictor,
    Predictor,
    PredictorHuggingface,
    PredictorMetadata,
    dataset_from_metadata,
    schedule_from_metadata,
)
from src.schedule import ScheduleGroup
from src.schedule.sampling import (
    EDMSamplingSchedule,
    GaussKLOptimalSamplingSchedule,
    GaussKLUBOptimalSamplingSchedule,
    LinearLogSNRSamplingSchedule,
    LinearSamplingSchedule,
    QuadraticSamplingSchedule,
)
from src.schedule.sampling.ays import (
    AYSConfig,
    AYSSamplingSchedule,
    CandidateSpacing,
)

app = typer.Typer(help="Align Your Steps")


def build_baseline_steps(
    comparison: list[SamplingScheduleType],
    *,
    schedules: ScheduleGroup | None = None,
    n_steps: int,
    max_t: float,
    c: float = 0.5,
) -> dict[str, np.ndarray]:
    baselines: dict[str, np.ndarray] = {}

    for name in comparison:
        if name == SamplingScheduleType.edm:
            schedule = EDMSamplingSchedule(T=max_t)
        elif name == SamplingScheduleType.linear:
            schedule = LinearSamplingSchedule(T=max_t, max_t=max_t)
        elif name == SamplingScheduleType.quadratic:
            schedule = QuadraticSamplingSchedule(T=max_t, max_t=max_t)
        elif name == SamplingScheduleType.logsnr_linear:
            if schedules is None:
                raise ValueError("schedules are required for logsnr_linear")
            schedule = LinearLogSNRSamplingSchedule(
                schedules=schedules, T=max_t, max_t=max_t
            )
        elif name in (
            SamplingScheduleType.gauss_kl_optimal,
            SamplingScheduleType.gauss_klub_optimal,
        ):
            optimal_cls = (
                GaussKLOptimalSamplingSchedule
                if name == SamplingScheduleType.gauss_kl_optimal
                else GaussKLUBOptimalSamplingSchedule
            )
            schedule = optimal_cls(T=max_t, max_t=max_t, c=c)
        else:
            raise ValueError(f"Unsupported baseline comparison: {name.value}")

        steps = schedule.get_timesteps(n_steps=n_steps).steps.cpu().numpy()[::-1]
        baselines[name.value] = steps

    return baselines


@app.command(name="tune", help="Find optimal steps")
def tune(
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
    ideal_gaussian: bool = typer.Option(
        False,
        "--ideal-gaussian",
        help="Use the analytical Gaussian denoiser instead of a checkpoint.",
    ),
    c: float = typer.Option(
        0.5,
        "--c",
        help="Standard deviation c of the Gaussian data",
    ),
    dataset: DatasetType | None = typer.Option(
        None,
        "--dataset",
        help="Target dataset. Defaults to the one stored in the model metadata.",
    ),
    schedule: ScheduleType | None = typer.Option(
        None,
        "--schedule",
        help="Signal/Noise schedule. Defaults to the one stored in the model metadata.",
    ),
    eta: EtaType = typer.Option(EtaType.deterministic, "--eta", help="Stochasticity"),
    equation_name: EquationType = typer.Option(
        EquationType.probability_flow, "--equation", help="Reverse diffusion equation"
    ),
    batch_size: int = typer.Option(128, "--batch-size", help="Batch size"),
    initial_sampling_schedule: SamplingScheduleType = typer.Option(
        SamplingScheduleType.edm,
        "--initial-schedule",
        help="Initial time sampling",
    ),
    max_t: float = typer.Option(
        80.0,
        "--max-t",
        help="Maximum noise/time level of the initial sampling schedule",
    ),
    checkpoint_path: str | None = typer.Option(
        None, "--checkpoint-path", help="Path to resume AYS tuning from a checkpoint"
    ),
    prefix: str | None = typer.Option(None, "--prefix", help="Prefix for checkpoint"),
    suffix: str | None = typer.Option(None, "--suffix", help="Suffix for checkpoint"),
    max_iter: int = typer.Option(
        32, "--max-iter", help="Max iterations for 10-step stage"
    ),
    max_finetune_iter: int = typer.Option(
        16, "--max-finetune-iter", help="Max fine tune iterations for 20/40-step stage"
    ),
    n_candidates: int = typer.Option(
        11, "--n-candidates", help="Number of timestep candidates"
    ),
    candidate_spacing: CandidateSpacing = typer.Option(
        CandidateSpacing.linear,
        "--candidate-spacing",
        help="Candidate spacing within each interval: linear or log in t.",
    ),
    n_monte_carlo_iter: int = typer.Option(
        4096, "--n-mc-iter", help="Number of Monte Carlo steps"
    ),
    save_interval_iter: int = typer.Option(
        1, "--save-interval", help="Checkpoint frequency"
    ),
    importance_sampling: bool = typer.Option(
        True,
        "--importance-sampling/--no-importance-sampling",
        help="Use importance sampling for AYS",
    ),
    its_grid_size: int = typer.Option(
        2000, "--its-grid-size", help="Grid size for Inverse Transform Sampling"
    ),
    vae_low_memory: bool = typer.Option(
        False, "--vae-low-memory", help="Enable low memory mode for VAE"
    ),
    n_workers: int = typer.Option(
        1, "--n-workers", help="Number of dataloader workers"
    ),
):
    logger.info("Starting the AYS Generation")
    logger.info(f"Equation: {equation_name.value}")
    logger.info(f"Initial Sampling Schedule: {initial_sampling_schedule.value}")

    if ideal_gaussian and (model_path or model_id):
        logger.error(
            "--ideal-gaussian cannot be combined with --model-path/--model-id."
        )
        raise typer.Exit(code=1)
    if not ideal_gaussian and not (model_path or model_id):
        logger.error("Provide --model-path, --model-id, or --ideal-gaussian.")
        raise typer.Exit(code=1)

    if ideal_gaussian:
        if dataset != DatasetType.ays_toy:
            logger.warning("Ignoring --dataset since --ideal-gaussian is used.")

        dataset = DatasetType.ays_toy
        shape_config = DATASET_CONFIGS[dataset]
        model = AnalyticalGaussianPredictor(
            c=c,
            n_channels=shape_config.channels,
            img_width=shape_config.img_size,
            img_height=shape_config.img_size,
        )
        logger.info(f"Using analytical Gaussian denoiser (c={c:g}).")
    elif model_id:
        logger.info(f"Loading HuggingFace model: {model_id}")
        model = PredictorHuggingface(model_id=model_id).cuda()
    else:
        if not model_path:
            logger.error("You must provide either '--model-path' or '--model-id'.")
            raise typer.Exit(code=1)
        logger.info(f"Loading local model from: {model_path}")
        model = Predictor.load_from_file(model_path).cuda()
        model.load()

    metadata_schedule = schedule_from_metadata(model.metadata)
    if schedule is None:
        schedule = metadata_schedule or ScheduleType.edm
        if metadata_schedule is not None:
            logger.info(f"Using schedule from model metadata: {schedule.value}")

    if ideal_gaussian and schedule != ScheduleType.edm:
        logger.error("--ideal-gaussian assumes the EDM schedule (s(t)=1, sigma(t)=t).")
        raise typer.Exit(code=1)

    logger.info(f"Signal/Noise Schedule: {schedule.value}")

    metadata_dataset = (
        dataset_from_metadata(model.metadata) if not ideal_gaussian else None
    )

    if dataset is None:
        if metadata_dataset is None:
            logger.warning(
                "Model metadata has no dataset; falling back to default dataset. "
            )
        dataset = metadata_dataset or DatasetType.flowers
        if metadata_dataset is not None:
            logger.info(f"Using dataset from model metadata: {dataset.value}")

    dataset_config = DATASET_CONFIGS[dataset]

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

    predictor_T = model.timestep_config.T
    solver_T = get_solver_T(equation_name, schedule, predictor_T)

    vae_model_id = model.metadata.get(PredictorMetadata.VAE, None)
    vae = None
    if vae_model_id:
        logger.info(f"Using VAE: {vae_model_id}")
        vae = VAE(model_id=vae_model_id, low_memory=vae_low_memory).cuda()

    if initial_sampling_schedule == SamplingScheduleType.edm:
        initial_t = (
            EDMSamplingSchedule(T=solver_T, sigma_max=max_t)
            .get_timesteps(n_steps=10)
            .cuda()
        )
    elif initial_sampling_schedule == SamplingScheduleType.linear:
        initial_t = (
            LinearSamplingSchedule(T=solver_T, max_t=max_t)
            .get_timesteps(n_steps=10)
            .cuda()
        )
    elif initial_sampling_schedule == SamplingScheduleType.quadratic:
        initial_t = (
            QuadraticSamplingSchedule(T=solver_T, max_t=max_t)
            .get_timesteps(n_steps=10)
            .cuda()
        )
    elif initial_sampling_schedule == SamplingScheduleType.logsnr_linear:
        initial_t = (
            LinearLogSNRSamplingSchedule(
                schedules=schedules, T=solver_T, max_t=max_t
            )
            .get_timesteps(n_steps=10)
            .cuda()
        )
    elif initial_sampling_schedule in (
        SamplingScheduleType.gauss_kl_optimal,
        SamplingScheduleType.gauss_klub_optimal,
    ):
        optimal_cls = (
            GaussKLOptimalSamplingSchedule
            if initial_sampling_schedule == SamplingScheduleType.gauss_kl_optimal
            else GaussKLUBOptimalSamplingSchedule
        )
        initial_t = (
            optimal_cls(T=solver_T, c=c, max_t=max_t)
            .get_timesteps(n_steps=10)
            .cuda()
        )
    else:
        raise ValueError(
            f"Unsupported initial schedule: {initial_sampling_schedule.value}"
        )

    model_stem = (
        safe_filename_part(f"analytical_gaussian_c{c:g}")
        if ideal_gaussian
        else (
            Path(model_path).stem
            if model_path
            else (model_id or "model").replace("/", "_")
        )
    )
    parts = [
        "ays",
        model_stem,
        initial_sampling_schedule.value,
    ]
    if prefix:
        parts.insert(0, prefix)
    if suffix:
        parts.append(suffix)

    save_file_path = checkpoint_path or (
        "generated/" + "_".join(safe_filename_part(part) for part in parts) + ".pth"
    )
    logger.info(f"AYS checkpoint will be saved to: {save_file_path}")
    ays_config = AYSConfig(
        max_iter=max_iter,
        max_finetune_iter=max_finetune_iter,
        n_candidates=n_candidates,
        n_monte_carlo_iter=n_monte_carlo_iter,
        save_interval_iter=save_interval_iter,
        importance_sampling=importance_sampling,
        inverse_transform_sampling_grid_size=its_grid_size,
        candidate_spacing=candidate_spacing,
        save_file=save_file_path,
    )

    ays_schedule = AYSSamplingSchedule(
        max_t=max_t,
        model=model,
        schedules=schedules,
        solver_T=solver_T,
        equation_type=equation_name,
        vae=vae,
        dataloader=get_dataloader(
            batch_size=batch_size,
            dataset_config=dataset_config,
            num_workers=n_workers,
        ),
        config=ays_config,
    )

    logger.info("Starting AYS tuning")

    timesteps = ays_schedule.get_timesteps(n_steps=40, initial_t=initial_t)

    logger.info(f"Final AYS schedule: {timesteps.steps}")


@app.command(name="plot", help="Plot generated AYS schedule")
def plot(
    checkpoint_path: str = typer.Argument(..., help="Path to checkpoint"),
    compare_with: list[SamplingScheduleType] = typer.Option(
        [],
        "--compare",
        help="Baseline(s) to compare. Repeat the option for multiple baselines.",
    ),
    schedule: ScheduleType = typer.Option(
        ScheduleType.edm,
        "--schedule",
        help="Noise schedule used to build the logSNR baseline.",
    ),
    c: float = typer.Option(
        0.5,
        "--c",
        help="Standard deviation c for the Gaussian optimal baselines.",
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output plot path (default: descriptive name)."
    ),
):
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise typer.Exit(code=1)

    state = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    ays_steps = state["steps"].numpy()[::-1]
    logger.info(f"AYS steps: {ays_steps}")

    n_steps = len(ays_steps) - 1
    max_t = ays_steps[-1]
    stage = state.get("stage", "Unknown")

    logger.info(
        f"Found AYS schedule for stage {stage} with {n_steps} steps (T={max_t:.2f})"
    )

    comparison = compare_with or [SamplingScheduleType.edm]

    schedules = None
    if SamplingScheduleType.logsnr_linear in comparison:
        alpha = SCHEDULE_CONFIGS[schedule].alpha_schedule_factory()
        sigma = SCHEDULE_CONFIGS[schedule].sigma_schedule_factory()
        eta = ETA_CONFIGS[EtaType.deterministic](alpha, sigma)
        schedules = ScheduleGroup(alpha, sigma, eta)

    try:
        baselines = build_baseline_steps(
            comparison,
            schedules=schedules,
            n_steps=n_steps,
            max_t=max_t,
            c=c,
        )
    except ValueError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is None:
        checkpoint_stem = safe_filename_part(Path(checkpoint_path).stem, max_length=180)
        parts = [
            "ays_plot",
            checkpoint_stem,
            "vs",
            safe_filename_part("-".join(baselines)),
        ]
        output = "_".join(parts) + ".png"

    plt.figure(figsize=(10, 6))

    colormap = plt.get_cmap("tab10")
    markers = ["s", "^", "d", "v", "P", "X", "*"]
    for index, (name, steps) in enumerate(baselines.items()):
        plt.plot(
            range(len(steps)),
            steps,
            marker=markers[index % len(markers)],
            linestyle="--",
            color=colormap((index + 1) % 10),
            alpha=0.8,
            label=f"Baseline ({name.upper()})",
        )

    plt.plot(
        range(len(ays_steps)),
        ays_steps,
        marker="o",
        linestyle="-",
        color="blue",
        linewidth=2,
        label=f"AYS (Stage {stage})",
    )

    plt.title("AYS vs " + ", ".join(name.upper() for name in baselines))
    plt.xlabel("Step Index")
    plt.ylabel("Timestep (t)")

    # plt.yscale("symlog", linthresh=1e-3)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output, dpi=200)
    logger.success(f"Saved AYS plot to {output}")


if __name__ == "__main__":
    app()
