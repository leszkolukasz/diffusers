import os

import typer
from loguru import logger

import wandb
from src import distributed
from src.config import (
    DatasetType,
    EtaType,
    ModelType,
    ScheduleType,
)
from src.config.presets import (
    DATASET_CONFIGS,
    ETA_CONFIGS,
    MODEL_CONFIGS,
    SCHEDULE_CONFIGS,
    get_timesampler,
)
from src.dataset import get_dataloader
from src.model import VAE, PredictionTarget
from src.model.presets import ModelSize
from src.schedule import ScheduleGroup
from src.train import Trainer, TrainingConfig
from src.train.checkpoint import resolve_snapshot_paths

app = typer.Typer(help="Train model")


@app.callback(invoke_without_command=True)
def train(
    run_id: str = typer.Option(None, "--run-id", help="Run id suffix for Wandb"),
    model_name: ModelType = typer.Option(
        ModelType.edm, "--model", help="Network architecture"
    ),
    model_id: str = typer.Option(
        None,
        "--model-id",
        help="HuggingFace repository ID (required if schedule=ddpm)",
    ),
    model_size: ModelSize = typer.Option(
        ModelSize.SMALL, "--model-size", help="Model size (number of parameters)"
    ),
    model_prefix: str = typer.Option(
        "", "--model-prefix", help="Prefix to prepend to model checkpoints"
    ),
    model_suffix: str = typer.Option(
        "", "--model-suffix", help="Suffix to append to model checkpoints"
    ),
    dataset: DatasetType = typer.Option(
        DatasetType.flowers, "--dataset", help="Target dataset for training"
    ),
    schedule: ScheduleType = typer.Option(
        ScheduleType.edm, "--schedule", help="Signal/Noise schedule"
    ),
    target: PredictionTarget | None = typer.Option(
        None,
        "--target",
        help="Prediction target",
    ),
    eta: EtaType = typer.Option(
        EtaType.deterministic,
        "--eta",
        help="Stochasticity",
    ),
    batch_size: int = typer.Option(128, "--batch-size", help="Batch size"),
    predictor_t: int = typer.Option(
        1000, "--predictor-t", help="Max time step predictor is trained on"
    ),
    n_epochs: int = typer.Option(1000, "--n-epochs", help="Number of epochs"),
    checkpoint_interval: int = typer.Option(
        100, "--checkpoint-interval", help="save a checkpoint every N steps"
    ),
    checkpoint_snapshot_percent: float = typer.Option(
        5.0,
        "--checkpoint-snapshot-percent",
        help="Save full snapshots every N% of training steps (0 disables).",
    ),
    resume_from: str | None = typer.Option(
        None,
        "--resume-from",
        help="Resume from a specific checkpoint/snapshot model, raw model, or trainer file.",
    ),
    use_ema: bool = typer.Option(
        True,
        "--ema/--no-ema",
        help="Whether to use exponential moving average for model weights",
    ),
    ema_halflife_ratio: float = typer.Option(
        0.05,
        "--ema-halflife-ratio",
        help="EMA half-life as a fraction of the total training steps",
    ),
    ema_warmup_ratio: float = typer.Option(
        0.05,
        "--ema-warmup-ratio",
        help="EMA warmup as a fraction of the steps seen so far (0 disables)",
    ),
    use_amp: bool = typer.Option(
        True,
        "--amp/--no-amp",
        help="Whether to use automatic mixed precision training",
    ),
    use_vae: bool = typer.Option(False, "--vae/--no-vae", help="Whether to use VAE"),
    use_x_norm: bool = typer.Option(
        False,
        "--x-norm/--no-x-norm",
        help="Whether to normalize the input x by s(t) before passing it to the model",
    ),
    vae_model_id: str = typer.Option(
        "madebyollin/taesd", "--vae-model-id", help="HF repo ID for the VAE"
    ),
    use_bf16: bool = typer.Option(
        False,
        "--bf16/--no-bf16",
        help="Whether to use bfloat16 precision",
    ),
    uncertainty_loss: bool = typer.Option(
        False,
        "--uncertainty-loss/--no-uncertainty-loss",
        help="Use EDM2 uncertainty loss with a learned log-variance head.",
    ),
    n_workers: int = typer.Option(
        1, "--n-workers", help="Number of dataloader workers"
    ),
    enable_wandb: bool = typer.Option(
        False,
        "--wandb/--no-wandb",
        help="Whether to enable WandB logging (automatic if running in a distributed environment)",
    ),
):
    dataset_config = DATASET_CONFIGS[dataset]
    model_class = MODEL_CONFIGS[model_name]
    timesampler_config = get_timesampler(schedule)

    logger.info("Starting the training")
    logger.info(f"Model architecture: {model_name.value}")
    logger.info(f"Model size: {model_size.value}")
    logger.info(f"Dataset: {dataset.value}")
    logger.info(f"Signal/Noise schedule: {schedule.value}")
    logger.info(
        f"Prediction target: {target.value if target is not None else 'model default'}"
    )
    logger.info(f"Stochasticity (eta): {eta.value}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Predictor T: {predictor_t}")
    logger.info(f"Number of epochs: {n_epochs}")
    logger.info(f"Timesampler: {timesampler_config.value}")
    logger.info(
        f"Full snapshot interval: {checkpoint_snapshot_percent}% of training steps"
    )
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")

    if use_ema:
        if not 0.0 < ema_halflife_ratio <= 1.0:
            logger.error("--ema-halflife-ratio must be in (0, 1].")
            raise typer.Exit(code=1)
        if not 0.0 <= ema_warmup_ratio <= 1.0:
            logger.error("--ema-warmup-ratio must be in [0, 1].")
            raise typer.Exit(code=1)
        logger.info(
            f"EMA halflife ratio: {ema_halflife_ratio}, "
            f"warmup ratio: {ema_warmup_ratio}"
        )

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

    if distributed.get_rank() == 0 and (enable_wandb or distributed.is_distributed()):
        run_identifier = f"{model_name.value}_{schedule.value}_{dataset.value}"
        if run_id:
            run_identifier = f"{run_id}_" + run_identifier

        wandb.init(
            project="prosem",
            name=run_identifier,
            id=run_identifier,
            resume="allow",
            config={
                "model": model_name.value,
                "batch_size": batch_size,
                "predictor_T": predictor_t,
                "dataset": dataset.value,
                "schedule": schedule.value,
                "eta": eta.value,
                "target": target.value if target is not None else "model_default",
                "model_prefix": model_prefix,
                "ema_halflife_ratio": ema_halflife_ratio,
                "ema_warmup_ratio": ema_warmup_ratio,
            },
        )

    vae = None
    if use_vae:
        logger.info(f"Using VAE: {vae_model_id}")
        vae = VAE(model_id=vae_model_id).cuda()
        logger.info(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")

    img_width = dataset_config.img_size
    img_height = dataset_config.img_size
    n_channels = dataset_config.channels
    logger.info(f"Image size: {n_channels}x{img_width}x{img_height}")

    if vae is not None:
        img_width = vae.get_latent_width(img_width)
        img_height = vae.get_latent_height(img_height)
        n_channels = vae.get_latent_channels()
        logger.info(f"Latent size: {n_channels}x{img_width}x{img_height}")

    if img_width == 1 and img_height == 1:
        if model_name in (ModelType.edm2, ModelType.edm2_pretrained):
            logger.error(
                "EDM2 does not support 1x1 inputs. Use --model edm or --model unet."
            )
            raise typer.Exit(code=1)

        if model_size != ModelSize.POINT:
            logger.info("1x1 inputs require the 'point' model size; using it.")
        model_size = ModelSize.POINT

    if target == PredictionTarget.Score and not use_x_norm:
        logger.error(
            "Score prediction requires input normalization. Use --x-norm to enable it."
        )
        raise typer.Exit(code=1)

    if use_x_norm:
        logger.info("Enabling input normalization")
    else:
        logger.warning(
            "x normalization is disabled. This means this model will not work with ScoreSDE and PF-ODE"
        )

    model_kwargs = {
        "T": predictor_t,
        "model_size": model_size,
        "n_channels": n_channels,
        "img_width": img_width,
        "img_height": img_height,
        "normalize_x": use_x_norm,
        "prefix": model_prefix or None,
        "suffix": f"_{dataset.value}_{model_suffix}"
        if model_suffix
        else f"_{dataset.value}",
    }
    if target is not None:
        model_kwargs["target"] = target

    model = model_class(**model_kwargs).cuda()

    logger.info(f"Model target: {model.target.value}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    resume_trainer_path = None
    if resume_from:
        resume_raw_path, resume_trainer_path, _ = resolve_snapshot_paths(resume_from)
        if not os.path.exists(resume_raw_path):
            logger.error(f"Raw model for resume not found: {resume_raw_path}")
            raise typer.Exit(code=1)
        if not os.path.exists(resume_trainer_path):
            logger.error(f"Trainer state for resume not found: {resume_trainer_path}")
            raise typer.Exit(code=1)
        model._load_from_path(resume_raw_path)
        logger.info(f"Loaded raw model for resume: {resume_raw_path}")
    elif model.try_load_raw():
        logger.info(f"Loaded raw model for resume: models/{model.raw_file_name}")
    elif model.try_load():
        logger.warning(
            "Loaded an EMA/inference checkpoint. Raw resume weights were not found."
        )

    is_edm2 = model_name in (ModelType.edm2, ModelType.edm2_pretrained)

    trainer = Trainer(
        model=model,
        vae=vae,
        schedules=schedules,
        config=TrainingConfig(
            time_sampler=timesampler_config,
            epochs=n_epochs,
            checkpoint_interval_steps=checkpoint_interval,
            checkpoint_snapshot_percent=checkpoint_snapshot_percent,
            use_amp=use_amp,
            use_bf16=use_bf16,
            uncertainty_loss=uncertainty_loss,
            use_ema=use_ema,
            ema_halflife_ratio=ema_halflife_ratio,
            ema_warmup_ratio=ema_warmup_ratio,
            lr=1e-2 if is_edm2 else 1e-4,
            dataset=dataset.value,
            enable_wandb=enable_wandb,
        ),
    )
    trainer.load_checkpoint(resume_trainer_path)

    dataloader = get_dataloader(
        batch_size=batch_size,
        dataset_config=dataset_config,
        num_workers=n_workers,
    )

    trainer.train(dataloader)

    logger.success("Training completed successfully")

    if distributed.get_rank() == 0 and (enable_wandb or distributed.is_distributed()):
        wandb.finish()
