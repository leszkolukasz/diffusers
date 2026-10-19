"""Helpers for reading training metadata from model checkpoints."""

from loguru import logger

from src.config import DatasetType, ScheduleType
from src.model.base import PredictorMetadata


def dataset_from_metadata(metadata: dict) -> DatasetType | None:
    value = metadata.get(PredictorMetadata.Dataset)
    if not value:
        return None
    try:
        return DatasetType(value)
    except ValueError:
        logger.warning(f"Unknown dataset in model metadata: {value}")
        return None


def schedule_from_metadata(metadata: dict) -> ScheduleType | None:
    alpha = metadata.get(PredictorMetadata.AlphaSchedule)
    sigma = metadata.get(PredictorMetadata.SigmaSchedule)

    if alpha == "ConstantAlphaSchedule" and sigma == "LinearSigmaSchedule":
        return ScheduleType.edm
    if alpha == "LinearAlphaSchedule" and sigma == "LinearSigmaSchedule":
        return ScheduleType.linear
    if alpha == "CosineAlphaSchedule" and sigma == "CosineSigmaSchedule":
        return ScheduleType.cosine
    if alpha == "HuggingFaceDDPMAlphaSchedule":
        return ScheduleType.ddpm
    return None
