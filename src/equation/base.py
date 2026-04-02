from abc import ABC

from loguru import logger

from src.model import Predictor, PredictorMetadata
from src.schedule import ScheduleGroup


class Equation(ABC):
    model: Predictor
    schedules: ScheduleGroup

    def __init__(self, model: Predictor, schedules: ScheduleGroup):
        self.model = model
        self.schedules = schedules

        schedule_meta_checks = (
            (PredictorMetadata.AlphaSchedule, "alpha"),
            (PredictorMetadata.SigmaSchedule, "sigma"),
            (PredictorMetadata.EtaSchedule, "eta"),
        )

        for meta_key, schedule_attr in schedule_meta_checks:
            trained_schedule_meta = model.metadata.get(meta_key, None)
            current_schedule = getattr(schedules, schedule_attr, None)

            if trained_schedule_meta is None or current_schedule is None:
                continue

            current_schedule_name = current_schedule.__class__.__name__
            if current_schedule_name != trained_schedule_meta:
                logger.warning(
                    f"Model was trained with {schedule_attr} schedule '{trained_schedule_meta}', "
                    f"but current schedule is '{current_schedule_name}'"
                )
