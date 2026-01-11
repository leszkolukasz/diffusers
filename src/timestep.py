from dataclasses import dataclass
from typing import Literal

import torch
from loguru import logger


@dataclass
class TimestepConfig:
    kind: Literal["discrete", "continuous"]
    max_t: int | float


@dataclass
class Timestep:
    config: TimestepConfig
    steps: torch.Tensor

    def adapt(self, new_config: TimestepConfig) -> "Timestep":
        if (
            self.config.kind == new_config.kind
            and self.config.max_t == new_config.max_t
        ):
            return self

        if (
            (self.config.kind == "discrete")
            and (new_config.kind == "discrete")
            and (self.config.max_t > new_config.max_t)
        ):
            logger.warning(
                "Adapting from a larger discrete max_t to a smaller discrete max_t will probably crash the denoiser."
            )

        adapted_steps = (
            self.steps.float() / float(self.config.max_t) * float(new_config.max_t)
        )

        if new_config.kind == "discrete":
            adapted_steps = adapted_steps.long()

        return Timestep(config=new_config, steps=adapted_steps)

    def as_discrete(self, max_t: int) -> "Timestep":
        return self.adapt(TimestepConfig(kind="discrete", max_t=max_t))

    def as_continuous(self, max_t: float) -> "Timestep":
        return self.adapt(TimestepConfig(kind="continuous", max_t=max_t))

    def __len__(self):
        return self.steps.size(0)

    def __getitem__(self, idx) -> "Timestep":
        assert 0 <= idx < len(self), "Index out of range"
        return Timestep(config=self.config, steps=self.steps[idx : idx + 1])
