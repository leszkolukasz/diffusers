from dataclasses import dataclass
from typing import Literal

import torch
from loguru import logger


@dataclass
class TimestepConfig:
    kind: Literal["discrete", "continuous"]
    # Should describe range of timesteps (e.g., 1000 for discrete, 1.0 for continuous).
    # Should not be set to smaller values like 0.95 to deal with generation problems
    # as it will be scaled back to 1.0 by denoisers.
    T: int | float


@dataclass
class Timestep:
    config: TimestepConfig
    steps: torch.Tensor

    def adapt(self, new_config: TimestepConfig) -> "Timestep":
        if self.config.kind == new_config.kind and self.config.T == new_config.T:
            return self

        if (
            (self.config.kind == "discrete")
            and (new_config.kind == "discrete")
            and (self.config.T > new_config.T)
        ):
            logger.warning(
                "Adapting from a larger discrete max_t to a smaller discrete max_t will probably crash the denoiser."
            )

        adapted_steps = self.steps.float() / float(self.config.T) * float(new_config.T)

        if new_config.kind == "discrete":
            adapted_steps = adapted_steps.long()

        return Timestep(config=new_config, steps=adapted_steps)

    def as_discrete(self, T: int) -> "Timestep":
        return self.adapt(TimestepConfig(kind="discrete", T=T))

    def as_continuous(self, T: float) -> "Timestep":
        return self.adapt(TimestepConfig(kind="continuous", T=T))

    def __len__(self):
        return self.steps.size(0)

    def __getitem__(self, idx) -> "Timestep":
        assert 0 <= idx < len(self), "Index out of range"
        return Timestep(config=self.config, steps=self.steps[idx : idx + 1])
