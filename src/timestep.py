from dataclasses import dataclass
from typing import Literal, Self

import torch


@dataclass
class TimestepConfig:
    kind: Literal["discrete", "continuous"]
    max_value: int | float

@dataclass
class Timestep:
    config: TimestepConfig
    values: torch.Tensor

    def adapt(self, new_config: TimestepConfig) -> Self:
        if self.config.kind == new_config.kind and self.config.max_value == new_config.max_value:
            return self

        if self.config.kind == "discrete" and new_config.kind == "continuous":
            adapted_values = self.values.float() / self.config.max_value * new_config.max_value
        elif self.config.kind == "continuous" and new_config.kind == "discrete":
            adapted_values = (self.values / self.config.max_value * new_config.max_value).long()
        else:
            raise ValueError(f"Unsupported conversion from {self.config} to {new_config}")

        return Timestep(config=new_config, values=adapted_values)