from dataclasses import dataclass
from typing import Literal, Self

import torch


@dataclass
class TimestepConfig:
    kind: Literal["discrete", "continuous"]
    max_t: int | float

@dataclass
class Timestep:
    config: TimestepConfig
    steps: torch.Tensor

    def adapt(self, new_config: TimestepConfig) -> Self:
        if self.config.kind == new_config.kind and self.config.max_t == new_config.max_t:
            return self
        
        adapted_steps = self.steps.float() / float(self.config.max_t) * float(new_config.max_t)

        if new_config.kind == "discrete":
            adapted_steps = adapted_steps.long()

        return Timestep(config=new_config, steps=adapted_steps)
    
    def as_discrete(self, max_t) -> Self:
        return self.adapt(TimestepConfig(kind="discrete", max_t=max_t))

    def as_continuous(self) -> Self:
        return self.adapt(TimestepConfig(kind="continuous", max_t=1.0))

    def __len__(self):
        return self.steps.size(0)
    
    def __getitem__(self, idx) -> Self:
        assert 0 <= idx < len(self), "Index out of range"
        return Timestep(config=self.config, steps=self.steps[idx:idx+1])