from enum import Enum
from typing import Any


class ModelSize(str, Enum):
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    @classmethod
    def from_value(cls, value: str) -> "ModelSize":
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Invalid ModelSize value: {value}")


UNET_PRESETS: dict[ModelSize, dict[str, Any]] = {
    ModelSize.MICRO: {  # 1.6M params
        "layers_per_block": 2,
        "block_out_channels": (16, 32, 64, 64),
        "down_block_types": (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        "up_block_types": ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        "norm_num_groups": 8,
    },
    ModelSize.SMALL: {  # 6.3M params
        "layers_per_block": 2,
        "block_out_channels": (32, 64, 128, 128),
        "down_block_types": (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        "up_block_types": ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        "norm_num_groups": 32,
    },
    ModelSize.MEDIUM: {  # 26.6M params
        "layers_per_block": 2,
        "block_out_channels": (64, 128, 256, 256),
        "down_block_types": (
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        "up_block_types": ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        "norm_num_groups": 32,
    },
    ModelSize.LARGE: {  # 63.1M params
        "layers_per_block": 2,
        "block_out_channels": (64, 128, 256, 512),
        "down_block_types": (
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        "up_block_types": ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        "norm_num_groups": 32,
    },
}


EDM2_PRESETS: dict[ModelSize, dict[str, Any]] = {
    ModelSize.MICRO: {  # 1.6M params
        "model_channels": 16,
        "channel_mult": [1, 2, 4, 4],
        "num_blocks": 2,
        "attn_resolutions": [16],
    },
    ModelSize.SMALL: {  # 4.8M params
        "model_channels": 32,
        "channel_mult": [1, 2, 2, 4],
        "num_blocks": 2,
        "attn_resolutions": [16],
    },
    ModelSize.MEDIUM: {  # 27M params
        "model_channels": 64,
        "channel_mult": [1, 2, 4, 4],
        "num_blocks": 2,
        "attn_resolutions": [32, 16],
    },
    ModelSize.LARGE: {  # 64M params
        "model_channels": 96,
        "channel_mult": [1, 2, 3, 4],
        "num_blocks": 3,
        "attn_resolutions": [32, 16],
    },
}
