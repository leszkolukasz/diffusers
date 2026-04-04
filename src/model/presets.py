from enum import Enum
from typing import Any


class ModelSize(str, Enum):
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


UNET_PRESETS: dict[ModelSize, dict[str, Any]] = {
    ModelSize.MICRO: {  # ~ 1M parameters
        "layers_per_block": 2,
        "block_out_channels": (16, 32, 64),
        "down_block_types": ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        "up_block_types": ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        "norm_num_groups": 8,
    },
    ModelSize.SMALL: {  # ~ 4M parameters
        "layers_per_block": 2,
        "block_out_channels": (32, 64, 128),
        "down_block_types": ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        "up_block_types": ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        "norm_num_groups": 32,
    },
    ModelSize.MEDIUM: {  # ~ 16M parameters
        "layers_per_block": 2,
        "block_out_channels": (64, 128, 256),
        "down_block_types": ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        "up_block_types": ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        "norm_num_groups": 32,
    },
    ModelSize.LARGE: {  # ~ 64M parameters
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
    ModelSize.MICRO: {  # ~ 1M parameters
        "model_channels": 16,
        "channel_mult": [1, 2, 4],
        "num_blocks": 2,
        "attn_resolutions": [16, 8],
    },
    ModelSize.SMALL: {  # ~ 4M parameters
        "model_channels": 32,
        "channel_mult": [1, 2, 4],
        "num_blocks": 2,
        "attn_resolutions": [16, 8],
    },
    ModelSize.MEDIUM: {  # ~ 16M parameters
        "model_channels": 64,
        "channel_mult": [1, 2, 4],
        "num_blocks": 2,
        "attn_resolutions": [16, 8],
    },
    ModelSize.LARGE: {  # ~ 64M parameters
        "model_channels": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_blocks": 3,
        "attn_resolutions": [16, 8],
    },
}
