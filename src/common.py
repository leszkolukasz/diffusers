import re
from typing import Any, cast

import torch
import torch.distributed
from diffusers import DDPMScheduler, UNet2DModel


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_filename_part(value: object, max_length: int = 80) -> str:
    """Turn a value into a filesystem-friendly filename component."""
    text = str(value).strip().replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:max_length] or "unknown"


def assert_type[T](val: Any, typ: type[T]) -> T:
    if not isinstance(val, typ):
        raise TypeError(f"Expexted type {typ}, got {type(val)}")
    return val


def load_unet_config(model_id: str, unet_class: type = UNet2DModel, **kwargs) -> dict:
    try:
        return cast(dict, unet_class.load_config(model_id, **kwargs))  # type: ignore
    except Exception:
        return cast(dict, unet_class.load_config(model_id, subfolder="unet", **kwargs))  # type: ignore


def load_unet_pretrained(
    model_id: str, unet_class: type = UNet2DModel, **kwargs
) -> UNet2DModel:
    try:
        return unet_class.from_pretrained(model_id, **kwargs)  # type: ignore
    except Exception:
        return unet_class.from_pretrained(model_id, subfolder="unet", **kwargs)  # type: ignore


def load_scheduler_config(
    model_id: str, scheduler_class: type = DDPMScheduler, **kwargs
) -> dict:
    try:
        return cast(dict, scheduler_class.load_config(model_id, **kwargs))  # type: ignore
    except Exception:
        return cast(
            dict,
            scheduler_class.load_config(model_id, subfolder="scheduler", **kwargs),  # type: ignore
        )
