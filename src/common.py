from typing import Any, Type, cast
from diffusers import UNet2DModel, ModelMixin, DDPMScheduler


import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_type[T](val: Any, typ: Type[T]) -> T:
    if not isinstance(val, typ):
        raise TypeError(f"Expexted type {typ}, got {type(val)}")
    return val


def load_unet_config(model_id: str, unet_class: Type = UNet2DModel, **kwargs) -> dict:
    try:
        return cast(dict, unet_class.load_config(model_id, **kwargs))  # type: ignore
    except Exception:
        return cast(dict, unet_class.load_config(model_id, subfolder="unet", **kwargs))  # type: ignore


def load_unet_pretrained(
    model_id: str, unet_class: Type = UNet2DModel, **kwargs
) -> UNet2DModel:
    try:
        return unet_class.from_pretrained(model_id, **kwargs)  # type: ignore
    except Exception:
        return unet_class.from_pretrained(model_id, subfolder="unet", **kwargs)  # type: ignore


def load_scheduler_config(
    model_id: str, scheduler_class: Type = DDPMScheduler, **kwargs
) -> dict:
    try:
        return cast(dict, scheduler_class.load_config(model_id, **kwargs))  # type: ignore
    except Exception:
        return cast(
            dict,
            scheduler_class.load_config(model_id, subfolder="scheduler", **kwargs),  # type: ignore
        )
