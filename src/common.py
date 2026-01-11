from typing import Any, Type

import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_type[T](val: Any, typ: Type[T]) -> T:
    if not isinstance(val, typ):
        raise TypeError(f"Expexted type {typ}, got {type(val)}")
    return val
