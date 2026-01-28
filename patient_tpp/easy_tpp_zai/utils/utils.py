from typing import Any

import torch
from torch import Tensor


def mode_with_ignored_value(
    tensor: Tensor, ignored_value: Any, dim: int = -1
) -> Tensor:
    """
    Calculates the mode of a tensor along a specified dimension,
    ignoring a particular value.

    Args:
        tensor (torch.Tensor): The input tensor.
        ignored_value (int or float): The value to be ignored during mode calculation.
        dim (int, optional): The dimension along which to calculate the mode.
                             Defaults to -1 (the last dimension).

    Returns:
        torch.Tensor: The mode values. Returns NaN for empty tensor.
    """
    mask = tensor != ignored_value
    filtered_tensor = tensor[mask]
    if filtered_tensor.numel() == 0:
        return torch.tensor(float("nan"))

    # Filtered tensor is 1-d, so use dim 0
    mode_values, _ = torch.mode(filtered_tensor, dim=0)

    return mode_values
