from __future__ import annotations

import torch
from torch import Tensor


class MinMaxScaler:
    """
    A MinMax Scaler for PyTorch tensors, with fit and transform methods.
    Scales to the range [0, 1] by default.
    """

    def __init__(
        self, feature_range: tuple[float, float] = (0, 1), device: str = "cuda"
    ) -> None:
        self.feature_range = feature_range
        self.min_val = torch.empty()
        self.max_val = torch.empty()
        self.new_min = feature_range[0]
        self.new_max = feature_range[1]
        self.device = device

    def fit(self, tensor: Tensor) -> MinMaxScaler:
        """Computes the minimum and maximum to be used for later scaling."""
        self.min_val = tensor.min(dim=0, keepdim=True).values
        self.max_val = tensor.max(dim=0, keepdim=True).values
        return self

    def transform(self, tensor: Tensor) -> Tensor:
        """Applies the scaling transformation."""
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

        scale = torch.nan_to_num(
            (self.new_max - self.new_min) / (self.max_val - self.min_val), nan=1.0
        ).to(self.device)
        scaled_tensor = (tensor - self.min_val.to(self.device)) * scale + self.new_min
        return scaled_tensor

    def inverse_transform(self, scaled_tensor: Tensor) -> Tensor:
        """Unapplies the scaling transformation."""
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

        scale = (self.new_max - self.new_min) / (self.max_val - self.min_val).to(
            self.device
        )
        unscaled_tensor = ((scaled_tensor - self.new_min) / scale) + self.min_val.to(
            self.device
        )
        return unscaled_tensor

    def fit_transform(self, tensor: Tensor) -> Tensor:
        """Fits to data, then transforms it."""
        return self.fit(tensor).transform(tensor)
