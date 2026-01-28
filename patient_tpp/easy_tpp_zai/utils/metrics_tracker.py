import numpy as np
from easy_tpp.utils import MetricsTracker


class ZaiMetricsTracker(MetricsTracker):
    """Track and record the metrics."""

    def __init__(self) -> None:
        self.current_best = {
            "loglike": np.finfo(float).min,
            "acc": 0.0,
            "diff_ratio": 0.0,
            "rmse": np.finfo(float).max,
        }
        self.episode_best = {
            "loglike": np.nan,
            "acc": np.nan,
            "diff_ratio": np.nan,
            "rmse": np.nan,
        }

    def update_best(self, key: str, value: float, epoch: int) -> bool:
        """Update the recorder for the best metrics.

        Args:
            key (str): metrics key.
            value (float): metrics value.
            epoch (int): num of epoch.

        Raises:
            NotImplementedError: for unsupported keys

        Returns:
            bool: whether the recorder has been updated.
        """
        updated = False
        to_maximize = ("loglike", "acc", "diff_ratio")
        to_minimize = "rmse"
        if key in to_maximize:
            if value > self.current_best[key]:
                updated = True
                self.current_best[key] = value
                self.episode_best[key] = epoch
        elif key in to_minimize:
            if value < self.current_best[key]:
                updated = True
                self.current_best[key] = value
                self.episode_best[key] = epoch
        else:
            raise NotImplementedError(f"{key}")

        return updated
