from collections.abc import Iterator
from typing import Any

from torch.utils.data import IterableDataset


class ZaiTPPDataset(IterableDataset):
    def __init__(self, iterable: Iterator):
        super(ZaiTPPDataset, self).__init__()
        self.iterable = iterable
        self.fake_length = 10_000_000_000

    def __iter__(self) -> Iterator:
        return iter(self.iterable)

    def __len__(self) -> int:
        """
        We're usually streaming so return nonsense.
        """
        return self.fake_length

    def __getitem__(self, _idx: int) -> dict[str, Any]:
        """

        Args:
            idx: iteration index

        Returns:
            dict: a dict of time_seqs, time_delta_seqs, invars, and type_seqs element

        """
        return next(self.iterable)

    def __next__(self) -> None:
        raise Exception("ZaiTPPDataset __next__()")


class ZaiTPPNumericDataset(ZaiTPPDataset):
    def __init__(self, iterable: Iterator) -> None:
        super(ZaiTPPNumericDataset, self).__init__(iterable)
        self.time_seqs: list[int] = []
        self.time_delta_seqs: list[int] = []
        self.type_seqs: list[float] = []

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return dict(
            {
                "time_seqs": self.time_seqs[idx],
                "time_delta_seqs": self.time_delta_seqs[idx],
                "values": self.type_seqs[idx],
            }
        )
