from __future__ import annotations

import gzip
import json
from typing import Any, BinaryIO, Generator, TextIO, Union

import numpy as np
from easy_tpp.preprocess.data_loader import TPPDataLoader
from easy_tpp.preprocess.dataset import get_data_loader
from easy_tpp.utils import load_pickle, py_assert

from patient_tpp.easy_tpp_zai.config_factory.data_config import ZaiDataConfig
from patient_tpp.easy_tpp_zai.preprocess.dataset import ZaiTPPDataset
from patient_tpp.easy_tpp_zai.preprocess.event_tokenizer import ZaiEventTokenizer

_TIME_DITHER_EPSILON = 0.001


class ZaiTPPDataLoader(TPPDataLoader):
    def __init__(self, data_config: ZaiDataConfig, **kwargs: Any) -> None:
        """Initialize the dataloader

        Args:
            data_config (EasyTPP.DataConfig): data config.
            backend (str): backend engine, e.g., tensorflow or torch.
        """
        super(ZaiTPPDataLoader, self).__init__(data_config, **kwargs)
        self.dither = (
            kwargs["dither"] if "dither" in kwargs and kwargs["dither"] else False
        )
        self.shingle = (
            kwargs["shingle"] if "shingle" in kwargs and kwargs["shingle"] else False
        )

    def build_input(
        self, source_dir: str, data_format: str, split: str
    ) -> Union[dict[str, Any], Generator[dict[str, Any]], None, None]:
        """Helper function to load and process dataset based on file format.

        Args:
            source_dir (str): Path to dataset directory.
            data_format (str): 'pkl', 'json', 'json-streaming', or 'json-streaming-compressed'
            split (str): Dataset split, e.g., 'train', 'dev', 'test'.

        Returns:
            dict: Dictionary containing sequences of event times, types, and intervals, or generator of same.
        """

        if data_format == "pkl":
            return self._build_input_from_pkl(source_dir, split)
        elif data_format == "json":
            return self._build_input_from_json(source_dir, split)
        elif data_format == "json-streaming":
            return self._build_streaming_json_input(source_dir, split)
        elif data_format == "json-streaming-compressed":
            return self._build_streaming_json_input(source_dir, split, compressed=True)
        else:
            raise ValueError(f"Unsupported file format: {data_format}")

    def _build_input_from_pkl(self, source_dir: str, split: str) -> dict[str, Any]:
        """Load and process data from a pickle file.

        Args:
            source_dir (str): Path to the pickle file.
            split (str): Dataset split, e.g., 'train', 'dev', 'test'.

        Returns:
            dict: Dictionary with processed event sequences.
        """
        data = load_pickle(source_dir)
        source_data = data[split]
        py_assert(
            data["dim_process"] == self.num_event_types,
            ValueError,
            "Inconsistent dim_process in different splits.",
        )

        return {
            "time_seqs": [x["time_since_start"] for x in source_data],
            "type_seqs": [x["type_event"] for x in source_data],
            "time_delta_seqs": [x["time_since_last_event"] for x in source_data],
            "invars": [
                x.get("invar_dat", [np.nan, np.nan, np.nan]) for x in source_data
            ],
            "t_start": [x["t_start"] for x in source_data],
            "seq_index": [x["seq_index"] for x in source_data],
        }

    @classmethod
    def dither_timestamps(
        cls, time_seq: np.ndarray, delta_seq: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Avoid coincident event times by dithering a small amount.
        Reset t_start to first event time so deltas all begin with 0.
        """
        new_t = time_seq.copy().astype("float64")
        new_d = delta_seq.copy().astype("float64")
        t_start = new_t[0]
        t_end = new_t[-1] + new_d[-1]
        while (len(new_d) - np.count_nonzero(new_d)) > 0:
            new_t[np.where(new_d == 0)] += _TIME_DITHER_EPSILON
            new_d = np.ediff1d(np.concatenate([[t_start], new_t, [t_end]]))
        new_d[0] = 0
        assert len(new_t) == len(time_seq)
        assert len(new_d) == len(delta_seq)
        return new_t, new_d

    def _build_streaming_json_input(
        self, source_dir: str, _split: str, compressed: bool = False
    ) -> Generator[dict[str, Any], None, None]:
        self.sample_counter = 0
        self.shingle_sample_counter = 0

        def open_function(filename: str) -> TextIO | BinaryIO:
            return (
                gzip.open(filename, "rt", encoding="utf-8")
                if compressed
                else open(filename, "r")
            )

        with open_function(source_dir) as fobj:
            for line in fobj:
                d = json.loads(line)
                # First delta should be 0.
                if d["t_start"] == 0.0:
                    t_start = d["time_since_start"][0]
                    d["t_start"] = t_start
                    d["time_since_start"] = (
                        np.array(d["time_since_start"]) - t_start
                    ).tolist()
                    d["time_since_last_event"][0] = 0.0
                else:
                    self.sample_counter += 1
                times = np.array([])
                if self.dither:
                    if len(d["time_seqs"]) > 0:
                        times, deltas = self.dither_timestamps(
                            np.array(d["time_since_start"]),
                            np.array(d["time_since_last_event"]),
                        )
                else:
                    times, deltas = np.array(d["time_since_start"]), np.array(
                        d["time_since_last_event"]
                    )
                if (len(times) > self.data_config.data_specs.max_len) and self.shingle:
                    for start_pos in range(
                        len(times) - self.data_config.data_specs.max_len
                    ):
                        self.sample_counter += 1
                        self.shingle_sample_counter += 1
                        d_shingle = {
                            "time_seqs": list(times)[start_pos:],
                            "type_seqs": d["type_event"][start_pos:],
                            "time_delta_seqs": list(deltas)[start_pos:],
                            "t_start": d["t_start"],
                            "seq_index": d["seq_index"],
                        }
                        d_shingle["invars"] = (
                            d["invars"]
                            if self.data_config.data_specs.use_invariant_features
                            else None
                        )
                        yield d_shingle
                else:
                    d_prime = {
                        "time_seqs": list(times),
                        "type_seqs": d["type_event"],
                        "time_delta_seqs": list(deltas),
                        "t_start": d["t_start"],
                        "seq_index": d["seq_index"],
                    }
                    d_prime["invars"] = (
                        d["invars"]
                        if self.data_config.data_specs.use_invariant_features
                        else None
                    )
                    yield d_prime

    def _build_input_from_json(self, source_dir: str, split: str) -> dict[str, Any]:
        """Load and process data from a JSON file.

        Args:
            source_dir (str): Path to the JSON file or Hugging Face dataset name.
            split (str): Dataset split, e.g., 'train', 'dev', 'test'.

        Returns:
            dict: Dictionary with processed event sequences.
        """
        from datasets import load_dataset

        split_mapped = "validation" if split == "dev" else split
        if source_dir.endswith(".json"):
            data = load_dataset(
                "json", data_files={split_mapped: source_dir}, split=split_mapped
            )
        elif source_dir.startswith("easytpp"):
            data = load_dataset(source_dir, split=split_mapped)
        else:
            raise ValueError("Unsupported source directory format for JSON.")

        py_assert(
            data["dim_process"][0] == self.num_event_types,
            ValueError,
            "Inconsistent dim_process in different splits.",
        )

        invar_dat = (
            data["invars"]
            if self.data_config.data_specs.use_invariant_features
            else [None] * len(data["invars"])
        )
        return {
            "time_seqs": data["time_since_start"],
            "type_seqs": data["type_event"],
            "time_delta_seqs": data["time_since_last_event"],
            "invars": invar_dat,
            "t_start": data["t_start"],
            "seq_index": data["seq_index"],
        }

    def train_loader(self, **kwargs: Any) -> ZaiTPPDataLoader:
        """Return the train loader

        Returns:
            ZaiTPPDataLoader: data loader for train set.
        """
        return self.get_loader("train", **kwargs)

    def valid_loader(self, **kwargs: Any) -> ZaiTPPDataLoader:
        """Return the valid loader

        Returns:
            ZaiTPPDataLoader: data loader for valid set.
        """
        return self.get_loader("dev", **kwargs)

    def test_loader(self, **kwargs: Any) -> ZaiTPPDataLoader:
        """Return the test loader

        Returns:
            ZaiTPPDataLoader: data loader for test set.
        """
        return self.get_loader("test", **kwargs)

    def get_loader(self, split: str = "train", **kwargs: Any) -> ZaiTPPDataLoader:
        """Get the corresponding data loader.

        Args:
            split (str, optional): denote the train, valid and test set. Defaults to 'train'.
            num_event_types (int, optional): num of event types in the data. Defaults to None.

        Raises:
            NotImplementedError: the input of 'num_event_types' is inconsistent with the data.

        Returns:
            ZaiTPPDataLoader: data loader
        """
        data_dir = self.data_config.get_data_dir(split)
        data = self.build_input(data_dir, self.data_config.data_format, split)
        dataset = ZaiTPPDataset(iter(data))
        tokenizer = ZaiEventTokenizer(self.data_config.data_specs)
        loader = get_data_loader(
            dataset,
            self.backend,
            tokenizer,
            batch_size=self.kwargs["batch_size"],
            shuffle=self.kwargs["shuffle"],
            **kwargs,
        )

        return loader
