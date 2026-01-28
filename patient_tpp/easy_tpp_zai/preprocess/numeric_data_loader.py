from __future__ import annotations

import gzip
import json
from typing import Any, BinaryIO, Generator, TextIO

import numpy as np
from easy_tpp.preprocess.dataset import get_data_loader
from easy_tpp.utils import load_pickle, py_assert

from patient_tpp.easy_tpp_zai.config_factory.data_config import ZaiDataConfig
from patient_tpp.easy_tpp_zai.preprocess.data_loader import ZaiTPPDataLoader
from patient_tpp.easy_tpp_zai.preprocess.dataset import ZaiTPPDataset
from patient_tpp.easy_tpp_zai.preprocess.event_tokenizer import ZaiEventTokenizer


class ZaiTPPNumericDataLoader(ZaiTPPDataLoader):
    def __init__(
        self,
        data_config: ZaiDataConfig,
        numeric_bins: int = 5,
        num_numeric_event_types: int = 8,
        pad_token_id: int = 41,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataloader

        Args:
            data_config (ZaiDataConfig): data config.
        """
        super(ZaiTPPNumericDataLoader, self).__init__(data_config, **kwargs)
        self.numeric_bins = numeric_bins
        self.num_numeric_event_types = num_numeric_event_types
        self.pad_token_id = pad_token_id
        self.dither = (
            kwargs["dither"] if "dither" in kwargs and kwargs["dither"] else False
        )
        self.shingle = (
            kwargs["shingle"] if "shingle" in kwargs and kwargs["shingle"] else False
        )

    def build_input(
        self, source_dir: str, data_format: str, split: str
    ) -> dict[str, Any] | Generator[dict[str, Any]]:
        """Helper function to load and process dataset based on file format.

        Args:
            source_dir (str): Path to dataset directory.
            data_format (str): 'pkl', 'json', 'json-streaming', or 'json-streaming-compressed'.
            split (str): Dataset split, e.g., 'train', 'dev', 'test'.

        Returns:
            dict: Dictionary containing sequences of event times, types, and intervals, or generator of same
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
        py_assert(
            data["dim_process"] == self.num_event_types,
            ValueError,
            "Inconsistent d2im_process in different splits.",
        )

        source_data = data[split]
        return {
            "time_seqs": [[x["time_since_start"] for x in seq] for seq in source_data],
            "values": [[x["values"] for x in seq] for seq in source_data],
            "time_delta_seqs": [
                [x["time_since_last_event"] for x in seq] for seq in source_data
            ],
            "t_start": [x["t_start"] for x in source_data],
            "seq_index": [x["seq_index"] for x in source_data],
        }

    def eventify(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Discretize values and rewrite record in conventional TPP series format.
        """

        new_t = []
        type_seqs = []
        valmat = np.array(record["values"])
        poss = np.argwhere(valmat >= 0.0)
        for r, c in poss:
            type_seqs.append(
                int(
                    np.floor(valmat[r, c] * self.numeric_bins) + (c * self.numeric_bins)
                )
            )
            new_t.append(record["time_since_start"][r])

        return {
            "ptid": record["ptid"],
            "t_start": record["t_start"],
            "t_end": record["t_end"],
            "time_since_start": new_t,
            "time_since_last_event": np.ediff1d(
                np.concatenate([[record["t_start"]], new_t, [record["t_end"]]])
            ).astype(np.float32),
            "type_seqs": type_seqs,
            "seq_index": record["seq_index"],
        }

    def _build_streaming_json_input(
        self, source_dir: str, _split: str, compressed: bool = False
    ) -> Generator[dict[str, Any]]:
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
                d = self.eventify(json.loads(line))
                # First delta should be 0.
                if d["t_start"] == 0.0 and (len(d["time_since_start"]) > 0):
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
                            "type_seqs": d["type_seqs"][start_pos:],
                            "time_delta_seqs": list(deltas)[start_pos:],
                            "t_start": d["t_start"],
                            "seq_index": d["seq_index"],
                        }
                        yield d_shingle
                elif len(times) > 0:
                    d_prime = {
                        "time_seqs": list(times),
                        "type_seqs": d["type_seqs"],
                        "time_delta_seqs": list(deltas),
                        "t_start": d["t_start"],
                        "seq_index": d["seq_index"],
                    }
                    yield d_prime
                else:
                    yield {
                        "time_seqs": [],
                        "type_seqs": [],
                        "time_delta_seqs": [],
                        "t_start": 0,
                        "seq_index": -1,
                    }

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

        return {
            "time_seqs": data["time_since_start"],
            "values": data["values"],
            "time_delta_seqs": data["time_since_last_event"],
            "t_start": data["t_start"],
        }

    def train_loader(self, **kwargs: Any) -> ZaiTPPNumericDataLoader:
        """Return the train loader

        Returns:
            ZaiTPPNumericDataLoader: data loader for numeric training set.
        """
        return self.get_loader("numeric_train", **kwargs)

    def valid_loader(self, **kwargs: Any) -> ZaiTPPNumericDataLoader:
        """Return the valid loader

        Returns:
            ZaiTPPNumericDataLoader: data loader for numeric valid set.
        """
        return self.get_loader("numeric_dev", **kwargs)

    def test_loader(self, **kwargs: Any) -> ZaiTPPNumericDataLoader:
        """Return the test loader

        Returns:
            ZaiTPPNumericDataLoader: data loader for numeric test set.
        """
        return self.get_loader("numeric_test", **kwargs)

    def get_loader(
        self, split: str = "numeric_train", **kwargs: Any
    ) -> ZaiTPPNumericDataLoader:
        """Get the corresponding data loader.

        Args:
            split (str, optional): denote the train, valid and test set. Defaults to 'train'.
            num_event_types (int, optional): num of event types in the data. Defaults to None.

        Raises:
            NotImplementedError: the input of 'num_event_types' is inconsistent with the data.

        Returns:
            EasyTPP.DataLoader: the data loader for tpp data.
        """
        data_dir = self.data_config.get_data_dir(split)
        data = self.build_input(data_dir, self.data_config.data_format, split)

        dataset = ZaiTPPDataset(iter(data))
        tokenizer = ZaiEventTokenizer(
            self.data_config.data_specs, use_pad_token_id=self.pad_token_id
        )
        loader = get_data_loader(
            dataset,
            self.backend,
            tokenizer,
            batch_size=self.kwargs["batch_size"],
            shuffle=self.kwargs["shuffle"],
            **kwargs,
        )

        return loader
