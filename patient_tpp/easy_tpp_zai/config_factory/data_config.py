from __future__ import annotations

import os
from typing import Any, Optional

from easy_tpp.config_factory.data_config import DataConfig, DataSpecConfig

from patient_tpp import project_data_dir
from patient_tpp.easy_tpp_zai.config_factory.config import ZaiConfig


class ZaiDataSpecConfig(DataSpecConfig):
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Config class."""
        self.num_event_types = kwargs.get("num_event_types")
        self.num_sex_classes = kwargs.get("num_sex_classes")
        self.num_race_classes = kwargs.get("num_race_classes")
        self.num_numeric_event_types = kwargs.get("num_numeric_event_types", 42)
        self.pad_token_id = kwargs.get("pad_token_id")
        self.numeric_pad_token_id = kwargs.get("numeric_pad_token_id")
        self.padding_side = kwargs.get("padding_side")
        self.truncation_side = kwargs.get("truncation_side")
        self.padding_strategy = kwargs.get("padding_strategy")
        self.max_len = kwargs.get("max_len")
        self.numeric_max_len = kwargs.get("numeric_max_len")
        self.truncation_strategy = kwargs.get("truncation_strategy")
        self.model_input_names = kwargs.get("model_input_names")
        self.use_invariant_features = kwargs.get("use_invariant_features")
        self.use_numeric_features = kwargs.get("use_numeric_features")

        if self.padding_side is not None and self.padding_side not in ["right", "left"]:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
            )

        if self.truncation_side is not None and self.truncation_side not in [
            "right",
            "left",
        ]:
            raise ValueError(
                f"Truncation side should be selected between 'right' and 'left', current value: {self.truncation_side}"
            )

    def get_yaml_config(self) -> dict[str, Any]:
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the data specs in dict format.
        """
        return {
            "num_event_types": self.num_event_types,
            "num_sex_classes": self.num_sex_classes,
            "num_race_classes": self.num_race_classes,
            "num_numeric_event_types": self.num_numeric_event_types,
            "pad_token_id": self.pad_token_id,
            "numeric_pad_token_id": self.numeric_pad_token_id,
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side,
            "padding_strategy": self.padding_strategy,
            "truncation_strategy": self.truncation_strategy,
            "max_len": self.max_len,
            "numeric_max_len": self.numeric_max_len,
            "use_invariant_features": self.use_invariant_features,
            "use_numeric_features": self.use_numeric_features,
        }

    @staticmethod
    def parse_from_yaml_config(yaml_config: dict[str, Any]) -> ZaiDataSpecConfig:
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            ZaiDataSpecConfig: Config class for data specs.
        """
        return ZaiDataSpecConfig(**yaml_config)

    def copy(self) -> ZaiDataSpecConfig:
        """Copy the config.

        Returns:
            ZaiDataSpecConfig: a copy of current config.
        """
        return ZaiDataSpecConfig(
            num_event_types=self.num_event_types,
            num_sex_classes=self.num_sex_classes,
            num_race_classes=self.num_race_classes,
            num_numeric_event_types=self.num_numeric_event_types,
            event_pad_index=self.pad_token_id,
            padding_side=self.padding_side,
            truncation_side=self.truncation_side,
            padding_strategy=self.padding_strategy,
            truncation_strategy=self.truncation_strategy,
            max_len=self.max_len,
            numeric_max_len=self.numeric_max_len,
            use_invariant_features=self.use_invariant_features,
            use_numeric_features=self.use_numeric_features,
        )


@ZaiConfig.register("zai_data_config")
class ZaiDataConfig(DataConfig):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        numeric_train_dir: str,
        numeric_valid_dir: str,
        numeric_test_dir: str,
        data_format: str,
        specs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the DataConfig object.

        Args:
            train_dir (str): dir of training set.
            valid_dir (str): dir of validation set.
            test_dir (str): dir of test set.
            numeric_train_dir (str): dir of numeric training set.
            numeric_valid_dir (str): dir of numeric validation set.
            numeric_test_dir (str): dir of numeric test set.
            specs (dict, optional): specs of dataset. Defaults to None.
        """
        self.train_dir = (
            train_dir
            if os.path.isabs(train_dir)
            else os.path.join(project_data_dir, train_dir)
        )
        self.valid_dir = (
            valid_dir
            if os.path.isabs(valid_dir)
            else os.path.join(project_data_dir, valid_dir)
        )
        self.test_dir = (
            test_dir
            if os.path.isabs(test_dir)
            else os.path.join(project_data_dir, test_dir)
        )
        self.numeric_train_dir = (
            numeric_train_dir
            if os.path.isabs(numeric_train_dir)
            else os.path.join(project_data_dir, numeric_train_dir)
        )
        self.numeric_valid_dir = (
            numeric_valid_dir
            if os.path.isabs(numeric_valid_dir)
            else os.path.join(project_data_dir, numeric_valid_dir)
        )
        self.numeric_test_dir = (
            numeric_test_dir
            if os.path.isabs(numeric_test_dir)
            else os.path.join(project_data_dir, numeric_test_dir)
        )
        self.data_specs = specs or ZaiDataSpecConfig()
        self.data_format = (
            train_dir.split(".")[-1] if data_format is None else data_format
        )

    def get_yaml_config(self) -> dict[str, Any]:
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the data in dict format.
        """
        return {
            "train_dir": self.train_dir,
            "valid_dir": self.valid_dir,
            "test_dir": self.test_dir,
            "numeric_train_dir": self.numeric_train_dir,
            "numeric_valid_dir": self.numeric_valid_dir,
            "numeric_test_dir": self.numeric_test_dir,
            "data_format": self.data_format,
            "data_specs": self.data_specs.get_yaml_config(),
        }

    @staticmethod
    def parse_from_yaml_config(yaml_config: dict[str, Any]) -> ZaiDataConfig:
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            EasyTPP.ZaiDataConfig: Config class for data.
        """
        raw_data_specs = (
            {} if yaml_config.get("data_specs") is None else yaml_config["data_specs"]
        )
        return ZaiDataConfig(
            train_dir=str(yaml_config.get("train_dir")),
            valid_dir=str(yaml_config.get("valid_dir")),
            test_dir=str(yaml_config.get("test_dir")),
            numeric_train_dir=str(yaml_config.get("numeric_train_dir")),
            numeric_valid_dir=str(yaml_config.get("numeric_valid_dir")),
            numeric_test_dir=str(yaml_config.get("numeric_test_dir")),
            data_format=str(yaml_config.get("data_format")),
            specs=ZaiDataSpecConfig.parse_from_yaml_config(raw_data_specs),
        )

    def copy(self) -> ZaiDataConfig:
        """Copy the config.

        Returns:
            EasyTPP.ZaiDataConfig: a copy of current config.
        """
        return ZaiDataConfig(
            train_dir=self.train_dir,
            valid_dir=self.valid_dir,
            test_dir=self.test_dir,
            numeric_train_dir=self.numeric_train_dir,
            numeric_valid_dir=self.numeric_valid_dir,
            numeric_test_dir=self.numeric_test_dir,
            data_format=self.data_format,
            specs=self.data_specs,
        )

    def get_data_dir(self, split: str) -> str:
        """Get the dir of the source raw data.

        Args:
            split (str): dataset split notation, 'train', 'dev' or 'valid', 'test'.

        Returns:
            str: dir of the source raw data file.
        """
        split = split.lower()
        if split == "train":
            return self.train_dir
        elif split in ("dev", "valid"):
            return self.valid_dir
        elif split == "test":
            return self.test_dir
        elif split == "numeric_train":
            return self.numeric_train_dir
        elif split == "numeric_test":
            return self.numeric_test_dir
        elif split in ("numeric_valid", "numeric_dev"):
            return self.numeric_valid_dir
        else:
            return self.numeric_test_dir
