from __future__ import annotations

import copy
import os
from typing import Any, Callable, Optional

from easy_tpp.config_factory.config import Config
from easy_tpp.config_factory.model_config import BaseConfig
from easy_tpp.utils import (
    DefaultRunnerConfig,
    MetricsHelper,
    RunnerPhase,
    create_folder,
    get_stage,
    get_unique_id,
    is_torch_available,
    is_torch_gpu_available,
    logger,
    py_assert,
)
from easy_tpp.utils.const import Backend

from patient_tpp.easy_tpp_zai.config_factory.data_config import ZaiDataConfig
from patient_tpp.easy_tpp_zai.config_factory.model_config import (
    ZaiModelConfig,
    ZaiTrainerConfig,
)
from patient_tpp.easy_tpp_zai.config_factory.vae_model_config import ZaiVaeModelConfig


@Config.register("zai_runner_config")
class ZaiRunnerConfig(Config):
    def __init__(
        self,
        base_config: BaseConfig,
        model_config: ZaiModelConfig,
        data_config: ZaiDataConfig,
        trainer_config: ZaiTrainerConfig,
        vae_model_config: Optional[ZaiVaeModelConfig] = None,
        save_me: bool = True,
    ) -> None:
        """Initialize the Config class.

        Args:
            base_config (EasyTPP.BaseConfig): BaseConfig object.
            model_config (EasyTPP.ZaiModelConfig): ZaiModelConfig object.
            data_config (EasyTPP.ZaiDataConfig): ZaiDataConfig object.
            trainer_config (EasyTPP.ZaiTrainerConfig): ZaiTrainerConfig object.
            vae_model_config (EasyTPP.ZaiVaeModelConfig: ZaiVaeModelConfig object (for data synthesis).
        """
        self.data_config = data_config
        self.model_config = model_config
        self.base_config = base_config
        self.trainer_config = trainer_config
        self.vae_model_config = vae_model_config

        self.ensure_valid_config()

        if save_me:
            self.update_config()
            # save the complete config
            save_dir = self.base_config.specs["output_config_dir"]
            self.save_to_yaml_file(save_dir)

            logger.info(f"I saved the config to {save_dir}")

    def get_yaml_config(self) -> dict[str, Any]:
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the runner config in dict format.
        """
        return {
            "data_config": self.data_config.get_yaml_config(),
            "base_config": self.base_config.get_yaml_config(),
            "model_config": self.model_config.get_yaml_config(),
            "trainer_config": self.trainer_config.get_yaml_config(),
            "vae_model_config": (
                self.vae_model_config.get_yaml_config()
                if self.vae_model_config
                else None
            ),
        }

    @staticmethod
    def parse_from_yaml_config(
        yaml_config: dict[str, Any], **kwargs: Any
    ) -> ZaiRunnerConfig:
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            ZaiRunnerConfig: Config object for runner specs.
        """
        direct_parse = kwargs.get("direct_parse", False)
        if not direct_parse:
            exp_id = kwargs.get("experiment_id")
            yaml_exp_config = yaml_config[str(exp_id)]
            dataset_id = yaml_exp_config.get("base_config").get("dataset_id")
            if dataset_id is None:
                dataset_id = DefaultRunnerConfig.DEFAULT_DATASET_ID
            try:
                yaml_data_config = yaml_config["data"][dataset_id]
            except KeyError as e:
                raise RuntimeError(
                    "dataset_id={} is not found in config.".format(dataset_id)
                ) from e

            data_config = ZaiDataConfig.parse_from_yaml_config(yaml_data_config)
            # add exp id to base config
            yaml_exp_config.get("base_config").update(exp_id=exp_id)

        else:
            yaml_exp_config = yaml_config
            raw_data_config = (
                {}
                if yaml_config.get("data_config") is None
                else yaml_config["data_config"]
            )

            data_config = ZaiDataConfig.parse_from_yaml_config(
                raw_data_config["data_config"]
            )

        base_config = BaseConfig.parse_from_yaml_config(
            yaml_exp_config.get("base_config")
        )
        model_config = ZaiModelConfig.parse_from_yaml_config(
            yaml_exp_config.get("model_config")
        )
        vae_model_config: Optional[ZaiVaeModelConfig] = None
        if "vae_model_config" in yaml_exp_config:
            vae_model_config = ZaiVaeModelConfig.parse_from_yaml_config(
                yaml_exp_config.get("vae_model_config")
            )
        trainer_config = ZaiTrainerConfig.parse_from_yaml_config(
            yaml_exp_config.get("trainer_config")
        )

        return ZaiRunnerConfig(
            data_config=data_config,
            base_config=base_config,
            model_config=model_config,
            trainer_config=trainer_config,
            vae_model_config=vae_model_config,
            save_me=kwargs.get("save_me", True),
        )

    def ensure_valid_config(self) -> None:
        """Do some sanity check about the config, to avoid conflicts in settings."""

        if self.base_config.stage != RunnerPhase.TRAIN:
            # during testing we dont do shuffle by default
            self.trainer_config.shuffle = False

            # during testing we dont apply tfb by default
            self.trainer_config.use_tfb = False

    def update_config(self) -> None:
        """Updated config dict."""
        model_folder_name = get_unique_id()

        log_folder = create_folder(self.base_config.base_dir, model_folder_name)
        model_folder = create_folder(log_folder, "models")
        eval_folder = create_folder(log_folder, "eval")

        self.base_config.specs["log_folder"] = log_folder
        self.base_config.specs["saved_model_dir"] = os.path.join(
            model_folder, "saved_model"
        )
        self.base_config.specs["saved_log_dir"] = os.path.join(log_folder, "log")
        self.base_config.specs["output_config_dir"] = os.path.join(
            log_folder, f"{self.base_config.exp_id}_output.yaml"
        )
        self.base_config.specs["eval_dir"] = eval_folder

        if self.trainer_config.use_tfb:
            self.base_config.specs["tfb_train_dir"] = create_folder(
                log_folder, "tfb_train"
            )
            self.base_config.specs["tfb_valid_dir"] = create_folder(
                log_folder, "tfb_valid"
            )

        current_stage = get_stage(self.base_config.stage)
        is_training = current_stage == RunnerPhase.TRAIN
        self.model_config.is_training = is_training
        self.model_config.gpu = self.trainer_config.gpu

        # update the dataset config => model config
        self.model_config.num_event_types = self.data_config.data_specs.num_event_types
        self.model_config.pad_token_id = self.data_config.data_specs.pad_token_id
        self.model_config.max_len = self.data_config.data_specs.max_len
        self.model_config.numeric_max_len = self.data_config.data_specs.numeric_max_len

        # update base config => model config
        model_id = self.base_config.model_id
        self.model_config.model_id = model_id

        if (
            self.base_config.model_id == "ODETPP"
            and self.base_config.backend == Backend.TF
        ):
            py_assert(
                self.data_config.data_specs.padding_strategy == "max_length",
                ValueError,
                "For ODETPP in TensorFlow, we must pad all sequence to the same length (max len of the sequences)!",
            )

        run = current_stage
        device = "GPU" if self.trainer_config.gpu >= 0 else "CPU"

        py_assert(
            is_torch_available(),
            ValueError,
            f"Backend {self.base_config.backend} is not supported in the current environment yet!",
        )

        if device != "CPU":
            py_assert(
                is_torch_gpu_available(),
                ValueError,
                "Torch cuda is not supported in the current environment yet!",
            )

        critical_msg = (
            "{run} model {model_name} using {device} with {tf_torch} backend".format(
                run=run,
                model_name=model_id,
                device=device,
                tf_torch=self.base_config.backend,
            )
        )

        logger.critical(critical_msg)

    def get_metric_functions(self) -> Callable:
        return MetricsHelper.get_metrics_callback_from_names(
            self.trainer_config.metrics
        )

    def get_metric_direction(self, metric_name: str = "rmse") -> Optional[str]:
        return MetricsHelper.get_metric_direction(metric_name)

    def copy(self) -> ZaiRunnerConfig:
        """Copy the config.

        Returns:
            RunnerConfig: a copy of current config.
        """
        return ZaiRunnerConfig(
            base_config=copy.deepcopy(self.base_config),
            model_config=copy.deepcopy(self.model_config),
            data_config=copy.deepcopy(self.data_config),
            trainer_config=copy.deepcopy(self.trainer_config),
            vae_model_config=copy.deepcopy(self.vae_model_config),
        )
