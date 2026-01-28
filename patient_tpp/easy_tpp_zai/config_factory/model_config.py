from __future__ import annotations

import os
from typing import Any

from easy_tpp.config_factory.model_config import Config, ModelConfig, ThinningConfig

from patient_tpp import project_data_dir


class ZaiTrainerConfig(Config):

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Config class."""
        self.seed = kwargs.get("seed", 9899)
        self.gpu = kwargs.get("gpu", -1)
        self.batch_size = kwargs.get("batch_size", 256)
        self.max_epoch = kwargs.get("max_epoch", 10)
        self.shuffle = kwargs.get("shuffle", False)
        self.optimizer = kwargs.get("optimizer", "adam")
        self.learning_rate = kwargs.get("learning_rate", 1.0e-3)
        self.valid_freq = kwargs.get("valid_freq", 1)
        self.use_tfb = kwargs.get("use_tfb", False)
        self.metrics = kwargs.get("metrics", ["acc", "rmse"])
        self.dither = kwargs.get("dither", False)
        self.shingle = kwargs.get("shingle", True)

    def get_yaml_config(self) -> dict[str, Any]:
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the trainer specs in dict format.
        """
        return {
            "seed": self.seed,
            "gpu": self.gpu,
            "batch_size": self.batch_size,
            "max_epoch": self.max_epoch,
            "shuffle": self.shuffle,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "valid_freq": self.valid_freq,
            "use_tfb": self.use_tfb,
            "metrics": self.metrics,
            "dither": self.dither,
            "shingle": self.shingle,
        }

    @staticmethod
    def parse_from_yaml_config(yaml_config: dict[str, Any]) -> ZaiTrainerConfig:
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            EasyTPP.TrainerConfig: Config class for trainer specs.
        """
        return ZaiTrainerConfig(**yaml_config)

    def copy(self) -> ZaiTrainerConfig:
        """Copy the config.

        Returns:
            EasyTPP.TrainerConfig: a copy of current config.
        """
        return ZaiTrainerConfig(
            batch_size=self.batch_size,
            max_epoch=self.max_epoch,
            shuffle=self.shuffle,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            valid_freq=self.valid_freq,
            use_tfb=self.use_tfb,
            metrics=self.metrics,
            dither=self.dither,
            shingle=self.shingle,
        )


class ZaiModelConfig(ModelConfig):
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Config class."""
        self.rnn_type = kwargs.get("rnn_type", "LSTM")
        self.hidden_size = kwargs.get("hidden_size", 32)
        self.time_emb_size = kwargs.get("time_emb_size", 16)
        self.num_layers = kwargs.get("num_layers", 2)
        self.num_heads = kwargs.get("num_heads", 2)
        self.num_event_types_pad = kwargs.get("num_event_types_pad", 68)
        self.sharing_param_layer = kwargs.get("sharing_param_layer", False)
        self.use_mc_samples = kwargs.get(
            "use_mc_samples", True
        )  # if using MC samples in computing log-likelihood
        self.loss_integral_num_sample_per_step = kwargs.get(
            "loss_integral_num_sample_per_step", 20
        )  # mc_num_sample_per_step
        self.dropout_rate = kwargs.get("dropout_rate", 0.0)
        self.use_ln = kwargs.get("use_ln", False)
        self.thinning = ThinningConfig.parse_from_yaml_config(kwargs.get("thinning"))
        self.is_training = kwargs.get("training", False)
        self.pad_token_id = kwargs.get("event_pad_index", None)
        self.model_id = kwargs.get("model_id", None)
        self.pretrained_model_dir = kwargs.get("pretrained_model_dir", None)
        self.pretrained_scalers_dir = kwargs.get("pretrained_scalers_dir", None)
        self.gpu = kwargs.get("gpu", -1)
        self.model_specs = kwargs.get("model_specs", {})

        # For invariant features
        self.pseudoage_bins = kwargs.get("pseudoage_bins", 20)
        self.pseudoage_embedding_size = kwargs.get("pseudoage_embedding_size", 4)
        self.gender_cardinality = kwargs.get("gender_cardinality", 4)
        self.gender_embedding_size = kwargs.get("gender_embedding_size", 2)
        self.race_cardinality = kwargs.get("race_cardinality", 5)
        self.race_embedding_size = kwargs.get("race_embedding_size", 3)
        ce_path = kwargs.get("clinical_embeddings_path", None)
        if ce_path is not None and not os.path.isabs(ce_path):
            ce_path = os.path.join(project_data_dir, ce_path)
        self.clinical_embeddings_path = ce_path
        self.clinical_embedding_size = kwargs.get("clinical_embedding_size", 128)

        # For numeric features
        self.numeric_bins = kwargs.get("numeric_bins", 5)

        self.return_intensities = kwargs.get("return_intensities", False)

    @staticmethod
    def parse_from_yaml_config(yaml_config: dict[str, Any]) -> ZaiModelConfig:
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            ZaiModelConfig: Config class for trainer specs.
        """
        return ZaiModelConfig(**yaml_config)

    def get_yaml_config(self) -> dict[str, Any]:
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the model config specs in dict format.
        """
        return {
            "rnn_type": self.rnn_type,
            "hidden_size": self.hidden_size,
            "time_emb_size": self.time_emb_size,
            "num_layers": self.num_layers,
            "sharing_param_layer": self.sharing_param_layer,
            "loss_integral_num_sample_per_step": self.loss_integral_num_sample_per_step,
            "dropout_rate": self.dropout_rate,
            "use_ln": self.use_ln,
            # for some models / cases we may not need to pass thinning config
            # e.g., for intensity-free model
            "thinning": (
                None if self.thinning is None else self.thinning.get_yaml_config()
            ),
            "event_pad_index": self.pad_token_id,
            "model_id": self.model_id,
            "pretrained_model_dir": self.pretrained_model_dir,
            "pretrained_scalers_dir": self.pretrained_scalers_dir,
            "gpu": self.gpu,
            "model_specs": self.model_specs,
            "pseudoage_bins": self.pseudoage_bins,
            "pseudoage_embedding_size": self.pseudoage_embedding_size,
            "gender_cardinality": self.gender_cardinality,
            "gender_embedding_size": self.gender_embedding_size,
            "race_cardinality": self.race_cardinality,
            "race_embedding_size": self.race_embedding_size,
            "clinical_embeddings_path": self.clinical_embeddings_path,
            "clinical_embedding_size": self.clinical_embedding_size,
            "numeric_bins": self.numeric_bins,
            "return_intensities": self.return_intensities,
        }

    def copy(self) -> ZaiModelConfig:
        """Copy the config.

        Returns:
            ModelConfig: a copy of current config.
        """
        return ZaiModelConfig(
            rnn_type=self.rnn_type,
            hidden_size=self.hidden_size,
            time_emb_size=self.time_emb_size,
            num_layers=self.num_layers,
            sharing_param_layer=self.sharing_param_layer,
            loss_integral_num_sample_per_step=self.loss_integral_num_sample_per_step,
            dropout_rate=self.dropout_rate,
            use_ln=self.use_ln,
            thinning=self.thinning,
            event_pad_index=self.pad_token_id,
            pretrained_model_dir=self.pretrained_model_dir,
            pretrained_scalers_dir=self.pretrained_scalers_dir,
            gpu=self.gpu,
            model_specs=self.model_specs,
            pseudoage_bins=self.pseudoage_bins,
            pseudoage_embedding_size=self.pseudoage_embedding_size,
            gender_cardinality=self.gender_cardinality,
            gender_embedding_size=self.gender_embedding_size,
            race_cardinality=self.race_cardinality,
            race_embedding_size=self.race_embedding_size,
            clinical_embeddings_path=self.clinical_embeddings_path,
            clinical_embedding_size=self.clinical_embedding_size,
            numeric_bins=self.numeric_bins,
            return_intensities=self.return_intensities,
        )
