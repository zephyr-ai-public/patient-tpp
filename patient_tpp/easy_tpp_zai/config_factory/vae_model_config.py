from __future__ import annotations

from typing import Any

from patient_tpp.easy_tpp_zai.config_factory.model_config import ZaiModelConfig


class ZaiVaeModelConfig(ZaiModelConfig):
    """
    Config pertaining to the variational autoencoder used in patient data synthesis.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Config class."""
        self.pretrained_model_dir = kwargs.get("pretrained_model_dir", "")
        self.pretrained_scalers_dir = kwargs.get("pretrained_scalers_dir", "")
        self.time_deltas_hidden_dim = kwargs.get("time_deltas_hidden_dim", 256)
        self.num_time_deltas_hidden_dim = kwargs.get("num_time_deltas_hidden_dim", 256)
        self.type_seqs_hidden_dim = kwargs.get("type_seqs_hidden_dim", 256)
        self.num_type_seqs_hidden_dim = kwargs.get("num_type_seqs_hidden_dim", 256)
        self.invars_hidden_dims = kwargs.get("invars_hidden_dims", (128, 64, 64))
        self.pad_mask_true_hidden_dim = kwargs.get("pad_mask_true_hidden_dim", 256)
        self.num_pad_mask_true_hidden_dim = kwargs.get("pad_mask_true_hidden_dim", 256)
        # self.num_event_types = kwargs.get("num_event_types", 68)
        self.num_sex_classes = kwargs.get("num_sex_classes", 73)
        self.num_race_classes = kwargs.get("num_race_classes", 42)
        self.latent_dim = kwargs.get("latent_dim", 128)
        self.max_epoch = kwargs.get("max_epoch", 10)

    def get_yaml_config(self) -> dict[str, Any]:
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the trainer specs in dict format.
        """
        return {
            "pretrained_model_dir": self.pretrained_model_dir,
            "pretrained_scalers_dir": self.pretrained_scalers_dir,
            "time_deltas_hidden_dim": self.time_deltas_hidden_dim,
            "type_seqs_hidden_dim": self.type_seqs_hidden_dim,
            "invars_hidden_dims": self.invars_hidden_dims,
            "pad_mask_true_hidden_dim": self.pad_mask_true_hidden_dim,
            # 'num_event_types': self.num_event_types,
            "num_sex_classes": self.num_sex_classes,
            "num_race_classes": self.num_race_classes,
            "latent_dim": self.latent_dim,
            "max_epoch": self.max_epoch,
        }

    @staticmethod
    def parse_from_yaml_config(yaml_config: dict[str, Any]) -> ZaiVaeModelConfig:
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            EasyTPP.ZaiVaeModelConfig: Config class for autoencoder specs.
        """
        return ZaiVaeModelConfig(**yaml_config)

    def copy(self) -> ZaiVaeModelConfig:
        """Copy the config.

        Returns:
            EasyTPP.ZaeVaeModelConfig: a copy of current config.
        """
        return ZaiVaeModelConfig(
            pretrained_model_dir=self.pretrained_model_dir,
            pretrained_scalers_dir=self.pretrained_scalers_dir,
            time_deltas_hidden_dim=self.time_deltas_hidden_dim,
            type_seqs_hidden_dim=self.type_seqs_hidden_dim,
            invars_hidden_dims=self.invars_hidden_dims,
            pad_mask_true_hidden_dim=self.pad_mask_true_hidden_dim,
            # num_event_types=self.num_event_types,
            num_sex_classes=self.num_sex_classes,
            num_race_classes=self.num_race_classes,
            latent_dim=self.latent_dim,
            max_epoch=self.max_epoch,
        )
