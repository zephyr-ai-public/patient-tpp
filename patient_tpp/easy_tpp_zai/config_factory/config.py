from abc import abstractmethod
from typing import Any

from easy_tpp.utils import Registrable, logger
from omegaconf import OmegaConf


class ZaiConfig(Registrable):

    def save_to_yaml_file(self, config_dir: str) -> None:
        """Save the config into the yaml file 'config_dir'.

        Args:
            config_dir (str): Target filename.

        Returns:
        """
        yaml_config = self.get_yaml_config()
        OmegaConf.save(yaml_config, config_dir)

    @staticmethod
    def build_from_yaml_file(yaml_dir: str, **kwargs: Any) -> Any:
        """Load yaml config file from disk.

        Args:
            yaml_dir (str): Path of the yaml config file.

        Returns:
            EasyTPP.Config: Config object corresponding to config class.
        """
        config = OmegaConf.load(yaml_dir)
        pipeline_config = str(config.get("pipeline_config_id"))
        config_cls = ZaiConfig.by_name(pipeline_config)
        logger.critical(f"Load pipeline config class {config_cls.__name__}")
        return config_cls.parse_from_yaml_config(config, **kwargs)

    @abstractmethod
    def get_yaml_config(self) -> dict[str, Any]:
        """Get the yaml format config from self.

        Returns:
        """
        pass

    @staticmethod
    @abstractmethod
    def parse_from_yaml_config(yaml_config: dict[str, Any]) -> Any:
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            EasyTPP.Config: Config class for data.
        """
        pass

    @abstractmethod
    def copy(self) -> Any:
        """Get a same and freely modifiable copy of self.

        Returns:
        """
        pass

    def __str__(self) -> str:
        """Str representation of the config.

        Returns:
            str: str representation of the dict format of the config.
        """
        return str(self.get_yaml_config())

    def update(self, config: dict[str, Any]) -> Any:
        """Update the config.

        Args:
            config (dict): config dict.

        Returns:
            EasyTPP.Config: Config class for data.
        """
        logger.critical(f"Update config class {self.__class__.__name__}")
        return self.parse_from_yaml_config(config)

    def pop(self, key: str, default_var: Any) -> Any:
        """pop out the key-value item fsrom the config.

        Args:
            key (str): key name.
            default_var (Any): default value to pop.

        Returns:
            Any: value to pop.
        """
        # mypy error on this:
        # return vars(self).pop(key) or default_var
        return vars(self)[key] or default_var

    def get(self, key: str, default_var: Any) -> Any:
        """Retrieve the key-value item from the config.

        Args:
            key (str): key name.
            default_var (Any): falla

        Returns:
            Any: value to get.
        """
        return vars(self)[key] or default_var

    # Currently unused; triggers mypy error
    # def set(self, key: str, var_to_set: Any) -> None:
    #     """Set the key-value item from the config.

    #     Args:
    #         key (str): key name.
    #         var_to_set (Any): value

    #     Returns:
    #         None
    #     """
    #     vars(self)[key] = var_to_set
