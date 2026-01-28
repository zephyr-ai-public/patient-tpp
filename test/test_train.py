import os
from test import test_root_dir

import pytest

from patient_tpp.easy_tpp_zai.config_factory.config import ZaiConfig
from patient_tpp.easy_tpp_zai.runner.tpp_runner import ZaiTPPRunner


class TestTrain:
    @pytest.fixture
    def test_config_file(self):
        return os.path.join(test_root_dir, "config.test.yaml")

    def test_train(self, test_config_file) -> None:
        config = ZaiConfig.build_from_yaml_file(
            test_config_file, experiment_id="AttNHP_zai_train"
        )

        model_runner = ZaiTPPRunner.build_from_config(config)

        model_runner.run()

    def test_gen(self, test_config_file) -> None:
        config = ZaiConfig.build_from_yaml_file(
            test_config_file, experiment_id="AttNHP_zai_gen"
        )

        model_runner = ZaiTPPRunner.build_from_config(config)

        model_runner.run()
